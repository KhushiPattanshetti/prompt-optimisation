from pathlib import Path
import os
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    @staticmethod
    def _resolve_compute_dtype(device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32

        index = device.index if device.index is not None else 0
        major, _ = torch.cuda.get_device_capability(index)
        if major >= 8:
            return torch.bfloat16
        return torch.float16

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            target_index = self.device.index if self.device.index is not None else 0
            device_map = {"": target_index}
        else:
            device_map = "auto"

        compute_dtype = self._resolve_compute_dtype(self.device)

        logger.info(
            "policy_model_init | model=%s | requested_device=%s | device_map=%s | compute_dtype=%s | checkpoint=%s",
            model_name,
            self.device,
            device_map,
            compute_dtype,
            checkpoint_path,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.training_max_length = int(os.environ.get("RL_POLICY_MAX_LENGTH", "1024"))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        base_model.config.use_cache = False

        if checkpoint_path is not None:
            self.model = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, peft_config)

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        last_hidden_states = outputs.hidden_states[-1]
        return token_log_probs, last_hidden_states, logits

    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_log_probs, _, _ = self.forward(input_ids, attention_mask)
        return token_log_probs.sum(dim=-1)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_max_length,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def tokenize_with_action_mask(
        self,
        original_texts: list[str],
        rewritten_texts: list[str],
    ) -> dict[str, torch.Tensor]:
        if len(original_texts) != len(rewritten_texts):
            raise ValueError("original_texts and rewritten_texts must have identical length")

        max_len = max(int(self.training_max_length), 2)
        separator_ids = self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        if self.tokenizer.bos_token_id is not None:
            fallback_prefix_token = int(self.tokenizer.bos_token_id)
        elif self.tokenizer.eos_token_id is not None:
            fallback_prefix_token = int(self.tokenizer.eos_token_id)
        else:
            fallback_prefix_token = int(pad_token_id)

        input_id_rows: list[list[int]] = []
        action_mask_rows: list[list[float]] = []
        valid_actions: list[bool] = []
        rewrite_span_starts: list[int] = []
        rewrite_span_ends: list[int] = []
        tokenized_lengths: list[int] = []

        for original, rewritten in zip(original_texts, rewritten_texts):
            original_ids = self.tokenizer(original, add_special_tokens=False)["input_ids"]
            rewritten_ids = self.tokenizer(rewritten, add_special_tokens=False)["input_ids"]

            # Preserve rewritten tokens first and truncate prefix context when needed.
            max_rewrite_tokens = max_len - 1
            if len(rewritten_ids) > max_rewrite_tokens:
                rewritten_ids = rewritten_ids[:max_rewrite_tokens]

            prefix_budget = max_len - len(rewritten_ids)
            prefix_ids = (original_ids + separator_ids)[: max(prefix_budget, 0)]

            if not prefix_ids:
                prefix_ids = [fallback_prefix_token]
                if len(prefix_ids) + len(rewritten_ids) > max_len:
                    rewritten_ids = rewritten_ids[: max_len - len(prefix_ids)]

            combined_ids = prefix_ids + rewritten_ids
            if not combined_ids:
                combined_ids = [fallback_prefix_token]

            seq_len = len(combined_ids)
            seq_token_logprob_len = max(seq_len - 1, 0)
            rewrite_start_token = min(len(prefix_ids), seq_len)
            start = max(rewrite_start_token - 1, 0)

            row_action_mask = [0.0] * seq_token_logprob_len
            has_rewrite_span = seq_token_logprob_len > start and len(rewritten_ids) > 0
            if has_rewrite_span:
                for pos in range(start, seq_token_logprob_len):
                    row_action_mask[pos] = 1.0
                rewrite_span_starts.append(start)
                rewrite_span_ends.append(seq_token_logprob_len - 1)
            else:
                rewrite_span_starts.append(-1)
                rewrite_span_ends.append(-1)

            input_id_rows.append(combined_ids)
            action_mask_rows.append(row_action_mask)
            valid_actions.append(has_rewrite_span)
            tokenized_lengths.append(seq_len)

        encoded = self.tokenizer.pad(
            {"input_ids": input_id_rows},
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        action_width = max(input_ids.shape[1] - 1, 0)
        action_mask = torch.zeros((input_ids.shape[0], action_width), dtype=torch.float32)

        for idx, row in enumerate(action_mask_rows):
            if not row:
                continue
            width = min(len(row), action_width)
            action_mask[idx, :width] = torch.tensor(row[:width], dtype=torch.float32)

        payload = {key: value.to(self.device) for key, value in encoded.items()}
        payload["action_mask"] = action_mask.to(self.device)
        payload["valid_action"] = torch.tensor(valid_actions, dtype=torch.bool, device=self.device)
        payload["rewrite_span_start"] = torch.tensor(
            rewrite_span_starts,
            dtype=torch.int32,
            device=self.device,
        )
        payload["rewrite_span_end"] = torch.tensor(
            rewrite_span_ends,
            dtype=torch.int32,
            device=self.device,
        )
        payload["tokenized_length"] = torch.tensor(
            tokenized_lengths,
            dtype=torch.int32,
            device=self.device,
        )
        return payload

    def parameters(self, recurse: bool = True):
        for parameter in self.model.parameters(recurse=recurse):
            if parameter.requires_grad:
                yield parameter

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.model.save_pretrained(path)
