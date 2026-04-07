from typing import Optional
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)


class ReferenceModel:
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
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            target_index = self.device.index if self.device.index is not None else 0
            device_map = {"": target_index}
        else:
            device_map = "auto"

        compute_dtype = self._resolve_compute_dtype(self.device)

        logger.info(
            "reference_model_init | model=%s | requested_device=%s | device_map=%s | compute_dtype=%s",
            model_name,
            self.device,
            device_map,
            compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

        if action_mask is not None:
            if action_mask.device != token_log_probs.device:
                action_mask = action_mask.to(token_log_probs.device)
            return (token_log_probs * action_mask).sum(dim=-1)

        return token_log_probs.sum(dim=-1)
