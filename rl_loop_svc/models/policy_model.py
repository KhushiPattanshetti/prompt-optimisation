"""
Policy model: Phi-3-mini with 4-bit NF4 quantization and LoRA adapter.

FIX SUMMARY (from review):
  - Previously loaded model at full precision with no quantization.
    Loading Phi-3 twice (policy + reference) at float32/bfloat16 =
    ~15GB VRAM just for the two LMs, leaving no room for Med42.
  - Now loads in 4-bit NF4 quantization (~3.8GB per model).
  - LoRA adapter attached so only ~3M params are trained, not 3.8B.
  - enable_input_require_grads() called for gradient checkpointing
    compatibility with quantized model.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

# LoRA config matching rewriter_inference_svc
_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class PolicyModel(nn.Module):
    """
    Phi-3-mini loaded in 4-bit NF4 with a LoRA adapter.

    Only LoRA weights (~3M params) are trainable.
    The base 3.8B weights remain frozen (quantized).

    Args:
        model_name: HuggingFace model identifier.
        checkpoint_path: Optional path to an RL checkpoint lora_adapter/ dir.
                         If None, a fresh LoRA adapter is attached.
        device: Torch device (auto-detected if None).
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            "Loading policy model (4-bit NF4): %s on %s", model_name, self.device
        )

        bnb_config = _build_bnb_config()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base.config.use_cache = False

        # Attach LoRA — load from checkpoint or fresh
        if checkpoint_path is not None:
            logger.info("Loading LoRA adapter from: %s", checkpoint_path)
            self.model: PreTrainedModel = PeftModel.from_pretrained(
                base, checkpoint_path, is_trainable=True
            )
        else:
            logger.info("Attaching fresh LoRA adapter")
            self.model = get_peft_model(base, _LORA_CONFIG)

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info("PolicyModel ready | trainable_params=%d", trainable)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            log_probs:    per-token log-probabilities, shape (B, T-1).
            hidden_states: last hidden layer, shape (B, T, H).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits                        # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)

        # Shift: position i predicts token i+1
        shift_log_probs = log_probs[:, :-1, :]         # (B, T-1, V)
        shift_labels    = input_ids[:, 1:]              # (B, T-1)
        token_log_probs = torch.gather(
            shift_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)                                   # (B, T-1)

        last_hidden = outputs.hidden_states[-1]         # (B, T, H)
        return token_log_probs, last_hidden

    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum log-probs over sequence → scalar per sample. Shape (B,)."""
        token_lp, _ = self.forward(input_ids, attention_mask)
        return token_lp.sum(dim=-1)

    def tokenize(self, texts: list) -> dict:
        """Tokenise a list of strings and move tensors to device."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def parameters(self, recurse: bool = True):
        """Return only trainable (LoRA) parameters for the optimizer."""
        return (p for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save LoRA adapter weights to path."""
        self.model.save_pretrained(path)
        logger.info("Policy LoRA adapter saved to %s", path)

    def load(self, path: str) -> None:
        """Load LoRA adapter weights from path."""
        self.model.load_adapter(path, adapter_name="default")
        logger.info("Policy LoRA adapter loaded from %s", path)

# """
# Policy model: a thin PyTorch wrapper around a HuggingFace causal language
# model used as the Prompt Rewriter.

# In training, both the token log-probabilities and the last-layer hidden
# state (needed by the value head) are exposed.
# """

# import logging
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# logger = logging.getLogger(__name__)


# class PolicyModel(nn.Module):
#     """
#     Wraps a HuggingFace causal language model.

#     Args:
#         model_name: HuggingFace model identifier (default: 'gpt2').
#         device: Torch device to move the model to.
#     """

#     def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
#         super().__init__()
#         self.device = torch.device(
#             device or ("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         logger.info("Loading policy model: %s on %s", model_name, self.device)

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
#             model_name, output_hidden_states=True
#         )
#         self.model.to(self.device)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass.

#         Returns:
#             log_probs: per-token log-probabilities, shape (B, T).
#             hidden_states: last hidden layer, shape (B, T, H).
#         """
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=input_ids,
#         )
#         logits = outputs.logits  # (B, T, V)
#         log_probs = torch.log_softmax(logits, dim=-1)

#         # Gather log-prob of each actual token
#         # Shift so that position i predicts token i+1
#         shift_log_probs = log_probs[:, :-1, :]  # (B, T-1, V)
#         shift_labels = input_ids[:, 1:]  # (B, T-1)
#         token_log_probs = torch.gather(
#             shift_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
#         ).squeeze(
#             -1
#         )  # (B, T-1)

#         last_hidden = outputs.hidden_states[-1]  # (B, T, H)
#         return token_log_probs, last_hidden

#     def get_sequence_log_prob(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Sum log-probabilities over the sequence to get a scalar per sample.

#         Returns:
#             Tensor of shape (B,).
#         """
#         token_lp, _ = self.forward(input_ids, attention_mask)
#         return token_lp.sum(dim=-1)

#     def tokenize(self, texts: list[str]) -> dict:
#         """Tokenise a list of strings and move to device."""
#         encoded = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512,
#         )
#         return {k: v.to(self.device) for k, v in encoded.items()}

#     def save(self, path: str) -> None:
#         """Save model weights to *path*."""
#         torch.save(self.model.state_dict(), path)
#         logger.info("Policy model saved to %s", path)

#     def load(self, path: str) -> None:
#         """Load model weights from *path*."""
#         state_dict = torch.load(path, map_location=self.device)
#         self.model.load_state_dict(state_dict)
#         logger.info("Policy model loaded from %s", path)
