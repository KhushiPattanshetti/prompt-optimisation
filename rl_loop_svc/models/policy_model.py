"""
Policy model: a thin PyTorch wrapper around a HuggingFace causal language
model used as the Prompt Rewriter.

In training, both the token log-probabilities and the last-layer hidden
state (needed by the value head) are exposed.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    """
    Wraps a HuggingFace causal language model.

    Args:
        model_name: HuggingFace model identifier (default: 'gpt2').
        device: Torch device to move the model to.
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Loading policy model: %s on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            log_probs: per-token log-probabilities, shape (B, T).
            hidden_states: last hidden layer, shape (B, T, H).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log-prob of each actual token
        # Shift so that position i predicts token i+1
        shift_log_probs = log_probs[:, :-1, :]  # (B, T-1, V)
        shift_labels = input_ids[:, 1:]  # (B, T-1)
        token_log_probs = torch.gather(
            shift_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B, T-1)

        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        return token_log_probs, last_hidden

    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sum log-probabilities over the sequence to get a scalar per sample.

        Returns:
            Tensor of shape (B,).
        """
        token_lp, _ = self.forward(input_ids, attention_mask)
        return token_lp.sum(dim=-1)

    def tokenize(self, texts: list[str]) -> dict:
        """Tokenise a list of strings and move to device."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.model.state_dict(), path)
        logger.info("Policy model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info("Policy model loaded from %s", path)
