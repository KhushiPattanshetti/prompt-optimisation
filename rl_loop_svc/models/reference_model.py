"""
Reference model: identical architecture to the policy model but with all
parameters frozen.  Used exclusively to compute KL divergence.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ReferenceModel:
    """
    A frozen copy of the initial policy used for KL-divergence tracking.

    The weights are NEVER updated during PPO training.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device.
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            "Loading reference model: %s on %s (frozen)", model_name, self.device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=False
        )
        self.model.to(self.device)

        # Freeze all parameters immediately
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute summed log-probability for each sequence.

        Returns:
            Tensor of shape (B,).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        token_log_probs = torch.gather(
            shift_log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.sum(dim=-1)
