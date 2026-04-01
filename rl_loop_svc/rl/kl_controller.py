"""
KL divergence controller.

Computes the per-sample KL divergence between the current policy and the
frozen reference model, and adjusts the reward accordingly.

reward_adjusted = reward - beta * KL

where KL = log_prob_policy - log_prob_reference
"""

import logging

import torch

logger = logging.getLogger(__name__)


class KLController:
    """
    Manages KL-divergence computation and reward adjustment.

    Args:
        beta: KL penalty coefficient.
    """

    def __init__(self, beta: float = 0.01) -> None:
        self.beta = beta
        self._last_kl: float = 0.0

    def compute_kl(
        self,
        log_prob_policy: torch.Tensor,
        log_prob_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Element-wise KL divergence estimate.

        Args:
            log_prob_policy:    Shape (B,) — log-probs under current policy.
            log_prob_reference: Shape (B,) — log-probs under frozen reference.

        Returns:
            kl: Shape (B,) — KL divergence per sample.
        """
        kl = log_prob_policy - log_prob_reference
        self._last_kl = kl.mean().item()
        return kl

    def adjust_rewards(
        self,
        rewards: torch.Tensor,
        log_prob_policy: torch.Tensor,
        log_prob_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subtract KL penalty from rewards.

        Returns:
            adjusted_rewards: Shape (B,).
        """
        kl = self.compute_kl(log_prob_policy, log_prob_reference)
        adjusted = rewards - self.beta * kl
        logger.debug(
            "Mean KL: %.4f | Mean reward before: %.4f | after: %.4f",
            self._last_kl,
            rewards.mean().item(),
            adjusted.mean().item(),
        )
        return adjusted

    @property
    def last_kl(self) -> float:
        """Most recently computed mean KL divergence."""
        return self._last_kl
