"""
PPO trainer: implements the clipped surrogate objective with value-function
and entropy components.

Total loss = policy_loss + value_coef * value_loss
             - entropy_coef * entropy + beta * KL_penalty
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.rollout_buffer import RolloutBatch

logger = logging.getLogger(__name__)


@dataclass
class PPOLossComponents:
    """Detailed breakdown of each loss term for logging."""

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy_bonus: float
    kl_penalty: float


class PPOTrainer:
    """
    Computes PPO losses and performs one gradient step.

    Args:
        policy_model:   The trainable policy network.
        value_head:     The trainable value head.
        optimizer:      Shared AdamW optimizer.
        epsilon:        PPO clip ratio (default 0.2).
        value_coef:     Value-loss coefficient (default 0.5).
        entropy_coef:   Entropy bonus coefficient (default 0.01).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.policy_model = policy_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(
        self,
        batch: RolloutBatch,
        log_probs_new: torch.Tensor,
        values_new: torch.Tensor,
        entropy: torch.Tensor,
        kl_penalty: torch.Tensor,
    ) -> PPOLossComponents:
        """
        Perform one PPO mini-batch update step.

        Args:
            batch:          Collected rollout batch.
            log_probs_new:  Current policy log-probs, shape (B,).
            values_new:     Current value estimates, shape (B,).
            entropy:        Mean entropy of the current policy.
            kl_penalty:     Per-sample KL term, shape (B,).

        Returns:
            PPOLossComponents breakdown.
        """
        # ── Policy (surrogate) loss ────────────────────────────────────────
        log_ratio = log_probs_new - batch.log_probs_old
        ratio = torch.exp(log_ratio)

        surr1 = ratio * batch.advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            * batch.advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss ─────────────────────────────────────────────────────
        value_loss = F.mse_loss(values_new, batch.returns)

        # ── KL penalty (mean across batch) ────────────────────────────────
        kl_loss = kl_penalty.mean()

        # ── Total loss ─────────────────────────────────────────────────────
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
            + kl_loss
        )

        # ── Gradient step ──────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_model.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        components = PPOLossComponents(
            total_loss=total_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_bonus=(
                entropy.item() if isinstance(entropy, torch.Tensor) else entropy
            ),
            kl_penalty=kl_loss.item(),
        )
        logger.debug("PPO step | %s", components)
        return components
