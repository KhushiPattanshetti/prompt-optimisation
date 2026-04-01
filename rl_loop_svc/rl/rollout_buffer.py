"""
Rollout buffer: stores collected trajectory data in tensors ready for
PPO mini-batch updates.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """Tensors filled by store() and consumed by the PPO trainer."""

    rewards: torch.Tensor  # (N,)
    log_probs_old: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    advantages: torch.Tensor  # (N,) — filled by advantage estimator
    returns: torch.Tensor  # (N,) — advantage + value baseline
    original_prompts: List[str] = field(default_factory=list)
    rewritten_prompts: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.rewards.shape[0]


class RolloutBuffer:
    """
    In-memory buffer that accumulates raw rollout entries and exposes
    them as a single RolloutBatch for training.

    Args:
        device: Torch device for the output tensors.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._rewards: List[float] = []
        self._log_probs_old: List[float] = []
        self._values: List[float] = []
        self._original_prompts: List[str] = []
        self._rewritten_prompts: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def store(
        self,
        reward: float,
        log_prob_old: float,
        value_estimate: float,
        original_prompt: str,
        rewritten_prompt: str,
    ) -> None:
        """Append a single trajectory step to the buffer."""
        self._rewards.append(reward)
        self._log_probs_old.append(log_prob_old)
        self._values.append(value_estimate)
        self._original_prompts.append(original_prompt)
        self._rewritten_prompts.append(rewritten_prompt)

    def build(self, advantages: torch.Tensor) -> RolloutBatch:
        """
        Finalise the buffer into a RolloutBatch.

        Args:
            advantages: Pre-computed GAE advantages, shape (N,).

        Returns:
            RolloutBatch ready for PPO updates.
        """
        rewards = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(
            self._log_probs_old, dtype=torch.float32, device=self.device
        )
        values = torch.tensor(self._values, dtype=torch.float32, device=self.device)
        returns = advantages + values

        return RolloutBatch(
            rewards=rewards,
            log_probs_old=log_probs_old,
            values=values,
            advantages=advantages.to(self.device),
            returns=returns,
            original_prompts=list(self._original_prompts),
            rewritten_prompts=list(self._rewritten_prompts),
        )

    def clear(self) -> None:
        """Reset the buffer for the next collection phase."""
        self._rewards.clear()
        self._log_probs_old.clear()
        self._values.clear()
        self._original_prompts.clear()
        self._rewritten_prompts.clear()

    def __len__(self) -> int:
        return len(self._rewards)
