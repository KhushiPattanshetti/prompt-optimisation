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
    sample_weights: torch.Tensor  # (N,)
    original_prompts: List[str] = field(default_factory=list)
    rewritten_prompts: List[str] = field(default_factory=list)
    group_ids: List[str] = field(default_factory=list)
    rollout_ids: List[str] = field(default_factory=list)
    og_codes: List[List[str]] = field(default_factory=list)
    enh_codes: List[List[str]] = field(default_factory=list)
    gt_codes: List[List[str]] = field(default_factory=list)

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
        self._sample_weights: List[float] = []
        self._log_probs_old: List[float] = []
        self._values: List[float] = []
        self._group_ids: List[str] = []
        self._original_prompts: List[str] = []
        self._rewritten_prompts: List[str] = []
        self._rollout_ids: List[str] = []
        self._og_codes: List[List[str]] = []
        self._enh_codes: List[List[str]] = []
        self._gt_codes: List[List[str]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def store(
        self,
        reward: float,
        log_prob_old: float,
        value_estimate: float,
        group_id: str,
        original_prompt: str,
        rewritten_prompt: str,
        sample_weight: float = 1.0,
        rollout_id: str = "",
        og_codes: List[str] | None = None,
        enh_codes: List[str] | None = None,
        gt_codes: List[str] | None = None,
    ) -> None:
        """Append a single trajectory step to the buffer."""
        self._rewards.append(reward)
        self._sample_weights.append(sample_weight)
        self._log_probs_old.append(log_prob_old)
        self._values.append(value_estimate)
        self._group_ids.append(group_id)
        self._original_prompts.append(original_prompt)
        self._rewritten_prompts.append(rewritten_prompt)
        self._rollout_ids.append(rollout_id)
        self._og_codes.append(og_codes or [])
        self._enh_codes.append(enh_codes or [])
        self._gt_codes.append(gt_codes or [])

    def build(self, advantages: torch.Tensor) -> RolloutBatch:
        """
        Finalise the buffer into a RolloutBatch.

        Args:
            advantages: Pre-computed GAE advantages, shape (N,).

        Returns:
            RolloutBatch ready for PPO updates.
        """
        rewards = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        sample_weights = torch.tensor(self._sample_weights, dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(
            self._log_probs_old, dtype=torch.float32, device=self.device
        )
        values = torch.tensor(self._values, dtype=torch.float32, device=self.device)
        advantages_on_device = advantages.to(self.device)
        returns = advantages_on_device + values

        return RolloutBatch(
            rewards=rewards,
            log_probs_old=log_probs_old,
            values=values,
            advantages=advantages_on_device,
            returns=returns,
            sample_weights=sample_weights,
            original_prompts=list(self._original_prompts),
            rewritten_prompts=list(self._rewritten_prompts),
            group_ids=list(self._group_ids),
            rollout_ids=list(self._rollout_ids),
            og_codes=list(self._og_codes),
            enh_codes=list(self._enh_codes),
            gt_codes=list(self._gt_codes),
        )

    def clear(self) -> None:
        """Reset the buffer for the next collection phase."""
        self._rewards.clear()
        self._sample_weights.clear()
        self._log_probs_old.clear()
        self._values.clear()
        self._group_ids.clear()
        self._original_prompts.clear()
        self._rewritten_prompts.clear()
        self._rollout_ids.clear()
        self._og_codes.clear()
        self._enh_codes.clear()
        self._gt_codes.clear()

    def __len__(self) -> int:
        return len(self._rewards)
