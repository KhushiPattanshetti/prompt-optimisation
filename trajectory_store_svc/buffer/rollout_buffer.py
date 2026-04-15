"""
buffer/rollout_buffer.py — Accumulate rollouts and produce training batches.

RolloutBatch  : immutable snapshot of one batch (advantages already computed).
RolloutBuffer : stateful accumulator — drains into batches once batch_size is reached.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from config import cfg
from schemas.rollout_schema import Rollout
from utils.logging import get_logger

log = get_logger("buffer")


# ── Batch ─────────────────────────────────────────────────────────────────────


@dataclass
class RolloutBatch:
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rollouts: List[Rollout] = field(default_factory=list)

    # Filled by advantage computation
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    @property
    def rollout_ids(self) -> List[str]:
        return [r.rollout_id for r in self.rollouts]

    @property
    def rewards(self) -> List[float]:
        return [r.reward for r in self.rollouts]

    @property
    def group_ids(self) -> List[Optional[str]]:
        return [r.group_id for r in self.rollouts]

    @property
    def sample_weights(self) -> List[float]:
        return [r.sample_weight for r in self.rollouts]

    def reward_mean(self) -> float:
        r = self.rewards
        return sum(r) / len(r) if r else 0.0

    def reward_std(self) -> float:
        import math

        r = self.rewards
        if len(r) < 2:
            return 0.0
        mean = sum(r) / len(r)
        variance = sum((x - mean) ** 2 for x in r) / len(r)
        return math.sqrt(variance)

    def size(self) -> int:
        return len(self.rollouts)


# ── Buffer ────────────────────────────────────────────────────────────────────


class RolloutBuffer:
    """
    Stateful FIFO buffer.  Call add() to enqueue rollouts; call flush()
    to drain complete batches once batch_size entries are available.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        self._batch_size = batch_size or cfg.batch_size
        self._pending: List[Rollout] = []
        self._filled_count = 0
        log.info("RolloutBuffer ready", extra={"batch_size": self._batch_size})

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, rollout: Rollout) -> None:
        self._pending.append(rollout)
        log.debug(
            "Rollout enqueued",
            extra={"rollout_id": rollout.rollout_id, "pending": len(self._pending)},
        )

    def add_many(self, rollouts: List[Rollout]) -> None:
        for r in rollouts:
            self.add(r)

    def flush(self) -> List[RolloutBatch]:
        """Return complete batches and keep any remainder in the buffer."""
        batches: List[RolloutBatch] = []
        while len(self._pending) >= self._batch_size:
            chunk = self._pending[: self._batch_size]
            self._pending = self._pending[self._batch_size :]
            batch = RolloutBatch(rollouts=chunk)
            self._filled_count += 1
            log.info(
                "Batch assembled",
                extra={
                    "batch_id": batch.batch_id,
                    "size": batch.size(),
                    "reward_mean": f"{batch.reward_mean():.4f}",
                    "reward_std": f"{batch.reward_std():.4f}",
                },
            )
            batches.append(batch)
        return batches

    def pending_count(self) -> int:
        return len(self._pending)

    def total_filled(self) -> int:
        return self._filled_count
