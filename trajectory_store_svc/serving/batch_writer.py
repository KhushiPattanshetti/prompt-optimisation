"""
serving/batch_writer.py — Persist finalised batches and manage reuse state.

Files written
─────────────
prepared_batches/prepared_batch_<batch_id>.json

Contents
────────
{
  "batch_id": "...",
  "rollout_ids": [...],
  "rewards": [...],
  "advantages": [...],
  "returns": [...],
  "group_ids": [...],
  "sample_weights": [...]
}

Batch-reuse modes
─────────────────
repeat      : serve the same batch up to max_batch_reuse_count times
single_pass : each batch consumed exactly once
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from config import cfg
from buffer.rollout_buffer import RolloutBatch
from utils.logging import get_logger

log = get_logger("serving.batch_writer")


class BatchWriter:
    """Write batch files and manage batch-reuse state."""

    def __init__(self) -> None:
        self._current_batch_id: Optional[str] = None
        self._reuse_count: int = 0
        self._batch_index: int = 0
        self._written_paths: List[Path] = []
        log.info(
            "BatchWriter initialised",
            extra={
                "mode": cfg.batch_reuse_mode,
                "max_reuse": cfg.max_batch_reuse_count,
            },
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, batch: RolloutBatch) -> Path:
        """Serialise batch to disk and update reuse state.  Returns the file path."""
        path = self._batch_path(batch.batch_id)
        payload = {
            "batch_id": batch.batch_id,
            "rollout_ids": batch.rollout_ids,
            "rewards": batch.rewards,
            "advantages": batch.advantages,
            "returns": batch.returns,
            "group_ids": batch.group_ids,
            "sample_weights": batch.sample_weights,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        self._current_batch_id = batch.batch_id
        self._reuse_count = 0
        self._batch_index += 1
        self._written_paths.append(path)

        log.info(
            "Batch written to disk",
            extra={
                "batch_id": batch.batch_id,
                "path": str(path),
                "size": batch.size(),
                "reward_mean": f"{batch.reward_mean():.4f}",
            },
        )
        return path

    def should_reuse(self) -> bool:
        """True when reuse mode is 'repeat' and reuse budget is not exhausted."""
        if cfg.batch_reuse_mode != "repeat":
            return False
        return self._reuse_count < cfg.max_batch_reuse_count

    def record_reuse(self) -> None:
        self._reuse_count += 1
        log.info(
            "Batch reused",
            extra={
                "batch_id": self._current_batch_id,
                "reuse_count": self._reuse_count,
                "max_reuse": cfg.max_batch_reuse_count,
            },
        )

    @property
    def batch_index(self) -> int:
        return self._batch_index

    @property
    def written_paths(self) -> List[Path]:
        return list(self._written_paths)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_path(batch_id: str) -> Path:
        return cfg.prepared_batches_dir / f"prepared_batch_{batch_id}.json"
