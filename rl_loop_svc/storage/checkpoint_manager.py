"""
Checkpoint manager: handles saving and loading model checkpoints with
incremental numbering and automatic retention of the N most recent.

Directory layout written by this module:

rl_checkpoints/
    checkpoint_0001/
        policy_model.pt
        value_head.pt
        optimizer.pt
        training_state.json
    checkpoint_0002/
        ...
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages incremental model checkpoints with automatic retention policy.

    Args:
        checkpoints_dir: Root directory under which checkpoints are written.
        max_checkpoints: Maximum number of checkpoints to retain.
    """

    def __init__(self, checkpoints_dir: Path, max_checkpoints: int = 5) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────

    def save(
        self,
        policy_state_dict: dict,
        value_head_state_dict: dict,
        optimizer_state_dict: dict,
        training_step: int,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """
        Write a new checkpoint and prune old ones.

        Returns:
            Path to the new checkpoint directory.
        """
        index = self._next_index()
        ckpt_dir = self.checkpoints_dir / f"checkpoint_{index:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(policy_state_dict, ckpt_dir / "policy_model.pt")
        torch.save(value_head_state_dict, ckpt_dir / "value_head.pt")
        torch.save(optimizer_state_dict, ckpt_dir / "optimizer.pt")

        meta = {
            "training_step": training_step,
            "checkpoint_index": index,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Checkpoint saved: %s (step %d)", ckpt_dir.name, training_step)
        self._prune()
        return ckpt_dir

    # ── Load ──────────────────────────────────────────────────────────────

    def latest(self) -> Optional[Path]:
        """Return the path of the most recent checkpoint directory, or None."""
        checkpoints = self._all_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def load_latest_meta(self) -> Optional[dict]:
        """Return the training_state.json of the latest checkpoint, or None."""
        latest = self.latest()
        if latest is None:
            return None
        state_path = latest / "training_state.json"
        with open(state_path) as f:
            return json.load(f)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _all_checkpoints(self) -> list[Path]:
        """Return sorted list of checkpoint directories."""
        return sorted(
            [
                p
                for p in self.checkpoints_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint_")
            ],
            key=lambda p: p.name,
        )

    def _next_index(self) -> int:
        existing = self._all_checkpoints()
        if not existing:
            return 1
        last_name = existing[-1].name  # e.g. "checkpoint_0003"
        return int(last_name.split("_")[-1]) + 1

    def _prune(self) -> None:
        """Delete oldest checkpoints beyond max_checkpoints."""
        all_ckpts = self._all_checkpoints()
        to_delete = all_ckpts[: max(0, len(all_ckpts) - self.max_checkpoints)]
        for ckpt in to_delete:
            shutil.rmtree(ckpt)
            logger.info("Pruned old checkpoint: %s", ckpt.name)
