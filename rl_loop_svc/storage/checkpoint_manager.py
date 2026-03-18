"""
Checkpoint manager: saves and loads RL training checkpoints.

Directory layout:
    rl_checkpoints/
        checkpoint_0001/
            lora_adapter/          ← LoRA adapter weights (PEFT format)
                adapter_config.json
                adapter_model.bin
            value_head.pt          ← ValueHead state dict
            optimizer.pt           ← Optimizer state dict
            training_state.json    ← Metadata (step, loss, KL, timestamp)
        checkpoint_0002/
            ...

FIX SUMMARY (from review):
  - Previously saved policy_model.pt as a full state dict via torch.save().
    For a 4-bit quantized LoRA model, saving the full state dict is both
    incorrect (quantized weights are not straightforwardly serializable)
    and wasteful (saves 3.8B params instead of ~3M LoRA params).
  - Now saves LoRA adapter in PEFT format via model.save_pretrained()
    into a lora_adapter/ subdirectory. This matches what model_loader.py
    and PolicyModel expect when loading from checkpoint.
  - Policy save/load now uses PeftModel.save_pretrained / load_adapter.
  - load() method updated to match new layout.
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
    Manages incremental RL checkpoints with automatic retention policy.

    Args:
        checkpoints_dir: Root directory for checkpoints.
        max_checkpoints: Maximum number of checkpoints to retain on disk.
    """

    def __init__(self, checkpoints_dir: Path, max_checkpoints: int = 5) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────

    def save(
        self,
        policy_model,               # PolicyModel instance (has .model PEFT attribute)
        value_head_state_dict: dict,
        optimizer_state_dict: dict,
        training_step: int,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """Write a new checkpoint and prune old ones.

        Args:
            policy_model:          PolicyModel instance — LoRA adapter saved
                                   via model.model.save_pretrained().
            value_head_state_dict: ValueHead.state_dict().
            optimizer_state_dict:  Optimizer.state_dict().
            training_step:         Current global training step.
            extra_meta:            Optional extra fields for training_state.json.

        Returns:
            Path to the new checkpoint directory.
        """
        index   = self._next_index()
        ckpt_dir = self.checkpoints_dir / f"checkpoint_{index:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter in PEFT format (not raw state dict)
        lora_dir = ckpt_dir / "lora_adapter"
        policy_model.model.save_pretrained(str(lora_dir))
        logger.info("LoRA adapter saved to %s", lora_dir)

        # Save value head and optimizer
        torch.save(value_head_state_dict, ckpt_dir / "value_head.pt")
        torch.save(optimizer_state_dict,  ckpt_dir / "optimizer.pt")

        # Save metadata
        meta = {
            "training_step":     training_step,
            "checkpoint_index":  index,
            "saved_at":          datetime.now(tz=timezone.utc).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(ckpt_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Checkpoint saved: %s (step %d)", ckpt_dir.name, training_step
        )
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
        if not state_path.exists():
            return None
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)

    def load_into(
        self,
        policy_model,
        value_head,
        optimizer=None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Optional[dict]:
        """Load a checkpoint into existing model/optimizer instances.

        Args:
            policy_model:    PolicyModel instance to load LoRA weights into.
            value_head:      ValueHead instance to load weights into.
            optimizer:       Optional optimizer to restore state.
            checkpoint_dir:  Specific checkpoint to load. Defaults to latest.

        Returns:
            Metadata dict from training_state.json, or None if no checkpoint.
        """
        ckpt = checkpoint_dir or self.latest()
        if ckpt is None:
            logger.info("No checkpoint found — starting from scratch")
            return None

        # Load LoRA adapter
        lora_dir = ckpt / "lora_adapter"
        if lora_dir.exists():
            policy_model.model.load_adapter(
                str(lora_dir), adapter_name="default"
            )
            logger.info("LoRA adapter loaded from %s", lora_dir)
        else:
            logger.warning("No lora_adapter/ in checkpoint %s", ckpt)

        # Load value head
        vh_path = ckpt / "value_head.pt"
        if vh_path.exists():
            state = torch.load(vh_path, map_location="cpu")
            value_head.load_state_dict(state)
            logger.info("Value head loaded from %s", vh_path)

        # Load optimizer
        if optimizer is not None:
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                state = torch.load(opt_path, map_location="cpu")
                optimizer.load_state_dict(state)
                logger.info("Optimizer state loaded from %s", opt_path)

        return self.load_latest_meta()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _all_checkpoints(self) -> list:
        """Return sorted list of checkpoint directories."""
        return sorted(
            [
                p for p in self.checkpoints_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint_")
            ],
            key=lambda p: p.name,
        )

    def _next_index(self) -> int:
        existing = self._all_checkpoints()
        if not existing:
            return 1
        last_name = existing[-1].name   # e.g. "checkpoint_0003"
        return int(last_name.split("_")[-1]) + 1

    def _prune(self) -> None:
        """Delete oldest checkpoints beyond max_checkpoints."""
        all_ckpts = self._all_checkpoints()
        to_delete = all_ckpts[: max(0, len(all_ckpts) - self.max_checkpoints)]
        for ckpt in to_delete:
            shutil.rmtree(ckpt)
            logger.info("Pruned old checkpoint: %s", ckpt.name)

# """
# Checkpoint manager: handles saving and loading model checkpoints with
# incremental numbering and automatic retention of the N most recent.

# Directory layout written by this module:

# rl_checkpoints/
#     checkpoint_0001/
#         policy_model.pt
#         value_head.pt
#         optimizer.pt
#         training_state.json
#     checkpoint_0002/
#         ...
# """

# import json
# import logging
# import shutil
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Optional

# import torch

# logger = logging.getLogger(__name__)


# class CheckpointManager:
#     """
#     Manages incremental model checkpoints with automatic retention policy.

#     Args:
#         checkpoints_dir: Root directory under which checkpoints are written.
#         max_checkpoints: Maximum number of checkpoints to retain.
#     """

#     def __init__(self, checkpoints_dir: Path, max_checkpoints: int = 5) -> None:
#         self.checkpoints_dir = Path(checkpoints_dir)
#         self.max_checkpoints = max_checkpoints
#         self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

#     # ── Save ──────────────────────────────────────────────────────────────

#     def save(
#         self,
#         policy_state_dict: dict,
#         value_head_state_dict: dict,
#         optimizer_state_dict: dict,
#         training_step: int,
#         extra_meta: Optional[dict] = None,
#     ) -> Path:
#         """
#         Write a new checkpoint and prune old ones.

#         Returns:
#             Path to the new checkpoint directory.
#         """
#         index = self._next_index()
#         ckpt_dir = self.checkpoints_dir / f"checkpoint_{index:04d}"
#         ckpt_dir.mkdir(parents=True, exist_ok=True)

#         torch.save(policy_state_dict, ckpt_dir / "policy_model.pt")
#         torch.save(value_head_state_dict, ckpt_dir / "value_head.pt")
#         torch.save(optimizer_state_dict, ckpt_dir / "optimizer.pt")

#         meta = {
#             "training_step": training_step,
#             "checkpoint_index": index,
#             "saved_at": datetime.now(tz=timezone.utc).isoformat(),
#         }
#         if extra_meta:
#             meta.update(extra_meta)

#         with open(ckpt_dir / "training_state.json", "w") as f:
#             json.dump(meta, f, indent=2)

#         logger.info("Checkpoint saved: %s (step %d)", ckpt_dir.name, training_step)
#         self._prune()
#         return ckpt_dir

#     # ── Load ──────────────────────────────────────────────────────────────

#     def latest(self) -> Optional[Path]:
#         """Return the path of the most recent checkpoint directory, or None."""
#         checkpoints = self._all_checkpoints()
#         return checkpoints[-1] if checkpoints else None

#     def load_latest_meta(self) -> Optional[dict]:
#         """Return the training_state.json of the latest checkpoint, or None."""
#         latest = self.latest()
#         if latest is None:
#             return None
#         state_path = latest / "training_state.json"
#         with open(state_path) as f:
#             return json.load(f)

#     # ── Internal helpers ──────────────────────────────────────────────────

#     def _all_checkpoints(self) -> list[Path]:
#         """Return sorted list of checkpoint directories."""
#         return sorted(
#             [
#                 p
#                 for p in self.checkpoints_dir.iterdir()
#                 if p.is_dir() and p.name.startswith("checkpoint_")
#             ],
#             key=lambda p: p.name,
#         )

#     def _next_index(self) -> int:
#         existing = self._all_checkpoints()
#         if not existing:
#             return 1
#         last_name = existing[-1].name  # e.g. "checkpoint_0003"
#         return int(last_name.split("_")[-1]) + 1

#     def _prune(self) -> None:
#         """Delete oldest checkpoints beyond max_checkpoints."""
#         all_ckpts = self._all_checkpoints()
#         to_delete = all_ckpts[: max(0, len(all_ckpts) - self.max_checkpoints)]
#         for ckpt in to_delete:
#             shutil.rmtree(ckpt)
#             logger.info("Pruned old checkpoint: %s", ckpt.name)
