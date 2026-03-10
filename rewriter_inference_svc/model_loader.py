"""Model loader for the rewriter inference service.

Responsible for:
- Determining which checkpoint to load (RL vs SFT)
- Loading the tokenizer
- Loading the model
- Caching the model instance
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import RL_CHECKPOINT_PATH, SFT_CHECKPOINT_PATH
from logger import get_logger

log = get_logger(__name__)

# Module-level cache
_cached_model: PreTrainedModel | None = None
_cached_tokenizer: PreTrainedTokenizerBase | None = None


def _resolve_checkpoint_path() -> Path:
    """Determine which checkpoint directory to load from.

    Logic:
        - If rl_checkpoints/ contains any sub-directories (checkpoints),
          return the path to the most recently modified one.
        - Otherwise fall back to sft_checkpoints/.

    Returns:
        Path to the selected checkpoint directory.

    Raises:
        FileNotFoundError: If no valid checkpoint directory is found.
    """
    rl_path = Path(RL_CHECKPOINT_PATH)

    if rl_path.exists():
        # Collect checkpoint sub-directories inside rl_checkpoints/
        rl_checkpoints = [p for p in rl_path.iterdir() if p.is_dir()]
        if rl_checkpoints:
            latest = max(rl_checkpoints, key=lambda p: p.stat().st_mtime)
            log.info("checkpoint_selected | source=rl_checkpoints | path=%s", latest)
            return latest

    sft_path = Path(SFT_CHECKPOINT_PATH)
    if sft_path.exists():
        # If sft_checkpoints itself is the checkpoint directory
        sft_subdirs = [p for p in sft_path.iterdir() if p.is_dir()]
        if sft_subdirs:
            latest = max(sft_subdirs, key=lambda p: p.stat().st_mtime)
            log.info("checkpoint_selected | source=sft_checkpoints | path=%s", latest)
            return latest
        # sft_checkpoints/ itself may be the checkpoint
        log.info("checkpoint_selected | source=sft_checkpoints | path=%s", sft_path)
        return sft_path

    raise FileNotFoundError(
        f"No checkpoint found. Checked: {rl_path}, {sft_path}"
    )


def load_model(
    checkpoint_path: Path | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the prompt-rewriter model and tokenizer.

    Uses a module-level cache so repeated calls return the same objects.

    Args:
        checkpoint_path: Optional explicit path. When ``None`` the path is
            resolved automatically via checkpoint priority logic.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _cached_model, _cached_tokenizer

    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer

    path = checkpoint_path or _resolve_checkpoint_path()
    path_str = str(path)

    log.info("loading_model | path=%s", path_str)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
    log.info("tokenizer_loaded | path=%s", path_str)

    model = AutoModelForCausalLM.from_pretrained(
        path_str,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    log.info("model_loaded | device=%s | path=%s", device, path_str)

    _cached_model = model
    _cached_tokenizer = tokenizer
    return model, tokenizer


def clear_cache() -> None:
    """Clear the cached model and tokenizer (useful for testing)."""
    global _cached_model, _cached_tokenizer
    _cached_model = None
    _cached_tokenizer = None
