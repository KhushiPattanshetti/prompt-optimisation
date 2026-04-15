from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import json
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import (
    MODEL_NAME,
    RL_CHECKPOINT_PATH,
    REWRITER_LOAD_LOCAL_CHECKPOINTS,
    VALUE_HEAD_HIDDEN_SIZE,
)
from .logger import get_logger

log = get_logger(__name__)

_cached_model: Optional[PreTrainedModel] = None
_cached_tokenizer: Optional[PreTrainedTokenizerBase] = None
_cached_value_head: Optional[nn.Module] = None

_VALUE_HEAD_MIGRATION_MARKER = ".value_head_migration_v2.json"


class ValueHead(nn.Module):
    """Mirror RL loop value-head architecture for checkpoint compatibility."""

    def __init__(self, hidden_size: int = VALUE_HEAD_HIDDEN_SIZE, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.layers.parameters()).dtype
        if hidden_states.dtype != param_dtype:
            hidden_states = hidden_states.to(param_dtype)
        return self.layers(hidden_states)


def _find_latest_checkpoint_dir() -> Optional[Path]:
    root = Path(RL_CHECKPOINT_PATH)
    if not root.exists():
        return None

    checkpoint_dirs: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = re.fullmatch(r"checkpoint_(\d+)", entry.name)
        if match:
            checkpoint_dirs.append((int(match.group(1)), entry))

    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=lambda pair: pair[0])[1]


def _latest_checkpoint_state_keys() -> set[str]:
    return set(ValueHead(hidden_size=VALUE_HEAD_HIDDEN_SIZE, dropout=0.1).state_dict().keys())


def _normalize_value_head_state_dict(raw_state: object, source_path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise RuntimeError(
            f"Unsupported value_head checkpoint format at {source_path}: {type(raw_state)}"
        )

    # Legacy single-linear checkpoints are not safely migratable to the MLP head.
    if set(raw_state.keys()) == {"weight"}:
        raise RuntimeError(
            "Found legacy single-linear value_head checkpoint at "
            f"{source_path}; remove this checkpoint or regenerate it with the current RL loop."
        )

    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Invalid non-tensor value for key '{key}' in {source_path}")
        canonical_key = key[7:] if key.startswith("module.") else key
        normalized[canonical_key] = value

    expected_keys = _latest_checkpoint_state_keys()
    if set(normalized.keys()) != expected_keys:
        raise RuntimeError(
            "Incompatible value_head state keys in "
            f"{source_path}; expected {sorted(expected_keys)} but got {sorted(normalized.keys())}"
        )

    return normalized


def _migrate_value_head_checkpoints_once() -> None:
    root = Path(RL_CHECKPOINT_PATH)
    if not root.exists():
        return

    marker_path = root / _VALUE_HEAD_MIGRATION_MARKER
    if marker_path.exists():
        return

    migrated = 0
    checked = 0

    for checkpoint_dir in sorted(root.glob("checkpoint_*")):
        if not checkpoint_dir.is_dir():
            continue

        value_head_path = checkpoint_dir / "value_head.pt"
        if not value_head_path.exists():
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} is missing value_head.pt; refusing to start with inconsistent checkpoint state"
            )

        checked += 1
        raw_state = torch.load(value_head_path, map_location="cpu")
        normalized = _normalize_value_head_state_dict(raw_state, value_head_path)

        if isinstance(raw_state, dict) and set(raw_state.keys()) != set(normalized.keys()):
            torch.save(normalized, value_head_path)
            migrated += 1

    marker_payload = {
        "migrated_checkpoints": migrated,
        "checked_checkpoints": checked,
        "version": 2,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")


def _build_value_head(device: torch.device, checkpoint_dir: Optional[Path]) -> nn.Module:
    value_head = ValueHead(hidden_size=VALUE_HEAD_HIDDEN_SIZE, dropout=0.1)
    value_head = value_head.to(dtype=torch.float16, device=device)

    if checkpoint_dir is not None:
        value_head_path = checkpoint_dir / "value_head.pt"
        if not value_head_path.exists():
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} missing value_head.pt; fail-fast enabled to prevent silent random init"
            )

        state = torch.load(value_head_path, map_location="cpu")
        normalized = _normalize_value_head_state_dict(state, value_head_path)
        value_head.load_state_dict(normalized, strict=True)
    else:
        for module in value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    value_head.eval()

    return value_head


def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, nn.Module]:
    global _cached_model, _cached_tokenizer, _cached_value_head

    if (
        _cached_model is not None
        and _cached_tokenizer is not None
        and _cached_value_head is not None
    ):
        return _cached_model, _cached_tokenizer, _cached_value_head

    _migrate_value_head_checkpoints_once()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = True

    checkpoint_dir = _find_latest_checkpoint_dir() if REWRITER_LOAD_LOCAL_CHECKPOINTS else None
    lora_adapter_path = None
    if checkpoint_dir is not None:
        candidate = checkpoint_dir / "lora_adapter"
        if candidate.exists():
            lora_adapter_path = candidate

    model: PreTrainedModel = base_model
    lora_active = False
    if lora_adapter_path is not None:
        adapter_config = lora_adapter_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                model = PeftModel.from_pretrained(
                    base_model,
                    str(lora_adapter_path),
                    is_trainable=False,
                )
                lora_active = True
                log.info("Loaded LoRA adapter from %s", lora_adapter_path)
            except Exception as exc:
                log.exception(
                    "Failed to load LoRA adapter from %s; continuing with base model only | error=%s",
                    lora_adapter_path,
                    exc,
                )
                model = base_model
                lora_active = False
        else:
            log.warning(
                "LoRA adapter directory missing adapter_config.json at %s; continuing with base model only",
                lora_adapter_path,
            )
    else:
        log.info("No RL LoRA adapter checkpoint found; using base model only")

    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception as exc:
            log.warning("Unable to disable gradient checkpointing: %s", exc)
    if hasattr(model, "disable_input_require_grads"):
        try:
            model.disable_input_require_grads()
        except Exception as exc:
            log.warning("Unable to disable input gradients: %s", exc)

    model.eval()

    device = next(model.parameters()).device
    value_head = _build_value_head(device, checkpoint_dir)

    log.info("Inference model ready | lora_active=%s", lora_active)

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_value_head = value_head
    return model, tokenizer, value_head


def clear_cache() -> None:
    global _cached_model, _cached_tokenizer, _cached_value_head

    _cached_model = None
    _cached_tokenizer = None
    _cached_value_head = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reload_from_latest_checkpoint() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, nn.Module]:
    clear_cache()
    return load_model()


def get_cached_model() -> Optional[PreTrainedModel]:
    return _cached_model
 

# """Model loader for the rewriter inference service.

# Responsible for:
# - Determining which checkpoint to load (RL vs SFT)
# - Loading the tokenizer
# - Loading the model
# - Caching the model instance
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# from config import RL_CHECKPOINT_PATH, SFT_CHECKPOINT_PATH
# from logger import get_logger

# log = get_logger(__name__)

# # Module-level cache
# _cached_model: PreTrainedModel | None = None
# _cached_tokenizer: PreTrainedTokenizerBase | None = None


# def _resolve_checkpoint_path() -> Path:
#     """Determine which checkpoint directory to load from.

#     Logic:
#         - If rl_checkpoints/ contains any sub-directories (checkpoints),
#           return the path to the most recently modified one.
#         - Otherwise fall back to sft_checkpoints/.

#     Returns:
#         Path to the selected checkpoint directory.

#     Raises:
#         FileNotFoundError: If no valid checkpoint directory is found.
#     """
#     rl_path = Path(RL_CHECKPOINT_PATH)

#     if rl_path.exists():
#         # Collect checkpoint sub-directories inside rl_checkpoints/
#         rl_checkpoints = [p for p in rl_path.iterdir() if p.is_dir()]
#         if rl_checkpoints:
#             latest = max(rl_checkpoints, key=lambda p: p.stat().st_mtime)
#             log.info("checkpoint_selected | source=rl_checkpoints | path=%s", latest)
#             return latest

#     sft_path = Path(SFT_CHECKPOINT_PATH)
#     if sft_path.exists():
#         # If sft_checkpoints itself is the checkpoint directory
#         sft_subdirs = [p for p in sft_path.iterdir() if p.is_dir()]
#         if sft_subdirs:
#             latest = max(sft_subdirs, key=lambda p: p.stat().st_mtime)
#             log.info("checkpoint_selected | source=sft_checkpoints | path=%s", latest)
#             return latest
#         # sft_checkpoints/ itself may be the checkpoint
#         log.info("checkpoint_selected | source=sft_checkpoints | path=%s", sft_path)
#         return sft_path

#     raise FileNotFoundError(
#         f"No checkpoint found. Checked: {rl_path}, {sft_path}"
#     )


# def load_model(
#     checkpoint_path: Path | None = None,
# ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
#     """Load the prompt-rewriter model and tokenizer.

#     Uses a module-level cache so repeated calls return the same objects.

#     Args:
#         checkpoint_path: Optional explicit path. When ``None`` the path is
#             resolved automatically via checkpoint priority logic.

#     Returns:
#         Tuple of (model, tokenizer).
#     """
#     global _cached_model, _cached_tokenizer

#     if _cached_model is not None and _cached_tokenizer is not None:
#         return _cached_model, _cached_tokenizer

#     path = checkpoint_path or _resolve_checkpoint_path()
#     path_str = str(path)

#     log.info("loading_model | path=%s", path_str)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
#     log.info("tokenizer_loaded | path=%s", path_str)

#     model = AutoModelForCausalLM.from_pretrained(
#         path_str,
#         trust_remote_code=True,
#         torch_dtype=torch.float32,
#     ).to(device)
#     model.eval()

#     log.info("model_loaded | device=%s | path=%s", device, path_str)

#     _cached_model = model
#     _cached_tokenizer = tokenizer
#     return model, tokenizer


# def clear_cache() -> None:
#     """Clear the cached model and tokenizer (useful for testing)."""
#     global _cached_model, _cached_tokenizer
#     _cached_model = None
#     _cached_tokenizer = None
