"""Model loader for the rewriter inference service.
 
FIXES APPLIED:
    - Added 4-bit NF4 quantization via BitsAndBytesConfig (was unquantized)
    - Added LoRA adapter attachment via PEFT get_peft_model()
    - Added explicit ValueHead nn.Linear layer
    - Checkpoint fallback now loads from HuggingFace when both
      rl_checkpoints/ and sft_checkpoints/ are empty (removes hard crash
      on first run without SFT)
    - enable_input_require_grads() called for gradient checkpointing compat
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Optional, Tuple
 
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig, get_peft_model, PeftModel
 
from config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    RL_CHECKPOINT_PATH,
    SFT_CHECKPOINT_PATH,
    VALUE_HEAD_HIDDEN_SIZE,
)
from logger import get_logger
 
log = get_logger(__name__)
 
# Module-level cache
_cached_model:      Optional[PreTrainedModel]          = None
_cached_tokenizer:  Optional[PreTrainedTokenizerBase]  = None
_cached_value_head: Optional[nn.Linear]                = None
 
 
# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------
 
def _build_value_head(device: torch.device) -> nn.Linear:
    """Create a single linear value head: hidden_size → 1."""
    head = nn.Linear(VALUE_HEAD_HIDDEN_SIZE, 1, bias=False)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    return head.to(torch.bfloat16).to(device)
 
 
# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
 
def _resolve_checkpoint_path() -> Optional[Path]:
    """Return the latest RL checkpoint path, or None to load from HuggingFace.
 
    Logic (SFT removed):
        1. If rl_checkpoints/ has sub-directories → return latest by mtime.
        2. Otherwise → return None (caller will load from HuggingFace).
 
    Returns:
        Path to checkpoint directory, or None.
    """
    rl_path = Path(RL_CHECKPOINT_PATH)
 
    if rl_path.exists():
        rl_checkpoints = [p for p in rl_path.iterdir() if p.is_dir()]
        if rl_checkpoints:
            latest = max(rl_checkpoints, key=lambda p: p.stat().st_mtime)
            log.info("checkpoint_selected | source=rl_checkpoints | path=%s", latest)
            return latest
 
    # No RL checkpoint found — signal caller to load from HuggingFace
    log.info(
        "checkpoint_selected | source=huggingface | model=%s "
        "(no rl_checkpoints found)", MODEL_NAME
    )
    return None
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def load_model(
    checkpoint_path: Optional[Path] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, nn.Linear]:
    """Load the prompt-rewriter model, tokenizer, and value head.
 
    Loading strategy (SFT removed):
        - If rl_checkpoints/ contains checkpoints → load latest LoRA adapter.
        - Otherwise → load base Phi-3 from HuggingFace and attach fresh LoRA.
 
    All weights are 4-bit NF4 quantized to fit on a single GPU alongside
    the frozen Med42-8B model.
 
    Returns:
        Tuple of (model_with_lora, tokenizer, value_head).
    """
    global _cached_model, _cached_tokenizer, _cached_value_head
 
    if (
        _cached_model is not None
        and _cached_tokenizer is not None
        and _cached_value_head is not None
    ):
        return _cached_model, _cached_tokenizer, _cached_value_head
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # ── 4-bit quantization config ─────────────────────────────────────────
    compute_dtype = (
        torch.bfloat16
        if BNB_4BIT_COMPUTE_DTYPE == "bfloat16"
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ) if LOAD_IN_4BIT else None
 
    # ── Tokenizer ─────────────────────────────────────────────────────────
    resolved = checkpoint_path or _resolve_checkpoint_path()
 
    # For tokenizer: prefer checkpoint, fall back to base model name
    tok_source = str(resolved) if resolved else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("tokenizer_loaded | source=%s", tok_source)
 
    # ── Base model ────────────────────────────────────────────────────────
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
 
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    base_model.config.use_cache = False
    log.info("base_model_loaded | model=%s | 4bit=%s", MODEL_NAME, LOAD_IN_4BIT)
 
    # ── LoRA adapter ──────────────────────────────────────────────────────
    if resolved and (resolved / "adapter_config.json").exists():
        # Load existing RL LoRA weights from checkpoint
        model = PeftModel.from_pretrained(base_model, str(resolved))
        log.info("lora_loaded | source=%s", resolved)
    else:
        # Attach fresh LoRA adapter (first run, no prior RL checkpoint)
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
        log.info("lora_created | r=%d | alpha=%d", LORA_R, LORA_ALPHA)
 
    # Required for gradient checkpointing with 4-bit quantized models
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    log.info("model_loaded | device=%s", device)
 
    # ── Value head ────────────────────────────────────────────────────────
    value_head = _build_value_head(device)
 
    # Load value head weights from checkpoint if available
    if resolved:
        vh_path = resolved / "value_head.pt"
        if vh_path.exists():
            state = torch.load(str(vh_path), map_location=device)
            value_head.load_state_dict(state)
            log.info("value_head_loaded | source=%s", vh_path)
        else:
            log.info("value_head_initialised_fresh | no checkpoint found")
    else:
        log.info("value_head_initialised_fresh | first run")
 
    _cached_model      = model
    _cached_tokenizer  = tokenizer
    _cached_value_head = value_head
 
    return model, tokenizer, value_head
 
 
def clear_cache() -> None:
    """Clear the cached model, tokenizer, and value head (useful for testing)."""
    global _cached_model, _cached_tokenizer, _cached_value_head
    _cached_model      = None
    _cached_tokenizer  = None
    _cached_value_head = None
 

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
