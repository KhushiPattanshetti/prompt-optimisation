"""Model loader for icd10_coding_svc.

Loads m42-health/Llama3-Med42-8B in 4-bit NF4 quantization.
Freezes all weights immediately after loading.
Caches model instance in memory for the lifetime of the service.

FIX SUMMARY (from review):
  - Added get_cached_model() so app.py can check weights_frozen in /health.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import MODEL_NAME, LOCAL_CACHE_PATH
from logger import get_logger

log = get_logger("model_loader")

_cached_model:     Optional[PreTrainedModel]         = None
_cached_tokenizer: Optional[PreTrainedTokenizerBase] = None


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load and cache the frozen Med42 model and tokenizer.

    Loading strategy:
        1. If LOCAL_CACHE_PATH exists → load from local cache.
        2. Otherwise → download from HuggingFace and save to cache.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _cached_model, _cached_tokenizer

    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer

    cache_exists = os.path.isdir(LOCAL_CACHE_PATH)
    source = LOCAL_CACHE_PATH if cache_exists else MODEL_NAME

    if cache_exists:
        log.info("Loading Med42 from local cache: %s", LOCAL_CACHE_PATH)
    else:
        log.info(
            "Local cache not found. Downloading from HuggingFace: %s",
            MODEL_NAME,
        )

    bnb_config = _build_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Freeze all weights — this model must never be updated
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    log.info("Model weights frozen | model=%s", MODEL_NAME)

    # Save to local cache if we downloaded from HuggingFace
    if not cache_exists:
        os.makedirs(LOCAL_CACHE_PATH, exist_ok=True)
        model.save_pretrained(LOCAL_CACHE_PATH)
        tokenizer.save_pretrained(LOCAL_CACHE_PATH)
        log.info("Model cached locally at: %s", LOCAL_CACHE_PATH)

    _cached_model     = model
    _cached_tokenizer = tokenizer
    return model, tokenizer


def get_cached_model() -> Optional[PreTrainedModel]:
    """Return the cached model instance, or None if not yet loaded."""
    return _cached_model

# import os

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# from config import MODEL_NAME, LOCAL_CACHE_PATH
# from logger import get_logger

# log = get_logger("model_loader")

# _model = None
# _tokenizer = None


# def load_model():
#     global _model, _tokenizer

#     if _model is not None and _tokenizer is not None:
#         return _model, _tokenizer

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     if os.path.isdir(LOCAL_CACHE_PATH) and os.listdir(LOCAL_CACHE_PATH):
#         log.info("Loading model from local cache: %s", LOCAL_CACHE_PATH)
#         model = AutoModelForCausalLM.from_pretrained(
#             LOCAL_CACHE_PATH,
#             quantization_config=bnb_config,
#             device_map="auto",
#         )
#         tokenizer = AutoTokenizer.from_pretrained(LOCAL_CACHE_PATH)
#         source = "cache"
#     else:
#         log.info("Downloading model from HuggingFace: %s", MODEL_NAME)
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             quantization_config=bnb_config,
#             device_map="auto",
#         )
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#         os.makedirs(LOCAL_CACHE_PATH, exist_ok=True)
#         model.save_pretrained(LOCAL_CACHE_PATH)
#         tokenizer.save_pretrained(LOCAL_CACHE_PATH)
#         source = "HuggingFace"

#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False
#     model.eval()

#     _model = model
#     _tokenizer = tokenizer

#     log.info("Model loaded and frozen (source: %s)", source)
#     return _model, _tokenizer
