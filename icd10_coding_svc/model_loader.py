import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_NAME, LOCAL_CACHE_PATH
from logger import get_logger

log = get_logger("model_loader")

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if os.path.isdir(LOCAL_CACHE_PATH) and os.listdir(LOCAL_CACHE_PATH):
        log.info("Loading model from local cache: %s", LOCAL_CACHE_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_CACHE_PATH,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_CACHE_PATH)
        source = "cache"
    else:
        log.info("Downloading model from HuggingFace: %s", MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        os.makedirs(LOCAL_CACHE_PATH, exist_ok=True)
        model.save_pretrained(LOCAL_CACHE_PATH)
        tokenizer.save_pretrained(LOCAL_CACHE_PATH)
        source = "HuggingFace"

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    _model = model
    _tokenizer = tokenizer

    log.info("Model loaded and frozen (source: %s)", source)
    return _model, _tokenizer
