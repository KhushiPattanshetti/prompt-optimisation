"""Configuration constants for the rewriter inference service."""

import os
from pathlib import Path

# Project root is one level above the service directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# SFT_CHECKPOINT_PATH: Path = PROJECT_ROOT / "sft_checkpoints"
RL_CHECKPOINT_PATH: Path = PROJECT_ROOT / "rl_checkpoints"
OUTPUT_PATH: Path = PROJECT_ROOT / "inference_outputs"

# FIXED: was "phi-3-mini" which is not a valid HuggingFace model ID
MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
 
# Generation parameters
# FIXED: TEMPERATURE 1.0 → 0.7, DO_SAMPLE False → True for diverse rewrites
MAX_NEW_TOKENS: int   = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE:    float = float(os.getenv("TEMPERATURE",   "0.7"))
DO_SAMPLE:      bool  = os.getenv("DO_SAMPLE", "true").lower() == "true"
 
# 4-bit quantization — required to fit Phi-3 + Med42 on a single 24GB GPU
LOAD_IN_4BIT:           bool  = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
BNB_4BIT_QUANT_TYPE:    str   = "nf4"
BNB_4BIT_COMPUTE_DTYPE: str   = "bfloat16"
 
# LoRA adapter configuration
LORA_R:              int   = 16
LORA_ALPHA:          int   = 32
LORA_DROPOUT:        float = 0.05
LORA_TARGET_MODULES: list  = ["q_proj", "k_proj", "v_proj", "o_proj"]
 
# Value head hidden size — must match Phi-3-mini hidden dimension
VALUE_HEAD_HIDDEN_SIZE: int = 3072
# # Generation parameters
# MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
# TEMPERATURE: float = float(os.getenv("TEMPERATURE", "1.0"))
# DO_SAMPLE: bool = os.getenv("DO_SAMPLE", "false").lower() == "true"
