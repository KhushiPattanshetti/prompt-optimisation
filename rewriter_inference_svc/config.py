"""Configuration constants for the rewriter inference service."""

import os
from pathlib import Path

# Project root is one level above the service directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

SFT_CHECKPOINT_PATH: Path = PROJECT_ROOT / "sft_checkpoints"
RL_CHECKPOINT_PATH: Path = PROJECT_ROOT / "rl_checkpoints"
OUTPUT_PATH: Path = PROJECT_ROOT / "inference_outputs"

MODEL_NAME: str = "phi-3-mini"

# Generation parameters
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "1.0"))
DO_SAMPLE: bool = os.getenv("DO_SAMPLE", "false").lower() == "true"
