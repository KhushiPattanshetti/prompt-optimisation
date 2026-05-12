import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RL_CHECKPOINT_PATH: Path = Path(
	os.environ.get("RL_CHECKPOINT_DIR",
	str(PROJECT_ROOT / "rl_checkpoints"))
)
OUTPUT_PATH: Path = PROJECT_ROOT / "inference_outputs"

MODEL_NAME: str = os.getenv("REWRITER_MODEL_NAME", "ishanmane/phi3-rewriter-sft")
REWRITER_LOAD_LOCAL_CHECKPOINTS: bool = (
	os.getenv("REWRITER_LOAD_LOCAL_CHECKPOINTS", "false").strip().lower() == "true"
)

MAX_NEW_TOKENS: int   = int(os.getenv("MAX_NEW_TOKENS", "256"))
MAX_NEW_TOKENS_HARD_CAP: int = int(os.getenv("REWRITER_MAX_NEW_TOKENS_HARD_CAP", "96"))
TEMPERATURE:    float = float(os.getenv("TEMPERATURE",  "0.7"))
DO_SAMPLE:      bool  = os.getenv("DO_SAMPLE", "true").lower() == "true"
TOP_P:          float = float(os.getenv("TOP_P", "0.92"))
TOP_K:          int   = int(os.getenv("TOP_K", "50"))
MAX_PROMPT_TOKENS: int = int(os.getenv("REWRITER_MAX_PROMPT_TOKENS", "4096"))
HARD_MAX_PROMPT_TOKENS: int = int(os.getenv("REWRITER_HARD_MAX_PROMPT_TOKENS", "3072"))
MAX_BATCH_ITEMS: int = int(os.getenv("REWRITER_MAX_BATCH_ITEMS", "4"))
MAX_RAW_NOTE_CHARS: int = int(os.getenv("REWRITER_MAX_RAW_NOTE_CHARS", "12000"))

LORA_R:           int   = 16
LORA_ALPHA:       int   = 32
LORA_DROPOUT:     float = 0.05
LORA_TARGET_MODULES: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

VALUE_HEAD_HIDDEN_SIZE: int = 3072

BEST_PROMPT_CACHE_THRESHOLD: float = 0.0
BEST_PROMPT_CACHE_FILE: Path = PROJECT_ROOT / "best_prompts_cache" / "cache.json"

MIN_REWRITE_CHARS: int = int(os.getenv("REWRITER_MIN_REWRITE_CHARS", "20"))
