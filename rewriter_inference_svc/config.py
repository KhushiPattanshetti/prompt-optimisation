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

# Controls how the rewriter model is used:
# - base: model refines rule-filtered clinical text (default)
# - sft: model can consume raw note text, while rule filters still validate/fallback
REWRITER_MODEL_VARIANT: str = os.getenv("REWRITER_MODEL_VARIANT", "base").strip().lower()

# Controls model input source:
# - dynamic: choose filtered for base, raw for sft
# - filtered: always use rule-filtered prompt as model input
# - raw: always use truncated raw note as model input
REWRITER_INPUT_MODE: str = os.getenv("REWRITER_INPUT_MODE", "dynamic").strip().lower()

# Enable semantic guardrail checks against rule-extracted descriptors.
REWRITER_ENABLE_POSTFILTER_GUARD: bool = (
	os.getenv("REWRITER_ENABLE_POSTFILTER_GUARD", "true").strip().lower() == "true"
)

# Minimum semantic keyword overlap required when guardrail is enforced.
REWRITER_GUARD_MIN_KEYWORD_HITS: int = int(os.getenv("REWRITER_GUARD_MIN_KEYWORD_HITS", "2"))

MAX_NEW_TOKENS: int   = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE:    float = float(os.getenv("TEMPERATURE",  "0.7"))
DO_SAMPLE:      bool  = os.getenv("DO_SAMPLE", "false").lower() == "true"
MAX_PROMPT_TOKENS: int = int(os.getenv("REWRITER_MAX_PROMPT_TOKENS", "1024"))
MAX_RAW_NOTE_CHARS: int = int(os.getenv("REWRITER_MAX_RAW_NOTE_CHARS", "3000"))

LORA_R:           int   = 16
LORA_ALPHA:       int   = 32
LORA_DROPOUT:     float = 0.05
LORA_TARGET_MODULES: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

VALUE_HEAD_HIDDEN_SIZE: int = 3072

BEST_PROMPT_CACHE_THRESHOLD: float = 0.0
BEST_PROMPT_CACHE_FILE: Path = PROJECT_ROOT / "best_prompts_cache" / "cache.json"

# Medical term dictionary used by semantic descriptor extraction.
MEDICAL_DESCRIPTOR_TERMS: list[str] = [
	"diabetes",
	"diabetes mellitus",
	"type 2 diabetes",
	"hypertension",
	"hypertensive heart disease",
	"hypotension",
	"hyperlipidemia",
	"hypercholesterolemia",
	"dyslipidemia",
	"obesity",
	"metabolic syndrome",
	"hypothyroidism",
	"hyperthyroidism",
	"heart failure",
	"congestive heart failure",
	"coronary artery disease",
	"myocardial infarction",
	"angina",
	"atrial fibrillation",
	"mitral valve prolapse",
	"arrhythmia",
	"cardiomyopathy",
	"peripheral vascular disease",
	"deep vein thrombosis",
	"pulmonary embolism",
	"stroke",
	"transient ischemic attack",
	"cerebrovascular disease",
	"dementia",
	"alzheimer disease",
	"parkinson disease",
	"parkinsonism",
	"seizure disorder",
	"migraine",
	"visual hallucinations",
	"gait disturbance",
	"delirium",
	"depression",
	"anxiety",
	"bipolar disorder",
	"chronic kidney disease",
	"acute kidney injury",
	"end stage renal disease",
	"nephrolithiasis",
	"kidney stone",
	"hydronephrosis",
	"hematuria",
	"urinary tract infection",
	"pyelonephritis",
	"urinary retention",
	"chronic obstructive pulmonary disease",
	"copd",
	"asthma",
	"pneumonia",
	"respiratory failure",
	"acute respiratory failure",
	"sleep apnea",
	"sepsis",
	"bacteremia",
	"cellulitis",
	"influenza",
	"covid",
	"hepatitis",
	"cirrhosis",
	"ascites",
	"portal hypertension",
	"gastroesophageal reflux disease",
	"gerd",
	"gastritis",
	"peptic ulcer disease",
	"gastrointestinal bleeding",
	"pancreatitis",
	"colitis",
	"anemia",
	"iron deficiency anemia",
	"thrombocytopenia",
	"coagulopathy",
	"fracture",
	"hip fracture",
	"femur fracture",
	"compression fracture",
	"osteoporosis",
	"osteoarthritis",
	"rheumatoid arthritis",
	"chronic pain",
	"back pain",
	"bladder cancer",
	"prostate cancer",
	"lung cancer",
	"breast cancer",
	"colon cancer",
	"metastatic cancer",
	"malignancy",
	"neoplasm",
	"hiv disease",
	"dehydration",
	"electrolyte imbalance",
	"hyponatremia",
	"hyperkalemia",
]

# Backward-compatible alias for existing consumers.
COMMON_ICD_RELEVANT_TERMS: list[str] = list(MEDICAL_DESCRIPTOR_TERMS)
# MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
# TEMPERATURE: float = float(os.getenv("TEMPERATURE", "1.0"))
# DO_SAMPLE: bool = os.getenv("DO_SAMPLE", "false").lower() == "true"
