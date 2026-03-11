import os

MODEL_NAME = "m42-health/Llama3-Med42-8B"
LOCAL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "sft_checkpoints", "med42")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "inference_outputs", "icd10")
GT_CODES_PATH = os.path.join(os.path.dirname(__file__), "..", "gt_codes")
NOTES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "notes.csv")
DIAGNOSES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diagnoses.csv")
REWARD_SERVICE_URL = os.environ.get("REWARD_SERVICE_URL", "http://localhost:8002")

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
DO_SAMPLE = False
REPETITION_PENALTY = 1.1

SYSTEM_INSTRUCTION = (
    "You are a clinical coding expert. "
    "Extract all ICD-10-CM codes from the given clinical note. "
    "Output codes as a JSON list only. "
    "Do not include explanations."
)

ICD10_REGEX_PATTERN = r"[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?"
