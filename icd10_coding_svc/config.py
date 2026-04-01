"""Configuration for icd10_coding_svc.

FIXES APPLIED:
    - Removed NOTES_CSV_PATH and DIAGNOSES_CSV_PATH.
      This service must never read CSV files directly.
      All dataset access must go through dataset_svc via HTTP.
    - Added DATASET_SERVICE_URL for gt_codes lookup.
    - GT_CODES_PATH retained only as a local disk cache.
"""

import os

MODEL_NAME        = "m42-health/Llama3-Med42-8B"
LOCAL_CACHE_PATH  = os.path.join(os.path.dirname(__file__), "..", "sft_checkpoints", "med42")
OUTPUT_PATH       = os.path.join(os.path.dirname(__file__), "..", "inference_outputs", "icd10")
GT_CODES_PATH     = os.path.join(os.path.dirname(__file__), "..", "gt_codes")

# Service URLs
# FIXED: dataset_svc is the only permitted source for gt_codes
DATASET_SERVICE_URL = os.environ.get("DATASET_SERVICE_URL", "http://localhost:8003")
REWARD_SERVICE_URL  = os.environ.get("REWARD_SERVICE_URL",  "http://localhost:8002")

MAX_NEW_TOKENS      = 256
TEMPERATURE         = 0.1
DO_SAMPLE           = False
REPETITION_PENALTY  = 1.1

SYSTEM_INSTRUCTION = (
    "You are a clinical coding expert. "
    "Extract all ICD-10-CM codes from the given clinical note. "
    "Output codes as a JSON list only. "
    "Do not include explanations."
)

ICD10_REGEX_PATTERN = r"[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?"

# import os

# MODEL_NAME = "m42-health/Llama3-Med42-8B"
# LOCAL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "sft_checkpoints", "med42")
# OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "inference_outputs", "icd10")
# GT_CODES_PATH = os.path.join(os.path.dirname(__file__), "..", "gt_codes")
# NOTES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "notes.csv")
# DIAGNOSES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diagnoses.csv")
# REWARD_SERVICE_URL = os.environ.get("REWARD_SERVICE_URL", "http://localhost:8002")

# MAX_NEW_TOKENS = 256
# TEMPERATURE = 0.1
# DO_SAMPLE = False
# REPETITION_PENALTY = 1.1

# SYSTEM_INSTRUCTION = (
#     "You are a clinical coding expert. "
#     "Extract all ICD-10-CM codes from the given clinical note. "
#     "Output codes as a JSON list only. "
#     "Do not include explanations."
# )

# ICD10_REGEX_PATTERN = r"[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?"
