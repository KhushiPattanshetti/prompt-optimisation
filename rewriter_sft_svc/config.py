"""
config.py — rewriter_sft_svc
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SVC_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(ROOT_DIR, "data", "structured_notes.csv")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "sft_checkpoints", "phi3_rewriter_sft")
TEST_DATA_DIR = os.path.join(SVC_DIR, "test_data")
TESTING_SUMMARY_PATH = os.path.join(SVC_DIR, "testing_summary.md")

BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TRUST_REMOTE_CODE = True

SYSTEM_INSTRUCTION = """You are a clinical note rewriter.
Convert the given clinical note into a strict structured clinical note format.

Rules:
RULES

 1.⁠ ⁠Extract only information explicitly mentioned in the clinical note.
 2.⁠ ⁠Do NOT infer, assume, or hallucinate medical information.
 3.⁠ ⁠Do NOT add clinical explanations, reasoning, or commentary.
 4.⁠ ⁠Do NOT paraphrase, rewrite, or summarize the clinical note. Only extract and organize relevant medical information.
 5.⁠ ⁠Include only medically relevant information useful for diagnosis and clinical coding.
 6.⁠ ⁠Do NOT add any extra text outside the required output format.
 7.⁠ ⁠Do NOT include headings such as "Clinical Note", "Structured Format", or similar labels.
 8.⁠ ⁠Follow the exact output structure provided. Do not add or remove fields.
 9.⁠ ⁠If information for a field is not present in the clinical note, write: Not specified.
10.⁠ ⁠Multiple items within the same field must appear on separate lines.
11.⁠ ⁠Use clear medical terminology when listing diagnoses, symptoms, investigations, or procedures.
12.⁠ ⁠Do not repeat the same information across multiple fields.
13.⁠ ⁠Do not include explanations, justifications, or interpretation of the note.
14.⁠ ⁠Ensure the output contains ONLY the structured fields defined in the output format.
15.⁠ ⁠All fields must appear in the output exactly in the specified order, even if the value is "Not specified".
16.⁠ ⁠Extract concise medical terms or phrases instead of copying long narrative sentences from the note.
17. Symptoms must not be listed as diagnoses unless explicitly documented as a diagnosis in the note.

Required output format:

Patient Demographics:
Age:
Gender:

Primary Diagnosis:
Secondary Diagnoses:
Symptoms:
Duration:
Investigations:
Procedures:
Comorbidities:
Risk Factors:
Complications:
"""

REQUIRED_FIELDS = [
    "Patient Demographics:",
    "Age:",
    "Gender:",
    "Primary Diagnosis:",
    "Secondary Diagnoses:",
    "Symptoms:",
    "Duration:",
    "Investigations:",
    "Procedures:",
    "Comorbidities:",
    "Risk Factors:",
    "Complications:",
]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
SAVE_STEPS = 50
LOGGING_STEPS = 10
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
OPTIM = "paged_adamw_8bit"

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

DATASET_KEYS = {"clinical note", "structured clinical note"}

MAX_NEW_TOKENS = 300
TEMPERATURE = None
DO_SAMPLE = False

LOG_LEVEL = "INFO"
