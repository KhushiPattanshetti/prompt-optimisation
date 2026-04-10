"""
preprocessing.py — rewriter_sft_svc

Converts raw {clinical note, structured clinical note} records into
Phi-3 Mini chat-style training messages and tokenizes them.
"""

import logging
from typing import List, Dict, Any

from config import SYSTEM_INSTRUCTION, MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)


# ── Message Conversion ─────────────────────────────────────────────────────────

def record_to_messages(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw dataset record into the OpenAI-style messages format expected
    by the Phi-3 chat template.

    Input format:
        {"clinical note": "...", "structured clinical note": "..."}

    Output format:
        {
            "messages": [
                {"role": "system",    "content": SYSTEM_INSTRUCTION},
                {"role": "user",      "content": "<clinical note>"},
                {"role": "assistant", "content": "<structured clinical note>"}
            ]
        }
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_INSTRUCTION},
            {"role": "user",      "content": record["clinical note"].strip()},
            {"role": "assistant", "content": record["structured clinical note"].strip()},
        ]
    }


def convert_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply record_to_messages to every record in the dataset."""
    converted = [record_to_messages(r) for r in records]
    logger.info(f"Converted {len(converted)} records to chat-message format.")
    return converted


# ── Tokenisation ───────────────────────────────────────────────────────────────

def apply_chat_template(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Apply the model's chat template to a messages dict and return a dict with
    a single 'text' key, which is what TRL's SFTTrainer expects via
    dataset_text_field or formatting_func.
    """
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def preprocess_for_training(records: List[Dict[str, Any]], tokenizer) -> List[Dict[str, str]]:
    """
    Full preprocessing pipeline:
    1. Convert records to chat messages.
    2. Apply Phi-3 chat template.
    3. Return a list of {"text": "..."} dicts ready for SFTTrainer.
    """
    chat_records = convert_dataset(records)
    texts = [apply_chat_template(r, tokenizer) for r in chat_records]
    logger.info(f"Preprocessing complete: {len(texts)} training texts generated.")
    return texts


# ── Output Validation Helpers ──────────────────────────────────────────────────

def validate_output_schema(output_text: str, required_fields: List[str]) -> Dict[str, Any]:
    """
    Validate that a model output contains all required fields in the correct order.

    Returns:
        {
            "valid": bool,
            "missing_fields": list,
            "wrong_order": bool,
            "extra_text_detected": bool,
        }
    """
    result = {
        "valid": True,
        "missing_fields": [],
        "wrong_order": False,
        "extra_text_detected": False,
    }

    # Check for missing fields
    for field in required_fields:
        if field not in output_text:
            result["missing_fields"].append(field)
            result["valid"] = False

    # Check field ordering
    if not result["missing_fields"]:
        positions = [output_text.index(f) for f in required_fields if f in output_text]
        if positions != sorted(positions):
            result["wrong_order"] = True
            result["valid"] = False

    # Heuristic check for leaked prompt / rules text
    forbidden_phrases = [
        "Do not add",
        "Do NOT infer",
        "RULES:",
        "Output Format",
        "System:",
        "system instruction",
        "clinical note:",       # original note leaking through
        "Structured Format",
    ]
    for phrase in forbidden_phrases:
        if phrase.lower() in output_text.lower():
            result["extra_text_detected"] = True
            result["valid"] = False
            break

    return result
