"""
storage.py
Persists validated user instructions to instruction_dataset.json.
These are later consumed by the ICD-10 LLM Coding Block.
"""

import json
import os
import uuid
from datetime import datetime, timezone

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_BASE, "..", "dataset", "instruction_dataset.json")
_USER_INSTRUCTIONS_DIR = os.path.join(_BASE, "..", "dataset", "user_instructions")


def _ensure_paths():
	os.makedirs(os.path.dirname(_DATASET_PATH), exist_ok=True)
	os.makedirs(_USER_INSTRUCTIONS_DIR, exist_ok=True)
	if not os.path.exists(_DATASET_PATH):
		with open(_DATASET_PATH, "w") as f:
			json.dump([], f)


def store_instruction(user_instruction: str) -> dict:
	"""
	Append user instruction to the instruction dataset JSON file.
	Also writes a flat file in dataset/user_instructions/.
	Returns the stored record.
	"""
	_ensure_paths()

	record = {
		"instruction_id": f"ins_{uuid.uuid4().hex[:8]}",
		"user_instruction": user_instruction,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	# ── Append to instruction_dataset.json ─────────────────────────────
	with open(_DATASET_PATH, "r") as f:
		dataset = json.load(f)

	dataset.append(record)

	with open(_DATASET_PATH, "w") as f:
		json.dump(dataset, f, indent=2)

	# ── Write individual file to dataset/user_instructions/ ────────────
	flat_path = os.path.join(_USER_INSTRUCTIONS_DIR, f"{record['instruction_id']}.json")
	with open(flat_path, "w") as f:
		json.dump(record, f, indent=2)

	return record


def get_all_instructions() -> list[dict]:
	"""Return all stored user instructions for inspection."""
	_ensure_paths()
	with open(_DATASET_PATH, "r") as f:
		return json.load(f)
