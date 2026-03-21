"""
validator.py
Deterministic validation logic for the ICD-10 User Instruction Parser.
All checks are ordered and short-circuit on first failure.
"""

import json
import os
import re

# ── Load keyword dictionary at module level (once at startup) ─────────
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "icd_keywords.json")

def _load_keywords() -> list[str]:
	with open(_CONFIG_PATH, "r") as f:
		return [kw.lower() for kw in json.load(f)]

ICD_KEYWORDS: list[str] = _load_keywords()

VALID_ACTION_VERBS: list[str] = [
	"generate", "extract", "list", "find", "identify",
	"assign", "map", "convert", "return", "provide"
]

INVALID_INTENT_PATTERNS: list[str] = [
	r"explain icd",
	r"teach (icd|coding|me)",
	r"what is icd",
	r"write a (story|blog|post|article)",
	r"create fictional",
	r"translate (this|the)",
	r"suggest treatment",
	r"recommend (medication|medicine|drug|treatment)",
]

ALLOWED_EXTENSIONS: set[str] = {".txt", ".json", ".csv"}
MIN_NOTE_LENGTH: int = 15


class ValidationResult:
	def __init__(self, ok: bool, error: str = ""):
		self.ok = ok
		self.error = error

	def to_dict(self) -> dict:
		if self.ok:
			return {"status": "OK"}
		return {"status": "NOT_OK", "error": self.error}


def validate_request(
	user_instruction: str | None,
	filename: str | None,
	file_content: str | None,
) -> ValidationResult:
	"""
	Run all validation checks in strict order.
	Returns ValidationResult with ok=True or ok=False + error message.
	"""

	# ── STEP 1: Presence check — instruction ──────────────────────────
	if not user_instruction or not user_instruction.strip():
		return ValidationResult(False, "User instruction is missing.")

	# ── STEP 2: Presence check — clinical document ────────────────────
	if not filename or not file_content:
		return ValidationResult(False, "Clinical document is missing.")

	# ── STEP 3: File format check ─────────────────────────────────────
	_, ext = os.path.splitext(filename.lower())
	if ext not in ALLOWED_EXTENSIONS:
		return ValidationResult(
			False,
			f"Unsupported file format '{ext}'. Allowed formats: .txt, .json, .csv"
		)

	# ── STEP 4: Clinical note length check ────────────────────────────
	stripped_content = file_content.strip()
	if len(stripped_content) < MIN_NOTE_LENGTH:
		return ValidationResult(
			False,
			f"Clinical document is too short (minimum {MIN_NOTE_LENGTH} characters required)."
		)

	# ── STEP 5: ICD keyword match ─────────────────────────────────────
	instruction_lower = user_instruction.lower()
	keyword_found = any(kw in instruction_lower for kw in ICD_KEYWORDS)
	if not keyword_found:
		return ValidationResult(
			False,
			"Instruction does not reference ICD-10 coding tasks."
		)

	# ── STEP 6: Action verb check ─────────────────────────────────────
	verb_found = any(verb in instruction_lower for verb in VALID_ACTION_VERBS)
	if not verb_found:
		return ValidationResult(
			False,
			"Instruction does not contain a valid action verb (e.g. extract, generate, map, identify)."
		)

	# ── STEP 7: Invalid intent check ──────────────────────────────────
	for pattern in INVALID_INTENT_PATTERNS:
		if re.search(pattern, instruction_lower):
			return ValidationResult(
				False,
				"Instruction is not related to ICD-10 coding (out-of-scope intent detected)."
			)

	return ValidationResult(True)
