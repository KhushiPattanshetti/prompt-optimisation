import json
import re
from typing import List, Optional

from config import ICD10_REGEX_PATTERN
from logger import get_logger

log = get_logger("code_parser")

_ICD10_RE = re.compile(ICD10_REGEX_PATTERN)
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def validate_code(code: str) -> bool:
    return re.fullmatch(ICD10_REGEX_PATTERN, code) is not None


def parse_icd10_codes(raw_output: str) -> List[str]:
    # Strategy 1 — JSON extraction
    codes = _try_json_parse(raw_output)
    if codes is not None:
        codes = list(dict.fromkeys(codes))
        log.info("Parsed %d codes via JSON strategy", len(codes))
        return codes

    # Strategy 2 — Regex extraction
    codes = _try_regex_parse(raw_output)
    if codes:
        codes = list(dict.fromkeys(codes))
        log.info("Parsed %d codes via regex strategy", len(codes))
        return codes

    # Strategy 3 — Fallback
    log.warning("No ICD-10 codes parsed from output (fallback)")
    return []


def _try_json_parse(raw_output: str) -> Optional[List[str]]:
    # Try the full output first
    try:
        parsed = json.loads(raw_output.strip())
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, str) and validate_code(c)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON array within the output
    match = _JSON_ARRAY_RE.search(raw_output)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [c for c in parsed if isinstance(c, str) and validate_code(c)]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _try_regex_parse(raw_output: str) -> List[str]:
    matches = _ICD10_RE.findall(raw_output)
    # re.findall with groups returns tuples; rebuild full matches
    full_codes = []
    for m in re.finditer(ICD10_REGEX_PATTERN, raw_output):
        full_codes.append(m.group())
    return [c for c in full_codes if validate_code(c)]
