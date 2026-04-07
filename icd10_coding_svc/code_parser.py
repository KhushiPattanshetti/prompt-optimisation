import json
import re
from typing import List, Optional

from icd10_coding_svc.config import ICD10_REGEX_PATTERN
from icd10_coding_svc.logger import get_logger

log = get_logger("code_parser")

_RELAXED_ICD10_RE = re.compile(r"\b[A-Za-z][0-9]{2}(?:\.?[A-Za-z0-9]{1,4})?\b")
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _normalize_code(code: str) -> Optional[str]:
    candidate = str(code or "").strip().upper()
    if not candidate:
        return None

    if "." not in candidate and len(candidate) > 3:
        candidate = f"{candidate[:3]}.{candidate[3:]}"

    if validate_code(candidate):
        return candidate
    return None


def validate_code(code: str) -> bool:
    return re.fullmatch(ICD10_REGEX_PATTERN, code) is not None


def parse_icd10_codes(raw_output: str, warn_on_empty: bool = True) -> List[str]:
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
    if warn_on_empty:
        log.warning("No ICD-10 codes parsed from output (fallback)")
    return []


def _try_json_parse(raw_output: str) -> Optional[List[str]]:
    # Try the full output first
    try:
        parsed = json.loads(raw_output.strip())
        if isinstance(parsed, list):
            normalized = [_normalize_code(c) for c in parsed if isinstance(c, str)]
            return [c for c in normalized if c]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON array within the output
    match = _JSON_ARRAY_RE.search(raw_output)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                normalized = [_normalize_code(c) for c in parsed if isinstance(c, str)]
                return [c for c in normalized if c]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _try_regex_parse(raw_output: str) -> List[str]:
    full_codes = []

    for m in re.finditer(ICD10_REGEX_PATTERN, raw_output):
        normalized = _normalize_code(m.group())
        if normalized:
            full_codes.append(normalized)

    for m in _RELAXED_ICD10_RE.finditer(raw_output):
        normalized = _normalize_code(m.group())
        if normalized:
            full_codes.append(normalized)

    return full_codes
