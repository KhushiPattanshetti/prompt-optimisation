import json
import os
from typing import List

import requests

from icd10_coding_svc.config import DATASET_SVC_URL, GT_CODES_PATH
from icd10_coding_svc.logger import get_logger

log = get_logger("gt_fetcher")


def _canonicalize_code(code: str) -> str:
    normalized = str(code).strip().upper()
    if len(normalized) > 3 and "." not in normalized:
        normalized = f"{normalized[:3]}.{normalized[3:]}"
    return normalized


def _canonicalize_codes(codes: List[str]) -> List[str]:
    out: List[str] = []
    for code in codes:
        canonical = _canonicalize_code(code)
        if canonical:
            out.append(canonical)
    return out


def _cache_file(note_id: str) -> str:
    os.makedirs(GT_CODES_PATH, exist_ok=True)
    return os.path.join(GT_CODES_PATH, f"{note_id}.json")


def get_gt_codes(note_id: str) -> List[str]:
    cache_file = _cache_file(note_id)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "gt_codes" in data:
                return _canonicalize_codes(list(data.get("gt_codes", [])))
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    url = f"{DATASET_SVC_URL}/gt_codes/{note_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()

    codes = _canonicalize_codes(list(payload.get("gt_codes", [])))
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"note_id": note_id, "gt_codes": codes}, f, indent=2)

    return codes
