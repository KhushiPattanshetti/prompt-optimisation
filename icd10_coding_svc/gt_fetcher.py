import json
import os
from typing import List, Optional

import pandas as pd

from config import GT_CODES_PATH, DIAGNOSES_CSV_PATH
from logger import get_logger

log = get_logger("gt_fetcher")

# Module-level DataFrames — loaded once at startup via init_datasets()
_diagnoses_df: Optional[pd.DataFrame] = None


def init_datasets() -> None:
    global _diagnoses_df
    _diagnoses_df = pd.read_csv(DIAGNOSES_CSV_PATH, dtype={"note_id": str})
    log.info("Loaded diagnoses.csv with %d rows", len(_diagnoses_df))


def get_gt_codes(note_id: str) -> List[str]:
    os.makedirs(GT_CODES_PATH, exist_ok=True)
    cache_path = os.path.join(GT_CODES_PATH, f"{note_id}.json")

    # Disk cache check
    if os.path.isfile(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        codes = data.get("gt_codes", [])
        log.info("gt_codes for %s loaded from disk (%d codes)", note_id, len(codes))
        return codes

    # Query DataFrame
    if _diagnoses_df is None:
        log.warning("Diagnoses DataFrame not loaded; returning empty gt_codes for %s", note_id)
        return []

    matches = _diagnoses_df[_diagnoses_df["note_id"] == str(note_id)]
    codes = matches["icd_code"].tolist()

    # Persist to disk
    payload = {"note_id": str(note_id), "gt_codes": codes}
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)

    source = "dataset"
    log.info("gt_codes for %s fetched from %s (%d codes)", note_id, source, len(codes))
    return codes
