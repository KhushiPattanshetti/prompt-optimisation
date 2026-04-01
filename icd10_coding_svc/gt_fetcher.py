"""Ground truth code fetcher for icd10_coding_svc.

FIXES APPLIED:
    - Removed all direct CSV reading (pd.read_csv on diagnoses.csv).
      Architecture rule: only dataset_svc may read CSV files.
    - gt_codes are now fetched via HTTP GET from dataset_svc.
    - Local disk cache (gt_codes/{note_id}.json) is retained for
      performance — avoids repeated network calls for the same note.
    - init_datasets() no longer loads a DataFrame; it now validates
      that dataset_svc is reachable.
    - Added one retry on network failure before raising.
"""

import json
import os
import time
from typing import List

import requests

from config import DATASET_SERVICE_URL, GT_CODES_PATH
from logger import get_logger

log = get_logger("gt_fetcher")


def init_datasets() -> None:
    """Verify dataset_svc is reachable at startup.

    Raises:
        RuntimeError: If dataset_svc /health endpoint does not return ok.
    """
    health_url = f"{DATASET_SERVICE_URL}/health"
    try:
        resp = requests.get(health_url, timeout=10)
        data = resp.json()
        if data.get("status") not in ("ok", "loading"):
            raise RuntimeError(
                f"dataset_svc returned unexpected status: {data}"
            )
        log.info("dataset_svc reachable | status=%s", data.get("status"))
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach dataset_svc at {health_url}.\n"
            f"Ensure dataset_svc is running: uvicorn dataset_svc.app:app --port 8003\n"
            f"Error: {exc}"
        ) from exc


def get_gt_codes(note_id: str) -> List[str]:
    """Return ground truth ICD-10 codes for a given note_id.

    Lookup order:
        1. Local disk cache at GT_CODES_PATH/{note_id}.json
        2. HTTP GET to dataset_svc /gt_codes/{note_id}
        3. Persist result to disk cache for future calls.

    Args:
        note_id: The note identifier to look up.

    Returns:
        Ordered list of ICD-10 code strings (may be empty).
    """
    os.makedirs(GT_CODES_PATH, exist_ok=True)
    cache_file = os.path.join(GT_CODES_PATH, f"{note_id}.json")

    # Step 1: check local cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            codes = data.get("gt_codes", [])
            log.info("gt_codes_cache_hit | note_id=%s | count=%d", note_id, len(codes))
            return codes
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("gt_codes_cache_corrupt | note_id=%s | error=%s", note_id, exc)
            # Fall through to network fetch

    # Step 2: fetch from dataset_svc
    codes = _fetch_from_dataset_svc(note_id)

    # Step 3: persist to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"note_id": note_id, "gt_codes": codes}, f)
        log.info("gt_codes_cached | note_id=%s | count=%d", note_id, len(codes))
    except OSError as exc:
        log.warning("gt_codes_cache_write_failed | note_id=%s | error=%s", note_id, exc)

    return codes


def _fetch_from_dataset_svc(note_id: str, retries: int = 1) -> List[str]:
    """Fetch gt_codes from dataset_svc with one retry on failure.

    Args:
        note_id: The note identifier.
        retries: Number of retry attempts after an initial failure.

    Returns:
        List of ICD-10 code strings (empty list if note has no codes).
    """
    url = f"{DATASET_SERVICE_URL}/gt_codes/{note_id}"

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                codes = resp.json().get("gt_codes", [])
                log.info(
                    "gt_codes_fetched | note_id=%s | count=%d", note_id, len(codes)
                )
                return codes
            if resp.status_code == 404:
                log.warning("gt_codes_not_found | note_id=%s", note_id)
                return []
            log.warning(
                "gt_codes_unexpected_status | note_id=%s | status=%d",
                note_id, resp.status_code,
            )
        except requests.RequestException as exc:
            log.warning(
                "gt_codes_fetch_error | note_id=%s | attempt=%d | error=%s",
                note_id, attempt + 1, exc,
            )
            if attempt < retries:
                time.sleep(2)

    log.error("gt_codes_fetch_failed | note_id=%s | all retries exhausted", note_id)
    return []

# import json
# import os
# from typing import List, Optional

# import pandas as pd

# from config import GT_CODES_PATH, DIAGNOSES_CSV_PATH
# from logger import get_logger

# log = get_logger("gt_fetcher")

# # Module-level DataFrames — loaded once at startup via init_datasets()
# _diagnoses_df: Optional[pd.DataFrame] = None


# def init_datasets() -> None:
#     global _diagnoses_df
#     _diagnoses_df = pd.read_csv(DIAGNOSES_CSV_PATH, dtype={"note_id": str})
#     log.info("Loaded diagnoses.csv with %d rows", len(_diagnoses_df))


# def get_gt_codes(note_id: str) -> List[str]:
#     os.makedirs(GT_CODES_PATH, exist_ok=True)
#     cache_path = os.path.join(GT_CODES_PATH, f"{note_id}.json")

#     # Disk cache check
#     if os.path.isfile(cache_path):
#         with open(cache_path, "r") as f:
#             data = json.load(f)
#         codes = data.get("gt_codes", [])
#         log.info("gt_codes for %s loaded from disk (%d codes)", note_id, len(codes))
#         return codes

#     # Query DataFrame
#     if _diagnoses_df is None:
#         log.warning("Diagnoses DataFrame not loaded; returning empty gt_codes for %s", note_id)
#         return []

#     matches = _diagnoses_df[_diagnoses_df["note_id"] == str(note_id)]
#     codes = matches["icd_code"].tolist()

#     # Persist to disk
#     payload = {"note_id": str(note_id), "gt_codes": codes}
#     with open(cache_path, "w") as f:
#         json.dump(payload, f, indent=2)

#     source = "dataset"
#     log.info("gt_codes for %s fetched from %s (%d codes)", note_id, source, len(codes))
#     return codes
