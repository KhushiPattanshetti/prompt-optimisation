"""
dataset_loader.py — rewriter_sft_svc

Loads and validates the paired clinical dataset.
Supports CSV datasets with columns:
original_note, optimized_note
"""

import logging
import os
from typing import List, Dict, Any, Tuple

import pandas as pd

from config import DATASET_PATH, DATASET_KEYS, VALIDATION_SPLIT, TEST_SPLIT

logger = logging.getLogger(__name__)


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert CSV column names to internal schema
    """

    normalized = {}

    if "original_note" in record:
        normalized["clinical note"] = record["original_note"]

    if "optimized_note" in record:
        normalized["structured clinical note"] = record["optimized_note"]

    return normalized


def load_raw_dataset(path: str = DATASET_PATH) -> List[Dict[str, Any]]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    records = df.to_dict(orient="records")

    records = [normalize_record(r) for r in records]

    logger.info(f"Loaded {len(records)} records")

    return records


def validate_schema(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    valid = []

    for i, r in enumerate(records):

        if "clinical note" not in r or "structured clinical note" not in r:
            logger.warning(f"Record {i} missing keys. Skipping.")
            continue

        if not isinstance(r["clinical note"], str) or not r["clinical note"].strip():
            continue

        if not isinstance(r["structured clinical note"], str) or not r["structured clinical note"].strip():
            continue

        valid.append(r)

    logger.info(f"{len(valid)} valid records after validation")

    return valid


def split_dataset(
    records: List[Dict[str, Any]],
    val_split: float = VALIDATION_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = 42,
) -> Tuple[List, List, List]:

    import random

    random.seed(seed)
    random.shuffle(records)

    n = len(records)

    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test = records[:n_test]
    val = records[n_test:n_test + n_val]
    train = records[n_test + n_val:]

    logger.info(f"Dataset split → train:{len(train)} val:{len(val)} test:{len(test)}")

    return train, val, test


def get_datasets(path: str = DATASET_PATH) -> Tuple[List, List, List]:

    raw = load_raw_dataset(path)

    valid = validate_schema(raw)

    return split_dataset(valid)