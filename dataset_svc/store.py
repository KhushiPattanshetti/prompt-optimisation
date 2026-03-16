"""Dataset store — loads CSV files and provides all data access methods.

This is the only module permitted to read CSV files.
All other modules access data through this class.

notes.csv is loaded using a custom line-by-line reader that correctly handles
multiline text fields. diagnoses.csv is loaded using pd.read_csv().
"""

from __future__ import annotations

import re
import time

import pandas as pd

from dataset_svc.logger import get_logger

logger = get_logger(__name__)


class DatasetStore:
    """In-memory store for clinical notes and ICD-10 diagnosis codes.

    Loads notes.csv via a custom multiline-aware reader and diagnoses.csv
    via pd.read_csv(), builds indexed lookups, and exposes read-only
    access methods.
    """

    def __init__(self, notes_path: str, diagnoses_path: str) -> None:
        self.notes_index: dict[str, str] = {}
        self.gt_codes_index: dict[str, list[str]] = {}
        self.note_ids_list: list[str] = []
        self.loading_time_sec: float = 0.0

        start = time.perf_counter()
        self._load_notes(notes_path)
        self._load_diagnoses(diagnoses_path)
        self.loading_time_sec = round(time.perf_counter() - start, 3)

        logger.info(
            "Service ready | total_notes=%d | total_coded_notes=%d | loading_time=%.3fs",
            self.get_total_notes(),
            self.get_total_coded_notes(),
            self.loading_time_sec,
        )

    # ------------------------------------------------------------------
    # Private loading methods
    # ------------------------------------------------------------------

    def _load_notes(self, notes_path: str) -> None:
        """Load notes.csv using a custom row-boundary reader.

        notes.csv contains multiline text fields that standard CSV parsers
        cannot handle correctly. Row boundaries are identified by a regex
        pattern matching the note_id,subject_id,hadm_id prefix.
        Duplicate note_ids are deduplicated, keeping only the first occurrence.
        """
        logger.info("Loading notes.csv from %s", notes_path)
        t0 = time.perf_counter()

        pattern = re.compile(r"^\d+-[A-Z]+-\d+,\d+,\d+,")
        current_row: str | None = None
        seen_note_ids: set[str] = set()

        with open(notes_path, encoding="utf-8") as fh:
            fh.readline()  # discard header line
            for line in fh:
                if pattern.match(line):
                    if current_row is not None:
                        self._flush_note_row(current_row, seen_note_ids)
                    current_row = line
                else:
                    if current_row is not None:
                        current_row += line

        if current_row is not None:
            self._flush_note_row(current_row, seen_note_ids)

        self.note_ids_list = list(self.notes_index.keys())
        logger.info(
            "notes.csv loaded | unique_notes=%d | time=%.3fs",
            len(self.notes_index),
            time.perf_counter() - t0,
        )

    def _flush_note_row(self, row: str, seen: set[str]) -> None:
        """Parse a fully-accumulated row string and add it to the index.

        Splits on comma with maxsplit=3 to extract note_id and text,
        skipping the note if note_id was already seen (deduplication).
        If the row has fewer than 4 parts, text defaults to empty string.
        """
        parts = row.split(",", maxsplit=3)
        note_id = parts[0].strip()
        text = parts[3].strip() if len(parts) >= 4 else ""
        if note_id not in seen:
            self.notes_index[note_id] = text
            seen.add(note_id)

    def _load_diagnoses(self, diagnoses_path: str) -> None:
        """Load diagnoses.csv using pd.read_csv() and build gt_codes_index."""
        logger.info("Loading diagnoses.csv from %s", diagnoses_path)
        t0 = time.perf_counter()

        diag_df = pd.read_csv(
            diagnoses_path,
            usecols=["note_id", "seq_num", "icd_code"],
            dtype={"note_id": str, "seq_num": int, "icd_code": str},
            engine="c",
        )
        diag_df = diag_df.sort_values(["note_id", "seq_num"])
        self.gt_codes_index = (
            diag_df.groupby("note_id")["icd_code"]
            .apply(list)
            .to_dict()
        )
        del diag_df

        logger.info(
            "diagnoses.csv loaded | coded_notes=%d | time=%.3fs",
            len(self.gt_codes_index),
            time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Public data access methods
    # ------------------------------------------------------------------

    def get_note(self, note_id: str) -> str | None:
        """Return the clinical note text for a given note_id.

        Returns None if the note_id is not found.
        """
        return self.notes_index.get(note_id)

    def get_gt_codes(self, note_id: str) -> list[str]:
        """Return the ordered list of ICD-10 codes for a given note_id.

        Returns an empty list if the note_id has no codes.
        """
        return self.gt_codes_index.get(note_id, [])

    def get_batch(self, offset: int, size: int) -> list[dict]:
        """Return a batch of records for training iteration.

        Each record contains note_id, text, and gt_codes.
        Returns an empty list if offset >= total notes.
        """
        if offset >= len(self.note_ids_list):
            return []

        batch_ids = self.note_ids_list[offset : offset + size]
        return [
            {
                "note_id": nid,
                "text": self.notes_index.get(nid, ""),
                "gt_codes": self.gt_codes_index.get(nid, []),
            }
            for nid in batch_ids
        ]

    def get_note_ids(self, offset: int, size: int) -> list[str]:
        """Return a slice of all note IDs."""
        return self.note_ids_list[offset : offset + size]

    def get_total_notes(self) -> int:
        """Return the total number of unique notes loaded."""
        return len(self.notes_index)

    def get_total_coded_notes(self) -> int:
        """Return the number of notes that have ground-truth codes."""
        return len(self.gt_codes_index)
