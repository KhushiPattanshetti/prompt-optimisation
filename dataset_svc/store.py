"""Dataset store — loads CSV files and provides all data access methods.

This is the only module permitted to read CSV files.
All other modules access data through this class.
"""

from __future__ import annotations

import time

import pandas as pd

from dataset_svc.logger import get_logger

logger = get_logger(__name__)


class DatasetStore:
    """In-memory store for clinical notes and ICD-10 diagnosis codes.

    Loads notes.csv and diagnoses.csv once at construction time,
    builds indexed lookups, and exposes read-only access methods.
    """

    def __init__(self, notes_path: str, diagnoses_path: str) -> None:
        """Initialise the store by loading and indexing both CSV files.

        Args:
            notes_path: Filesystem path to notes.csv.
            diagnoses_path: Filesystem path to diagnoses.csv.
        """
        self.notes_index: dict[str, str] = {}
        self.gt_codes_index: dict[str, list[str]] = {}
        self.note_ids_list: list[str] = []
        self.loading_time_sec: float = 0.0

        start = time.perf_counter()
        self._load(notes_path, diagnoses_path)
        self.loading_time_sec = round(time.perf_counter() - start, 3)

        logger.info(
            "Service ready | total_notes=%d | total_coded_notes=%d | loading_time=%.3fs",
            self.get_total_notes(),
            self.get_total_coded_notes(),
            self.loading_time_sec,
        )

    def _load(self, notes_path: str, diagnoses_path: str) -> None:
        """Load both CSVs and build in-memory indexes.

        Args:
            notes_path: Filesystem path to notes.csv.
            diagnoses_path: Filesystem path to diagnoses.csv.
        """
        # --- Load notes.csv ---
        logger.info("Loading notes.csv from %s", notes_path)
        t0 = time.perf_counter()
        notes_df = pd.read_csv(
            notes_path,
            usecols=["note_id", "text"],
            dtype={"note_id": str, "text": str},
            engine="c",
        )
        logger.info(
            "notes.csv loaded | rows=%d | columns=%s | time=%.3fs",
            len(notes_df),
            list(notes_df.columns),
            time.perf_counter() - t0,
        )

        # Build notes_index: note_id → text
        self.notes_index = dict(zip(notes_df["note_id"], notes_df["text"]))
        self.note_ids_list = list(notes_df["note_id"])
        del notes_df

        # --- Load diagnoses.csv ---
        logger.info("Loading diagnoses.csv from %s", diagnoses_path)
        t1 = time.perf_counter()
        diag_df = pd.read_csv(
            diagnoses_path,
            usecols=["note_id", "seq_num", "icd_code"],
            dtype={"note_id": str, "icd_code": str},
            engine="c",
        )
        logger.info(
            "diagnoses.csv loaded | rows=%d | time=%.3fs",
            len(diag_df),
            time.perf_counter() - t1,
        )

        # Sort by note_id then seq_num, group into ordered lists
        diag_df = diag_df.sort_values(["note_id", "seq_num"])
        self.gt_codes_index = (
            diag_df.groupby("note_id")["icd_code"]
            .apply(list)
            .to_dict()
        )
        del diag_df

        logger.info(
            "Index built | notes_indexed=%d | codes_indexed=%d",
            len(self.notes_index),
            len(self.gt_codes_index),
        )

    # ------------------------------------------------------------------
    # Public data access methods
    # ------------------------------------------------------------------

    def get_note(self, note_id: str) -> str | None:
        """Return the clinical note text for a given note_id.

        Args:
            note_id: The unique identifier of the note.

        Returns:
            The note text, or None if the note_id is not found.
        """
        return self.notes_index.get(note_id)

    def get_gt_codes(self, note_id: str) -> list[str]:
        """Return the ordered list of ICD-10 codes for a given note_id.

        Args:
            note_id: The unique identifier of the note.

        Returns:
            A list of ICD-10 code strings ordered by seq_num,
            or an empty list if the note_id has no codes.
        """
        return self.gt_codes_index.get(note_id, [])

    def get_batch(self, offset: int, size: int) -> list[dict]:
        """Return a batch of records for training iteration.

        Args:
            offset: Starting index into the note_ids_list.
            size: Number of records to return.

        Returns:
            A list of dicts, each with keys note_id, text, gt_codes.
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
        """Return a slice of all note IDs.

        Args:
            offset: Starting index into the note_ids_list.
            size: Number of IDs to return.

        Returns:
            A list of note_id strings.
        """
        return self.note_ids_list[offset : offset + size]

    def get_total_notes(self) -> int:
        """Return the total number of notes loaded.

        Returns:
            Count of entries in notes_index.
        """
        return len(self.notes_index)

    def get_total_coded_notes(self) -> int:
        """Return the number of notes that have ground-truth codes.

        Returns:
            Count of unique note_ids in gt_codes_index.
        """
        return len(self.gt_codes_index)
