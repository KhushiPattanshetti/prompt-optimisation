"""Unit tests for DatasetStore.

All tests use small in-memory CSV fixtures written to tmp_path.
The real notes.csv and diagnoses.csv are never loaded.
"""

import csv
from pathlib import Path

import pytest

from dataset_svc.store import DatasetStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NOTES_ROWS = [
    {"note_id": "N001", "subject_id_x": "S1", "hadm_id": "H1", "text": "Patient presented with fever."},
    {"note_id": "N002", "subject_id_x": "S2", "hadm_id": "H2", "text": "Routine check-up, no complaints."},
    {"note_id": "N003", "subject_id_x": "S3", "hadm_id": "H3", "text": "Post-operative follow-up."},
]

DIAGNOSES_ROWS = [
    {"note_id": "N001", "seq_num": "2", "icd_code": "A01"},
    {"note_id": "N001", "seq_num": "1", "icd_code": "B02"},
    {"note_id": "N001", "seq_num": "3", "icd_code": "C03"},
    {"note_id": "N002", "seq_num": "1", "icd_code": "D04"},
    {"note_id": "N002", "seq_num": "2", "icd_code": "E05"},
    # N003 intentionally has no diagnoses
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    """Return a DatasetStore loaded from small fixture CSVs."""
    notes_path = tmp_path / "notes.csv"
    diag_path = tmp_path / "diagnoses.csv"
    _write_csv(notes_path, NOTES_ROWS)
    _write_csv(diag_path, DIAGNOSES_ROWS)
    return DatasetStore(str(notes_path), str(diag_path))


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestNotesIndex:
    """Verify notes_index builds correctly from sample CSV."""

    def test_all_note_ids_indexed(self, store: DatasetStore) -> None:
        assert set(store.notes_index.keys()) == {"N001", "N002", "N003"}

    def test_text_content(self, store: DatasetStore) -> None:
        assert store.notes_index["N001"] == "Patient presented with fever."


class TestGTCodesIndex:
    """Verify gt_codes_index builds with correct seq_num ordering."""

    def test_codes_ordered_by_seq_num(self, store: DatasetStore) -> None:
        # seq_num order: 1→B02, 2→A01, 3→C03
        assert store.gt_codes_index["N001"] == ["B02", "A01", "C03"]

    def test_multiple_codes(self, store: DatasetStore) -> None:
        assert store.gt_codes_index["N002"] == ["D04", "E05"]


class TestGetNote:
    """Verify get_note returns correct text."""

    def test_known_note_id(self, store: DatasetStore) -> None:
        assert store.get_note("N002") == "Routine check-up, no complaints."

    def test_unknown_note_id(self, store: DatasetStore) -> None:
        assert store.get_note("UNKNOWN") is None


class TestGetGTCodes:
    """Verify get_gt_codes returns codes in seq_num order."""

    def test_known_note_id(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("N001") == ["B02", "A01", "C03"]

    def test_unknown_note_id(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("UNKNOWN") == []

    def test_note_with_no_diagnoses(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("N003") == []


class TestGetBatch:
    """Verify get_batch returns correct slice."""

    def test_first_batch(self, store: DatasetStore) -> None:
        batch = store.get_batch(0, 2)
        assert len(batch) == 2
        assert batch[0]["note_id"] == "N001"
        assert batch[1]["note_id"] == "N002"
        assert "text" in batch[0]
        assert "gt_codes" in batch[0]

    def test_batch_beyond_total(self, store: DatasetStore) -> None:
        batch = store.get_batch(100, 10)
        assert batch == []

    def test_batch_at_boundary(self, store: DatasetStore) -> None:
        batch = store.get_batch(2, 10)
        assert len(batch) == 1
        assert batch[0]["note_id"] == "N003"


class TestTotals:
    """Verify total counts match loaded data."""

    def test_total_notes(self, store: DatasetStore) -> None:
        assert store.get_total_notes() == 3

    def test_total_coded_notes(self, store: DatasetStore) -> None:
        # Only N001 and N002 have diagnoses
        assert store.get_total_coded_notes() == 2


class TestGetNoteIds:
    """Verify get_note_ids returns a slice of note_ids_list."""

    def test_slice(self, store: DatasetStore) -> None:
        ids = store.get_note_ids(0, 2)
        assert ids == ["N001", "N002"]

    def test_offset_past_end(self, store: DatasetStore) -> None:
        ids = store.get_note_ids(100, 5)
        assert ids == []
