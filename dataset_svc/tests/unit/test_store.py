"""Unit tests for DatasetStore.

All tests use small in-memory CSV fixture strings written to tmp_path.
The real notes.csv and diagnoses.csv are never loaded.

notes.csv fixtures simulate the multiline text behaviour of the real dataset,
including continuations, duplicates, embedded commas, and rows with fewer
than 4 comma-separated parts.
"""

import csv
from pathlib import Path

import pytest

from dataset_svc.store import DatasetStore


# ------------------------------------------------------------------
# Fixture content
# ------------------------------------------------------------------

# Raw notes.csv content simulating multiline text behaviour:
#   19-JAN-2024  — text spans three lines (embedded newlines)
#   20-FEB-2024  — single-line text
#   21-MAR-2024  — text contains embedded commas
NOTES_CSV = (
    "note_id,subject_id_x,hadm_id,text\n"
    "19-JAN-2024,1001,2001,Patient presents with\n"
    "chest pain. History of\n"
    "diabetes mellitus.\n"
    "20-FEB-2024,1002,2002,Shortness of breath.\n"
    "21-MAR-2024,1003,2003,Normal exam, no acute distress, comma test.\n"
)

# Fixture with a duplicate note_id; the first occurrence must be kept.
NOTES_CSV_WITH_DUPE = (
    "note_id,subject_id_x,hadm_id,text\n"
    "19-JAN-2024,1001,2001,First occurrence.\n"
    "20-FEB-2024,1002,2002,Other note.\n"
    "19-JAN-2024,1001,2001,Duplicate occurrence.\n"
)

DIAGNOSES_ROWS = [
    {"note_id": "19-JAN-2024", "seq_num": "2", "icd_code": "A01"},
    {"note_id": "19-JAN-2024", "seq_num": "1", "icd_code": "B02"},
    {"note_id": "19-JAN-2024", "seq_num": "3", "icd_code": "C03"},
    {"note_id": "20-FEB-2024", "seq_num": "1", "icd_code": "D04"},
    {"note_id": "20-FEB-2024", "seq_num": "2", "icd_code": "E05"},
    # 21-MAR-2024 intentionally has no diagnoses
]


def _write_diagnoses(path: Path, rows: list[dict]) -> None:
    """Write diagnoses rows to a CSV file using csv.DictWriter."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    """DatasetStore loaded from multiline-aware fixture CSVs."""
    notes_path = tmp_path / "notes.csv"
    diag_path = tmp_path / "diagnoses.csv"
    notes_path.write_text(NOTES_CSV, encoding="utf-8")
    _write_diagnoses(diag_path, DIAGNOSES_ROWS)
    return DatasetStore(str(notes_path), str(diag_path))


@pytest.fixture()
def store_with_dupe(tmp_path: Path) -> DatasetStore:
    """DatasetStore loaded from a notes CSV with a duplicate note_id."""
    notes_path = tmp_path / "notes.csv"
    diag_path = tmp_path / "diagnoses.csv"
    notes_path.write_text(NOTES_CSV_WITH_DUPE, encoding="utf-8")
    diag_path.write_text("note_id,seq_num,icd_code\n", encoding="utf-8")
    return DatasetStore(str(notes_path), str(diag_path))


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestNotesIndex:
    """Verify notes_index builds correctly from multiline notes fixture."""

    def test_all_note_ids_indexed(self, store: DatasetStore) -> None:
        assert set(store.notes_index.keys()) == {
            "19-JAN-2024",
            "20-FEB-2024",
            "21-MAR-2024",
        }

    def test_multiline_text_preserved(self, store: DatasetStore) -> None:
        """Text spanning multiple lines must be reconstructed with newlines."""
        expected = "Patient presents with\nchest pain. History of\ndiabetes mellitus."
        assert store.notes_index["19-JAN-2024"] == expected

    def test_single_line_text(self, store: DatasetStore) -> None:
        assert store.notes_index["20-FEB-2024"] == "Shortness of breath."

    def test_embedded_commas_preserved(self, store: DatasetStore) -> None:
        """Commas inside the text field must not be treated as delimiters."""
        assert store.notes_index["21-MAR-2024"] == "Normal exam, no acute distress, comma test."


class TestDeduplication:
    """Verify duplicate note_ids are filtered, keeping only the first."""

    def test_first_occurrence_kept(self, store_with_dupe: DatasetStore) -> None:
        assert store_with_dupe.notes_index["19-JAN-2024"] == "First occurrence."

    def test_duplicate_count(self, store_with_dupe: DatasetStore) -> None:
        """Only two unique notes should be indexed despite three rows."""
        assert store_with_dupe.get_total_notes() == 2


class TestMalformedRow:
    """Verify _flush_note_row handles rows with fewer than 4 parts safely."""

    def test_fewer_than_4_parts_no_exception(self) -> None:
        """A row with fewer than 4 comma-separated parts must not raise."""
        store = DatasetStore.__new__(DatasetStore)
        store.notes_index = {}
        seen: set[str] = set()
        store._flush_note_row("19-JAN-2024,1001,2001", seen)
        assert store.notes_index["19-JAN-2024"] == ""

    def test_fewer_than_4_parts_note_id_indexed(self) -> None:
        """note_id is still indexed even when text field is absent."""
        store = DatasetStore.__new__(DatasetStore)
        store.notes_index = {}
        seen: set[str] = set()
        store._flush_note_row("19-JAN-2024,1001,2001", seen)
        assert "19-JAN-2024" in store.notes_index


class TestGTCodesIndex:
    """Verify gt_codes_index builds with correct seq_num ordering."""

    def test_codes_ordered_by_seq_num(self, store: DatasetStore) -> None:
        # seq_num order: 1→B02, 2→A01, 3→C03
        assert store.gt_codes_index["19-JAN-2024"] == ["B02", "A01", "C03"]

    def test_multiple_codes(self, store: DatasetStore) -> None:
        assert store.gt_codes_index["20-FEB-2024"] == ["D04", "E05"]


class TestGetNote:
    """Verify get_note returns correct text including multiline and commas."""

    def test_known_note_id(self, store: DatasetStore) -> None:
        assert store.get_note("20-FEB-2024") == "Shortness of breath."

    def test_unknown_note_id(self, store: DatasetStore) -> None:
        assert store.get_note("UNKNOWN") is None

    def test_multiline_text_via_get_note(self, store: DatasetStore) -> None:
        expected = "Patient presents with\nchest pain. History of\ndiabetes mellitus."
        assert store.get_note("19-JAN-2024") == expected

    def test_embedded_commas_via_get_note(self, store: DatasetStore) -> None:
        assert store.get_note("21-MAR-2024") == "Normal exam, no acute distress, comma test."


class TestGetGTCodes:
    """Verify get_gt_codes returns codes in seq_num order."""

    def test_known_note_id(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("19-JAN-2024") == ["B02", "A01", "C03"]

    def test_unknown_note_id(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("UNKNOWN") == []

    def test_note_with_no_diagnoses(self, store: DatasetStore) -> None:
        assert store.get_gt_codes("21-MAR-2024") == []


class TestGetBatch:
    """Verify get_batch returns correct slice."""

    def test_first_batch(self, store: DatasetStore) -> None:
        batch = store.get_batch(0, 2)
        assert len(batch) == 2
        assert batch[0]["note_id"] == "19-JAN-2024"
        assert batch[1]["note_id"] == "20-FEB-2024"
        assert "text" in batch[0]
        assert "gt_codes" in batch[0]

    def test_batch_beyond_total(self, store: DatasetStore) -> None:
        batch = store.get_batch(100, 10)
        assert batch == []

    def test_batch_at_boundary(self, store: DatasetStore) -> None:
        batch = store.get_batch(2, 10)
        assert len(batch) == 1
        assert batch[0]["note_id"] == "21-MAR-2024"


class TestTotals:
    """Verify total counts match loaded data."""

    def test_total_notes(self, store: DatasetStore) -> None:
        assert store.get_total_notes() == 3

    def test_total_coded_notes(self, store: DatasetStore) -> None:
        # Only 19-JAN-2024 and 20-FEB-2024 have diagnoses
        assert store.get_total_coded_notes() == 2


class TestGetNoteIds:
    """Verify get_note_ids returns a slice of note_ids_list."""

    def test_slice(self, store: DatasetStore) -> None:
        ids = store.get_note_ids(0, 2)
        assert ids == ["19-JAN-2024", "20-FEB-2024"]

    def test_offset_past_end(self, store: DatasetStore) -> None:
        ids = store.get_note_ids(100, 5)
        assert ids == []
