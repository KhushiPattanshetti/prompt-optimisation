"""Integration tests for the dataset_svc pipeline.

Loads DatasetStore with sample CSV fixtures and verifies
data access methods return correct types and values.
Also tests config.validate_data_files behaviour.
"""

import csv
from pathlib import Path

import pytest

from dataset_svc.config import validate_data_files, NOTES_CSV_PATH, DIAGNOSES_CSV_PATH
from dataset_svc.store import DatasetStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NOTES_ROWS = [
    {"note_id": "N001", "subject_id_x": "S1", "hadm_id": "H1", "text": "Fever and chills."},
    {"note_id": "N002", "subject_id_x": "S2", "hadm_id": "H2", "text": "Routine visit."},
    {"note_id": "N003", "subject_id_x": "S3", "hadm_id": "H3", "text": "Follow-up."},
    {"note_id": "N004", "subject_id_x": "S4", "hadm_id": "H4", "text": "Discharge summary."},
    {"note_id": "N005", "subject_id_x": "S5", "hadm_id": "H5", "text": "Admission note."},
]

DIAGNOSES_ROWS = [
    {"note_id": "N001", "seq_num": "1", "icd_code": "A01"},
    {"note_id": "N001", "seq_num": "2", "icd_code": "B02"},
    {"note_id": "N002", "seq_num": "1", "icd_code": "C03"},
    {"note_id": "N003", "seq_num": "1", "icd_code": "D04"},
    {"note_id": "N003", "seq_num": "2", "icd_code": "E05"},
    {"note_id": "N003", "seq_num": "3", "icd_code": "F06"},
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    notes_path = tmp_path / "notes.csv"
    diag_path = tmp_path / "diagnoses.csv"
    _write_csv(notes_path, NOTES_ROWS)
    _write_csv(diag_path, DIAGNOSES_ROWS)
    return DatasetStore(str(notes_path), str(diag_path))


# ------------------------------------------------------------------
# Data access pipeline tests
# ------------------------------------------------------------------


class TestPipelineSequence:
    """Call get_note, get_gt_codes, get_batch in sequence and validate."""

    def test_get_note_returns_correct_text(self, store: DatasetStore) -> None:
        text = store.get_note("N001")
        assert isinstance(text, str)
        assert text == "Fever and chills."

    def test_get_gt_codes_returns_correct_codes(self, store: DatasetStore) -> None:
        codes = store.get_gt_codes("N001")
        assert isinstance(codes, list)
        assert codes == ["A01", "B02"]

    def test_get_batch_returns_correct_records(self, store: DatasetStore) -> None:
        batch = store.get_batch(0, 3)
        assert isinstance(batch, list)
        assert len(batch) == 3
        for rec in batch:
            assert "note_id" in rec
            assert "text" in rec
            assert "gt_codes" in rec

    def test_batch_iteration_covers_all_with_no_overlap(
        self, store: DatasetStore
    ) -> None:
        """Iterate through all records with batch_size=2 and verify
        every note is covered exactly once."""
        seen_ids: list[str] = []
        total = store.get_total_notes()
        offset = 0
        batch_size = 2

        while offset < total:
            batch = store.get_batch(offset, batch_size)
            for rec in batch:
                seen_ids.append(rec["note_id"])
            offset += batch_size

        # All note_ids appear exactly once
        assert len(seen_ids) == total
        assert len(set(seen_ids)) == total


# ------------------------------------------------------------------
# validate_data_files tests
# ------------------------------------------------------------------


class TestValidateDataFiles:
    """Verify validate_data_files raises correctly."""

    def test_raises_when_files_missing(self) -> None:
        # The real CSV files should not exist in the test environment
        # If they do, this test is a no-op, which is acceptable.
        # We test by monkeypatching the paths to non-existent locations.
        import dataset_svc.config as cfg

        original_notes = cfg.NOTES_CSV_PATH
        original_diag = cfg.DIAGNOSES_CSV_PATH
        try:
            cfg.NOTES_CSV_PATH = Path("/tmp/nonexistent_notes_12345.csv")
            cfg.DIAGNOSES_CSV_PATH = Path("/tmp/nonexistent_diag_12345.csv")
            with pytest.raises(FileNotFoundError, match="Missing data files"):
                validate_data_files()
        finally:
            cfg.NOTES_CSV_PATH = original_notes
            cfg.DIAGNOSES_CSV_PATH = original_diag

    def test_raises_when_one_file_missing(self, tmp_path: Path) -> None:
        import dataset_svc.config as cfg

        original_notes = cfg.NOTES_CSV_PATH
        original_diag = cfg.DIAGNOSES_CSV_PATH

        # Create only notes.csv
        notes_path = tmp_path / "notes.csv"
        notes_path.write_text("note_id,text\n")

        try:
            cfg.NOTES_CSV_PATH = notes_path
            cfg.DIAGNOSES_CSV_PATH = Path("/tmp/nonexistent_diag_12345.csv")
            with pytest.raises(FileNotFoundError, match="Missing data files"):
                validate_data_files()
        finally:
            cfg.NOTES_CSV_PATH = original_notes
            cfg.DIAGNOSES_CSV_PATH = original_diag

    def test_passes_when_files_present(self, tmp_path: Path) -> None:
        import dataset_svc.config as cfg

        original_notes = cfg.NOTES_CSV_PATH
        original_diag = cfg.DIAGNOSES_CSV_PATH

        notes_path = tmp_path / "notes.csv"
        diag_path = tmp_path / "diagnoses.csv"
        notes_path.write_text("note_id,text\n")
        diag_path.write_text("note_id,seq_num,icd_code\n")

        try:
            cfg.NOTES_CSV_PATH = notes_path
            cfg.DIAGNOSES_CSV_PATH = diag_path
            # Should not raise
            validate_data_files()
        finally:
            cfg.NOTES_CSV_PATH = original_notes
            cfg.DIAGNOSES_CSV_PATH = original_diag
