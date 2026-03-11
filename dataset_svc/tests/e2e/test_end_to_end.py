"""End-to-end tests for the dataset_svc FastAPI application.

Starts the service using TestClient with fixture data and
verifies all endpoints return correct responses.
"""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dataset_svc.store import DatasetStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NOTES_ROWS = [
    {"note_id": "N001", "subject_id_x": "S1", "hadm_id": "H1", "text": "Fever and chills."},
    {"note_id": "N002", "subject_id_x": "S2", "hadm_id": "H2", "text": "Routine visit."},
    {"note_id": "N003", "subject_id_x": "S3", "hadm_id": "H3", "text": "Follow-up."},
]

DIAGNOSES_ROWS = [
    {"note_id": "N001", "seq_num": "1", "icd_code": "A01"},
    {"note_id": "N001", "seq_num": "2", "icd_code": "B02"},
    {"note_id": "N002", "seq_num": "1", "icd_code": "C03"},
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """Create a TestClient with a fixture-backed DatasetStore."""
    notes_path = tmp_path / "notes.csv"
    diag_path = tmp_path / "diagnoses.csv"
    _write_csv(notes_path, NOTES_ROWS)
    _write_csv(diag_path, DIAGNOSES_ROWS)

    store = DatasetStore(str(notes_path), str(diag_path))

    # Patch validate_data_files so it doesn't check real paths,
    # and patch the CSV paths so lifespan uses our fixtures
    with patch("dataset_svc.app.validate_data_files"), \
         patch("dataset_svc.app.NOTES_CSV_PATH", notes_path), \
         patch("dataset_svc.app.DIAGNOSES_CSV_PATH", diag_path):
        from dataset_svc.app import app
        # Pre-set the store so lifespan's DatasetStore also works
        # (lifespan will overwrite with fixture paths)
        with TestClient(app) as tc:
            yield tc


# ------------------------------------------------------------------
# Endpoint tests
# ------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health → verify status ok."""

    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["total_notes"] == 3
        assert body["total_coded_notes"] == 2
        assert "loading_time_sec" in body


class TestNoteEndpoint:
    """GET /note/{note_id} → verify text returned."""

    def test_valid_note(self, client: TestClient) -> None:
        resp = client.get("/note/N001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == "N001"
        assert body["text"] == "Fever and chills."

    def test_invalid_note(self, client: TestClient) -> None:
        resp = client.get("/note/INVALID")
        assert resp.status_code == 404


class TestGTCodesEndpoint:
    """GET /gt_codes/{note_id} → verify ordered list returned."""

    def test_valid_note(self, client: TestClient) -> None:
        resp = client.get("/gt_codes/N001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["note_id"] == "N001"
        assert body["gt_codes"] == ["A01", "B02"]

    def test_note_without_codes(self, client: TestClient) -> None:
        resp = client.get("/gt_codes/N003")
        assert resp.status_code == 404


class TestBatchEndpoint:
    """GET /batch → verify batched records returned."""

    def test_first_batch(self, client: TestClient) -> None:
        resp = client.get("/batch", params={"offset": 0, "size": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["batch"]) == 3
        assert body["total"] == 3

    def test_batch_offset_exceeds_total(self, client: TestClient) -> None:
        resp = client.get("/batch", params={"offset": 100, "size": 10})
        assert resp.status_code == 404


class TestNoteIdsEndpoint:
    """GET /note_ids → verify list returned."""

    def test_note_ids_slice(self, client: TestClient) -> None:
        resp = client.get("/note_ids", params={"offset": 0, "size": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["note_ids"], list)
        assert len(body["note_ids"]) <= 5
        assert body["total"] == 3
