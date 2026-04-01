"""Unit tests for Pydantic schemas.

Verifies that all response models validate and reject correctly.
"""

import pytest
from pydantic import ValidationError

from dataset_svc.schemas import (
    BatchRecord,
    BatchResponse,
    HealthResponse,
    NoteIdsResponse,
    NoteResponse,
)


class TestNoteResponse:
    """Verify NoteResponse schema validation."""

    def test_valid(self) -> None:
        resp = NoteResponse(note_id="N001", text="some text")
        assert resp.note_id == "N001"
        assert resp.text == "some text"

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            NoteResponse()  # type: ignore[call-arg]

    def test_rejects_missing_text(self) -> None:
        with pytest.raises(ValidationError):
            NoteResponse(note_id="N001")  # type: ignore[call-arg]


class TestBatchRecord:
    """Verify BatchRecord validates correctly."""

    def test_valid(self) -> None:
        rec = BatchRecord(note_id="N001", text="hello", gt_codes=["A01", "B02"])
        assert rec.note_id == "N001"
        assert rec.gt_codes == ["A01", "B02"]

    def test_rejects_missing_gt_codes(self) -> None:
        with pytest.raises(ValidationError):
            BatchRecord(note_id="N001", text="hello")  # type: ignore[call-arg]


class TestBatchResponse:
    """Verify BatchResponse total field is accurate."""

    def test_valid(self) -> None:
        rec = BatchRecord(note_id="N001", text="t", gt_codes=["C03"])
        resp = BatchResponse(batch=[rec], offset=0, size=1, total=100)
        assert resp.total == 100
        assert len(resp.batch) == 1

    def test_empty_batch(self) -> None:
        resp = BatchResponse(batch=[], offset=50, size=0, total=100)
        assert resp.batch == []
        assert resp.total == 100


class TestNoteIdsResponse:
    """Verify NoteIdsResponse validates."""

    def test_valid(self) -> None:
        resp = NoteIdsResponse(note_ids=["N001", "N002"], offset=0, size=2, total=10)
        assert resp.note_ids == ["N001", "N002"]


class TestHealthResponse:
    """Verify HealthResponse validates with different statuses."""

    def test_status_loading(self) -> None:
        resp = HealthResponse(
            status="loading",
            total_notes=0,
            total_coded_notes=0,
            loading_time_sec=0.0,
        )
        assert resp.status == "loading"

    def test_status_ok(self) -> None:
        resp = HealthResponse(
            status="ok",
            total_notes=122000,
            total_coded_notes=110000,
            loading_time_sec=18.456,
        )
        assert resp.status == "ok"
        assert resp.total_notes == 122000
        assert resp.loading_time_sec == 18.456
