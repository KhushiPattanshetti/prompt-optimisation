"""Pydantic request and response schemas for dataset_svc API."""

from __future__ import annotations

from pydantic import BaseModel


class NoteResponse(BaseModel):
    """Response schema for a single clinical note."""

    note_id: str
    text: str


class GTCodesResponse(BaseModel):
    """Response schema for ground-truth ICD-10 codes of a note."""

    note_id: str
    gt_codes: list[str]


class BatchRecord(BaseModel):
    """A single record within a training batch."""

    note_id: str
    text: str
    gt_codes: list[str]


class BatchResponse(BaseModel):
    """Response schema for a batch of training records."""

    batch: list[BatchRecord]
    offset: int
    size: int
    total: int


class NoteIdsResponse(BaseModel):
    """Response schema for a slice of note IDs."""

    note_ids: list[str]
    offset: int
    size: int
    total: int


class HealthResponse(BaseModel):
    """Response schema for service health check."""

    status: str
    total_notes: int
    total_coded_notes: int
    loading_time_sec: float
