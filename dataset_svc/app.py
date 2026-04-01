"""FastAPI application for dataset_svc.

Loads the dataset at startup via lifespan context manager and
exposes REST endpoints for all data access in the pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query

from dataset_svc.config import (
    DEFAULT_BATCH_SIZE,
    DIAGNOSES_CSV_PATH,
    MAX_BATCH_SIZE,
    NOTES_CSV_PATH,
    validate_data_files,
)
from dataset_svc.logger import get_logger
from dataset_svc.schemas import (
    BatchResponse,
    GTCodesResponse,
    HealthResponse,
    NoteIdsResponse,
    NoteResponse,
)
from dataset_svc.store import DatasetStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load dataset into memory before the service accepts requests."""
    # Step 1: validate data files exist
    try:
        validate_data_files()
        logger.info("Data files validated")
    except FileNotFoundError as exc:
        logger.error("Data file validation failed: %s", exc)
        raise SystemExit(str(exc)) from exc

    # Step 2: load and index data
    logger.info(
        "Loading started | notes=%s | diagnoses=%s",
        NOTES_CSV_PATH,
        DIAGNOSES_CSV_PATH,
    )
    store = DatasetStore(
        notes_path=str(NOTES_CSV_PATH),
        diagnoses_path=str(DIAGNOSES_CSV_PATH),
    )

    # Step 3: attach to app state
    application.state.store = store
    application.state.ready = True

    yield


app = FastAPI(title="dataset_svc", lifespan=lifespan)
app.state.ready = False


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/note/{note_id}", response_model=NoteResponse)
async def get_note(note_id: str) -> NoteResponse:
    """Return the clinical note text for a given note_id."""
    store: DatasetStore = app.state.store
    text = store.get_note(note_id)
    if text is None:
        logger.warning("note_id not found: %s", note_id)
        raise HTTPException(status_code=404, detail="note_id not found")
    return NoteResponse(note_id=note_id, text=text)


@app.get("/gt_codes/{note_id}", response_model=GTCodesResponse)
async def get_gt_codes(note_id: str) -> GTCodesResponse:
    """Return the ordered list of ground-truth ICD-10 codes for a note_id."""
    store: DatasetStore = app.state.store
    codes = store.get_gt_codes(note_id)
    if not codes:
        logger.warning("note_id not found or has no codes: %s", note_id)
        raise HTTPException(
            status_code=404, detail="note_id not found or has no codes"
        )
    return GTCodesResponse(note_id=note_id, gt_codes=codes)


@app.get("/batch", response_model=BatchResponse)
async def get_batch(
    offset: int = Query(default=0, ge=0),
    size: int = Query(default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE),
) -> BatchResponse:
    """Return a batch of training records."""
    store: DatasetStore = app.state.store
    total = store.get_total_notes()

    if offset >= total:
        logger.warning("Batch out of range | offset=%d | total=%d", offset, total)
        raise HTTPException(status_code=404, detail="offset exceeds total notes")

    batch = store.get_batch(offset, size)
    return BatchResponse(batch=batch, offset=offset, size=len(batch), total=total)


@app.get("/note_ids", response_model=NoteIdsResponse)
async def get_note_ids(
    offset: int = Query(default=0, ge=0),
    size: int = Query(default=DEFAULT_BATCH_SIZE, ge=1),
) -> NoteIdsResponse:
    """Return a slice of all note IDs."""
    store: DatasetStore = app.state.store
    total = store.get_total_notes()
    note_ids = store.get_note_ids(offset, size)
    return NoteIdsResponse(
        note_ids=note_ids, offset=offset, size=len(note_ids), total=total
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status and dataset statistics."""
    if not getattr(app.state, "ready", False):
        return HealthResponse(
            status="loading",
            total_notes=0,
            total_coded_notes=0,
            loading_time_sec=0.0,
        )

    store: DatasetStore = app.state.store
    return HealthResponse(
        status="ok",
        total_notes=store.get_total_notes(),
        total_coded_notes=store.get_total_coded_notes(),
        loading_time_sec=store.loading_time_sec,
    )
