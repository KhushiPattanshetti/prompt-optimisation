"""FastAPI application for the rewriter inference service.

Endpoint:
    POST /rewrite_prompt
"""

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import MODEL_NAME
from .inference_engine import run_inference, run_inference_batch, update_best_prompt_cache
from .logger import get_logger
from .model_loader import get_cached_model, load_model, reload_from_latest_checkpoint
from .schemas import RewriteRequest, RewriteResponse

log = get_logger(__name__)

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline_logger import ServiceIOLogger
    from pipeline_logger.hash_utils import prompt_hash

    _io = ServiceIOLogger("rewriter_svc")
    _PRETTY_LOG = True
except Exception:
    _PRETTY_LOG = False


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Rewriter Inference Service",
    description="Inference service for the phi-3-mini prompt rewriter (PPO actor).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/rewrite_prompt", response_model=RewriteResponse)
def rewrite_prompt(request: RewriteRequest) -> RewriteResponse:
    """Accept a clinical note and return a rewritten prompt with PPO metrics.

    Steps:
        1. Validate request using schema.
        2. Call inference engine.
        3. Return RewriteResponse.
    """
    log.info("request_received | note_length=%d", len(request.clinical_note))

    if _PRETTY_LOG:
        _io.log_input(
            note_id=request.note_id or "—",
            prompt_hash=prompt_hash(request.clinical_note),
            note_length=len(request.clinical_note),
        )

    t0 = time.perf_counter()
    try:
        result = run_inference(
            request.clinical_note,
            note_id=request.note_id,
            sampling_nonce=request.sampling_nonce,
            disable_best_prompt_cache=request.disable_best_prompt_cache,
        )
    except Exception as exc:
        log.exception("inference_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if _PRETTY_LOG:
        _io.log_output(
            note_id=request.note_id or "—",
            rewritten_hash=prompt_hash(result["rewritten_prompt"]),
            generation_source=result["generation_source"],
            rejection_reason=result.get("rejection_reason"),
            elapsed_ms=elapsed_ms,
        )

    return RewriteResponse(
        rewritten_prompt=result["rewritten_prompt"],
        log_prob_old=result["log_prob_old"],
        value_estimate=result["value_estimate"],
        generation_source=result["generation_source"],
        rejection_reason=result.get("rejection_reason"),
        rejection_reason_counts=dict(result.get("rejection_reason_counts", {})),
    )


@app.post("/rewrite_prompt_batch", response_model=list[RewriteResponse])
def rewrite_prompt_batch(requests: list[RewriteRequest]) -> list[RewriteResponse]:
    for request in requests:
        log.info("batch_request_received | note_length=%d", len(request.clinical_note))

    payload = [
        {
            "note_id": request.note_id,
            "clinical_note": request.clinical_note,
            "sampling_nonce": request.sampling_nonce,
            "disable_best_prompt_cache": request.disable_best_prompt_cache,
        }
        for request in requests
    ]

    try:
        batch_results = run_inference_batch(payload)
    except Exception as exc:
        log.exception("batch_inference_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail="Batch inference failed.") from exc

    return [
        RewriteResponse(
            rewritten_prompt=result["rewritten_prompt"],
            log_prob_old=result["log_prob_old"],
            value_estimate=result["value_estimate"],
            generation_source=result["generation_source"],
            rejection_reason=result.get("rejection_reason"),
            rejection_reason_counts=dict(result.get("rejection_reason_counts", {})),
        )
        for result in batch_results
    ]


@app.post("/reload_checkpoint")
def reload_checkpoint():
    reload_from_latest_checkpoint()
    return {"status": "reloaded"}


class UpdateBestPromptRequest(BaseModel):
    note_id: str
    rewritten_prompt: str
    reward: float


@app.post("/update_best_prompt")
def update_best_prompt(payload: UpdateBestPromptRequest):
    update_best_prompt_cache(
        payload.note_id,
        payload.rewritten_prompt,
        payload.reward,
    )
    return {"status": "updated"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": get_cached_model() is not None,
    }
