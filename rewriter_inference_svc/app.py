"""FastAPI application for the rewriter inference service.

Endpoint:
    POST /rewrite_prompt
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rewriter_inference_svc.config import MODEL_NAME
from rewriter_inference_svc.inference_engine import run_inference
from rewriter_inference_svc.logger import get_logger
from rewriter_inference_svc.model_loader import get_cached_model, load_model
from rewriter_inference_svc.schemas import RewriteRequest, RewriteResponse

log = get_logger(__name__)


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

    try:
        result = run_inference(request.clinical_note, note_id=request.note_id)
    except Exception as exc:
        log.exception("inference_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return RewriteResponse(
        rewritten_prompt=result["rewritten_prompt"],
        log_prob_old=result["log_prob_old"],
        value_estimate=result["value_estimate"],
        generation_source=result["generation_source"],
    )


@app.post("/reload_checkpoint")
def reload_checkpoint():
    from rewriter_inference_svc.model_loader import reload_from_latest_checkpoint

    reload_from_latest_checkpoint()
    return {"status": "reloaded"}


@app.post("/update_best_prompt")
def update_best_prompt(payload: dict):
    from rewriter_inference_svc.inference_engine import update_best_prompt_cache

    update_best_prompt_cache(
        payload["note_id"],
        payload["rewritten_prompt"],
        payload["reward"],
    )
    return {"status": "updated"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": get_cached_model() is not None,
    }
