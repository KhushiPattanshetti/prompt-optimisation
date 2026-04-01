"""FastAPI application for the rewriter inference service.

Endpoint:
    POST /rewrite_prompt
"""

from fastapi import FastAPI, HTTPException

from inference_engine import run_inference
from logger import get_logger
from schemas import RewriteRequest, RewriteResponse

log = get_logger(__name__)

app = FastAPI(
    title="Rewriter Inference Service",
    description="Inference service for the phi-3-mini prompt rewriter (PPO actor).",
    version="1.0.0",
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
        result = run_inference(request.clinical_note)
    except Exception as exc:
        log.exception("inference_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return RewriteResponse(
        rewritten_prompt=result["rewritten_prompt"],
        log_prob_old=result["log_prob_old"],
        value_estimate=result["value_estimate"],
    )
