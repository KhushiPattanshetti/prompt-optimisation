import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI

from . import inference_engine, model_loader
from .config import MODEL_NAME
from .logger import get_logger
from .schemas import CodeRequest, CodeResponse

log = get_logger("app")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline_logger import ServiceIOLogger
    from pipeline_logger.hash_utils import prompt_hash

    _io = ServiceIOLogger("icd10_svc")
    _PRETTY_LOG = True
except Exception:
    _PRETTY_LOG = False


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_loader.load_model()
    yield


app = FastAPI(title="icd10_coding_svc", lifespan=lifespan)


@app.post("/generate_codes", response_model=CodeResponse)
def generate_codes(request: CodeRequest):
    if _PRETTY_LOG:
        _io.log_input(
            note_id=request.note_id,
            rewritten_hash=prompt_hash(request.rewritten_prompt),
            original_hash=prompt_hash(request.original_prompt),
        )

    t0 = time.perf_counter()
    result = inference_engine.run_inference(
        note_id=request.note_id,
        run_id=request.run_id,
        group_id=request.group_id,
        original_prompt=request.original_prompt,
        rewritten_prompt=request.rewritten_prompt,
        generation_source=request.generation_source,
        log_prob_old=request.log_prob_old,
        value_estimate=request.value_estimate,
        skip_reward_forward=request.skip_reward_forward,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if _PRETTY_LOG:
        _io.log_output(
            note_id=request.note_id,
            enh_codes=", ".join(result.get("enh_codes", [])) or "—",
            og_codes=", ".join(result.get("org_codes", [])) or "—",
            gt_codes=", ".join(result.get("gt_codes", [])) or "—",
            elapsed_ms=elapsed_ms,
        )

    return CodeResponse(**result)


@app.post("/generate_codes_batch", response_model=List[CodeResponse])
def generate_codes_batch(requests: List[CodeRequest]):
    responses: List[CodeResponse] = []
    for request in requests:
        result = inference_engine.run_inference(
            note_id=request.note_id,
            run_id=request.run_id,
            group_id=request.group_id,
            original_prompt=request.original_prompt,
            rewritten_prompt=request.rewritten_prompt,
            generation_source=request.generation_source,
            log_prob_old=request.log_prob_old,
            value_estimate=request.value_estimate,
            skip_reward_forward=request.skip_reward_forward,
        )
        responses.append(CodeResponse(**result))
    return responses


@app.get("/health")
def health():
    model = model_loader.get_cached_model()
    weights_frozen = False
    if model is not None:
        weights_frozen = all(not p.requires_grad for p in model.parameters())

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "weights_frozen": weights_frozen,
    }


@app.get("/observability")
def observability():
    return inference_engine.get_observability_snapshot()
