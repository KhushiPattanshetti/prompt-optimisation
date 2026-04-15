from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import inference_engine, model_loader
from .config import MODEL_NAME
from .logger import get_logger
from .schemas import CodeRequest, CodeResponse

log = get_logger("app")


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_loader.load_model()
    yield


app = FastAPI(title="icd10_coding_svc", lifespan=lifespan)


@app.post("/generate_codes", response_model=CodeResponse)
def generate_codes(request: CodeRequest):
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
    return CodeResponse(**result)


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
