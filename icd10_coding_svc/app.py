import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import gt_fetcher
import inference_engine
import model_loader
from config import MODEL_NAME, OUTPUT_PATH, GT_CODES_PATH
from logger import get_logger
from schemas import CodeRequest, CodeResponse

log = get_logger("app")


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup
    log.info("Starting icd10_coding_svc…")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(GT_CODES_PATH, exist_ok=True)
    gt_fetcher.init_datasets()
    model_loader.load_model()
    log.info("Startup complete.")
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(title="icd10_coding_svc", lifespan=lifespan)


@app.post("/generate_codes", response_model=CodeResponse)
def generate_codes(request: CodeRequest):
    log.info("POST /generate_codes — note_id=%s", request.note_id)
    result = inference_engine.run_inference(
        note_id=request.note_id,
        original_prompt=request.original_prompt,
        rewritten_prompt=request.rewritten_prompt,
    )
    return CodeResponse(**result)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "weights_frozen": True,
    }
