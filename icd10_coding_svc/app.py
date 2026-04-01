"""FastAPI application for icd10_coding_svc.

FIX SUMMARY (from review):
  - Removed gt_fetcher.init_datasets() call at startup.
    gt_fetcher no longer loads CSVs — it calls dataset_svc via HTTP.
  - /health endpoint now verifies model is loaded and weights are frozen.
  - Lock scope issue noted: _inference_lock in inference_engine
    now only wraps GPU calls, not disk I/O or HTTP forwarding.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

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

    # FIX: removed gt_fetcher.init_datasets() — no CSV loading here
    # gt_fetcher now calls dataset_svc via HTTP on demand

    model_loader.load_model()
    log.info("Startup complete. Model loaded and frozen.")
    yield
    # Shutdown — nothing to clean up


app = FastAPI(title="icd10_coding_svc", lifespan=lifespan)


@app.post("/generate_codes", response_model=CodeResponse)
def generate_codes(request: CodeRequest):
    """Run dual inference (enhanced + original) and return ICD-10 codes."""
    log.info("POST /generate_codes — note_id=%s", request.note_id)
    result = inference_engine.run_inference(
        note_id=request.note_id,
        original_prompt=request.original_prompt,
        rewritten_prompt=request.rewritten_prompt,
    )
    return CodeResponse(**result)


@app.get("/health")
def health():
    """Return service status and model freeze confirmation."""
    model = model_loader.get_cached_model()
    weights_frozen = False
    if model is not None:
        weights_frozen = all(
            not p.requires_grad for p in model.parameters()
        )
    return {
        "status":         "ok",
        "model":          MODEL_NAME,
        "weights_frozen": weights_frozen,
    }

# import os
# from contextlib import asynccontextmanager

# from fastapi import FastAPI

# import gt_fetcher
# import inference_engine
# import model_loader
# from config import MODEL_NAME, OUTPUT_PATH, GT_CODES_PATH
# from logger import get_logger
# from schemas import CodeRequest, CodeResponse

# log = get_logger("app")


# @asynccontextmanager
# async def lifespan(application: FastAPI):
#     # Startup
#     log.info("Starting icd10_coding_svc…")
#     os.makedirs(OUTPUT_PATH, exist_ok=True)
#     os.makedirs(GT_CODES_PATH, exist_ok=True)
#     gt_fetcher.init_datasets()
#     model_loader.load_model()
#     log.info("Startup complete.")
#     yield
#     # Shutdown (nothing to clean up)


# app = FastAPI(title="icd10_coding_svc", lifespan=lifespan)


# @app.post("/generate_codes", response_model=CodeResponse)
# def generate_codes(request: CodeRequest):
#     log.info("POST /generate_codes — note_id=%s", request.note_id)
#     result = inference_engine.run_inference(
#         note_id=request.note_id,
#         original_prompt=request.original_prompt,
#         rewritten_prompt=request.rewritten_prompt,
#     )
#     return CodeResponse(**result)


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model": MODEL_NAME,
#         "weights_frozen": True,
#     }
