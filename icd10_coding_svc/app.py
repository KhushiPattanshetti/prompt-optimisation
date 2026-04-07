from contextlib import asynccontextmanager

from fastapi import FastAPI

from icd10_coding_svc import inference_engine, model_loader
from icd10_coding_svc.config import MODEL_NAME
from icd10_coding_svc.logger import get_logger
from icd10_coding_svc.schemas import CodeRequest, CodeResponse

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
