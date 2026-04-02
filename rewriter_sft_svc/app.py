import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    BASE_MODEL_NAME,
    SYSTEM_INSTRUCTION,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rewriter_sft_svc")

tokenizer = None
model = None
device = None

HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "").strip()
LOCAL_ADAPTER_PATH = os.environ.get("LOCAL_ADAPTER_PATH", "").strip()
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "true").lower() == "true"

USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "12000"))


class RewriteRequest(BaseModel):
    instruction_id: str
    filename: str
    clinical_note: str


class RewriteResponse(BaseModel):
    status: str
    instruction_id: str
    filename: str
    structured_output: str
    device: str


def clean_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_INPUT_CHARS:
        logger.warning("Input too long, truncating to %d chars", MAX_INPUT_CHARS)
        text = text[:MAX_INPUT_CHARS]
    return text


def build_prompt(clinical_note: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION.strip()}\n\n"
        f"Clinical Note:\n{clinical_note.strip()}\n\n"
        f"Structured Clinical Note:\n"
    )


def postprocess_output(text: str) -> str:
    text = (text or "").strip()

    if "Structured Clinical Note:" in text:
        text = text.split("Structured Clinical Note:", 1)[-1].strip()

    return text if text else "Not specified"


def get_adapter_source() -> Optional[str]:
    if LOCAL_ADAPTER_PATH:
        return LOCAL_ADAPTER_PATH
    if HF_ADAPTER_REPO:
        return HF_ADAPTER_REPO
    return None


def load_model_and_tokenizer():
    global tokenizer, model, device

    logger.info("Initializing tokenizer and model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Detected device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": True,
    }

    if device == "cuda":
        logger.info("GPU Name: %s", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if USE_4BIT:
            logger.info("Loading model in 4-bit mode")
            model_kwargs["device_map"] = "auto"
            model_kwargs["load_in_4bit"] = True
        else:
            logger.info("Loading model in float16 with auto device map")
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
    else:
        logger.info("Loading model on CPU")
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        **model_kwargs,
    )

    adapter_source = get_adapter_source()
    if adapter_source:
        logger.info("Loading adapter from: %s", adapter_source)
        model = PeftModel.from_pretrained(model, adapter_source)

    model.eval()
    logger.info("Model loaded successfully")


def warmup_model():
    global tokenizer, model, device

    try:
        logger.info("Running warmup...")
        prompt = build_prompt(
            "54-year-old male with fever, cough, and shortness of breath for three days."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        if device == "cuda":
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        logger.info("Warmup done")
    except Exception as e:
        logger.warning("Warmup failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    warmup_model()
    yield


app = FastAPI(
    title="rewriter_sft_svc",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": device,
        "base_model": BASE_MODEL_NAME,
        "adapter_loaded": bool(get_adapter_source()),
        "use_4bit": USE_4BIT,
    }


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_note(payload: RewriteRequest):
    global tokenizer, model, device

    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    clinical_note = clean_text(payload.clinical_note)
    if not clinical_note:
        raise HTTPException(status_code=400, detail="clinical_note is empty")

    prompt = build_prompt(clinical_note)

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if device == "cuda":
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                top_p=0.95 if DO_SAMPLE else None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        structured_output = postprocess_output(decoded)

        return RewriteResponse(
            status="success",
            instruction_id=payload.instruction_id,
            filename=payload.filename,
            structured_output=structured_output,
            device=device,
        )

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail="GPU out of memory during inference. Reduce input size or use 4-bit loading.",
        )
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
