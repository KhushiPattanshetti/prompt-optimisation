from dotenv import load_dotenv
load_dotenv()

import traceback
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


from config import (
    BASE_MODEL_NAME,
    TRUST_REMOTE_CODE,
    HF_ADAPTER_REPO,
    HF_TOKEN,
    SYSTEM_INSTRUCTION,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
)

app = FastAPI(title="rewriter_sft_svc")


class RewriteRequest(BaseModel):
    instruction_id: str
    filename: str
    clinical_note: str


class RewriteResponse(BaseModel):
    status: str
    instruction_id: str
    filename: str
    structured_output: str


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
model_loaded = False
load_error = None


def build_prompt(clinical_note: str) -> str:
    return f"""<|system|>
{SYSTEM_INSTRUCTION}
<|user|>
Convert the following clinical note into the required structured clinical note format:

{clinical_note}
<|assistant|>
"""


try:
    print("Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        token=HF_TOKEN,
    )

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(f"Using device: {device}")
    print("Loading base model from Hugging Face...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.float32,
        token=HF_TOKEN,
    ).to(device)

    print(f"Loading LoRA adapter from Hugging Face repo: {HF_ADAPTER_REPO}")
    model = PeftModel.from_pretrained(
        base_model,
        HF_ADAPTER_REPO,
        token=HF_TOKEN,
    )
    model = model.to(device)
    model.eval()

    model_loaded = True
    print("Model loaded successfully from Hugging Face.")

except Exception as e:
    load_error = str(e)
    model_loaded = False
    print(f"Model loading failed: {e}")
    traceback.print_exc()


@app.get("/")
def root():
    return {
        "service": "rewriter_sft_svc",
        "status": "running",
        "model_loaded": model_loaded,
        "device": device,
        "adapter_repo": HF_ADAPTER_REPO,
    }


@app.get("/health")
def health():
    return {
        "service": "rewriter_sft_svc",
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "device": device,
        "adapter_repo": HF_ADAPTER_REPO,
        "load_error": load_error,
    }


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not loaded. Error: {load_error}"
        )

    note = req.clinical_note.strip()
    if not note:
        raise HTTPException(status_code=400, detail="clinical_note is empty")

    prompt = build_prompt(note)

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "eos_token_id": int(eos_token_id),
            "pad_token_id": int(pad_token_id),
        }

        if TEMPERATURE is not None and DO_SAMPLE:
            generate_kwargs["temperature"] = TEMPERATURE

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]

        structured_output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        return RewriteResponse(
            status="OK",
            instruction_id=req.instruction_id,
            filename=req.filename,
            structured_output=structured_output,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
