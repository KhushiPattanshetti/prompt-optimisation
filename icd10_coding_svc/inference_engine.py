import json
import os
import threading
import time
from datetime import datetime, timezone

import httpx
import torch

import code_parser
import gt_fetcher
import model_loader
from config import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
    REPETITION_PENALTY,
    SYSTEM_INSTRUCTION,
    OUTPUT_PATH,
    REWARD_SERVICE_URL,
)
from logger import get_logger

log = get_logger("inference_engine")

_inference_lock = threading.Lock()

PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n"
    "{system_instruction}\n"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{prompt}\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)


def _build_prompt(user_prompt: str) -> str:
    return PROMPT_TEMPLATE.format(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=user_prompt,
    )


def _generate(model, tokenizer, prompt_text: str) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            repetition_penalty=REPETITION_PENALTY,
        )
    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _forward_to_reward_service(payload: dict) -> None:
    url = f"{REWARD_SERVICE_URL}/compute_reward"
    for attempt in range(2):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
            log.info("Forwarded to reward_metric_svc for note_id=%s (attempt %d)", payload["note_id"], attempt + 1)
            return
        except Exception as exc:
            log.error("Forwarding failed for note_id=%s (attempt %d): %s", payload["note_id"], attempt + 1, exc)
    log.error("All forwarding attempts exhausted for note_id=%s", payload["note_id"])


def _save_output(result: dict, note_id: str) -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"{note_id}_{ts}.json"
    filepath = os.path.join(OUTPUT_PATH, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Output saved to %s", filepath)


def run_inference(note_id: str, original_prompt: str, rewritten_prompt: str) -> dict:
    log.info("Request received for note_id=%s", note_id)
    model, tokenizer = model_loader.load_model()

    # Build prompts
    enh_prompt_text = _build_prompt(rewritten_prompt)
    org_prompt_text = _build_prompt(original_prompt)

    # Acquire lock — both passes are serialised together
    with _inference_lock:
        # Pass 1: Enhanced inference
        t0 = time.perf_counter()
        enh_raw_output = _generate(model, tokenizer, enh_prompt_text)
        log.info("Enhanced inference completed in %.2fs for note_id=%s", time.perf_counter() - t0, note_id)

        # Pass 2: Original inference
        t0 = time.perf_counter()
        org_raw_output = _generate(model, tokenizer, org_prompt_text)
        log.info("Original inference completed in %.2fs for note_id=%s", time.perf_counter() - t0, note_id)

    # Parse codes
    enh_codes = code_parser.parse_icd10_codes(enh_raw_output)
    org_codes = code_parser.parse_icd10_codes(org_raw_output)

    # Fetch ground truth
    gt_codes = gt_fetcher.get_gt_codes(note_id)

    parsing_success = len(enh_codes) > 0 and len(org_codes) > 0

    result = {
        "note_id": note_id,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
        "gt_codes": gt_codes,
        "enh_raw_output": enh_raw_output,
        "org_raw_output": org_raw_output,
        "parsing_success": parsing_success,
    }

    # Save output to disk
    storage_record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "note_id": note_id,
        "rewritten_prompt": rewritten_prompt,
        "original_prompt": original_prompt,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
        "gt_codes": gt_codes,
        "parsing_success": parsing_success,
    }
    _save_output(storage_record, note_id)

    # Async forwarding to reward_metric_svc
    forward_payload = {
        "note_id": note_id,
        "gt_codes": gt_codes,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
    }
    thread = threading.Thread(target=_forward_to_reward_service, args=(forward_payload,), daemon=True)
    thread.start()

    return result
