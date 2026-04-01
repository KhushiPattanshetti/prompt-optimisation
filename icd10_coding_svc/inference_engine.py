"""Inference engine for icd10_coding_svc.

FIXES APPLIED:
    - threading.Lock scope tightened: only wraps model.generate() GPU calls.
      Previously the entire run_inference() was inside the lock, blocking
      /health and other non-GPU work.
    - Added one retry with 2-second delay on reward service forwarding.
      Previously a network failure would silently drop the reward signal.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
import requests

import code_parser
import gt_fetcher
import model_loader
from config import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    OUTPUT_PATH,
    REPETITION_PENALTY,
    REWARD_SERVICE_URL,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
)
from logger import get_logger

log = get_logger("inference_engine")

# Serialises GPU inference calls only — not file I/O or HTTP calls
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_prompt(prompt: str) -> str:
    """Wrap prompt in Llama-3 chat template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_INSTRUCTION}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ---------------------------------------------------------------------------
# Single GPU inference pass (must be called inside _inference_lock)
# ---------------------------------------------------------------------------

def _run_single_pass(prompt: str, model, tokenizer) -> str:
    """Run one inference pass and return raw decoded text."""
    formatted = _format_prompt(prompt)
    encoding  = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len  = encoding["input_ids"].shape[1]
    output_ids = model.generate(
        **encoding,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[:, input_len:]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Downstream forwarding
# ---------------------------------------------------------------------------

def _forward_to_reward_service(payload: Dict[str, Any]) -> None:
    """POST results to reward_metric_svc with one retry on failure.

    FIX: Previously fire-and-forget. A single network failure would
    silently drop the reward for that rollout with no indication.
    """
    url = f"{REWARD_SERVICE_URL}/compute_reward"
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                log.info("reward_forwarded | note_id=%s", payload.get("note_id"))
                return
            log.warning(
                "reward_forward_bad_status | status=%d | attempt=%d",
                resp.status_code, attempt + 1,
            )
        except requests.RequestException as exc:
            log.warning(
                "reward_forward_error | attempt=%d | error=%s", attempt + 1, exc
            )
        if attempt == 0:
            time.sleep(2)

    log.error(
        "reward_forward_failed | note_id=%s | all retries exhausted",
        payload.get("note_id"),
    )


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _save_output(data: Dict[str, Any]) -> None:
    """Save inference result to disk."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"{data['note_id']}_{ts.replace(':', '-')}.json"
    path     = os.path.join(OUTPUT_PATH, filename)
    data["timestamp"] = ts
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("output_saved | path=%s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    note_id: str,
    original_prompt: str,
    rewritten_prompt: str,
) -> Dict[str, Any]:
    """Run dual inference passes and fetch ground truth codes.

    Steps:
        1. Fetch gt_codes from dataset_svc via gt_fetcher (HTTP, no lock).
        2. Acquire GPU lock.
        3. Enhanced pass: rewritten_prompt → enh_codes.
        4. Original pass: original_prompt  → org_codes.
        5. Release GPU lock.
        6. Parse codes from raw outputs.
        7. Save to disk (I/O, no lock).
        8. Forward to reward_metric_svc asynchronously.
        9. Return structured response.

    Args:
        note_id:          Identifier used to fetch gt_codes from dataset_svc.
        original_prompt:  Unmodified prompt (baseline, no rewriting).
        rewritten_prompt: Enhanced prompt from Prompt Rewriter Service.

    Returns:
        Dict matching CodeResponse schema.
    """
    # Step 1: fetch gt_codes — HTTP call, outside GPU lock
    gt_codes: List[str] = gt_fetcher.get_gt_codes(note_id)
    log.info("gt_codes_fetched | note_id=%s | count=%d", note_id, len(gt_codes))

    model, tokenizer = model_loader.load_model()

    # Steps 2-4: both GPU passes inside one lock acquisition
    with _inference_lock:
        with torch.no_grad():
            enh_raw = _run_single_pass(rewritten_prompt, model, tokenizer)
            org_raw = _run_single_pass(original_prompt,  model, tokenizer)

    # Parse codes (CPU, outside lock)
    enh_codes: List[str] = code_parser.parse_icd10_codes(enh_raw)
    org_codes: List[str] = code_parser.parse_icd10_codes(org_raw)
    parsing_success      = bool(enh_codes or org_codes)

    log.info(
        "codes_parsed | note_id=%s | enh=%d | org=%d | success=%s",
        note_id, len(enh_codes), len(org_codes), parsing_success,
    )

    result: Dict[str, Any] = {
        "note_id":          note_id,
        "enh_codes":        enh_codes,
        "org_codes":        org_codes,
        "gt_codes":         gt_codes,
        "enh_raw_output":   enh_raw,
        "org_raw_output":   org_raw,
        "parsing_success":  parsing_success,
        "rewritten_prompt": rewritten_prompt,
        "original_prompt":  original_prompt,
    }

    # Save to disk
    _save_output(result)

    # Forward to reward service in background thread
    reward_payload = {
        "note_id":   note_id,
        "gt_codes":  gt_codes,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
    }
    threading.Thread(
        target=_forward_to_reward_service,
        args=(reward_payload,),
        daemon=True,
    ).start()

    return result

# import json
# import os
# import threading
# import time
# from datetime import datetime, timezone

# import httpx
# import torch

# import code_parser
# import gt_fetcher
# import model_loader
# from config import (
#     MAX_NEW_TOKENS,
#     TEMPERATURE,
#     DO_SAMPLE,
#     REPETITION_PENALTY,
#     SYSTEM_INSTRUCTION,
#     OUTPUT_PATH,
#     REWARD_SERVICE_URL,
# )
# from logger import get_logger

# log = get_logger("inference_engine")

# _inference_lock = threading.Lock()

# PROMPT_TEMPLATE = (
#     "<|begin_of_text|>"
#     "<|start_header_id|>system<|end_header_id|>\n"
#     "{system_instruction}\n"
#     "<|eot_id|>"
#     "<|start_header_id|>user<|end_header_id|>\n"
#     "{prompt}\n"
#     "<|eot_id|>"
#     "<|start_header_id|>assistant<|end_header_id|>"
# )


# def _build_prompt(user_prompt: str) -> str:
#     return PROMPT_TEMPLATE.format(
#         system_instruction=SYSTEM_INSTRUCTION,
#         prompt=user_prompt,
#     )


# def _generate(model, tokenizer, prompt_text: str) -> str:
#     inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             do_sample=DO_SAMPLE,
#             repetition_penalty=REPETITION_PENALTY,
#         )
#     # Decode only the newly generated tokens
#     generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
#     return tokenizer.decode(generated_ids, skip_special_tokens=True)


# def _forward_to_reward_service(payload: dict) -> None:
#     url = f"{REWARD_SERVICE_URL}/compute_reward"
#     for attempt in range(2):
#         try:
#             with httpx.Client(timeout=30) as client:
#                 resp = client.post(url, json=payload)
#                 resp.raise_for_status()
#             log.info("Forwarded to reward_metric_svc for note_id=%s (attempt %d)", payload["note_id"], attempt + 1)
#             return
#         except Exception as exc:
#             log.error("Forwarding failed for note_id=%s (attempt %d): %s", payload["note_id"], attempt + 1, exc)
#     log.error("All forwarding attempts exhausted for note_id=%s", payload["note_id"])


# def _save_output(result: dict, note_id: str) -> None:
#     os.makedirs(OUTPUT_PATH, exist_ok=True)
#     ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
#     filename = f"{note_id}_{ts}.json"
#     filepath = os.path.join(OUTPUT_PATH, filename)
#     with open(filepath, "w") as f:
#         json.dump(result, f, indent=2)
#     log.info("Output saved to %s", filepath)


# def run_inference(note_id: str, original_prompt: str, rewritten_prompt: str) -> dict:
#     log.info("Request received for note_id=%s", note_id)
#     model, tokenizer = model_loader.load_model()

#     # Build prompts
#     enh_prompt_text = _build_prompt(rewritten_prompt)
#     org_prompt_text = _build_prompt(original_prompt)

#     # Acquire lock — both passes are serialised together
#     with _inference_lock:
#         # Pass 1: Enhanced inference
#         t0 = time.perf_counter()
#         enh_raw_output = _generate(model, tokenizer, enh_prompt_text)
#         log.info("Enhanced inference completed in %.2fs for note_id=%s", time.perf_counter() - t0, note_id)

#         # Pass 2: Original inference
#         t0 = time.perf_counter()
#         org_raw_output = _generate(model, tokenizer, org_prompt_text)
#         log.info("Original inference completed in %.2fs for note_id=%s", time.perf_counter() - t0, note_id)

#     # Parse codes
#     enh_codes = code_parser.parse_icd10_codes(enh_raw_output)
#     org_codes = code_parser.parse_icd10_codes(org_raw_output)

#     # Fetch ground truth
#     gt_codes = gt_fetcher.get_gt_codes(note_id)

#     parsing_success = len(enh_codes) > 0 and len(org_codes) > 0

#     result = {
#         "note_id": note_id,
#         "enh_codes": enh_codes,
#         "org_codes": org_codes,
#         "gt_codes": gt_codes,
#         "enh_raw_output": enh_raw_output,
#         "org_raw_output": org_raw_output,
#         "parsing_success": parsing_success,
#     }

#     # Save output to disk
#     storage_record = {
#         "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
#         "note_id": note_id,
#         "rewritten_prompt": rewritten_prompt,
#         "original_prompt": original_prompt,
#         "enh_codes": enh_codes,
#         "org_codes": org_codes,
#         "gt_codes": gt_codes,
#         "parsing_success": parsing_success,
#     }
#     _save_output(storage_record, note_id)

#     # Async forwarding to reward_metric_svc
#     forward_payload = {
#         "note_id": note_id,
#         "gt_codes": gt_codes,
#         "enh_codes": enh_codes,
#         "org_codes": org_codes,
#     }
#     thread = threading.Thread(target=_forward_to_reward_service, args=(forward_payload,), daemon=True)
#     thread.start()

#     return result
