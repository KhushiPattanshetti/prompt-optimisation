from __future__ import annotations

from collections import defaultdict
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from icd10_coding_svc import code_parser, gt_fetcher, model_loader
from icd10_coding_svc.config import (
    DO_SAMPLE,
    ICD_INPUT_MAX_LENGTH,
    ICD_PARSE_RECOVERY_ENABLED,
    ICD_PARSE_RECOVERY_MAX_NOTE_CHARS,
    MAX_NEW_TOKENS,
    OUTPUT_PATH,
    REPETITION_PENALTY,
    REWARD_SERVICE_URL,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
)
from icd10_coding_svc.logger import get_logger

log = get_logger("inference_engine")

_inference_lock = threading.Lock()
_observability_lock = threading.Lock()
_observability_state = {
    "total_requests": 0,
    "parse_success": 0,
    "enhanced_parse_failures": 0,
    "original_parse_failures": 0,
    "both_parse_failures": 0,
    "joint_failure_modes": defaultdict(int),
    "enhanced_failure_reasons": defaultdict(int),
    "original_failure_reasons": defaultdict(int),
}


def _format_prompt(prompt: str) -> str:
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


def _classify_failure_reason(raw_output: str) -> str:
    text = str(raw_output or "").strip()
    if not text:
        return "empty_output"
    if "[" in text or "{" in text:
        return "json_like_no_valid_codes"
    return "no_valid_icd_pattern"


def _record_parse_observability(
    enh_raw: str,
    org_raw: str,
    enh_codes: List[str],
    org_codes: List[str],
) -> None:
    enh_ok = bool(enh_codes)
    org_ok = bool(org_codes)

    with _observability_lock:
        _observability_state["total_requests"] += 1
        if enh_ok or org_ok:
            _observability_state["parse_success"] += 1

        if not enh_ok:
            _observability_state["enhanced_parse_failures"] += 1
            reason = _classify_failure_reason(enh_raw)
            _observability_state["enhanced_failure_reasons"][reason] += 1

        if not org_ok:
            _observability_state["original_parse_failures"] += 1
            reason = _classify_failure_reason(org_raw)
            _observability_state["original_failure_reasons"][reason] += 1

        if not enh_ok and not org_ok:
            _observability_state["both_parse_failures"] += 1
            _observability_state["joint_failure_modes"]["both_failed"] += 1
        elif not enh_ok:
            _observability_state["joint_failure_modes"]["enhanced_failed_only"] += 1
        elif not org_ok:
            _observability_state["joint_failure_modes"]["original_failed_only"] += 1


def get_observability_snapshot() -> Dict[str, Any]:
    with _observability_lock:
        total_requests = int(_observability_state["total_requests"])
        parse_success = int(_observability_state["parse_success"])
        enhanced_parse_failures = int(_observability_state["enhanced_parse_failures"])
        original_parse_failures = int(_observability_state["original_parse_failures"])
        both_parse_failures = int(_observability_state["both_parse_failures"])
        joint_failure_modes = dict(_observability_state["joint_failure_modes"])
        enhanced_failure_reasons = dict(_observability_state["enhanced_failure_reasons"])
        original_failure_reasons = dict(_observability_state["original_failure_reasons"])

    parse_failure_total = enhanced_parse_failures + original_parse_failures
    return {
        "total_requests": total_requests,
        "parse_success_rate": (parse_success / total_requests) if total_requests else 0.0,
        "enhanced_parse_failures": enhanced_parse_failures,
        "original_parse_failures": original_parse_failures,
        "both_parse_failures": both_parse_failures,
        "parse_failure_total": parse_failure_total,
        "parse_failure_taxonomy": {
            "joint_failure_modes": joint_failure_modes,
            "enhanced_failure_reasons": enhanced_failure_reasons,
            "original_failure_reasons": original_failure_reasons,
        },
    }


def _truncate_note_for_recovery(note_text: str, max_chars: int) -> str:
    note = str(note_text or "").strip()
    if len(note) <= max_chars:
        return note
    if max_chars <= 64:
        return note[:max_chars]

    head_len = int(max_chars * 0.7)
    tail_len = max_chars - head_len - 5
    if tail_len <= 0:
        return note[:max_chars]

    return f"{note[:head_len].rstrip()}\n...\n{note[-tail_len:].lstrip()}"


def _extract_clinical_note(prompt: str) -> str:
    text = str(prompt or "")
    marker = "Clinical note:"
    idx = text.lower().find(marker.lower())
    if idx >= 0:
        extracted = text[idx + len(marker):].strip()
        if extracted:
            return extracted
    return text.strip()


def _build_recovery_prompt(prompt: str) -> str:
    note_text = _extract_clinical_note(prompt)
    clipped_note = _truncate_note_for_recovery(
        note_text=note_text,
        max_chars=ICD_PARSE_RECOVERY_MAX_NOTE_CHARS,
    )
    return (
        "Extract all ICD-10-CM diagnosis codes from the clinical note below.\n"
        "Return only a JSON array of code strings, for example [\"I10\", \"E11.9\"].\n"
        "If no diagnosis codes are present, return [].\n\n"
        f"Clinical note:\n{clipped_note}\n\n"
        "JSON:"
    )


def _recover_parse_if_needed(
    prompt: str,
    raw_output: str,
    parsed_codes: List[str],
    model,
    tokenizer,
    branch: str,
) -> Tuple[str, List[str], bool]:
    if parsed_codes or not ICD_PARSE_RECOVERY_ENABLED:
        return raw_output, parsed_codes, False

    recovery_prompt = _build_recovery_prompt(prompt)
    recovery_raw = _run_single_pass(recovery_prompt, model, tokenizer)
    recovery_codes = code_parser.parse_icd10_codes(recovery_raw, warn_on_empty=False)
    if recovery_codes:
        log.info(
            "parse_recovery_success | branch=%s | recovered_codes=%d",
            branch,
            len(recovery_codes),
        )
        return recovery_raw, recovery_codes, True

    return raw_output, parsed_codes, False


def _run_single_pass(prompt: str, model, tokenizer) -> str:
    formatted = _format_prompt(prompt)
    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=ICD_INPUT_MAX_LENGTH,
    ).to(model.device)

    input_len = encoding["input_ids"].shape[1]
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


def _forward_to_reward_service(payload: Dict[str, Any]) -> None:
    url = f"{REWARD_SERVICE_URL}/reward"
    for attempt in range(2):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code < 400:
                return
            log.warning(
                "reward_forward_bad_status | attempt=%d | status=%d",
                attempt + 1,
                response.status_code,
            )
        except requests.RequestException as exc:
            log.warning(
                "reward_forward_error | attempt=%d | error=%s",
                attempt + 1,
                exc,
            )
        if attempt == 0:
            time.sleep(2)


def _save_output(data: Dict[str, Any]) -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"{data['note_id']}_{ts.replace(':', '-')}.json"
    path = os.path.join(OUTPUT_PATH, filename)
    payload = dict(data)
    payload["timestamp"] = ts
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_inference(
    note_id: str,
    original_prompt: str,
    rewritten_prompt: str,
    log_prob_old: Optional[float] = None,
    value_estimate: Optional[float] = None,
    run_id: Optional[str] = None,
    group_id: Optional[str] = None,
    generation_source: Optional[str] = None,
) -> Dict[str, Any]:
    gt_codes: List[str] = gt_fetcher.get_gt_codes(note_id)
    model, tokenizer = model_loader.load_model()

    with _inference_lock:
        with torch.no_grad():
            enh_raw = _run_single_pass(rewritten_prompt, model, tokenizer)
            org_raw = _run_single_pass(original_prompt, model, tokenizer)
            enh_codes = code_parser.parse_icd10_codes(enh_raw, warn_on_empty=False)
            org_codes = code_parser.parse_icd10_codes(org_raw, warn_on_empty=False)
            enh_raw, enh_codes, enh_recovery_used = _recover_parse_if_needed(
                prompt=rewritten_prompt,
                raw_output=enh_raw,
                parsed_codes=enh_codes,
                model=model,
                tokenizer=tokenizer,
                branch="enhanced",
            )
            org_raw, org_codes, org_recovery_used = _recover_parse_if_needed(
                prompt=original_prompt,
                raw_output=org_raw,
                parsed_codes=org_codes,
                model=model,
                tokenizer=tokenizer,
                branch="original",
            )

    enh_parse_ok = bool(enh_codes)
    org_parse_ok = bool(org_codes)
    both_parse_success = bool(enh_parse_ok and org_parse_ok)
    parsing_success = bool(enh_codes or org_codes)
    _record_parse_observability(
        enh_raw=enh_raw,
        org_raw=org_raw,
        enh_codes=enh_codes,
        org_codes=org_codes,
    )

    if not parsing_success:
        log.warning(
            "parse_failed_both_prompts | note_id=%s | enh_recovery=%s | org_recovery=%s",
            note_id,
            enh_recovery_used,
            org_recovery_used,
        )

    output_payload: Dict[str, Any] = {
        "note_id": note_id,
        "run_id": run_id,
        "group_id": group_id,
        "gt_codes": gt_codes,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
        "original_prompt": original_prompt,
        "rewritten_prompt": rewritten_prompt,
        "generation_source": generation_source,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
        "enh_raw_output": enh_raw,
        "org_raw_output": org_raw,
        "parsing_success": parsing_success,
        "enh_parse_ok": enh_parse_ok,
        "org_parse_ok": org_parse_ok,
        "both_parse_success": both_parse_success,
        "enh_recovery_used": enh_recovery_used,
        "org_recovery_used": org_recovery_used,
    }

    _save_output(output_payload)

    _forward_to_reward_service(
        {
            "note_id": note_id,
            "run_id": run_id,
            "group_id": group_id,
            "gt_codes": gt_codes,
            "enh_codes": enh_codes,
            "org_codes": org_codes,
            "enh_parse_ok": enh_parse_ok,
            "org_parse_ok": org_parse_ok,
            "both_parse_success": both_parse_success,
            "original_prompt": original_prompt,
            "rewritten_prompt": rewritten_prompt,
            "generation_source": generation_source,
            "log_prob_old": log_prob_old,
            "value_estimate": value_estimate,
        }
    )

    return {
        "note_id": note_id,
        "enh_codes": enh_codes,
        "org_codes": org_codes,
        "gt_codes": gt_codes,
        "enh_raw_output": enh_raw,
        "org_raw_output": org_raw,
        "parsing_success": parsing_success,
        "enh_parse_ok": enh_parse_ok,
        "org_parse_ok": org_parse_ok,
        "both_parse_success": both_parse_success,
    }

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
