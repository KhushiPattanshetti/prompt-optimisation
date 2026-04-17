from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

from .config import (
    BEST_PROMPT_CACHE_FILE,
    BEST_PROMPT_CACHE_THRESHOLD,
    DO_SAMPLE,
    HARD_MAX_PROMPT_TOKENS,
    MAX_BATCH_ITEMS,
    MAX_PROMPT_TOKENS,
    MAX_RAW_NOTE_CHARS,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_HARD_CAP,
    OUTPUT_PATH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from .logger import get_logger
from .model_loader import load_model
from .policy import PromptPolicy, sample_policy_candidates
from .strategy_templates import TEMPLATES, VALID_STRATEGIES

log = get_logger(__name__)

REWRITER_SYSTEM_PROMPT = (
    "You are a clinical prompt optimizer for ICD-10 coding.\n"
    "You receive one full clinical note.\n\n"
    "Write one note-specific extraction instruction for a downstream coding model.\n"
    "Keep diagnosis intent faithful and preserve clinically relevant ambiguity.\n\n"
    "Rules:\n"
    "- Output an instruction prompt, not diagnosis codes\n"
    "- Never emit ICD-10 code strings\n"
    "- Use domain-specific clinical focus from this note\n"
    "- Mention at least two note-specific clinical targets\n"
    "- Vary wording across notes and avoid fixed generic templates\n"
    "- Keep output compact and JSON-extraction oriented\n"
    "- Output must be a single paragraph between 80 and 400 characters\n"
    "- Output must not contain any JSON arrays or code lists\n"
)

_REWRITER_USER_TASK_PREFIX = (
    "Rewrite the following ICD-10-CM extraction prompt into a note-specific instruction. "
    "Preserve clinical focus, vary phrasing, and avoid generic boilerplate. "
    "Output one paragraph prompt only, and do not output diagnosis codes.\n\n"
)

_ICD_CODE_REGEX = re.compile(r"\b[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?\b")
_HEALTHCHECK_LITERALS = {"test", "dummy", "healthcheck", "ping"}

_GENERIC_TEMPLATE_PHRASES = (
    "extract all icd-10-cm diagnosis codes",
    "extract all icd-10 diagnosis codes",
    "from the clinical information below",
    "output only a json array of code strings",
)

_CLINICAL_FOCUS_KEYWORDS = (
    "neuro",
    "cardio",
    "pulmonary",
    "renal",
    "hepatic",
    "endocrine",
    "oncolog",
    "infect",
    "psychi",
    "diagnos",
    "comorbid",
    "complication",
)

_inference_lock = threading.Lock()
_cache_lock = threading.Lock()
_group_lock = threading.Lock()
_rejection_counter_lock = threading.Lock()
_best_prompt_cache: Optional[Dict[str, Dict[str, Any]]] = None
_group_strategy_history: dict[str, list[str]] = {}
_rewrite_rejection_counts: Counter[str] = Counter()

_LOGIT_CLAMP_MIN = -50.0
_LOGIT_CLAMP_MAX = 50.0
_MIN_RETRY_NEW_TOKENS = 32
_MIN_REWRITE_CHARS = 64
_MAX_REWRITE_CHARS = 420
_STRICT_MINIMAL_FALLBACK = (
    "Extract all supported ICD-10-CM diagnosis codes from this case. "
    "Prioritize principal diagnosis, active complications, and clinically relevant comorbidities, "
    "then return only a JSON array of diagnosis code strings."
)
_NOTE_PAYLOAD_MARKERS = (
    "clinical note:\n",
    "clinical information",
    "--- clinical information ---",
    "icd-10-cm codes (json array)",
)


class _FiniteClampLogitsProcessor(LogitsProcessor):
    """Force generation logits into finite, bounded values."""

    def __init__(self, min_logit: float = _LOGIT_CLAMP_MIN, max_logit: float = _LOGIT_CLAMP_MAX) -> None:
        self.min_logit = float(min_logit)
        self.max_logit = float(max_logit)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        finite_scores = torch.nan_to_num(
            scores,
            nan=0.0,
            posinf=self.max_logit,
            neginf=self.min_logit,
        )
        return torch.clamp(finite_scores, min=self.min_logit, max=self.max_logit)


_SAFE_LOGITS_PROCESSOR = LogitsProcessorList([_FiniteClampLogitsProcessor()])


def _tokenizer_max_len(tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    raw = getattr(tokenizer, "model_max_length", None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None

    # Some tokenizers use very large sentinel values for "unbounded".
    if value <= 0 or value >= 1_000_000:
        return None
    return value


def _effective_prompt_token_cap(tokenizer: PreTrainedTokenizerBase) -> int:
    hard_cap = max(256, int(HARD_MAX_PROMPT_TOKENS))
    tokenizer_cap = _tokenizer_max_len(tokenizer)
    if tokenizer_cap is None:
        return max(1, min(int(MAX_PROMPT_TOKENS), hard_cap))
    return max(1, min(int(MAX_PROMPT_TOKENS), int(tokenizer_cap), hard_cap))


def _cap_ids_right(ids: torch.Tensor, cap: int) -> torch.Tensor:
    if ids.shape[1] <= cap:
        return ids
    return ids[:, -cap:]


def _truncate_note(note: str, max_chars: int = MAX_RAW_NOTE_CHARS) -> str:
    text = (note or "").strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars]
    boundaries = list(re.finditer(r"[.!?](?:\s|$)", clipped))
    if boundaries:
        return clipped[:boundaries[-1].end()].strip()
    return clipped.strip()


def _truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value

    clipped = value[:max_chars]
    boundary = max(clipped.rfind("\n"), clipped.rfind(". "), clipped.rfind("; "))
    if boundary >= int(max_chars * 0.6):
        clipped = clipped[:boundary]
    return clipped.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _is_healthcheck_payload(note: str) -> bool:
    return (note or "").strip().lower() in _HEALTHCHECK_LITERALS


def _build_raw_note_prompt(note_text: str) -> str:
    content = _truncate_text(str(note_text or "").strip(), MAX_RAW_NOTE_CHARS)
    if not content:
        content = "No clinical details provided."

    return (
        "Extract all ICD-10-CM diagnosis codes from the clinical information below.\n"
        "Output ONLY a JSON array of code strings. No explanations.\n"
        "Include diagnoses, complications, and relevant co-morbidities.\n\n"
        "--- CLINICAL INFORMATION ---\n"
        f"{content}\n"
        "--- END CLINICAL INFORMATION ---\n\n"
        "ICD-10-CM codes (JSON array):"
    )


def _policy_action_string(policy: PromptPolicy) -> str:
    enabled = sorted(key for key, value in policy.modifiers.items() if value)
    return f"{policy.template_name}|{','.join(enabled)}"


def _policy_modifier_block(policy: PromptPolicy) -> str:
    lines = []
    if policy.modifiers.get("prioritize_primary"):
        lines.append("- Prioritize principal diagnosis capture before secondary conditions")
    if policy.modifiers.get("strict_exclusion"):
        lines.append("- Exclude weakly implied conditions without explicit support")
    if policy.modifiers.get("expand_risk_factors"):
        lines.append("- Include documented risk factors when coding-relevant")
    if policy.modifiers.get("enforce_fall_detection"):
        lines.append("- Include fall and mobility-related diagnoses when supported")
    if policy.modifiers.get("enforce_z_codes"):
        lines.append("- Use status/history Z-codes when directly documented")
    if policy.modifiers.get("strict_precision"):
        lines.append("- Prefer precision over broad recall for ambiguous mentions")
    if policy.modifiers.get("expand_secondary"):
        lines.append("- Capture clinically meaningful secondary diagnoses and comorbidities")
    return "\n".join(lines)


def _render_rewriter_user_content(note_text: str, policy: PromptPolicy) -> str:
    strategy_instruction = TEMPLATES.get(policy.template_name, TEMPLATES["balanced"])
    modifier_instruction = _policy_modifier_block(policy)
    policy_block = f"Strategy: {strategy_instruction}\n"
    if modifier_instruction:
        policy_block += f"Policy constraints:\n{modifier_instruction}\n"

    return (
        f"{_REWRITER_USER_TASK_PREFIX}"
        f"{policy_block}\n"
        f"Clinical note:\n{str(note_text or '').strip()}"
    )


def _note_group_key(note_id: Optional[str], note_text: str) -> str:
    if note_id:
        digest = hashlib.sha1(note_id.encode("utf-8")).hexdigest()[:6]
        return f"group_{int(digest, 16) % 12}"
    digest = hashlib.sha1(note_text[:256].encode("utf-8")).hexdigest()[:6]
    return f"group_{int(digest, 16) % 12}"


def _select_policy_candidates(note_text: str, note_id: Optional[str], count: int = 2) -> list[PromptPolicy]:
    candidates = sample_policy_candidates(note_text, note_id, count=count, temperature=1.0)

    group_key = _note_group_key(note_id, note_text)
    with _group_lock:
        seen = list(_group_strategy_history.get(group_key, []))

    seen_set = set(seen)
    if seen_set and len(seen_set) < 2 and len(candidates) >= 2:
        primary = next(iter(seen_set))
        if all(c.template_name == primary for c in candidates):
            alternatives = [name for name in VALID_STRATEGIES if name != primary]
            if alternatives:
                replacement = PromptPolicy(template_name=alternatives[0], modifiers=candidates[-1].modifiers)
                candidates[-1] = replacement

    if len(candidates) >= 2 and len({c.template_name for c in candidates}) < 2:
        alternatives = [name for name in VALID_STRATEGIES if name != candidates[0].template_name]
        if alternatives:
            candidates[-1] = PromptPolicy(template_name=alternatives[0], modifiers=candidates[-1].modifiers)

    return candidates


def _record_group_strategy(note_id: Optional[str], note_text: str, strategy_name: str) -> None:
    group_key = _note_group_key(note_id, note_text)
    with _group_lock:
        history = _group_strategy_history.setdefault(group_key, [])
        history.append(strategy_name)
        if len(history) > 12:
            del history[:-12]


def _looks_like_generic_template(rewritten: str) -> bool:
    normalized = _normalize_text(rewritten)
    phrase_hits = sum(1 for phrase in _GENERIC_TEMPLATE_PHRASES if phrase in normalized)
    if phrase_hits >= 3:
        return True

    if re.search(r"^(extract|identify)\s+(all\s+)?icd-?10(?:-cm)?\s+diagnos", normalized):
        if not any(kw in normalized for kw in _CLINICAL_FOCUS_KEYWORDS):
            return True

    return False


def _sanitize_generated_rewrite(candidate: str) -> str:
    value = str(candidate or "").strip()
    if not value:
        return value

    value = value.strip('"').strip("'").strip()
    if "\n\n" in value:
        value = value.split("\n\n", 1)[0]

    value = re.sub(r"\[[^\]]{0,240}\]", " ", value)
    value = _ICD_CODE_REGEX.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip(" .;:-,")
    if len(value) > _MAX_REWRITE_CHARS:
        value = value[:_MAX_REWRITE_CHARS]
        pivot = max(value.rfind(". "), value.rfind("; "))
        if pivot >= int(_MAX_REWRITE_CHARS * 0.6):
            value = value[:pivot]
        value = value.strip(" .;:-,")
    return value


def _is_repetitive_rewrite(candidate: str) -> bool:
    normalized = _normalize_text(candidate)
    if not normalized:
        return False

    repeated_phrase = re.search(
        r"(\b[a-z0-9\-']+(?:\s+[a-z0-9\-']+){2,8})\s+\1\s+\1",
        normalized,
        flags=re.IGNORECASE,
    )
    if repeated_phrase:
        return True

    tokens = re.findall(r"[a-z0-9\-']+", normalized)
    if len(tokens) < 12:
        return False

    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / max(len(tokens), 1) > 0.24


def _looks_like_note_payload(candidate: str) -> bool:
    normalized = _normalize_text(candidate)
    if any(marker in normalized for marker in _NOTE_PAYLOAD_MARKERS):
        return True

    section_header = re.search(
        r"\b(history of present illness|past medical history|physical exam|discharge medications?)\b",
        normalized,
    )
    if section_header:
        return True

    if str(candidate).count("\n") >= 4:
        return True

    return False


def _is_valid_rewrite(rewritten: str, rule_prompt: str) -> bool:
    return _rewrite_rejection_reason(rewritten, rule_prompt) is None


def _rewrite_rejection_reason(rewritten: str, rule_prompt: str) -> Optional[str]:
    candidate = (rewritten or "").strip()
    baseline = (rule_prompt or "").strip()

    if len(candidate) < _MIN_REWRITE_CHARS:
        log.info("rewrite_rejected reason=too_short chars=%d min=%d", len(candidate), _MIN_REWRITE_CHARS)
        return "too_short"
    if len(candidate) > _MAX_REWRITE_CHARS:
        log.info("rewrite_rejected reason=too_long chars=%d max=%d", len(candidate), _MAX_REWRITE_CHARS)
        return "too_long"
    if _normalize_text(candidate) == _normalize_text(baseline):
        log.info("rewrite_rejected reason=identical_to_rule_prompt")
        return "identical_to_rule_prompt"
    if _is_repetitive_rewrite(candidate):
        log.info("rewrite_rejected reason=repetitive_output")
        return "repetitive_output"
    if _ICD_CODE_REGEX.search(candidate):
        log.info("rewrite_rejected reason=contains_icd_like_token")
        return "contains_icd_like_token"
    if _looks_like_note_payload(candidate):
        log.info("rewrite_rejected reason=note_payload_detected")
        return "note_payload_detected"
    if _looks_like_generic_template(candidate):
        log.info("rewrite_rejected reason=generic_template")
        return "generic_template"

    # Keyword gating turned out to be too brittle: the model can generate a perfectly
    # usable note-specific instruction (mentions concrete clinical targets) without
    # including literal strings like "ICD", "JSON", "extract", etc. Keep the other
    # structural guards above, but do not reject solely on missing these keywords.

    return None


def _record_rewrite_rejection(reason: str) -> None:
    key = str(reason or "unknown_rejection")
    with _rejection_counter_lock:
        _rewrite_rejection_counts[key] += 1


def _rewrite_rejection_snapshot() -> Dict[str, int]:
    with _rejection_counter_lock:
        return {k: int(v) for k, v in _rewrite_rejection_counts.items()}


def _build_prompt_inputs(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    policy: PromptPolicy,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_cap = _effective_prompt_token_cap(tokenizer)
    user_content = _render_rewriter_user_content(note_text, policy)
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=token_cap,
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"]
        attention_mask = torch.ones_like(prompt_ids)
        return prompt_ids.to(device), attention_mask.to(device)

    prompt = f"{REWRITER_SYSTEM_PROMPT}\n\n{user_content}"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=token_cap,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _build_full_sequence(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    rewritten_prompt: str,
    policy: PromptPolicy,
    device: torch.device,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    token_cap = _effective_prompt_token_cap(tokenizer)
    user_content = _render_rewriter_user_content(note_text, policy)
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": rewritten_prompt}],
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=token_cap,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=token_cap,
            add_special_tokens=False,
        )["input_ids"]
        prompt_ids = prompt_ids.to(device)
        full_ids = full_ids.to(device)
        return full_ids, prompt_ids.shape[1], prompt_ids

    prompt = f"{REWRITER_SYSTEM_PROMPT}\n\n{user_content}"
    full_text = f"{prompt}{rewritten_prompt}"
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=token_cap,
    )["input_ids"].to(device)
    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=token_cap,
    )["input_ids"].to(device)
    return full_ids, prompt_ids.shape[1], prompt_ids


def _build_full_sequence_safe(
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    rewritten_prompt: str,
    policy: PromptPolicy,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    try:
        return _build_full_sequence(tokenizer, note_text, rewritten_prompt, policy, device)
    except Exception as exc:
        log.exception("full_sequence_build_failed | error=%s", exc)
        return None, 0, None


def _generate_rewrite(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    note_text: str,
    policy: PromptPolicy,
    sampling_nonce: Optional[int] = None,
) -> Optional[Tuple[str, torch.Tensor, int, torch.Tensor]]:
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_prompt_inputs(tokenizer, note_text, policy, device)
    token_cap = _effective_prompt_token_cap(tokenizer)
    input_ids = _cap_ids_right(input_ids, token_cap)
    attention_mask = _cap_ids_right(attention_mask, token_cap)
    input_length = input_ids.shape[1]

    remaining_context = max(0, token_cap - input_length)
    if remaining_context <= 0:
        log.warning(
            "generate_skipped_no_context | input_tokens=%d token_cap=%d",
            input_length,
            token_cap,
        )
        return None

    do_sample = bool(DO_SAMPLE)
    base_temperature = max(float(TEMPERATURE), 0.05)

    primary_budget = min(MAX_NEW_TOKENS, MAX_NEW_TOKENS_HARD_CAP, 96, remaining_context)
    if primary_budget <= 0:
        return None
    retry_budget = min(remaining_context, max(_MIN_RETRY_NEW_TOKENS, primary_budget // 2))
    token_budgets = [primary_budget, retry_budget]
    seen_budgets: set[int] = set()

    for attempt_idx, token_budget in enumerate(token_budgets, start=1):
        if token_budget in seen_budgets:
            continue
        seen_budgets.add(token_budget)

        try:
            with _inference_lock:
                with torch.no_grad():
                    generate_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": token_budget,
                        "do_sample": do_sample,
                        "use_cache": False,
                        "logits_processor": _SAFE_LOGITS_PROCESSOR,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if do_sample:
                        jitter_basis = (int(sampling_nonce or 0) + attempt_idx) % 7
                        jitter = (jitter_basis - 3) * 0.04
                        sample_temperature = min(1.5, max(0.2, base_temperature * (1.0 + jitter)))
                        generate_kwargs["temperature"] = sample_temperature
                        if TOP_P > 0:
                            generate_kwargs["top_p"] = float(TOP_P)
                        if TOP_K > 0:
                            generate_kwargs["top_k"] = int(TOP_K)
                        generate_kwargs["num_beams"] = 1

                    generated_ids = model.generate(**generate_kwargs)

            if generated_ids.shape[1] <= input_length:
                log.warning(
                    "generate_empty_output | attempt=%d token_budget=%d input_tokens=%d",
                    attempt_idx,
                    token_budget,
                    input_length,
                )
                continue

            rewritten = tokenizer.decode(generated_ids[0, input_length:], skip_special_tokens=True).strip()
            return rewritten, generated_ids, input_length, input_ids
        except Exception as exc:
            log.exception(
                "generate_attempt_failed | attempt=%d token_budget=%d temp=%.3f strategy=%s error=%s",
                attempt_idx,
                token_budget,
                TEMPERATURE,
                policy.template_name,
                exc,
            )

    return None


def _compute_log_prob(model: PreTrainedModel, full_input_ids: torch.Tensor, input_length: int) -> float:
    with _inference_lock:
        with torch.no_grad():
            outputs = model(full_input_ids)
            logits = outputs.logits[:, :-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            generated_token_ids = full_input_ids[:, input_length:]
            generated_positions = log_probs[:, input_length - 1 : input_length - 1 + generated_token_ids.shape[1], :]
            token_log_probs = generated_positions.gather(dim=-1, index=generated_token_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def _pad_inputs_for_batch(
    tokenizer: PreTrainedTokenizerBase,
    input_ids_list: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    lengths = [int(ids.shape[1]) for ids in input_ids_list]
    max_len = max(lengths)

    padded_ids: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for ids, mask in zip(input_ids_list, attention_masks):
        pad_cols = max_len - ids.shape[1]
        if pad_cols > 0:
            pad_tensor = torch.full((1, pad_cols), int(pad_id), dtype=ids.dtype, device=ids.device)
            mask_pad = torch.zeros((1, pad_cols), dtype=mask.dtype, device=mask.device)
            ids = torch.cat([ids, pad_tensor], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
        padded_ids.append(ids)
        padded_masks.append(mask)

    return torch.cat(padded_ids, dim=0), torch.cat(padded_masks, dim=0), lengths


def _generate_rewrite_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    notes: list[str],
    policies: list[PromptPolicy],
    sampling_nonces: list[Optional[int]],
) -> list[Optional[str]]:
    if not notes:
        return []

    device = next(model.parameters()).device
    token_cap = _effective_prompt_token_cap(tokenizer)
    input_ids_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []

    for note_text, policy in zip(notes, policies):
        input_ids, attention_mask = _build_prompt_inputs(tokenizer, note_text, policy, device)
        input_ids = _cap_ids_right(input_ids, token_cap)
        attention_mask = _cap_ids_right(attention_mask, token_cap)
        input_ids_list.append(input_ids)
        masks_list.append(attention_mask)

    batched_ids, batched_masks, input_lengths = _pad_inputs_for_batch(tokenizer, input_ids_list, masks_list)
    remaining_context = [max(0, token_cap - length) for length in input_lengths]
    batch_budget = min(remaining_context) if remaining_context else 0
    batch_budget = min(MAX_NEW_TOKENS, MAX_NEW_TOKENS_HARD_CAP, 96, batch_budget)

    if batch_budget <= 0:
        return [None for _ in notes]

    do_sample = bool(DO_SAMPLE)
    base_temperature = max(float(TEMPERATURE), 0.05)
    nonce_seed = int(sum(int(n or 0) for n in sampling_nonces)) if sampling_nonces else 0

    generate_kwargs = {
        "input_ids": batched_ids,
        "attention_mask": batched_masks,
        "max_new_tokens": int(batch_budget),
        "do_sample": do_sample,
        "use_cache": False,
        "logits_processor": _SAFE_LOGITS_PROCESSOR,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        jitter_basis = nonce_seed % 7
        jitter = (jitter_basis - 3) * 0.04
        sample_temperature = min(1.5, max(0.2, base_temperature * (1.0 + jitter)))
        generate_kwargs["temperature"] = sample_temperature
        if TOP_P > 0:
            generate_kwargs["top_p"] = float(TOP_P)
        if TOP_K > 0:
            generate_kwargs["top_k"] = int(TOP_K)
        generate_kwargs["num_beams"] = 1

    with _inference_lock:
        with torch.no_grad():
            generated = model.generate(**generate_kwargs)

    rewrites: list[Optional[str]] = []
    for row_idx, input_len in enumerate(input_lengths):
        if generated.shape[1] <= input_len:
            rewrites.append(None)
            continue
        text = tokenizer.decode(generated[row_idx, input_len:], skip_special_tokens=True).strip()
        rewrites.append(text)
    return rewrites


def _compute_value_estimate(model: PreTrainedModel, value_head: torch.nn.Module, input_ids: torch.Tensor) -> float:
    with _inference_lock:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
    value_head_dtype = next(value_head.parameters()).dtype
    state = outputs.hidden_states[-1][:, -1, :].to(value_head_dtype)
    value = value_head(state)
    return float(value.squeeze().item())


def _save_output(payload: Dict[str, Any]) -> Path:
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    filepath = output_dir / f"{timestamp.replace(':', '-')}.json"

    record = dict(payload)
    record["timestamp"] = timestamp
    filepath.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return filepath


def _fallback_prompt(note_text: str) -> str:
    guided = _build_guided_rewrite(note_text, variant_seed=0)
    if _is_valid_rewrite(guided, ""):
        return guided
    return _STRICT_MINIMAL_FALLBACK


def _build_guided_rewrite(note_text: str, variant_seed: int = 0) -> str:
    lower = _normalize_text(note_text)
    focus_area = "the primary diagnosis and clinically active comorbidities"
    if any(token in lower for token in ("stroke", "seizure", "parkinson", "dementia", "neuro")):
        focus_area = "neurologic diagnoses, deficits, and related complications"
    elif any(token in lower for token in ("heart", "cardiac", "atrial", "hypertension", "coronary")):
        focus_area = "cardiovascular diagnoses, rhythm disorders, and hemodynamic complications"
    elif any(token in lower for token in ("copd", "asthma", "pneumonia", "respiratory", "pulmonary")):
        focus_area = "pulmonary diagnoses, respiratory failure risk, and infectious complications"
    elif any(token in lower for token in ("kidney", "renal", "ckd", "aki", "neph")):
        focus_area = "renal diagnoses, acuity, and electrolyte-related complications"

    templates = [
        (
            "Extract all possible ICD-10-CM diagnosis codes for this case with focus on {focus}. "
            "Prioritize principal diagnoses first, then active complications and relevant comorbidities, "
            "then output only a JSON list of code strings."
        ),
        (
            "Identify all possible ICD-10-CM diagnoses from this note emphasizing {focus}. "
            "Capture principal condition first, then clinically active secondary conditions and major complications, "
            "then return only a JSON list of diagnosis code strings."
        ),
        (
            "From this clinical note, extract all possible supported ICD-10-CM diagnosis codes centered on {focus}. "
            "Include the principal diagnosis first and active comorbid disease burden, "
            "and respond only with a JSON array of code strings."
        ),
    ]
    idx = abs(int(variant_seed)) % len(templates)
    return templates[idx].format(focus=focus_area)


def _load_best_prompt_cache() -> Dict[str, Dict[str, Any]]:
    global _best_prompt_cache

    if _best_prompt_cache is not None:
        return _best_prompt_cache

    cache_path = Path(BEST_PROMPT_CACHE_FILE)
    if cache_path.exists():
        try:
            _best_prompt_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _best_prompt_cache = {}
    else:
        _best_prompt_cache = {}
    return _best_prompt_cache


def _persist_best_prompt_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    cache_path = Path(BEST_PROMPT_CACHE_FILE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def run_inference(
    clinical_note: str,
    note_id: Optional[str] = None,
    sampling_nonce: Optional[int] = None,
    disable_best_prompt_cache: bool = False,
) -> Dict[str, Any]:
    full_note = str(clinical_note or "").strip()
    model_input_text = _truncate_note(full_note)
    model_input_source = "raw"
    rule_prompt = _build_raw_note_prompt(model_input_text)
    input_note_preview = model_input_text
    generation_source = "model"

    policy_candidates = _select_policy_candidates(model_input_text, note_id, count=2)
    if not policy_candidates:
        policy_candidates = [PromptPolicy(template_name="balanced", modifiers={})]

    if sampling_nonce is not None and len(policy_candidates) > 1:
        rotate_by = abs(int(sampling_nonce)) % len(policy_candidates)
        if rotate_by:
            policy_candidates = policy_candidates[rotate_by:] + policy_candidates[:rotate_by]

    chosen_policy = policy_candidates[0]

    if _is_healthcheck_payload(clinical_note):
        rewritten_prompt = _fallback_prompt(full_note)
        generation_source = "healthcheck_fallback"
        result = {
            "note_id": note_id,
            "input_note": input_note_preview,
            "rule_prompt": rule_prompt,
            "model_input_source": "healthcheck",
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
            "policy_action": _policy_action_string(chosen_policy),
        }
        _save_output(result)
        return {
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }

    try:
        model, tokenizer, value_head = load_model()
        device = next(model.parameters()).device
    except Exception as exc:
        log.exception("model_load_failed | using fallback prompt | error=%s", exc)
        guided = _build_guided_rewrite(model_input_text, variant_seed=int(sampling_nonce or 0))
        if _is_valid_rewrite(guided, rule_prompt):
            rewritten_prompt = guided
            generation_source = "guided_fallback_model_load_error"
        else:
            rewritten_prompt = _STRICT_MINIMAL_FALLBACK
            generation_source = "strict_fallback_model_load_error"

        output = {
            "note_id": note_id,
            "input_note": input_note_preview,
            "model_input_source": model_input_source,
            "model_input": model_input_text,
            "rule_prompt": rule_prompt,
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
            "policy_action": _policy_action_string(chosen_policy),
        }
        _save_output(output)
        return {
            "rewritten_prompt": rewritten_prompt,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": generation_source,
        }

    cached_rewrite: Optional[str] = None
    if note_id and not disable_best_prompt_cache:
        cache = _load_best_prompt_cache()
        entry = cache.get(note_id)
        if entry and float(entry.get("reward", -1.0)) >= 0.0:
            cached_rewrite = str(entry.get("rewritten_prompt", "")).strip()

    rewritten_prompt = ""
    rejection_reason: Optional[str] = None
    full_ids: Optional[torch.Tensor] = None
    input_length = 0
    prompt_ids: Optional[torch.Tensor] = None

    if cached_rewrite and _is_valid_rewrite(_sanitize_generated_rewrite(cached_rewrite), rule_prompt):
        rewritten_prompt = _sanitize_generated_rewrite(cached_rewrite)
        generation_source = "cache"
        full_ids, input_length, prompt_ids = _build_full_sequence_safe(
            tokenizer,
            model_input_text,
            rewritten_prompt,
            chosen_policy,
            device,
        )

    if not rewritten_prompt:
        for candidate in policy_candidates:
            first_result = _generate_rewrite(
                model,
                tokenizer,
                model_input_text,
                candidate,
                sampling_nonce=sampling_nonce,
            )
            if first_result is None:
                continue
            first_raw, full_1, in_len_1, prompt_1 = first_result
            first = _sanitize_generated_rewrite(first_raw)
            if first != first_raw:
                full_1, in_len_1, prompt_1 = _build_full_sequence_safe(
                    tokenizer,
                    model_input_text,
                    first,
                    candidate,
                    device,
                )
            if _is_valid_rewrite(first, rule_prompt):
                rewritten_prompt = first
                chosen_policy = candidate
                generation_source = "model_first"
                full_ids = full_1
                input_length = in_len_1
                prompt_ids = prompt_1
                break
            rejection_reason = _rewrite_rejection_reason(first, rule_prompt)

    if not rewritten_prompt:
        guided = _build_guided_rewrite(model_input_text, variant_seed=int(sampling_nonce or 0))
        if _is_valid_rewrite(guided, rule_prompt):
            rewritten_prompt = guided
            generation_source = "guided_fallback"
            if rejection_reason is None:
                rejection_reason = "model_generation_invalid_or_empty"
            _record_rewrite_rejection(rejection_reason)
            full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                tokenizer,
                model_input_text,
                rewritten_prompt,
                chosen_policy,
                device,
            )
        else:
            rewritten_prompt = _STRICT_MINIMAL_FALLBACK
            generation_source = "strict_fallback"
            if rejection_reason is None:
                rejection_reason = "model_generation_invalid_or_empty"
            _record_rewrite_rejection(rejection_reason)
            full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                tokenizer,
                model_input_text,
                rewritten_prompt,
                chosen_policy,
                device,
            )

    if not rewritten_prompt:
        rewritten_prompt = _STRICT_MINIMAL_FALLBACK
        generation_source = "strict_fallback"
        full_ids, input_length, prompt_ids = _build_full_sequence_safe(
            tokenizer,
            model_input_text,
            rewritten_prompt,
            chosen_policy,
            device,
        )

    log_prob_old = -1e-6
    if full_ids is not None:
        try:
            log_prob_old = _compute_log_prob(model, full_ids, input_length)
            if log_prob_old == 0.0:
                log_prob_old = -1e-6
        except Exception:
            log_prob_old = -1e-6

    value_estimate = 0.0
    if prompt_ids is not None:
        try:
            value_estimate = _compute_value_estimate(model, value_head, prompt_ids)
        except Exception:
            value_estimate = 0.0

    _record_group_strategy(note_id, model_input_text, chosen_policy.template_name)

    output = {
        "note_id": note_id,
        "input_note": input_note_preview,
        "model_input_source": model_input_source,
        "model_input": model_input_text,
        "rule_prompt": rule_prompt,
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
        "generation_source": generation_source,
        "rejection_reason": rejection_reason,
        "rejection_reason_counts": _rewrite_rejection_snapshot(),
        "policy_action": _policy_action_string(chosen_policy),
    }
    _save_output(output)
    return {
        "rewritten_prompt": rewritten_prompt,
        "log_prob_old": log_prob_old,
        "value_estimate": value_estimate,
        "generation_source": generation_source,
        "rejection_reason": rejection_reason,
        "rejection_reason_counts": _rewrite_rejection_snapshot(),
    }


def run_inference_batch(requests: list[dict[str, Any]]) -> list[Dict[str, Any]]:
    if not requests:
        return []

    safe_batch_size = max(1, min(int(MAX_BATCH_ITEMS), 16))
    responses: list[Dict[str, Any]] = [
        {
            "rewritten_prompt": _STRICT_MINIMAL_FALLBACK,
            "log_prob_old": -1e-6,
            "value_estimate": 0.0,
            "generation_source": "strict_fallback",
            "rejection_reason": None,
            "rejection_reason_counts": {},
        }
        for _ in requests
    ]

    for start in range(0, len(requests), safe_batch_size):
        batch = requests[start : start + safe_batch_size]
        try:
            model, tokenizer, value_head = load_model()
            device = next(model.parameters()).device
        except Exception as exc:
            log.exception("batch_model_load_failed | using fallback prompts | error=%s", exc)
            for local_idx, req in enumerate(batch):
                global_idx = start + local_idx
                note = str(req.get("clinical_note", "") or "").strip()
                nonce = req.get("sampling_nonce")
                guided = _build_guided_rewrite(_truncate_note(note), variant_seed=int(nonce or 0))
                rewritten_prompt = guided if _is_valid_rewrite(guided, "") else _STRICT_MINIMAL_FALLBACK
                source = "guided_fallback_model_load_error" if rewritten_prompt == guided else "strict_fallback_model_load_error"
                responses[global_idx] = {
                    "rewritten_prompt": rewritten_prompt,
                    "log_prob_old": -1e-6,
                    "value_estimate": 0.0,
                    "generation_source": source,
                    "rejection_reason": "model_load_error",
                    "rejection_reason_counts": _rewrite_rejection_snapshot(),
                }
            continue

        pending_for_batch: list[dict[str, Any]] = []
        for local_idx, req in enumerate(batch):
            global_idx = start + local_idx
            note_id = req.get("note_id")
            full_note = str(req.get("clinical_note", "") or "").strip()
            model_input_text = _truncate_note(full_note)
            rule_prompt = _build_raw_note_prompt(model_input_text)
            nonce = req.get("sampling_nonce")
            disable_cache = bool(req.get("disable_best_prompt_cache", False))

            policy_candidates = _select_policy_candidates(model_input_text, note_id, count=2)
            if not policy_candidates:
                policy_candidates = [PromptPolicy(template_name="balanced", modifiers={})]
            if nonce is not None and len(policy_candidates) > 1:
                rotate_by = abs(int(nonce)) % len(policy_candidates)
                if rotate_by:
                    policy_candidates = policy_candidates[rotate_by:] + policy_candidates[:rotate_by]
            chosen_policy = policy_candidates[0]

            cached_rewrite: Optional[str] = None
            if note_id and not disable_cache:
                cache = _load_best_prompt_cache()
                entry = cache.get(str(note_id))
                if entry and float(entry.get("reward", -1.0)) >= 0.0:
                    cached_rewrite = str(entry.get("rewritten_prompt", "")).strip()

            base = {
                "global_idx": global_idx,
                "note_id": note_id,
                "model_input_text": model_input_text,
                "rule_prompt": rule_prompt,
                "nonce": nonce,
                "policy_candidates": policy_candidates,
                "chosen_policy": chosen_policy,
                "cached_rewrite": cached_rewrite,
            }

            if _is_healthcheck_payload(full_note):
                rewritten_prompt = _fallback_prompt(full_note)
                responses[global_idx] = {
                    "rewritten_prompt": rewritten_prompt,
                    "log_prob_old": -1e-6,
                    "value_estimate": 0.0,
                    "generation_source": "healthcheck_fallback",
                    "rejection_reason": None,
                    "rejection_reason_counts": _rewrite_rejection_snapshot(),
                }
            elif cached_rewrite and _is_valid_rewrite(_sanitize_generated_rewrite(cached_rewrite), rule_prompt):
                rewritten_prompt = _sanitize_generated_rewrite(cached_rewrite)
                full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                    tokenizer,
                    model_input_text,
                    rewritten_prompt,
                    chosen_policy,
                    device,
                )
                log_prob_old = -1e-6
                if full_ids is not None:
                    try:
                        log_prob_old = _compute_log_prob(model, full_ids, input_length)
                        if log_prob_old == 0.0:
                            log_prob_old = -1e-6
                    except Exception:
                        log_prob_old = -1e-6
                value_estimate = 0.0
                if prompt_ids is not None:
                    try:
                        value_estimate = _compute_value_estimate(model, value_head, prompt_ids)
                    except Exception:
                        value_estimate = 0.0
                responses[global_idx] = {
                    "rewritten_prompt": rewritten_prompt,
                    "log_prob_old": log_prob_old,
                    "value_estimate": value_estimate,
                    "generation_source": "cache",
                    "rejection_reason": None,
                    "rejection_reason_counts": _rewrite_rejection_snapshot(),
                }
            else:
                pending_for_batch.append(base)

        if not pending_for_batch:
            continue

        notes = [entry["model_input_text"] for entry in pending_for_batch]
        policies = [entry["chosen_policy"] for entry in pending_for_batch]
        nonces = [entry["nonce"] for entry in pending_for_batch]

        batch_raw = _generate_rewrite_batch(model, tokenizer, notes, policies, nonces)

        for entry, raw in zip(pending_for_batch, batch_raw):
            rewritten_prompt = ""
            generation_source = "model_first"
            chosen_policy = entry["chosen_policy"]
            rejection_reason: Optional[str] = None

            if raw is not None:
                candidate = _sanitize_generated_rewrite(raw)
                if _is_valid_rewrite(candidate, entry["rule_prompt"]):
                    rewritten_prompt = candidate
                else:
                    rejection_reason = _rewrite_rejection_reason(candidate, entry["rule_prompt"])

            if not rewritten_prompt:
                retry_result = _generate_rewrite(
                    model,
                    tokenizer,
                    entry["model_input_text"],
                    chosen_policy,
                    sampling_nonce=entry["nonce"],
                )
                if retry_result is not None:
                    retry_raw, _, _, _ = retry_result
                    retry_candidate = _sanitize_generated_rewrite(retry_raw)
                    if _is_valid_rewrite(retry_candidate, entry["rule_prompt"]):
                        rewritten_prompt = retry_candidate
                        generation_source = "model_retry"
                    else:
                        rejection_reason = _rewrite_rejection_reason(retry_candidate, entry["rule_prompt"])

            if not rewritten_prompt:
                guided = _build_guided_rewrite(entry["model_input_text"], variant_seed=int(entry["nonce"] or 0))
                if _is_valid_rewrite(guided, entry["rule_prompt"]):
                    rewritten_prompt = guided
                    generation_source = "guided_fallback"
                    if rejection_reason is None:
                        rejection_reason = "model_generation_invalid_or_empty"
                    _record_rewrite_rejection(rejection_reason)
                else:
                    rewritten_prompt = _STRICT_MINIMAL_FALLBACK
                    generation_source = "strict_fallback"
                    if rejection_reason is None:
                        rejection_reason = "model_generation_invalid_or_empty"
                    _record_rewrite_rejection(rejection_reason)

            full_ids, input_length, prompt_ids = _build_full_sequence_safe(
                tokenizer,
                entry["model_input_text"],
                rewritten_prompt,
                chosen_policy,
                device,
            )
            log_prob_old = -1e-6
            if full_ids is not None:
                try:
                    log_prob_old = _compute_log_prob(model, full_ids, input_length)
                    if log_prob_old == 0.0:
                        log_prob_old = -1e-6
                except Exception:
                    log_prob_old = -1e-6

            value_estimate = 0.0
            if prompt_ids is not None:
                try:
                    value_estimate = _compute_value_estimate(model, value_head, prompt_ids)
                except Exception:
                    value_estimate = 0.0

            _record_group_strategy(entry.get("note_id"), entry["model_input_text"], chosen_policy.template_name)
            responses[int(entry["global_idx"])] = {
                "rewritten_prompt": rewritten_prompt,
                "log_prob_old": log_prob_old,
                "value_estimate": value_estimate,
                "generation_source": generation_source,
                "rejection_reason": rejection_reason,
                "rejection_reason_counts": _rewrite_rejection_snapshot(),
            }

    return responses


def update_best_prompt_cache(note_id: str, rewritten: str, reward: float) -> None:
    if reward <= BEST_PROMPT_CACHE_THRESHOLD:
        return

    with _cache_lock:
        cache = _load_best_prompt_cache()
        existing = cache.get(note_id)
        if existing is not None and float(existing.get("reward", -1.0)) >= float(reward):
            return

        cache[note_id] = {
            "rewritten_prompt": rewritten,
            "reward": float(reward),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _persist_best_prompt_cache(cache)
