from __future__ import annotations

from collections import deque
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reward_metrics_svc.embedding_model import cosine_similarity_matrix, embed_texts
from reward_metrics_svc.icd_descriptions import description_count, description_source, get_descriptions, normalize_description

log = logging.getLogger("reward_metrics_svc")
logging.basicConfig(level=logging.INFO)

TREE_PATH = Path(__file__).resolve().parent / "icd10_tree.json"
RL_BLOCK_ENDPOINT = os.environ.get("RL_BLOCK_ENDPOINT", "http://localhost:8004/rollout")
RL_STATUS_ENDPOINT = os.environ.get(
    "RL_STATUS_ENDPOINT",
    RL_BLOCK_ENDPOINT.rsplit("/", 1)[0] + "/status",
)

REWARD_RELATIVE_WEIGHT = float(os.environ.get("REWARD_RELATIVE_WEIGHT", "0.7"))
REWARD_ABSOLUTE_WEIGHT = float(os.environ.get("REWARD_ABSOLUTE_WEIGHT", "0.3"))

EXACT_WEIGHT = float(os.environ.get("EXACT_WEIGHT", "0.5"))
SEMANTIC_WEIGHT = float(os.environ.get("SEMANTIC_WEIGHT", "0.25"))
CONCEPT_WEIGHT = float(os.environ.get("CONCEPT_WEIGHT", "0.15"))
STRUCT_WEIGHT = float(os.environ.get("STRUCT_WEIGHT", "0.05"))
DELTA_WEIGHT = float(os.environ.get("DELTA_WEIGHT", "0.05"))
CONSISTENCY_BONUS_VALUE = float(os.environ.get("CONSISTENCY_BONUS_VALUE", "0.2"))

SEMANTIC_REWARD_ENABLED = os.environ.get("SEMANTIC_REWARD_ENABLED", "true").strip().lower() == "true"
SEMANTIC_HUNGARIAN_MAX_CODES = int(os.environ.get("SEMANTIC_HUNGARIAN_MAX_CODES", "20"))

MAX_CODES = int(os.environ.get("MAX_CODES", "20"))

ROLLOUT_REQUIRE_PARSE_COMPARABILITY = (
    os.environ.get("ROLLOUT_REQUIRE_PARSE_COMPARABILITY", "true").strip().lower() == "true"
)
ROLLOUT_SKIP_FALLBACK_SOURCES = (
    os.environ.get("ROLLOUT_SKIP_FALLBACK_SOURCES", "true").strip().lower() == "true"
)
ROLLOUT_FAIL_FAST_TRANSPORT = (
    os.environ.get("ROLLOUT_FAIL_FAST_TRANSPORT", "true").strip().lower() == "true"
)
ROLLOUT_TRANSPORT_ERROR_STREAK_THRESHOLD = int(
    os.environ.get("ROLLOUT_TRANSPORT_ERROR_STREAK_THRESHOLD", "3")
)
ROLLOUT_HEALTH_MAX_DROP_RATE = float(os.environ.get("ROLLOUT_HEALTH_MAX_DROP_RATE", "0.2"))
ROLLOUT_SAMPLE_WEIGHT_FALLBACK = float(os.environ.get("ROLLOUT_SAMPLE_WEIGHT_FALLBACK", "0.3"))
ROLLOUT_SAMPLE_WEIGHT_INCOMPARABLE = float(os.environ.get("ROLLOUT_SAMPLE_WEIGHT_INCOMPARABLE", "0.5"))

_CURRICULUM_STATE_CACHE_SECONDS = float(os.environ.get("CURRICULUM_STATE_CACHE_SECONDS", "5.0"))

ROLLOUT_QUEUE_DIR = Path(
    os.environ.get(
        "ROLLOUT_QUEUE_DIR",
        str(Path(__file__).resolve().parent / "rollout_queue"),
    )
)
ROLLOUT_QUEUE_PENDING = ROLLOUT_QUEUE_DIR / "pending"
ROLLOUT_QUEUE_ACKED = ROLLOUT_QUEUE_DIR / "acked"
ROLLOUT_QUEUE_FAILED = ROLLOUT_QUEUE_DIR / "failed"
ROLLOUT_QUEUE_BATCH_SIZE = int(os.environ.get("ROLLOUT_QUEUE_BATCH_SIZE", "32"))
ROLLOUT_QUEUE_MAX_ATTEMPTS = int(os.environ.get("ROLLOUT_QUEUE_MAX_ATTEMPTS", "5"))

_METRICS_LOCK = threading.Lock()
_QUEUE_OBSERVABILITY = {
    "enqueue_attempts": 0,
    "enqueue_success": 0,
    "enqueue_skipped_non_rollout": 0,
    "enqueue_skipped_incomparable": 0,
    "enqueue_skipped_fallback_source": 0,
    "enqueue_weighted_incomparable": 0,
    "enqueue_weighted_fallback_source": 0,
    "enqueue_dropped_partial_payload": 0,
    "enqueue_duplicate": 0,
    "flush_calls": 0,
    "flush_posted_batches": 0,
    "flush_posted_rollouts": 0,
    "flush_duplicate_rollouts": 0,
    "flush_failed_batches": 0,
    "retry_count": 0,
    "acked_rollouts": 0,
    "failed_rollouts": 0,
    "dropped_rollouts": 0,
    "transport_error_total": 0,
    "ack_latency_sum_sec": 0.0,
    "ack_latency_count": 0,
    "ack_latency_samples_sec": deque(maxlen=2048),
    "reward_count": 0,
    "reward_total_sum": 0.0,
    "reward_total_sq_sum": 0.0,
    "reward_exact_sum": 0.0,
    "reward_semantic_sum": 0.0,
    "reward_semantic_sq_sum": 0.0,
    "reward_concept_sum": 0.0,
    "reward_concept_sq_sum": 0.0,
    "reward_semantic_concept_cross_sum": 0.0,
    "reward_concept_precision_sum": 0.0,
    "reward_concept_recall_sum": 0.0,
    "reward_struct_sum": 0.0,
    "reward_delta_sum": 0.0,
    "reward_alignment_bonus_sum": 0.0,
    "reward_consistency_bonus_sum": 0.0,
    "reward_penalty_sum": 0.0,
}

_TRANSPORT_STATE = {
    "consecutive_failures": 0,
    "last_status_code": 0,
    "last_error": "",
    "contract_mismatch_detected": 0,
}
_CURRICULUM_STATE = {
    "last_fetch_ts": 0.0,
    "training_step": 0,
}
_SEMANTIC_WARNING_EMITTED = False
_SEMANTIC_DISABLED = False


def _reset_observability_state() -> None:
    with _METRICS_LOCK:
        _QUEUE_OBSERVABILITY["enqueue_attempts"] = 0
        _QUEUE_OBSERVABILITY["enqueue_success"] = 0
        _QUEUE_OBSERVABILITY["enqueue_skipped_non_rollout"] = 0
        _QUEUE_OBSERVABILITY["enqueue_skipped_incomparable"] = 0
        _QUEUE_OBSERVABILITY["enqueue_skipped_fallback_source"] = 0
        _QUEUE_OBSERVABILITY["enqueue_weighted_incomparable"] = 0
        _QUEUE_OBSERVABILITY["enqueue_weighted_fallback_source"] = 0
        _QUEUE_OBSERVABILITY["enqueue_dropped_partial_payload"] = 0
        _QUEUE_OBSERVABILITY["enqueue_duplicate"] = 0
        _QUEUE_OBSERVABILITY["flush_calls"] = 0
        _QUEUE_OBSERVABILITY["flush_posted_batches"] = 0
        _QUEUE_OBSERVABILITY["flush_posted_rollouts"] = 0
        _QUEUE_OBSERVABILITY["flush_duplicate_rollouts"] = 0
        _QUEUE_OBSERVABILITY["flush_failed_batches"] = 0
        _QUEUE_OBSERVABILITY["retry_count"] = 0
        _QUEUE_OBSERVABILITY["acked_rollouts"] = 0
        _QUEUE_OBSERVABILITY["failed_rollouts"] = 0
        _QUEUE_OBSERVABILITY["dropped_rollouts"] = 0
        _QUEUE_OBSERVABILITY["transport_error_total"] = 0
        _QUEUE_OBSERVABILITY["ack_latency_sum_sec"] = 0.0
        _QUEUE_OBSERVABILITY["ack_latency_count"] = 0
        _QUEUE_OBSERVABILITY["ack_latency_samples_sec"].clear()
        _QUEUE_OBSERVABILITY["reward_count"] = 0
        _QUEUE_OBSERVABILITY["reward_total_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_total_sq_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_exact_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_semantic_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_semantic_sq_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_concept_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_concept_sq_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_semantic_concept_cross_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_concept_precision_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_concept_recall_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_struct_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_delta_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_alignment_bonus_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_consistency_bonus_sum"] = 0.0
        _QUEUE_OBSERVABILITY["reward_penalty_sum"] = 0.0

        _TRANSPORT_STATE["consecutive_failures"] = 0
        _TRANSPORT_STATE["last_status_code"] = 0
        _TRANSPORT_STATE["last_error"] = ""
        _TRANSPORT_STATE["contract_mismatch_detected"] = 0

_DESCRIPTOR_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|\bwith\b|\bfor\b)\s*", re.IGNORECASE)
_DESCRIPTOR_PREFIX_RE = re.compile(
    r"^(?:focus\s+on|focused\s+on|target(?:ing)?|include|including|extract|diagnos(?:is|es)|"
    r"codes?|code\s+for|icd-?10(?:-cm)?|clinical\s+focus)\s+",
    re.IGNORECASE,
)
_DESCRIPTOR_STOPWORDS = {
    "diagnosis",
    "diagnoses",
    "code",
    "codes",
    "icd",
    "icd10",
    "icd-10",
    "json",
    "array",
    "list",
    "output",
    "extract",
    "clinical",
    "prompt",
    "focus",
    "target",
    "targets",
    "relevant",
    "active",
    "principal",
    "comorbidities",
    "comorbidity",
    "complication",
    "complications",
    "condition",
    "conditions",
}


class RolloutPostError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"status={status_code} body={body}")
        self.status_code = int(status_code)
        self.body = str(body)


def _record_transport_success() -> None:
    with _METRICS_LOCK:
        _TRANSPORT_STATE["consecutive_failures"] = 0
        _TRANSPORT_STATE["last_error"] = ""


def _record_transport_failure(status_code: Optional[int], error_text: str) -> None:
    with _METRICS_LOCK:
        _TRANSPORT_STATE["consecutive_failures"] += 1
        _TRANSPORT_STATE["last_status_code"] = int(status_code or 0)
        _TRANSPORT_STATE["last_error"] = str(error_text)
        if int(status_code or 0) == 404:
            _TRANSPORT_STATE["contract_mismatch_detected"] = 1
        _QUEUE_OBSERVABILITY["transport_error_total"] += 1


def _transport_snapshot() -> Dict[str, float | int | str]:
    with _METRICS_LOCK:
        return {
            "transport_consecutive_failures": int(_TRANSPORT_STATE["consecutive_failures"]),
            "transport_last_status_code": int(_TRANSPORT_STATE["last_status_code"]),
            "transport_last_error": str(_TRANSPORT_STATE["last_error"]),
            "transport_contract_mismatch_detected": int(_TRANSPORT_STATE["contract_mismatch_detected"]),
        }


def load_icd10_tree() -> nx.DiGraph:
    with TREE_PATH.open("r", encoding="utf-8") as f:
        adjacency = json.load(f)

    graph = nx.DiGraph()
    for parent, children in adjacency.items():
        graph.add_node(parent)
        for child in children:
            graph.add_edge(parent, child)

    graph.add_node("ROOT")
    for node in list(graph.nodes):
        if node != "ROOT" and graph.in_degree(node) == 0:
            graph.add_edge("ROOT", node)

    return graph


def _compute_max_depth(graph: nx.DiGraph) -> int:
    max_depth = 1
    for node in graph.nodes:
        if node == "ROOT":
            continue
        try:
            depth = nx.shortest_path_length(graph, "ROOT", node)
            if depth > max_depth:
                max_depth = depth
        except nx.NetworkXNoPath:
            continue
    return max_depth


ICD10_GRAPH = load_icd10_tree()
MAX_DEPTH = _compute_max_depth(ICD10_GRAPH)


def canonicalize_code(code: str) -> str:
    normalized = str(code).strip().upper()
    if len(normalized) > 3 and "." not in normalized:
        normalized = f"{normalized[:3]}.{normalized[3:]}"
    return normalized


def canonicalize_codes(codes) -> list:
    return [canonicalize_code(code) for code in list(codes or []) if str(code).strip()]


def _nearest_existing_node(code: str, graph: nx.DiGraph) -> str:
    if code in graph:
        return code

    if "." in code:
        parent = code.split(".")[0]
        if parent in graph:
            return parent

    if len(code) >= 3 and code[:3] in graph:
        return code[:3]
    if len(code) >= 2 and code[:2] in graph:
        return code[:2]
    if code and code[0] in graph:
        return code[0]
    return "ROOT"


def _lca_depth(path_a: List[str], path_b: List[str]) -> int:
    shared_depth = 0
    for idx, (node_a, node_b) in enumerate(zip(path_a, path_b)):
        if node_a != node_b:
            break
        shared_depth = idx
    return shared_depth


def distance_between(codes_a, codes_b, graph, max_depth):
    if not codes_a and not codes_b:
        return 0.0
    if not codes_a or not codes_b:
        return 1.0

    left = canonicalize_codes(codes_a)
    right = canonicalize_codes(codes_b)

    lca_depths: List[int] = []
    for left_code in left:
        left_node = _nearest_existing_node(left_code, graph)
        left_path = nx.shortest_path(graph, "ROOT", left_node)

        best_depth = 0
        for right_code in right:
            right_node = _nearest_existing_node(right_code, graph)
            right_path = nx.shortest_path(graph, "ROOT", right_node)
            depth = _lca_depth(left_path, right_path)
            if depth > best_depth:
                best_depth = depth

        lca_depths.append(best_depth)

    mean_lca_depth = sum(lca_depths) / max(len(lca_depths), 1)
    return 1 - (mean_lca_depth / max_depth)


def _relative_reward(d_gt_enh: float, d_gt_org: float, d_enh_org: float) -> float:
    if d_gt_enh < d_gt_org:
        improvement = d_gt_org - d_gt_enh
        return float(min(improvement * 2.0, 1.0))

    if d_gt_enh > d_gt_org:
        degradation = d_gt_enh - d_gt_org
        return float(-min(degradation * 2.0, 1.0))

    if d_enh_org == 0:
        return 0.0

    # Tie on relative baseline, but still provide a weak absolute signal.
    return float(-d_gt_enh)


def _absolute_reward(d_gt_enh: float) -> float:
    # Map distance in [0,1] to reward in [-1,1].
    return float(max(min(1.0 - (2.0 * d_gt_enh), 1.0), -1.0))


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(min(float(value), high), low))


def _exact_code_match_reward(gt_codes: List[str], pred_codes: List[str]) -> float:
    target = set(canonicalize_codes(gt_codes))
    predicted = set(canonicalize_codes(pred_codes))

    if not target and not predicted:
        return 1.0
    if not predicted:
        return -1.0
    if not target:
        return -1.0

    tp = len(target & predicted)
    fp = len(predicted - target)
    fn = len(target - predicted)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return _clamp((2.0 * f1) - 1.0)


def _structure_validity_reward(
    enh_codes: List[str],
    org_codes: List[str],
    enh_parse_ok: Optional[bool],
    org_parse_ok: Optional[bool],
    both_parse_success: Optional[bool],
) -> float:
    enh_ok = bool(enh_parse_ok) if enh_parse_ok is not None else bool(canonicalize_codes(enh_codes))
    org_ok = bool(org_parse_ok) if org_parse_ok is not None else bool(canonicalize_codes(org_codes))
    both_ok = bool(both_parse_success) if both_parse_success is not None else (enh_ok and org_ok)

    if not enh_ok:
        return -1.0
    if both_ok:
        return 1.0
    if enh_ok and not org_ok:
        return 0.6
    return 0.2


def _prediction_penalty(pred_codes: List[str]) -> float:
    canonical = canonicalize_codes(pred_codes)
    penalty = 0.0
    if len(canonical) > MAX_CODES:
        penalty -= 0.1
    if len(canonical) != len(set(canonical)):
        penalty -= 0.1
    return penalty


def _hungarian_similarity_mean(similarity: np.ndarray) -> float:
    if similarity.size == 0:
        return 0.0

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-similarity)
        if len(row_ind) == 0:
            return 0.0
        return float(np.mean(similarity[row_ind, col_ind]))
    except Exception:
        return float(np.mean(np.max(similarity, axis=1)))


def _semantic_similarity_reward(
    gt_codes: List[str],
    pred_codes: List[str],
    exact_reward: float,
) -> tuple[float, float]:
    global _SEMANTIC_WARNING_EMITTED, _SEMANTIC_DISABLED
    del exact_reward

    if not pred_codes:
        return -1.0, 0.0
    if not SEMANTIC_REWARD_ENABLED:
        return 0.0, 0.0
    if _SEMANTIC_DISABLED:
        return 0.0, 0.0

    gt_descriptions = [normalize_description(desc) for desc in get_descriptions(gt_codes)]
    pred_descriptions = [normalize_description(desc) for desc in get_descriptions(pred_codes)]

    gt_descriptions = [desc for desc in gt_descriptions if desc]
    pred_descriptions = [desc for desc in pred_descriptions if desc]

    if not gt_descriptions or not pred_descriptions:
        return 0.0, 0.0

    try:
        pred_embeddings = embed_texts(pred_descriptions)
        gt_embeddings = embed_texts(gt_descriptions)
        similarity = cosine_similarity_matrix(pred_embeddings, gt_embeddings)

        if (
            similarity.shape[0] <= SEMANTIC_HUNGARIAN_MAX_CODES
            and similarity.shape[1] <= SEMANTIC_HUNGARIAN_MAX_CODES
        ):
            score = _hungarian_similarity_mean(similarity)
        else:
            score = float(np.mean(np.max(similarity, axis=1)))

        semantic_score = float(np.clip(score, 0.0, 1.0))
        semantic_reward = _clamp(2.0 * (semantic_score - 0.5))
        return semantic_reward, semantic_score
    except Exception as exc:
        if not _SEMANTIC_WARNING_EMITTED:
            log.warning("semantic_reward_unavailable err=%s", exc)
            _SEMANTIC_WARNING_EMITTED = True
        _SEMANTIC_DISABLED = True
        return 0.0, 0.0


def _clean_descriptor_value(raw: str) -> str:
    value = normalize_description(str(raw or ""))
    if not value:
        return ""

    value = _DESCRIPTOR_PREFIX_RE.sub("", value).strip()
    tokens = [token for token in value.split() if token]
    if not tokens:
        return ""

    while tokens and tokens[0] in _DESCRIPTOR_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _DESCRIPTOR_STOPWORDS:
        tokens = tokens[:-1]

    if not tokens:
        return ""
    if len(tokens) > 6:
        return ""
    if all(token in _DESCRIPTOR_STOPWORDS for token in tokens):
        return ""

    return " ".join(tokens)


def _extract_descriptors_from_prompt(rewritten_prompt: str) -> List[str]:
    text = str(rewritten_prompt or "").strip()
    if not text:
        return []

    segments: List[str] = []
    lowered = normalize_description(text)

    focus_match = re.search(
        r"(?:focus\s+on|for|target(?:ing)?|including|include)\s+(.+)",
        lowered,
        flags=re.IGNORECASE,
    )
    if focus_match:
        segments.append(focus_match.group(1))

    segments.extend(re.split(r"[\.;:]", lowered))

    out: List[str] = []
    for segment in segments:
        for piece in _DESCRIPTOR_SPLIT_RE.split(segment):
            candidate = piece.strip()
            if candidate:
                out.append(candidate)
    return out


def _prepare_concept_descriptors(
    semantic_descriptors: Optional[List[str]],
    rewritten_prompt: Optional[str],
) -> List[str]:
    del rewritten_prompt

    raw_candidates = list(semantic_descriptors or [])

    cleaned: List[str] = []
    seen: set[str] = set()

    for raw in raw_candidates:
        candidate = _clean_descriptor_value(str(raw or ""))
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)

    return cleaned


def _descriptor_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9\-']+", normalize_description(value)) if token}


def _descriptor_match_score(predicted_desc: str, descriptor: str) -> float:
    pred_norm = normalize_description(predicted_desc)
    desc_norm = normalize_description(descriptor)

    if not pred_norm or not desc_norm:
        return 0.0
    if desc_norm in pred_norm or pred_norm in desc_norm:
        return 1.0

    pred_tokens = _descriptor_tokens(pred_norm)
    desc_tokens = _descriptor_tokens(desc_norm)
    if not pred_tokens or not desc_tokens:
        return 0.0

    overlap = pred_tokens & desc_tokens
    if not overlap:
        return 0.0

    return len(overlap) / max(len(pred_tokens), len(desc_tokens), 1)


def _descriptor_overlap_count(predicted_descriptions: List[str], descriptors: List[str]) -> int:
    if not predicted_descriptions or not descriptors:
        return 0

    unmatched = set(range(len(descriptors)))
    overlap_count = 0

    for predicted in predicted_descriptions:
        best_idx = None
        best_score = 0.0
        for idx in unmatched:
            score = _descriptor_match_score(predicted, descriptors[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.5:
            overlap_count += 1
            unmatched.remove(best_idx)

    return overlap_count


def _concept_similarity_reward(
    pred_codes: List[str],
    semantic_descriptors: Optional[List[str]],
    rewritten_prompt: Optional[str],
) -> tuple[float, float, float, float]:
    if not pred_codes:
        return -1.0, 0.0, 0.0, 0.0

    descriptors = _prepare_concept_descriptors(
        semantic_descriptors=semantic_descriptors,
        rewritten_prompt=rewritten_prompt,
    )

    # Keep a mild negative signal when rewrite guidance is missing, rather than neutral.
    if len(descriptors) < 1:
        return -0.2, 0.0, 0.0, 0.0

    predicted_descriptions = [normalize_description(desc) for desc in get_descriptions(pred_codes)]
    predicted_descriptions = [desc for desc in predicted_descriptions if desc]

    overlap = _descriptor_overlap_count(predicted_descriptions, descriptors)

    concept_precision = overlap / max(len(pred_codes), 1)
    concept_recall = overlap / max(len(descriptors), 1)
    if concept_precision + concept_recall <= 0.0:
        concept_f1 = 0.0
    else:
        concept_f1 = 2.0 * concept_precision * concept_recall / (concept_precision + concept_recall)

    concept_reward = _clamp((2.0 * concept_f1) - 1.0)
    return concept_reward, concept_precision, concept_recall, concept_f1


def calculate_reward_components(
    gt_codes,
    enh_codes,
    org_codes,
    *,
    semantic_descriptors: Optional[List[str]] = None,
    rewritten_prompt: Optional[str] = None,
    enh_parse_ok: Optional[bool] = None,
    org_parse_ok: Optional[bool] = None,
    both_parse_success: Optional[bool] = None,
    training_step: Optional[int] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    gt_codes = canonicalize_codes(gt_codes)
    enh_codes = canonicalize_codes(enh_codes)
    org_codes = canonicalize_codes(org_codes)
    resolved_step = _resolve_training_step(training_step, epoch)

    if not enh_codes:
        return {
            "exact_reward": -1.0,
            "semantic_reward": 0.0,
            "semantic_score": 0.0,
            "structure_reward": _structure_validity_reward(
                enh_codes=enh_codes,
                org_codes=org_codes,
                enh_parse_ok=enh_parse_ok,
                org_parse_ok=org_parse_ok,
                both_parse_success=both_parse_success,
            ),
            "delta_reward": 0.0,
            "concept_reward": 0.0,
            "concept_precision": 0.0,
            "concept_recall": 0.0,
            "concept_f1": 0.0,
            "consistency_bonus": 0.0,
            "alignment_bonus": 0.0,
            "penalty": 0.0,
            "legacy_relative": 0.0,
            "legacy_absolute": 0.0,
            "training_step": float(resolved_step),
            "total_reward": -1.0,
        }

    if not enh_codes and not org_codes:
        return {
            "exact_reward": 0.0,
            "semantic_reward": 0.0,
            "semantic_score": 0.0,
            "structure_reward": 0.0,
            "delta_reward": 0.0,
            "concept_reward": 0.0,
            "concept_precision": 0.0,
            "concept_recall": 0.0,
            "concept_f1": 0.0,
            "consistency_bonus": 0.0,
            "alignment_bonus": 0.0,
            "penalty": 0.0,
            "legacy_relative": 0.0,
            "legacy_absolute": 0.0,
            "training_step": float(resolved_step),
            "total_reward": 0.0,
        }

    d_gt_enh = distance_between(gt_codes, enh_codes, ICD10_GRAPH, MAX_DEPTH)
    d_gt_org = distance_between(gt_codes, org_codes, ICD10_GRAPH, MAX_DEPTH)
    d_enh_org = distance_between(enh_codes, org_codes, ICD10_GRAPH, MAX_DEPTH)

    legacy_relative = _relative_reward(d_gt_enh, d_gt_org, d_enh_org)
    legacy_absolute = _absolute_reward(d_gt_enh)

    enhanced_exact = _exact_code_match_reward(gt_codes, enh_codes)
    original_exact = _exact_code_match_reward(gt_codes, org_codes)

    exact_reward = enhanced_exact
    exact_delta = _clamp(enhanced_exact - original_exact)
    delta_reward = _clamp((legacy_relative + exact_delta) / 2.0)

    structure_reward = _structure_validity_reward(
        enh_codes=enh_codes,
        org_codes=org_codes,
        enh_parse_ok=enh_parse_ok,
        org_parse_ok=org_parse_ok,
        both_parse_success=both_parse_success,
    )

    semantic_reward, semantic_score = _semantic_similarity_reward(
        gt_codes=gt_codes,
        pred_codes=enh_codes,
        exact_reward=exact_reward,
    )

    concept_reward, concept_precision, concept_recall, concept_f1 = _concept_similarity_reward(
        pred_codes=enh_codes,
        semantic_descriptors=semantic_descriptors,
        rewritten_prompt=rewritten_prompt,
    )

    consistency_bonus = 0.0
    if semantic_score > 0.6 and concept_f1 > 0.5:
        consistency_bonus = CONSISTENCY_BONUS_VALUE

    if exact_reward < -0.5:
        semantic_reward *= 0.3
        concept_reward *= 0.3
        consistency_bonus = 0.0

    penalty = 0.0

    blended = (
        (EXACT_WEIGHT * exact_reward)
        + (SEMANTIC_WEIGHT * semantic_reward)
        + (CONCEPT_WEIGHT * concept_reward)
        + (STRUCT_WEIGHT * structure_reward)
        + (DELTA_WEIGHT * delta_reward)
    )
    total_reward = _clamp(blended + consistency_bonus)

    return {
        "exact_reward": exact_reward,
        "semantic_reward": semantic_reward,
        "semantic_score": semantic_score,
        "structure_reward": structure_reward,
        "delta_reward": delta_reward,
        "concept_reward": concept_reward,
        "concept_precision": concept_precision,
        "concept_recall": concept_recall,
        "concept_f1": concept_f1,
        "consistency_bonus": consistency_bonus,
        "alignment_bonus": consistency_bonus,
        "penalty": penalty,
        "legacy_relative": legacy_relative,
        "legacy_absolute": legacy_absolute,
        "training_step": float(resolved_step),
        "total_reward": total_reward,
    }


def calculate_reward(gt_codes, enh_codes, org_codes, epoch=0):
    components = calculate_reward_components(
        gt_codes=gt_codes,
        enh_codes=enh_codes,
        org_codes=org_codes,
        epoch=epoch,
    )
    return round(components["total_reward"], 4)


class RewardRequest(BaseModel):
    note_id: Optional[str] = None
    run_id: Optional[str] = None
    gt_codes: Optional[List[str]] = None
    gt_file: Optional[str] = None
    enh_codes: List[str]
    org_codes: List[str]
    enh_parse_ok: Optional[bool] = None
    org_parse_ok: Optional[bool] = None
    both_parse_success: Optional[bool] = None
    original_prompt: Optional[str] = None
    rewritten_prompt: Optional[str] = None
    semantic_descriptors: Optional[List[str]] = None
    group_id: Optional[str] = None
    generation_source: Optional[str] = None
    sample_weight: Optional[float] = None
    log_prob_old: Optional[float] = None
    value_estimate: Optional[float] = None
    training_step: Optional[int] = None
    epoch: Optional[int] = 0


class QueueStatusResponse(BaseModel):
    pending_count: int
    acked_count: int
    failed_count: int


class QueueFlushResponse(BaseModel):
    posted_batches: int
    posted_rollouts: int
    duplicate_count: int
    failed_batches: int
    pending_count: int


def _get_cached_training_step() -> int:
    now_ts = time.time()
    with _METRICS_LOCK:
        last_fetch_ts = float(_CURRICULUM_STATE.get("last_fetch_ts", 0.0))
        cached_step = int(_CURRICULUM_STATE.get("training_step", 0))
        if now_ts - last_fetch_ts <= _CURRICULUM_STATE_CACHE_SECONDS:
            return cached_step

    try:
        response = requests.get(RL_STATUS_ENDPOINT, timeout=3)
        response.raise_for_status()
        payload = response.json()
        fetched_step = int(payload.get("training_step", cached_step))
    except Exception:
        fetched_step = cached_step

    with _METRICS_LOCK:
        _CURRICULUM_STATE["training_step"] = fetched_step
        _CURRICULUM_STATE["last_fetch_ts"] = now_ts

    return fetched_step


def _resolve_training_step(request_step: Optional[int], epoch: Optional[int]) -> int:
    if request_step is not None:
        return max(int(request_step), 0)
    if epoch is not None and int(epoch) > 0:
        return int(epoch)
    return _get_cached_training_step()


def _resolve_curriculum_weights(training_step: int) -> Dict[str, float]:
    del training_step

    return {
        "exact": EXACT_WEIGHT,
        "semantic": SEMANTIC_WEIGHT,
        "concept": CONCEPT_WEIGHT,
        "delta": DELTA_WEIGHT,
        "struct": STRUCT_WEIGHT,
    }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return float(ordered[low])
    low_val = ordered[low]
    high_val = ordered[high]
    frac = index - low
    return float(low_val + (high_val - low_val) * frac)


def _metrics_add(metric: str, value) -> None:
    with _METRICS_LOCK:
        _QUEUE_OBSERVABILITY[metric] += value


def _record_reward_metrics(components: Dict[str, float]) -> None:
    total = float(components.get("total_reward", 0.0))
    exact = float(components.get("exact_reward", 0.0))
    semantic = float(components.get("semantic_reward", 0.0))
    concept = float(components.get("concept_reward", 0.0))
    concept_precision = float(components.get("concept_precision", 0.0))
    concept_recall = float(components.get("concept_recall", 0.0))
    struct = float(components.get("structure_reward", 0.0))
    delta = float(components.get("delta_reward", 0.0))
    consistency_bonus = float(components.get("consistency_bonus", 0.0))
    alignment_bonus = float(components.get("alignment_bonus", consistency_bonus))
    penalty = float(components.get("penalty", 0.0))

    with _METRICS_LOCK:
        _QUEUE_OBSERVABILITY["reward_count"] += 1
        _QUEUE_OBSERVABILITY["reward_total_sum"] += total
        _QUEUE_OBSERVABILITY["reward_total_sq_sum"] += total * total
        _QUEUE_OBSERVABILITY["reward_exact_sum"] += exact
        _QUEUE_OBSERVABILITY["reward_semantic_sum"] += semantic
        _QUEUE_OBSERVABILITY["reward_semantic_sq_sum"] += semantic * semantic
        _QUEUE_OBSERVABILITY["reward_concept_sum"] += concept
        _QUEUE_OBSERVABILITY["reward_concept_sq_sum"] += concept * concept
        _QUEUE_OBSERVABILITY["reward_semantic_concept_cross_sum"] += semantic * concept
        _QUEUE_OBSERVABILITY["reward_concept_precision_sum"] += concept_precision
        _QUEUE_OBSERVABILITY["reward_concept_recall_sum"] += concept_recall
        _QUEUE_OBSERVABILITY["reward_struct_sum"] += struct
        _QUEUE_OBSERVABILITY["reward_delta_sum"] += delta
        _QUEUE_OBSERVABILITY["reward_alignment_bonus_sum"] += alignment_bonus
        _QUEUE_OBSERVABILITY["reward_consistency_bonus_sum"] += consistency_bonus
        _QUEUE_OBSERVABILITY["reward_penalty_sum"] += penalty


def _metrics_snapshot() -> Dict[str, float]:
    with _METRICS_LOCK:
        samples = list(_QUEUE_OBSERVABILITY["ack_latency_samples_sec"])
        ack_count = int(_QUEUE_OBSERVABILITY["ack_latency_count"])
        ack_sum = float(_QUEUE_OBSERVABILITY["ack_latency_sum_sec"])
        enqueue_attempts = int(_QUEUE_OBSERVABILITY["enqueue_attempts"])
        dropped_rollouts = int(_QUEUE_OBSERVABILITY["dropped_rollouts"])
        reward_count = int(_QUEUE_OBSERVABILITY["reward_count"])

        semantic_mean = (
            float(_QUEUE_OBSERVABILITY["reward_semantic_sum"]) / reward_count
            if reward_count
            else 0.0
        )
        semantic_var = (
            (float(_QUEUE_OBSERVABILITY["reward_semantic_sq_sum"]) / reward_count) - (semantic_mean * semantic_mean)
            if reward_count
            else 0.0
        )
        semantic_std = math.sqrt(max(semantic_var, 0.0))

        concept_mean = (
            float(_QUEUE_OBSERVABILITY["reward_concept_sum"]) / reward_count
            if reward_count
            else 0.0
        )
        concept_var = (
            (float(_QUEUE_OBSERVABILITY["reward_concept_sq_sum"]) / reward_count) - (concept_mean * concept_mean)
            if reward_count
            else 0.0
        )
        concept_std = math.sqrt(max(concept_var, 0.0))

        semantic_concept_corr = 0.0
        if reward_count and semantic_std > 1e-8 and concept_std > 1e-8:
            cross_mean = float(_QUEUE_OBSERVABILITY["reward_semantic_concept_cross_sum"]) / reward_count
            covariance = cross_mean - (semantic_mean * concept_mean)
            semantic_concept_corr = covariance / (semantic_std * concept_std)

        snapshot = {
            "enqueue_attempts": enqueue_attempts,
            "enqueue_success": int(_QUEUE_OBSERVABILITY["enqueue_success"]),
            "enqueue_skipped_non_rollout": int(_QUEUE_OBSERVABILITY["enqueue_skipped_non_rollout"]),
            "enqueue_skipped_incomparable": int(_QUEUE_OBSERVABILITY["enqueue_skipped_incomparable"]),
            "enqueue_skipped_fallback_source": int(_QUEUE_OBSERVABILITY["enqueue_skipped_fallback_source"]),
            "enqueue_weighted_incomparable": int(_QUEUE_OBSERVABILITY["enqueue_weighted_incomparable"]),
            "enqueue_weighted_fallback_source": int(_QUEUE_OBSERVABILITY["enqueue_weighted_fallback_source"]),
            "enqueue_dropped_partial_payload": int(_QUEUE_OBSERVABILITY["enqueue_dropped_partial_payload"]),
            "enqueue_duplicate": int(_QUEUE_OBSERVABILITY["enqueue_duplicate"]),
            "flush_calls": int(_QUEUE_OBSERVABILITY["flush_calls"]),
            "flush_posted_batches": int(_QUEUE_OBSERVABILITY["flush_posted_batches"]),
            "flush_posted_rollouts": int(_QUEUE_OBSERVABILITY["flush_posted_rollouts"]),
            "flush_duplicate_rollouts": int(_QUEUE_OBSERVABILITY["flush_duplicate_rollouts"]),
            "flush_failed_batches": int(_QUEUE_OBSERVABILITY["flush_failed_batches"]),
            "retry_count": int(_QUEUE_OBSERVABILITY["retry_count"]),
            "acked_rollouts": int(_QUEUE_OBSERVABILITY["acked_rollouts"]),
            "failed_rollouts": int(_QUEUE_OBSERVABILITY["failed_rollouts"]),
            "dropped_rollouts": dropped_rollouts,
            "transport_error_total": int(_QUEUE_OBSERVABILITY["transport_error_total"]),
            "ack_latency_mean_sec": (ack_sum / ack_count) if ack_count else 0.0,
            "ack_latency_p95_sec": _percentile(samples, 0.95),
            "ack_latency_p99_sec": _percentile(samples, 0.99),
            "rollout_drop_rate": (dropped_rollouts / enqueue_attempts) if enqueue_attempts else 0.0,
            "reward_count": reward_count,
            "reward_mean": (
                float(_QUEUE_OBSERVABILITY["reward_total_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "reward_std": math.sqrt(
                max(
                    (
                        (float(_QUEUE_OBSERVABILITY["reward_total_sq_sum"]) / reward_count)
                        - (
                            (
                                float(_QUEUE_OBSERVABILITY["reward_total_sum"]) / reward_count
                            )
                            ** 2
                        )
                    )
                    if reward_count
                    else 0.0,
                    0.0,
                )
            ),
            "exact_mean": (
                float(_QUEUE_OBSERVABILITY["reward_exact_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "semantic_mean": semantic_mean,
            "semantic_std": semantic_std,
            "concept_mean": concept_mean,
            "concept_std": concept_std,
            "semantic_concept_correlation": semantic_concept_corr,
            "semantic_concept_alignment": semantic_concept_corr,
            "concept_precision_mean": (
                float(_QUEUE_OBSERVABILITY["reward_concept_precision_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "concept_recall_mean": (
                float(_QUEUE_OBSERVABILITY["reward_concept_recall_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "structure_mean": (
                float(_QUEUE_OBSERVABILITY["reward_struct_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "delta_mean": (
                float(_QUEUE_OBSERVABILITY["reward_delta_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "alignment_bonus_mean": (
                float(_QUEUE_OBSERVABILITY["reward_alignment_bonus_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "consistency_bonus_mean": (
                float(_QUEUE_OBSERVABILITY["reward_consistency_bonus_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
            "penalty_mean": (
                float(_QUEUE_OBSERVABILITY["reward_penalty_sum"]) / reward_count
                if reward_count
                else 0.0
            ),
        }
    return snapshot


def _oldest_pending_age_seconds() -> float:
    _ensure_queue_dirs()
    oldest: Optional[float] = None

    for path in ROLLOUT_QUEUE_PENDING.glob("*.json"):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            created_at = envelope.get("created_at")
            if not created_at:
                continue
            created_ts = datetime.fromisoformat(str(created_at)).timestamp()
            if oldest is None or created_ts < oldest:
                oldest = created_ts
        except Exception:
            continue

    if oldest is None:
        return 0.0
    return max(time.time() - oldest, 0.0)


def _transport_is_degraded(obs: Dict[str, float] | None = None) -> bool:
    snapshot = _transport_snapshot()
    metrics = obs if obs is not None else _metrics_snapshot()
    has_enqueues = int(metrics.get("enqueue_attempts", 0)) > 0
    drop_rate = float(metrics.get("rollout_drop_rate", 0.0))

    if int(snapshot.get("transport_contract_mismatch_detected", 0)) > 0:
        return True
    if int(snapshot.get("transport_consecutive_failures", 0)) >= ROLLOUT_TRANSPORT_ERROR_STREAK_THRESHOLD:
        return True
    if has_enqueues and drop_rate > ROLLOUT_HEALTH_MAX_DROP_RATE:
        return True
    return False


def queue_observability_snapshot() -> Dict[str, float]:
    counts = _queue_counts()
    metrics = _metrics_snapshot()
    transport = _transport_snapshot()
    metrics.update(
        {
            "queue_depth": counts.pending_count,
            "queue_acked_files": counts.acked_count,
            "queue_failed_files": counts.failed_count,
            "queue_lag_seconds": _oldest_pending_age_seconds(),
            "transport_consecutive_failures": transport["transport_consecutive_failures"],
            "transport_last_status_code": transport["transport_last_status_code"],
            "transport_last_error": transport["transport_last_error"],
            "transport_contract_mismatch_detected": transport["transport_contract_mismatch_detected"],
            "transport_degraded": _transport_is_degraded(metrics),
        }
    )
    return metrics


app = FastAPI(title="Reward Metrics Service")


def _load_gt_codes_from_file(gt_file: str) -> List[str]:
    path = Path(gt_file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / gt_file

    if not path.exists():
        raise HTTPException(status_code=400, detail=f"gt_file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "gt_codes" in payload:
        return list(payload["gt_codes"])
    if isinstance(payload, list):
        return list(payload)
    raise HTTPException(status_code=400, detail="gt_file must contain list or {gt_codes: [...]} payload")


def _ensure_queue_dirs() -> None:
    ROLLOUT_QUEUE_PENDING.mkdir(parents=True, exist_ok=True)
    ROLLOUT_QUEUE_ACKED.mkdir(parents=True, exist_ok=True)
    ROLLOUT_QUEUE_FAILED.mkdir(parents=True, exist_ok=True)


def _queue_counts() -> QueueStatusResponse:
    _ensure_queue_dirs()
    return QueueStatusResponse(
        pending_count=len(list(ROLLOUT_QUEUE_PENDING.glob("*.json"))),
        acked_count=len(list(ROLLOUT_QUEUE_ACKED.glob("*.json"))),
        failed_count=len(list(ROLLOUT_QUEUE_FAILED.glob("*.json"))),
    )


def _build_rollout_id(reward: float, request: RewardRequest) -> str:
    comparability_ok = _resolve_comparability(request)
    source = str(request.generation_source or "").strip().lower()
    sample_weight = _resolve_rollout_sample_weight(
        request=request,
        comparability_ok=comparability_ok,
        source=source,
    )

    payload = {
        "note_id": request.note_id,
        "run_id": request.run_id,
        "group_id": request.group_id,
        "original_prompt": request.original_prompt,
        "rewritten_prompt": request.rewritten_prompt,
        "generation_source": request.generation_source,
        "sample_weight": float(sample_weight),
        "reward": float(reward),
        "log_prob_old": request.log_prob_old,
        "value_estimate": request.value_estimate,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"r_{digest}"


def _resolve_comparability(request: RewardRequest) -> bool:
    comparability_ok = request.both_parse_success
    if comparability_ok is None and request.enh_parse_ok is not None and request.org_parse_ok is not None:
        comparability_ok = bool(request.enh_parse_ok and request.org_parse_ok)
    if comparability_ok is None:
        comparability_ok = bool(request.enh_codes and request.org_codes)
    return bool(comparability_ok)


def _resolve_rollout_sample_weight(
    request: RewardRequest,
    comparability_ok: bool,
    source: str,
) -> float:
    weight = 1.0
    if request.sample_weight is not None:
        try:
            weight = float(request.sample_weight)
        except Exception:
            weight = 1.0

    weight = max(min(weight, 1.0), 0.0)

    if not comparability_ok:
        weight = min(weight, float(ROLLOUT_SAMPLE_WEIGHT_INCOMPARABLE))

    if source and "fallback" in source:
        weight = min(weight, float(ROLLOUT_SAMPLE_WEIGHT_FALLBACK))

    return max(min(weight, 1.0), 0.0)


def _enqueue_rollout(reward: float, request: RewardRequest, concept_reward: float) -> bool:
    comparability_ok = _resolve_comparability(request)
    source = str(request.generation_source or "").strip().lower()
    sample_weight = _resolve_rollout_sample_weight(
        request=request,
        comparability_ok=comparability_ok,
        source=source,
    )

    payload = {
        "rollout_id": _build_rollout_id(reward, request),
        "run_id": request.run_id or "default",
        "group_id": request.group_id,
        "original_prompt": request.original_prompt,
        "rewritten_prompt": request.rewritten_prompt,
        "generation_source": request.generation_source,
        "sample_weight": float(sample_weight),
        "reward": reward,
        "concept_reward": float(concept_reward),
        "log_prob_old": request.log_prob_old,
        "value_estimate": request.value_estimate,
    }

    rollout_fields = [
        "original_prompt",
        "rewritten_prompt",
        "log_prob_old",
        "value_estimate",
    ]
    missing = [field for field in rollout_fields if payload[field] is None]

    # Val/test reward calls intentionally omit rollout metadata to avoid
    # training leakage; skip enqueue for this expected path.
    if len(missing) == len(rollout_fields):
        _metrics_add("enqueue_skipped_non_rollout", 1)
        return False

    if not comparability_ok:
        if ROLLOUT_REQUIRE_PARSE_COMPARABILITY:
            _metrics_add("enqueue_weighted_incomparable", 1)
        else:
            _metrics_add("enqueue_skipped_incomparable", 1)

    if source and "fallback" in source:
        if ROLLOUT_SKIP_FALLBACK_SOURCES:
            _metrics_add("enqueue_weighted_fallback_source", 1)
        else:
            _metrics_add("enqueue_skipped_fallback_source", 1)

    _metrics_add("enqueue_attempts", 1)

    if missing:
        log.warning("Partial rollout payload missing fields: %s", ", ".join(missing))
        _metrics_add("enqueue_dropped_partial_payload", 1)
        _metrics_add("dropped_rollouts", 1)
        return False

    _ensure_queue_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    filename = f"{timestamp}_{payload['rollout_id']}.json"

    pending_path = ROLLOUT_QUEUE_PENDING / filename
    acked_path = ROLLOUT_QUEUE_ACKED / filename
    failed_path = ROLLOUT_QUEUE_FAILED / filename

    if pending_path.exists() or acked_path.exists() or failed_path.exists():
        _metrics_add("enqueue_duplicate", 1)
        _metrics_add("dropped_rollouts", 1)
        return False

    envelope = {
        "run_id": payload["run_id"],
        "rollout_id": payload["rollout_id"],
        "payload": payload,
        "attempts": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    pending_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    _metrics_add("enqueue_success", 1)
    return True


def _chunked(items: List[Dict], size: int) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for idx in range(0, len(items), max(size, 1)):
        out.append(items[idx : idx + max(size, 1)])
    return out


def _post_rollout_submission(payload: Dict) -> Dict:
    """Post a single rollout submission matching rl_loop_svc RolloutSubmission schema."""
    response = requests.post(RL_BLOCK_ENDPOINT, json=payload, timeout=10)
    if response.status_code >= 400:
        raise RolloutPostError(response.status_code, response.text)
    return response.json() if response.content else {}


def flush_rollout_queue() -> QueueFlushResponse:
    _ensure_queue_dirs()
    _metrics_add("flush_calls", 1)

    posted_batches = 0
    posted_rollouts = 0
    duplicate_count = 0
    failed_batches = 0

    pending_paths = sorted(ROLLOUT_QUEUE_PENDING.glob("*.json"))
    if not pending_paths:
        return QueueFlushResponse(
            posted_batches=0,
            posted_rollouts=0,
            duplicate_count=0,
            failed_batches=0,
            pending_count=0,
        )

    queued_rows: List[Dict] = []
    for path in pending_paths:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            queued_rows.append({"path": path, "envelope": envelope})
        except Exception:
            shutil.move(str(path), str(ROLLOUT_QUEUE_FAILED / path.name))

    for batch in _chunked(queued_rows, ROLLOUT_QUEUE_BATCH_SIZE):
        batch_had_failure = False

        for row in batch:
            path = row["path"]
            envelope = dict(row["envelope"])
            rollout_payload = dict(envelope.get("payload") or {})
            if not rollout_payload:
                failed_path = ROLLOUT_QUEUE_FAILED / path.name
                failed_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
                path.unlink(missing_ok=True)
                _metrics_add("failed_rollouts", 1)
                _metrics_add("dropped_rollouts", 1)
                batch_had_failure = True
                continue

            try:
                body = _post_rollout_submission(rollout_payload)
                _record_transport_success()
                posted_rollouts += int(body.get("accepted_count", 1))
                duplicate_count += int(body.get("duplicate_count", 0))

                try:
                    created_at = envelope.get("created_at")
                    if created_at:
                        lag = max(time.time() - datetime.fromisoformat(str(created_at)).timestamp(), 0.0)
                        _metrics_add("ack_latency_sum_sec", lag)
                        _metrics_add("ack_latency_count", 1)
                        with _METRICS_LOCK:
                            _QUEUE_OBSERVABILITY["ack_latency_samples_sec"].append(lag)
                except Exception:
                    pass

                shutil.move(str(path), str(ROLLOUT_QUEUE_ACKED / path.name))
                _metrics_add("flush_posted_rollouts", int(body.get("accepted_count", 1)))
                _metrics_add("flush_duplicate_rollouts", int(body.get("duplicate_count", 0)))
                _metrics_add("dropped_rollouts", int(body.get("duplicate_count", 0)))
                _metrics_add("acked_rollouts", 1)
            except RolloutPostError as exc:
                _record_transport_failure(exc.status_code, str(exc))
                batch_had_failure = True
                log.warning("Rollout queue batch post failed: %s", exc)
                _metrics_add("flush_failed_batches", 1)
                envelope["attempts"] = int(envelope.get("attempts", 0)) + 1
                if envelope["attempts"] >= ROLLOUT_QUEUE_MAX_ATTEMPTS:
                    failed_path = ROLLOUT_QUEUE_FAILED / path.name
                    failed_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
                    path.unlink(missing_ok=True)
                    _metrics_add("failed_rollouts", 1)
                    _metrics_add("dropped_rollouts", 1)
                else:
                    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
                    _metrics_add("retry_count", 1)
            except Exception as exc:
                _record_transport_failure(None, str(exc))
                batch_had_failure = True
                log.warning("Rollout queue batch post failed: %s", exc)
                _metrics_add("flush_failed_batches", 1)
                envelope["attempts"] = int(envelope.get("attempts", 0)) + 1
                if envelope["attempts"] >= ROLLOUT_QUEUE_MAX_ATTEMPTS:
                    failed_path = ROLLOUT_QUEUE_FAILED / path.name
                    failed_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
                    path.unlink(missing_ok=True)
                    _metrics_add("failed_rollouts", 1)
                    _metrics_add("dropped_rollouts", 1)
                else:
                    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
                    _metrics_add("retry_count", 1)

        posted_batches += 1
        _metrics_add("flush_posted_batches", 1)
        if batch_had_failure:
            failed_batches += 1

    return QueueFlushResponse(
        posted_batches=posted_batches,
        posted_rollouts=posted_rollouts,
        duplicate_count=duplicate_count,
        failed_batches=failed_batches,
        pending_count=len(list(ROLLOUT_QUEUE_PENDING.glob("*.json"))),
    )


def post_to_rl_block(reward: float, request: RewardRequest, concept_reward: float) -> None:
    enqueued = _enqueue_rollout(reward, request, concept_reward)
    if enqueued:
        flush_rollout_queue()
        if ROLLOUT_FAIL_FAST_TRANSPORT and _transport_is_degraded():
            raise HTTPException(
                status_code=503,
                detail="Rollout transport degraded; refusing to acknowledge reward while training transport is unhealthy",
            )


@app.post("/reward")
def reward_endpoint(request: RewardRequest):
    resolved_gt_codes = request.gt_codes
    if resolved_gt_codes is None and request.gt_file:
        resolved_gt_codes = _load_gt_codes_from_file(request.gt_file)

    if resolved_gt_codes is None:
        raise HTTPException(status_code=400, detail="Provide either gt_codes or gt_file")

    components = calculate_reward_components(
        gt_codes=resolved_gt_codes,
        enh_codes=request.enh_codes,
        org_codes=request.org_codes,
        semantic_descriptors=request.semantic_descriptors,
        rewritten_prompt=request.rewritten_prompt,
        enh_parse_ok=request.enh_parse_ok,
        org_parse_ok=request.org_parse_ok,
        both_parse_success=request.both_parse_success,
        training_step=request.training_step,
        epoch=request.epoch or 0,
    )
    reward = round(float(components["total_reward"]), 4)

    _record_reward_metrics(components)
    log.info(
        "reward_breakdown | exact=%.3f semantic=%.3f semantic_score=%.3f concept=%.3f concept_precision=%.3f concept_recall=%.3f struct=%.3f delta=%.3f consistency_bonus=%.3f penalty=%.3f total=%.3f",
        float(components.get("exact_reward", 0.0)),
        float(components.get("semantic_reward", 0.0)),
        float(components.get("semantic_score", 0.0)),
        float(components.get("concept_reward", 0.0)),
        float(components.get("concept_precision", 0.0)),
        float(components.get("concept_recall", 0.0)),
        float(components.get("structure_reward", 0.0)),
        float(components.get("delta_reward", 0.0)),
        float(components.get("consistency_bonus", 0.0)),
        float(components.get("penalty", 0.0)),
        reward,
    )

    post_to_rl_block(reward, request, float(components.get("concept_reward", 0.0)))

    return {
        "reward": reward,
        "reward_components": {
            "exact": round(float(components.get("exact_reward", 0.0)), 4),
            "semantic": round(float(components.get("semantic_reward", 0.0)), 4),
            "semantic_score": round(float(components.get("semantic_score", 0.0)), 4),
            "concept": round(float(components.get("concept_reward", 0.0)), 4),
            "concept_precision": round(float(components.get("concept_precision", 0.0)), 4),
            "concept_recall": round(float(components.get("concept_recall", 0.0)), 4),
            "structure": round(float(components.get("structure_reward", 0.0)), 4),
            "delta": round(float(components.get("delta_reward", 0.0)), 4),
            "alignment_bonus": round(float(components.get("alignment_bonus", 0.0)), 4),
            "consistency_bonus": round(float(components.get("consistency_bonus", 0.0)), 4),
            "penalty": round(float(components.get("penalty", 0.0)), 4),
        },
    }


@app.get("/queue/status", response_model=QueueStatusResponse)
def queue_status() -> QueueStatusResponse:
    return _queue_counts()


@app.post("/queue/flush", response_model=QueueFlushResponse)
def queue_flush() -> QueueFlushResponse:
    return flush_rollout_queue()


@app.get("/observability")
def observability():
    return queue_observability_snapshot()


@app.post("/observability/reset")
def reset_observability():
    _reset_observability_state()
    return {"status": "ok"}


@app.get("/health")
def health():
    q = _queue_counts()
    obs = queue_observability_snapshot()
    degraded = bool(obs.get("transport_degraded", False))
    return {
        "status": "degraded" if degraded else "ok",
        "tree_nodes": ICD10_GRAPH.number_of_nodes(),
        "tree_edges": ICD10_GRAPH.number_of_edges(),
        "max_depth": MAX_DEPTH,
        "queue_pending": q.pending_count,
        "queue_failed": q.failed_count,
        "queue_lag_seconds": obs.get("queue_lag_seconds", 0.0),
        "rollout_drop_rate": obs.get("rollout_drop_rate", 0.0),
        "transport_degraded": degraded,
        "icd_description_source": description_source(),
        "icd_description_count": description_count(),
    }


# uvicorn reward_metrics_svc.main:app --host 0.0.0.0 --port 8002
