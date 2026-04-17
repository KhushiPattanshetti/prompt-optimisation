#!/usr/bin/env python3
"""Full-dataset train/test/validate runner for the prompt optimisation pipeline.

Optimized execution flow:
1. Micro-batched rewriter calls.
2. Micro-batched ICD calls.
3. Stage overlap: while batch N is in ICD/reward, batch N+1 runs in rewriter.
4. Async RL train triggers every N successful train rollouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_logger import ServiceIOLogger, init_run
from pipeline_logger.hash_utils import prompt_hash
from pipeline_logger.mode_resolver import resolve_mode1, resolve_mode2
from pipeline_logger.system_logger import write_batch_summary, write_csv_rows


@dataclass
class TimingStats:
    rewrite_sec: List[float]
    icd10_sec: List[float]
    reward_sec: List[float]
    train_cycle_sec: List[float]
    note_total_sec: List[float]


@dataclass
class PipelineItem:
    note_id: str
    note_text: str
    gt_codes: List[str]
    split: str
    group_id: str
    note_index: int
    note_started: float
    note_record: Dict
    rwj: Optional[Dict] = None
    icj: Optional[Dict] = None


@dataclass
class RunStats:
    processed: int = 0
    train_seen: int = 0
    train_success: int = 0
    val_success: int = 0
    test_success: int = 0
    failures: int = 0
    parse_success_count: int = 0
    both_parse_success_count: int = 0
    original_parse_failure_count: int = 0
    train_comparability_failure_count: int = 0
    train_cycle_attempts: int = 0
    train_cycle_success_count: int = 0
    train_group_valid_count: int = 0
    train_group_invalid_count: int = 0
    train_rollouts_dropped_count: int = 0


_THREAD_LOCAL = threading.local()
_REWRITER_IO = ServiceIOLogger("rewriter_svc")
_ICD10_IO = ServiceIOLogger("icd10_svc")
_REWARD_IO = ServiceIOLogger("reward_svc")
_RL_IO = ServiceIOLogger("rl_loop_svc")
_REWRITER_URL_LOCK = threading.Lock()
_REWRITER_URL_INDEX = 0
_REWRITER_METRICS_LOCK = threading.Lock()
_REWRITER_METRICS: Dict[str, Dict[str, float]] = {}


def _build_pooled_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = _build_pooled_session()
        _THREAD_LOCAL.session = session
    return session


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def now() -> float:
    return time.perf_counter()


def log_line(message: str) -> None:
    print(message, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def default_results_dir() -> str:
    env_dir = os.environ.get("RUN_RESULTS_DIR", "").strip()
    if env_dir:
        return env_dir
    if Path("/app/run_results").exists():
        return "/app/run_results"
    return "run_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full dataset train/test/validate pipeline")
    p.add_argument("--dataset-url", default="http://localhost:8003")
    p.add_argument("--rewriter-url", default="http://localhost:8000")
    p.add_argument(
        "--rewriter-urls",
        default="",
        help="Comma-separated list of rewriter service URLs for round-robin routing.",
    )
    p.add_argument("--icd10-url", default="http://localhost:8001")
    p.add_argument("--reward-url", default="http://localhost:8002")
    p.add_argument("--rl-url", default="http://localhost:8004")
    p.add_argument("--run-id", default="", help="Optional run identifier for rollout isolation")
    p.add_argument(
        "--results-dir",
        default=default_results_dir(),
        help="Directory to write run artifacts (notes JSONL + summary JSON).",
    )
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--rewriter-batch-size", type=int, default=6)
    p.add_argument("--icd-batch-size", type=int, default=6)
    p.add_argument("--rewriter-workers", type=int, default=4)
    p.add_argument("--icd-workers", type=int, default=4)
    p.add_argument("--reward-workers", type=int, default=4)
    p.add_argument("--train-workers", type=int, default=3)
    p.add_argument(
        "--stream-note-wise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Phase 1 streaming mode: process notes through rewriter+ICD+reward one-by-one within each dataset batch.",
    )
    p.add_argument(
        "--trajectory-flush-size",
        type=int,
        default=96,
        help="Phase 2 streaming mode: flush buffered valid train rollouts to trajectory store in batches of this size.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--split-strategy",
        choices=["stratified", "hash"],
        default="stratified",
        help="Split assignment strategy. Use 'hash' for fast streaming assignment in timeboxed runs.",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--grpo-group-size", type=int, default=3)
    p.add_argument("--rewrites-per-note", type=int, default=6)
    p.add_argument(
        "--fast-train-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, use fewer train rewrite candidates per note to reduce latency.",
    )
    p.add_argument(
        "--fast-rewrites-per-note",
        type=int,
        default=3,
        help="Rewrite candidates per train note in fast mode.",
    )
    p.add_argument(
        "--train-candidate-min-len",
        type=int,
        default=32,
        help="Minimum rewritten prompt length for train candidate acceptance.",
    )
    p.add_argument(
        "--train-candidate-max-len",
        type=int,
        default=2000,
        help="Maximum rewritten prompt length for train candidate acceptance.",
    )
    p.add_argument(
        "--train-candidate-max-note-token-overlap",
        type=float,
        default=0.55,
        help="Max token overlap ratio with source note before dropping a train rewrite.",
    )
    p.add_argument(
        "--max-pending-train-triggers",
        type=int,
        default=1,
        help="Maximum queued train triggers to avoid long trigger backlogs.",
    )
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument(
        "--max-run-minutes",
        type=float,
        default=0.0,
        help="Wall-clock limit in minutes; 0 disables timeboxing.",
    )
    p.add_argument(
        "--checkpoint-every-minutes",
        type=float,
        default=0.0,
        help="Write interim checkpoint snapshots every N minutes; 0 disables periodic checkpoints.",
    )
    p.add_argument("--request-timeout", type=int, default=1800)
    p.add_argument("--max-notes", type=int, default=0, help="0 means full dataset")
    p.add_argument(
        "--service-ready-attempts",
        type=int,
        default=600,
        help="Health-check attempts per service before failing (default: 600).",
    )
    p.add_argument(
        "--service-ready-sleep-sec",
        type=int,
        default=2,
        help="Seconds between service health-check attempts (default: 2).",
    )
    p.add_argument("--guard-max-rollout-drop-rate", type=float, default=0.2)
    p.add_argument("--guard-min-both-parse-rate", type=float, default=0.5)
    p.add_argument("--guard-max-original-failure-rate", type=float, default=0.6)
    p.add_argument(
        "--reset-reward-observability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset reward observability counters at run start (recommended for gate runs).",
    )
    return p.parse_args()


def ensure_services(session: requests.Session, args: argparse.Namespace) -> None:
    def wait_ready(name: str, url: str, attempts: int, sleep_sec: int) -> None:
        last_error: str = ""
        for attempt in range(1, attempts + 1):
            try:
                r = session.get(url, timeout=10)
                r.raise_for_status()
                log_line(f"service_ok {name} {url} attempt={attempt}")
                return
            except Exception as exc:
                last_error = str(exc)
                if attempt % 10 == 0:
                    log_line(f"service_wait {name} attempt={attempt} err={last_error}")
                time.sleep(sleep_sec)

        raise RuntimeError(f"service_unavailable name={name} url={url} last_err={last_error}")

    rewriter_checks = []
    for url in _rewriter_url_pool(args):
        rewriter_checks.append((f"rewriter[{url}]", f"{url}/health"))

    checks = [
        ("dataset", f"{args.dataset_url}/health"),
        ("icd10", f"{args.icd10_url}/health"),
        ("reward", f"{args.reward_url}/health"),
        ("rl", f"{args.rl_url}/status"),
    ] + rewriter_checks
    for name, url in checks:
        wait_ready(
            name,
            url,
            attempts=args.service_ready_attempts,
            sleep_sec=args.service_ready_sleep_sec,
        )


def reset_reward_observability(session: requests.Session, reward_url: str) -> None:
    try:
        response = session.post(f"{reward_url}/observability/reset", timeout=30)
        if response.status_code < 400:
            log_line("reward_observability_reset status=ok")
        else:
            log_line(f"warning reward_observability_reset_failed status={response.status_code}")
    except Exception as exc:
        log_line(f"warning reward_observability_reset_unavailable err={exc}")


def get_dataset_total(session: requests.Session, dataset_url: str) -> int:
    r = session.get(f"{dataset_url}/health", timeout=30)
    r.raise_for_status()
    payload = r.json()
    return int(payload["total_notes"])


def split_key_for_gt_codes(gt_codes: List[str]) -> str:
    if not gt_codes:
        return "NONE"
    first = canonicalize_code(gt_codes[0])
    if not first:
        return "NONE"
    return first[0]


def build_stratified_split_map(
    session: requests.Session,
    dataset_url: str,
    total_notes: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    groups: Dict[str, List[str]] = defaultdict(list)

    offset = 0
    while offset < total_notes:
        size = min(batch_size, total_notes - offset)
        br = session.get(
            f"{dataset_url}/batch",
            params={"offset": offset, "size": size},
            timeout=60,
        )
        br.raise_for_status()
        batch = br.json().get("batch", [])
        if not batch:
            break

        for rec in batch:
            note_id = str(rec["note_id"])
            gt_codes = rec.get("gt_codes", [])
            groups[split_key_for_gt_codes(gt_codes)].append(note_id)

        offset += size

    rng = random.Random(seed)
    split_map: Dict[str, str] = {}
    for _, note_ids in groups.items():
        shuffled = list(note_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        for idx, note_id in enumerate(shuffled):
            if idx < train_n:
                split_map[note_id] = "train"
            elif idx < train_n + val_n:
                split_map[note_id] = "val"
            else:
                split_map[note_id] = "test"

    return split_map


def split_for_note_id_hash(note_id: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    # Stable split assignment that avoids whole-dataset prescan.
    key = f"{seed}:{note_id}".encode("utf-8")
    bucket = int(hashlib.sha1(key).hexdigest()[:12], 16) / float(16 ** 12)
    if bucket < train_ratio:
        return "train"
    if bucket < (train_ratio + val_ratio):
        return "val"
    return "test"


def canonicalize_code(code: str) -> str:
    value = str(code).strip().upper()
    if len(value) > 3 and "." not in value:
        value = f"{value[:3]}.{value[3:]}"
    return value


def canonicalized_code_set(codes: List[str]) -> set[str]:
    return {
        canonicalize_code(code)
        for code in list(codes or [])
        if str(code).strip()
    }


def update_micro_counts(target: set[str], predicted: set[str], counts: Dict[str, int]) -> None:
    counts["tp"] += len(target & predicted)
    counts["fp"] += len(predicted - target)
    counts["fn"] += len(target - predicted)


def per_note_prf(target: set[str], predicted: set[str]) -> Tuple[float, float, float]:
    tp = len(target & predicted)
    fp = len(predicted - target)
    fn = len(target - predicted)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def update_macro_counts(target: set[str], predicted: set[str], counts: Dict[str, float]) -> None:
    precision, recall, f1 = per_note_prf(target, predicted)
    counts["precision_sum"] += precision
    counts["recall_sum"] += recall
    counts["f1_sum"] += f1
    counts["exact_match"] += 1.0 if target == predicted else 0.0
    counts["count"] += 1.0


def macro_prf_exact(counts: Dict[str, float]) -> Tuple[float, float, float, float]:
    denom = counts["count"] if counts["count"] > 0 else 1.0
    return (
        counts["precision_sum"] / denom,
        counts["recall_sum"] / denom,
        counts["f1_sum"] / denom,
        counts["exact_match"] / denom,
    )


def micro_prf(counts: Dict[str, int]) -> Tuple[float, float, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_original_prompt(clinical_note: str) -> str:
    return (
        "You are a clinical coding expert. "
        "Extract all ICD-10-CM diagnosis codes from the clinical note below. "
        "Output the codes as a JSON list of strings only. "
        "Do not include any explanation or other text.\n\n"
        f"Clinical note:\n{clinical_note}"
    )


def build_group_id(
    note_id: str,
    run_id: str,
    sample_index: Optional[int] = None,
    group_size: int = 3,
) -> str:
    if sample_index is not None and group_size > 1:
        bucket = int(sample_index) // int(group_size)
        digest = hashlib.sha1(f"{run_id}:group:{bucket}".encode("utf-8")).hexdigest()
        return f"g_{digest[:16]}"

    digest = hashlib.sha1(f"{run_id}:{note_id}".encode("utf-8")).hexdigest()
    return f"g_{digest[:16]}"


def get_rl_status(session: requests.Session, rl_url: str) -> Dict:
    r = session.get(f"{rl_url}/status", timeout=30)
    r.raise_for_status()
    return r.json()


def poll_train_idle(
    session: requests.Session,
    rl_url: str,
    previous_finished_at: Optional[str],
    timeout_sec: int = 7200,
) -> Tuple[bool, Dict]:
    t0 = time.time()
    saw_non_idle = False
    while True:
        st = get_rl_status(session, rl_url)
        if st.get("trainer_state") != "IDLE":
            saw_non_idle = True

        finished_at = st.get("last_train_finished_at")
        cycle_finished = bool(finished_at) and finished_at != previous_finished_at
        if st.get("trainer_state") == "IDLE" and (cycle_finished or saw_non_idle):
            return True, st
        if time.time() - t0 > timeout_sec:
            return False, st
        time.sleep(2)


def run_train_cycle(session: requests.Session, rl_url: str, timeout_sec: int = 7200) -> Dict:
    pre_status = get_rl_status(session, rl_url)
    pre_step = int(pre_status.get("training_step", 0))
    pre_finished_at = pre_status.get("last_train_finished_at")

    tr = session.post(f"{rl_url}/train", timeout=30)
    tr.raise_for_status()
    train_response = tr.json()
    if not bool(train_response.get("triggered", False)):
        raise RuntimeError(f"training cycle rejected: {train_response.get('message', 'trainer busy')}")

    ok, status = poll_train_idle(session, rl_url, previous_finished_at=pre_finished_at, timeout_sec=timeout_sec)
    if not ok:
        raise RuntimeError(f"training cycle timeout, last_status={status}")

    if status.get("last_train_success") is False:
        raise RuntimeError(f"training cycle failed: {status.get('last_train_error', 'unknown error')}")

    post_step = int(status.get("training_step", pre_step))
    if post_step <= pre_step:
        raise RuntimeError(f"training step did not advance (pre_step={pre_step} post_step={post_step})")

    return status


def flush_and_wait_reward_queue(session: requests.Session, reward_url: str, timeout_sec: int = 120) -> Dict:
    del session, reward_url, timeout_sec
    return {"pending_count": 0}


def median_or_zero(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def percentile_or_zero(xs: List[float], percentile: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    low_idx = int(math.floor(position))
    high_idx = int(math.ceil(position))
    if low_idx == high_idx:
        return float(ordered[low_idx])
    low_val = ordered[low_idx]
    high_val = ordered[high_idx]
    frac = position - low_idx
    return float(low_val + (high_val - low_val) * frac)


def estimate_remaining_seconds(
    processed: int,
    total: int,
    train_every: int,
    timings: TimingStats,
    train_ratio: float,
) -> float:
    if processed <= 0:
        return 0.0
    per_note = median_or_zero(timings.rewrite_sec) + median_or_zero(timings.icd10_sec)
    per_reward = median_or_zero(timings.reward_sec)
    remaining = max(total - processed, 0)
    remaining_valtest = int(remaining * max(0.0, 1.0 - train_ratio))
    remaining_train = remaining - remaining_valtest
    estimate_notes = (remaining_train * per_note) + (remaining_valtest * (per_note + per_reward))
    cycle_sec = median_or_zero(timings.train_cycle_sec)
    if train_every > 0 and cycle_sec > 0:
        cycles_remaining = math.ceil(max(remaining_train, 0) / train_every)
        estimate_notes += cycles_remaining * cycle_sec
    return estimate_notes


def _chunks(items: List[PipelineItem], chunk_size: int) -> List[List[PipelineItem]]:
    size = max(1, int(chunk_size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _rewriter_url_pool(args: argparse.Namespace) -> List[str]:
    raw = str(getattr(args, "rewriter_urls", "") or "").strip()
    if not raw:
        return [str(args.rewriter_url).rstrip("/")]
    urls = [chunk.strip().rstrip("/") for chunk in raw.split(",") if chunk.strip()]
    return urls if urls else [str(args.rewriter_url).rstrip("/")]


def _next_rewriter_url(args: argparse.Namespace) -> str:
    global _REWRITER_URL_INDEX
    pool = _rewriter_url_pool(args)
    with _REWRITER_URL_LOCK:
        selected = pool[_REWRITER_URL_INDEX % len(pool)]
        _REWRITER_URL_INDEX += 1
    return selected


def _record_rewriter_request(url: str, latency_sec: float, ok: bool) -> None:
    key = str(url).rstrip("/")
    with _REWRITER_METRICS_LOCK:
        entry = _REWRITER_METRICS.setdefault(
            key,
            {
                "request_count": 0.0,
                "error_count": 0.0,
                "latency_sec_sum": 0.0,
            },
        )
        entry["request_count"] += 1.0
        entry["latency_sec_sum"] += max(0.0, float(latency_sec))
        if not ok:
            entry["error_count"] += 1.0


def _rewriter_metrics_snapshot() -> Dict[str, Dict[str, float]]:
    with _REWRITER_METRICS_LOCK:
        raw = {
            url: {
                "request_count": float(values.get("request_count", 0.0)),
                "error_count": float(values.get("error_count", 0.0)),
                "latency_sec_sum": float(values.get("latency_sec_sum", 0.0)),
            }
            for url, values in _REWRITER_METRICS.items()
        }

    total_requests = sum(v["request_count"] for v in raw.values())
    total_latency_sec = sum(v["latency_sec_sum"] for v in raw.values())

    snapshot: Dict[str, Dict[str, float]] = {}
    for url, values in raw.items():
        count = values["request_count"]
        latency_sum = values["latency_sec_sum"]
        errors = values["error_count"]
        snapshot[url] = {
            "request_count": int(count),
            "error_count": int(errors),
            "avg_latency_ms": (latency_sum / count * 1000.0) if count > 0 else 0.0,
            "request_share": (count / total_requests) if total_requests > 0 else 0.0,
            "latency_share": (latency_sum / total_latency_sec) if total_latency_sec > 0 else 0.0,
        }
    return snapshot


def _format_rewriter_split_for_log(snapshot: Dict[str, Dict[str, float]]) -> str:
    if not snapshot:
        return "none"
    parts: List[str] = []
    for url in sorted(snapshot.keys()):
        values = snapshot[url]
        parts.append(
            f"{url}:count={int(values.get('request_count', 0))}"
            f",req_share={float(values.get('request_share', 0.0)):.2%}"
            f",lat_share={float(values.get('latency_share', 0.0)):.2%}"
            f",avg_ms={float(values.get('avg_latency_ms', 0.0)):.1f}"
            f",err={int(values.get('error_count', 0))}"
        )
    return "; ".join(parts)


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in str(text).lower().split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if len(cleaned) >= 4:
            tokens.add(cleaned)
    return tokens


def _rewrite_candidate_viable(args: argparse.Namespace, note_text: str, rewrite: Dict) -> bool:
    rewritten = str(rewrite.get("rewritten_prompt", "")).strip()
    if not rewritten:
        return False
    if len(rewritten) < max(1, int(args.train_candidate_min_len)):
        return False
    if len(rewritten) > max(1, int(args.train_candidate_max_len)):
        return False

    source = str(rewrite.get("generation_source", "")).lower()
    if "model_load_error" in source:
        return False

    lowered = rewritten.lower()
    if not any(k in lowered for k in ("icd", "code", "json", "diagnosis")):
        return False

    note_tokens = _tokenize_for_overlap(note_text)
    if note_tokens:
        rewrite_tokens = _tokenize_for_overlap(rewritten)
        overlap = len(note_tokens & rewrite_tokens) / max(1, len(note_tokens))
        if overlap > float(args.train_candidate_max_note_token_overlap):
            return False

    return True


def _rewrite_chunk(args: argparse.Namespace, chunk: List[PipelineItem]) -> None:
    session = _get_thread_session()
    chunk_t0 = now()
    rewriter_url = _next_rewriter_url(args)
    payload = [{"note_id": item.note_id, "clinical_note": item.note_text} for item in chunk]

    for item in chunk:
        _REWRITER_IO.log_input(
            note_id=item.note_id,
            split=item.split,
            prompt_hash=prompt_hash(item.note_text),
            clinical_note_len=len(item.note_text),
        )

    outputs: List[Dict]
    try:
        if len(payload) > 1:
            request_t0 = now()
            resp = session.post(f"{rewriter_url}/rewrite_prompt_batch", json=payload, timeout=args.request_timeout)
            _record_rewriter_request(rewriter_url, now() - request_t0, ok=(resp.status_code < 400))
            resp.raise_for_status()
            outputs = list(resp.json())
        else:
            request_t0 = now()
            resp = session.post(f"{rewriter_url}/rewrite_prompt", json=payload[0], timeout=args.request_timeout)
            _record_rewriter_request(rewriter_url, now() - request_t0, ok=(resp.status_code < 400))
            resp.raise_for_status()
            outputs = [resp.json()]
    except Exception:
        _record_rewriter_request(rewriter_url, 0.0, ok=False)
        outputs = []
        for p in payload:
            failover_url = _next_rewriter_url(args)
            request_t0 = now()
            r = session.post(f"{failover_url}/rewrite_prompt", json=p, timeout=args.request_timeout)
            _record_rewriter_request(failover_url, now() - request_t0, ok=(r.status_code < 400))
            r.raise_for_status()
            outputs.append(r.json())

    for item, out in zip(chunk, outputs):
        item.rwj = out

    chunk_elapsed_ms = (now() - chunk_t0) * 1000.0
    per_item_elapsed_ms = chunk_elapsed_ms / max(1, len(chunk))
    for item in chunk:
        rewritten_prompt = str((item.rwj or {}).get("rewritten_prompt", ""))
        _REWRITER_IO.log_output(
            note_id=item.note_id,
            split=item.split,
            rewritten_hash=prompt_hash(rewritten_prompt) if rewritten_prompt else "",
            generation_source=(item.rwj or {}).get("generation_source"),
            rejection_reason=(item.rwj or {}).get("rejection_reason"),
            elapsed_ms=f"{per_item_elapsed_ms:.2f}",
        )


def run_rewriter_stage(args: argparse.Namespace, items: List[PipelineItem], timings: TimingStats) -> List[PipelineItem]:
    chunks = _chunks(items, args.rewriter_batch_size)
    with ThreadPoolExecutor(max_workers=max(1, args.rewriter_workers)) as pool:
        futures = []
        for chunk in chunks:
            t0 = now()
            fut = pool.submit(_rewrite_chunk, args, chunk)
            futures.append((fut, chunk, t0))

        for fut, chunk, t0 in futures:
            fut.result()
            elapsed = now() - t0
            per_item = elapsed / max(1, len(chunk))
            for item in chunk:
                out = item.rwj or {}
                rewritten_prompt = str(out.get("rewritten_prompt", ""))
                item.note_record["rewriter"] = {
                    "request_meta": {
                        "note_id": item.note_id,
                        "clinical_note_sha1": item.note_record["inputs"]["clinical_note"]["sha1"],
                        "clinical_note_len": item.note_record["inputs"]["clinical_note"]["len"],
                    },
                    "response_meta": {
                        "rewritten_prompt_sha1": sha1_text(rewritten_prompt),
                        "rewritten_prompt_len": len(rewritten_prompt),
                        "generation_source": out.get("generation_source"),
                        "rejection_reason": out.get("rejection_reason"),
                        "rejection_reason_counts": out.get("rejection_reason_counts", {}),
                        "log_prob_old": out.get("log_prob_old"),
                        "value_estimate": out.get("value_estimate"),
                    },
                    "timing_sec": per_item,
                }
                timings.rewrite_sec.append(per_item)

    return items


def _icd_chunk(args: argparse.Namespace, run_id: str, chunk: List[PipelineItem]) -> None:
    session = _get_thread_session()
    chunk_t0 = now()
    payload = []
    for item in chunk:
        rewritten_prompt = str((item.rwj or {}).get("rewritten_prompt", ""))
        _ICD10_IO.log_input(
            note_id=item.note_id,
            run_id=run_id,
            split=item.split,
            group_id=item.group_id,
            prompt_hash=prompt_hash(item.note_text),
            rewritten_hash=prompt_hash(rewritten_prompt) if rewritten_prompt else "",
        )
        payload.append(
            {
                "note_id": item.note_id,
                "run_id": run_id,
                "group_id": item.group_id,
                "original_prompt": build_original_prompt(item.note_text),
                "rewritten_prompt": item.rwj["rewritten_prompt"],
                "generation_source": item.rwj.get("generation_source"),
                "log_prob_old": item.rwj.get("log_prob_old"),
                "value_estimate": item.rwj.get("value_estimate"),
                "skip_reward_forward": True,
            }
        )

    outputs: List[Dict]
    try:
        if len(payload) > 1:
            resp = session.post(f"{args.icd10_url}/generate_codes_batch", json=payload, timeout=args.request_timeout)
            resp.raise_for_status()
            outputs = list(resp.json())
        else:
            resp = session.post(f"{args.icd10_url}/generate_codes", json=payload[0], timeout=args.request_timeout)
            resp.raise_for_status()
            outputs = [resp.json()]
    except Exception:
        outputs = []
        for p in payload:
            r = session.post(f"{args.icd10_url}/generate_codes", json=p, timeout=args.request_timeout)
            r.raise_for_status()
            outputs.append(r.json())

    for item, out in zip(chunk, outputs):
        item.icj = out

    chunk_elapsed_ms = (now() - chunk_t0) * 1000.0
    per_item_elapsed_ms = chunk_elapsed_ms / max(1, len(chunk))
    for item in chunk:
        icj = item.icj or {}
        _ICD10_IO.log_output(
            note_id=item.note_id,
            run_id=run_id,
            split=item.split,
            parse_success=bool(icj.get("parsing_success", False)),
            enh_count=len(icj.get("enh_codes", []) or []),
            org_count=len(icj.get("org_codes", []) or []),
            elapsed_ms=f"{per_item_elapsed_ms:.2f}",
        )


def run_icd_stage(args: argparse.Namespace, run_id: str, items: List[PipelineItem], timings: TimingStats) -> List[PipelineItem]:
    chunks = _chunks(items, args.icd_batch_size)
    with ThreadPoolExecutor(max_workers=max(1, args.icd_workers)) as pool:
        futures = []
        for chunk in chunks:
            t1 = now()
            fut = pool.submit(_icd_chunk, args, run_id, chunk)
            futures.append((fut, chunk, t1))

        for fut, chunk, t1 in futures:
            fut.result()
            elapsed = now() - t1
            per_item = elapsed / max(1, len(chunk))
            for item in chunk:
                icj = item.icj or {}
                original_prompt = build_original_prompt(item.note_text)
                item.note_record["icd10"]["request_meta"] = {
                    "note_id": item.note_id,
                    "run_id": run_id,
                    "group_id": item.group_id,
                    "original_prompt_sha1": sha1_text(original_prompt),
                    "original_prompt_len": len(original_prompt),
                    "rewritten_prompt_sha1": item.note_record["rewriter"]["response_meta"]["rewritten_prompt_sha1"],
                    "rewritten_prompt_len": item.note_record["rewriter"]["response_meta"]["rewritten_prompt_len"],
                }
                item.note_record["icd10"]["timing_sec"] = per_item
                item.note_record["icd10"]["response"] = icj
                timings.icd10_sec.append(per_item)

    return items


def _reward_item(args: argparse.Namespace, item: PipelineItem) -> Tuple[PipelineItem, float]:
    session = _get_thread_session()
    icj = item.icj or {}
    _REWARD_IO.log_input(
        note_id=item.note_id,
        split=item.split,
        gt_count=len(item.gt_codes or []),
        enh_count=len(icj.get("enh_codes", []) or []),
        org_count=len(icj.get("org_codes", []) or []),
    )
    t2 = now()
    reward_request = {
        "note_id": item.note_id,
        "gt_codes": item.gt_codes,
        "enh_codes": icj.get("enh_codes", []),
        "org_codes": icj.get("org_codes", []),
        "parsing_success": icj.get("parsing_success", True),
    }
    rr = session.post(f"{args.reward_url}/compute_reward", json=reward_request, timeout=60)
    rr.raise_for_status()
    reward_payload = rr.json()
    rv = float(reward_payload["reward"])
    reward_sec = now() - t2
    item.note_record["reward"] = {
        "called": True,
        "request": reward_request,
        "response": reward_payload,
        "timing_sec": reward_sec,
    }
    _REWARD_IO.log_output(
        note_id=item.note_id,
        split=item.split,
        reward=f"{rv:.6f}",
        elapsed_ms=f"{reward_sec * 1000.0:.2f}",
    )
    return item, rv


def _generate_train_rewrites(
    args: argparse.Namespace,
    item: PipelineItem,
    seed_rewrite: Dict,
) -> List[Dict]:
    session = _get_thread_session()
    rewrites: List[Dict] = []
    seen_actions: set[str] = set()

    def _maybe_add(entry: Dict) -> None:
        rewritten = str(entry.get("rewritten_prompt", "")).strip()
        if not rewritten or rewritten in seen_actions:
            return
        if not _rewrite_candidate_viable(args, item.note_text, entry):
            return
        seen_actions.add(rewritten)
        rewrites.append(entry)

    _maybe_add(seed_rewrite)

    if args.fast_train_mode:
        target = max(int(args.fast_rewrites_per_note), 1)
    else:
        target = max(int(args.rewrites_per_note), 1)
    max_attempts = max(target * 4, 8)
    attempts = 0
    while len(rewrites) < target and attempts < max_attempts:
        attempts += 1
        rewriter_url = _next_rewriter_url(args)
        payload = {
            "note_id": item.note_id,
            "clinical_note": item.note_text,
            "sampling_nonce": attempts,
            "disable_best_prompt_cache": True,
        }
        t0 = now()
        resp = session.post(
            f"{rewriter_url}/rewrite_prompt",
            json=payload,
            timeout=args.request_timeout,
        )
        _record_rewriter_request(rewriter_url, now() - t0, ok=(resp.status_code < 400))
        resp.raise_for_status()
        _maybe_add(resp.json())

    return rewrites[:target]


def _run_icd_for_train_candidate(
    args: argparse.Namespace,
    *,
    run_id: str,
    note_id: str,
    group_id: str,
    note_text: str,
    rewrite: Dict,
) -> Dict:
    session = _get_thread_session()
    payload = {
        "note_id": note_id,
        "run_id": run_id,
        "group_id": group_id,
        "original_prompt": build_original_prompt(note_text),
        "rewritten_prompt": rewrite.get("rewritten_prompt", ""),
        "generation_source": rewrite.get("generation_source"),
        "log_prob_old": rewrite.get("log_prob_old"),
        "value_estimate": rewrite.get("value_estimate"),
        "skip_reward_forward": True,
    }
    resp = session.post(
        f"{args.icd10_url}/generate_codes",
        json=payload,
        timeout=args.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _compute_reward_for_train_candidate(
    args: argparse.Namespace,
    *,
    note_id: str,
    gt_codes: List[str],
    original_prompt: str,
    rewritten_prompt: str,
    log_prob_old: float,
    value_estimate: float,
    icd_payload: Dict,
) -> float:
    session = _get_thread_session()
    reward_request = {
        "note_id": note_id,
        "gt_codes": gt_codes,
        "enh_codes": icd_payload.get("enh_codes", []),
        "org_codes": icd_payload.get("org_codes", []),
        "parsing_success": bool(icd_payload.get("parsing_success", True)),
        "state": original_prompt,
        "action": rewritten_prompt,
        "log_prob_old": float(log_prob_old),
        "value_estimate": float(value_estimate),
    }
    resp = session.post(
        f"{args.reward_url}/compute_reward",
        json=reward_request,
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return float(payload.get("reward", 0.0))


def submit_train_rollout_group(
    args: argparse.Namespace,
    run_id: str,
    item: PipelineItem,
    *,
    submit_to_store: bool = True,
    rollout_buffer: Optional[List[Dict]] = None,
) -> Dict:
    original_prompt = build_original_prompt(item.note_text)
    note_group_id = item.note_id
    seed_rewrite = item.rwj or {}

    rewrites = _generate_train_rewrites(args, item, seed_rewrite)
    unique_action_count = len({str(r.get("rewritten_prompt", "")).strip() for r in rewrites if str(r.get("rewritten_prompt", "")).strip()})

    dropped_rollouts = 0
    rollouts: List[Dict] = []
    for rewrite in rewrites:
        rewritten_prompt = str(rewrite.get("rewritten_prompt", "")).strip()
        if not rewritten_prompt:
            dropped_rollouts += 1
            continue

        try:
            log_prob_old = float(rewrite.get("log_prob_old", 0.0))
        except Exception:
            dropped_rollouts += 1
            continue
        if (not math.isfinite(log_prob_old)) or abs(log_prob_old) <= 1e-12:
            dropped_rollouts += 1
            continue

        value_estimate_raw = rewrite.get("value_estimate")
        if value_estimate_raw is None:
            dropped_rollouts += 1
            continue
        try:
            value_estimate = float(value_estimate_raw)
        except Exception:
            dropped_rollouts += 1
            continue

        try:
            icd_payload = _run_icd_for_train_candidate(
                args,
                run_id=run_id,
                note_id=item.note_id,
                group_id=note_group_id,
                note_text=item.note_text,
                rewrite=rewrite,
            )
            reward = _compute_reward_for_train_candidate(
                args,
                note_id=item.note_id,
                gt_codes=item.gt_codes,
                original_prompt=original_prompt,
                rewritten_prompt=rewritten_prompt,
                log_prob_old=log_prob_old,
                value_estimate=value_estimate,
                icd_payload=icd_payload,
            )
        except Exception:
            dropped_rollouts += 1
            continue

        if not math.isfinite(reward):
            dropped_rollouts += 1
            continue

        rollouts.append(
            {
                "run_id": run_id,
                "group_id": note_group_id,
                "original_prompt": original_prompt,
                "rewritten_prompt": rewritten_prompt,
                "reward": float(reward),
                "log_prob_old": log_prob_old,
                "value_estimate": value_estimate,
            }
        )

    # Hard group-level validation
    state_set = {r["original_prompt"] for r in rollouts}
    note_group_set = {r["group_id"] for r in rollouts}
    unique_actions_after_filter = {r["rewritten_prompt"] for r in rollouts}
    group_valid = (
        len(state_set) == 1
        and len(note_group_set) == 1
        and len(unique_actions_after_filter) >= 2
        and len(rollouts) >= 2
    )

    ack_payload: Dict = {
        "accepted": False,
        "accepted_count": 0,
        "duplicate_count": 0,
        "file_path": "",
        "run_id": run_id,
    }

    if group_valid:
        if submit_to_store:
            session = _get_thread_session()
            submit_payload = {
                "run_id": run_id,
                "rollouts": rollouts,
            }
            resp = session.post(
                f"{args.rl_url}/rollout_batch",
                json=submit_payload,
                timeout=60,
            )
            resp.raise_for_status()
            ack_payload = resp.json()
        elif rollout_buffer is not None:
            rollout_buffer.extend(rollouts)
            ack_payload = {
                "accepted": True,
                "accepted_count": len(rollouts),
                "duplicate_count": 0,
                "file_path": "buffered",
                "run_id": run_id,
            }

    diag = {
        "note_id": item.note_id,
        "group_id": note_group_id,
        "rollout_count": len(rollouts),
        "unique_action_count": len(unique_actions_after_filter),
        "dropped_rollouts_count": dropped_rollouts,
        "group_validity": bool(group_valid),
        "accepted_count": int(ack_payload.get("accepted_count", 0)),
    }
    log_line(
        "train_group_diag "
        f"note_id={diag['note_id']} group_id={diag['group_id']} "
        f"rollout_count={diag['rollout_count']} unique_action_count={diag['unique_action_count']} "
        f"dropped_rollouts_count={diag['dropped_rollouts_count']} group_validity={diag['group_validity']}"
    )
    _RL_IO.log_output(**diag)
    return diag


def flush_rollout_buffer(args: argparse.Namespace, run_id: str, rollout_buffer: List[Dict]) -> Dict[str, Any]:
    if not rollout_buffer:
        return {"accepted": False, "accepted_count": 0, "duplicate_count": 0, "run_id": run_id}

    session = _get_thread_session()
    submit_payload = {
        "run_id": run_id,
        "rollouts": list(rollout_buffer),
    }
    resp = session.post(
        f"{args.rl_url}/rollout_batch",
        json=submit_payload,
        timeout=60,
    )
    resp.raise_for_status()
    ack_payload = resp.json()
    buffered_count = len(rollout_buffer)
    rollout_buffer.clear()
    log_line(
        "trajectory_buffer_flushed "
        f"requested={buffered_count} accepted={ack_payload.get('accepted_count', 0)} "
        f"duplicate={ack_payload.get('duplicate_count', 0)}"
    )
    return ack_payload


def emit_pipeline_batch_logs(
    completed_items: List[PipelineItem],
    *,
    pipeline_run_id: Optional[str],
    mode1: Optional[str],
    mode2: Optional[str],
    batch_num: int,
) -> None:
    if not completed_items or not pipeline_run_id or not mode1 or not mode2:
        return

    csv_rows: List[Dict] = []
    summary_entries: List[Dict] = []
    for item in completed_items:
        rewritten_prompt = str((item.rwj or {}).get("rewritten_prompt", ""))
        icj = item.icj or {}
        reward_value = 0.0
        reward_payload = item.note_record.get("reward", {}).get("response")
        if isinstance(reward_payload, dict) and "reward" in reward_payload:
            try:
                reward_value = float(reward_payload.get("reward", 0.0))
            except Exception:
                reward_value = 0.0

        row = {
            "ts": utc_now_iso(),
            "run_id": pipeline_run_id,
            "mode1": mode1,
            "mode2": mode2,
            "batch_num": batch_num,
            "iter_num": 1,
            "batch_size": len(completed_items),
            "max_iters": 1,
            "rollout_id": str(icj.get("rollout_id", "")),
            "prompt_hash": prompt_hash(item.note_text),
            "rewritten_hash": prompt_hash(rewritten_prompt) if rewritten_prompt else "",
            "og_codes": list(icj.get("org_codes", []) or []),
            "enh_codes": list(icj.get("enh_codes", []) or []),
            "gt_codes": list(item.gt_codes or []),
            "reward": reward_value,
        }
        csv_rows.append(row)
        summary_entries.append(
            {
                "prompt_hash": row["prompt_hash"],
                "rewritten_hash": row["rewritten_hash"],
                "og_codes": row["og_codes"],
                "enh_codes": row["enh_codes"],
                "gt_codes": row["gt_codes"],
                "reward": reward_value,
            }
        )

    write_csv_rows(csv_rows)
    write_batch_summary(
        run_id=pipeline_run_id,
        mode1=mode1,
        mode2=mode2,
        batch_num=batch_num,
        iter_num=1,
        max_iters=1,
        batch_size=len(completed_items),
        entries=summary_entries,
    )


def build_runtime_snapshot(
    *,
    run_id: str,
    total_notes: int,
    stats: RunStats,
    timings: TimingStats,
    start_wall: float,
    deadline_wall: Optional[float],
    pending_train_triggers: int,
    train_future_active: bool,
    checkpoint_payload: Optional[Dict],
    run_guard_failures: Optional[List[str]] = None,
) -> Dict:
    elapsed_total = time.time() - start_wall
    throughput_notes_per_sec = (stats.processed / elapsed_total) if elapsed_total > 0 else 0.0
    remaining_sec = max(0.0, (deadline_wall - time.time())) if deadline_wall is not None else None

    snapshot: Dict = {
        "run_id": run_id,
        "total_processed": stats.processed,
        "total_target": total_notes,
        "failures": stats.failures,
        "train_success": stats.train_success,
        "val_success": stats.val_success,
        "test_success": stats.test_success,
        "parse_success_rate": (stats.parse_success_count / stats.processed) if stats.processed else 0.0,
        "both_parse_success_rate": (stats.both_parse_success_count / stats.processed) if stats.processed else 0.0,
        "original_parse_failure_rate": (stats.original_parse_failure_count / stats.processed) if stats.processed else 0.0,
        "train_cycle_attempts": stats.train_cycle_attempts,
        "train_cycle_success_count": stats.train_cycle_success_count,
        "train_group_valid_count": stats.train_group_valid_count,
        "train_group_invalid_count": stats.train_group_invalid_count,
        "train_rollouts_dropped_count": stats.train_rollouts_dropped_count,
        "rewrite_sec_median": median_or_zero(timings.rewrite_sec),
        "rewrite_sec_p95": percentile_or_zero(timings.rewrite_sec, 0.95),
        "icd10_sec_median": median_or_zero(timings.icd10_sec),
        "icd10_sec_p95": percentile_or_zero(timings.icd10_sec, 0.95),
        "reward_sec_median": median_or_zero(timings.reward_sec),
        "reward_sec_p95": percentile_or_zero(timings.reward_sec, 0.95),
        "train_cycle_sec_median": median_or_zero(timings.train_cycle_sec),
        "note_total_sec_median": median_or_zero(timings.note_total_sec),
        "note_total_sec_p95": percentile_or_zero(timings.note_total_sec, 0.95),
        "throughput_notes_per_sec": throughput_notes_per_sec,
        "elapsed_hr": elapsed_total / 3600.0,
        "pending_train_triggers": int(pending_train_triggers),
        "train_future_active": bool(train_future_active),
        "timeboxed": deadline_wall is not None,
        "deadline_remaining_sec": remaining_sec,
        "checkpoint": checkpoint_payload or {},
        "rewriter_request_balance": _rewriter_metrics_snapshot(),
        "snapshot_utc": utc_now_iso(),
    }
    if run_guard_failures is not None:
        snapshot["run_guard_failures"] = list(run_guard_failures)
    return snapshot


def finalize_stage(
    args: argparse.Namespace,
    run_id: str,
    items: List[PipelineItem],
    stats: RunStats,
    timings: TimingStats,
    val_rewards: List[float],
    test_rewards: List[float],
    val_micro: Dict[str, int],
    test_micro: Dict[str, int],
    val_macro: Dict[str, float],
    test_macro: Dict[str, float],
    rollout_buffer: Optional[List[Dict]] = None,
) -> List[PipelineItem]:
    val_test_items: List[PipelineItem] = []
    train_items: List[PipelineItem] = []
    for item in items:
        icj = item.icj or {}
        enh_parse_ok = bool(icj.get("enh_parse_ok", bool(icj.get("enh_codes", []))))
        org_parse_ok = bool(icj.get("org_parse_ok", bool(icj.get("org_codes", []))))
        both_parse_success = bool(icj.get("both_parse_success", bool(enh_parse_ok and org_parse_ok)))

        stats.parse_success_count += 1 if icj.get("parsing_success") else 0
        stats.both_parse_success_count += 1 if both_parse_success else 0
        stats.original_parse_failure_count += 0 if org_parse_ok else 1

        if item.split == "train":
            stats.train_seen += 1
            train_items.append(item)
            if not both_parse_success:
                stats.train_comparability_failure_count += 1
        else:
            val_test_items.append(item)

    if train_items:
        submit_direct = rollout_buffer is None
        with ThreadPoolExecutor(max_workers=max(1, args.train_workers)) as pool:
            futures = [
                pool.submit(
                    submit_train_rollout_group,
                    args,
                    run_id,
                    item,
                    submit_to_store=submit_direct,
                    rollout_buffer=rollout_buffer,
                )
                for item in train_items
            ]
            for item, fut in zip(train_items, futures):
                group_diag = fut.result()
                item.note_record["train_group"] = group_diag
                stats.train_rollouts_dropped_count += int(
                    group_diag.get("dropped_rollouts_count", 0)
                )
                if bool(group_diag.get("group_validity", False)):
                    stats.train_group_valid_count += 1
                else:
                    stats.train_group_invalid_count += 1
                if int(group_diag.get("accepted_count", 0)) >= 2:
                    stats.train_success += 1

        if rollout_buffer is not None:
            flush_size = max(1, int(args.trajectory_flush_size))
            if len(rollout_buffer) >= flush_size:
                flush_rollout_buffer(args, run_id, rollout_buffer)

    with ThreadPoolExecutor(max_workers=max(1, args.reward_workers)) as pool:
        futures = [pool.submit(_reward_item, args, item) for item in val_test_items]
        for fut in futures:
            item, rv = fut.result()
            reward_sec = float(item.note_record["reward"]["timing_sec"])
            timings.reward_sec.append(reward_sec)
            gt_set = canonicalized_code_set(item.gt_codes)
            pred_set = canonicalized_code_set((item.icj or {}).get("enh_codes", []))
            if item.split == "val":
                stats.val_success += 1
                val_rewards.append(rv)
                update_micro_counts(gt_set, pred_set, val_micro)
                update_macro_counts(gt_set, pred_set, val_macro)
            else:
                stats.test_success += 1
                test_rewards.append(rv)
                update_micro_counts(gt_set, pred_set, test_micro)
                update_macro_counts(gt_set, pred_set, test_macro)

    for item in items:
        item.note_record["status"] = "ok"

    return items


def main() -> int:
    args = parse_args()

    run_id = args.run_id.strip() if args.run_id else ""
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")

    mode1_label: Optional[str] = None
    mode2_label: Optional[str] = None
    pipeline_logs_base: Optional[str] = None
    pipeline_logs_run_id: Optional[str] = None
    pipeline_logs_dir: Optional[str] = None
    try:
        grpo_enabled = _env_bool("RL_GRPO_ENABLED", True)
        has_value_estimates = not _env_bool("RL_DISABLE_VALUE_ESTIMATES", False)
        ppo_epochs = int(os.environ.get("RL_PPO_EPOCHS", "3"))
        mode1_label = resolve_mode1(
            grpo_enabled=grpo_enabled,
            has_value_estimates=has_value_estimates,
        )
        mode2_label = resolve_mode2(ppo_epochs=ppo_epochs)
        base, pipeline_logs_run_id = init_run(mode1_label, mode2_label)
        pipeline_logs_base = str(base)
        pipeline_logs_dir = str(base / pipeline_logs_run_id)
        log_line(
            "pipeline_logs_init "
            f"base={pipeline_logs_base} run_id={pipeline_logs_run_id} "
            f"mode1={mode1_label} mode2={mode2_label}"
        )
    except Exception as exc:
        log_line(f"warning pipeline_logs_init_failed err={exc}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    notes_path = results_dir / f"{run_id}.notes.jsonl"
    summary_path = results_dir / f"{run_id}.summary.json"
    checkpoints_dir = results_dir / f"{run_id}.checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = utc_now_iso()

    if args.train_ratio < 0 or args.val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative")
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    session = _build_pooled_session()

    ensure_services(session, args)
    if args.reset_reward_observability:
        reset_reward_observability(session, args.reward_url)

    initial_training_step = 0
    try:
        initial_training_step = int(get_rl_status(session, args.rl_url).get("training_step", 0))
    except Exception:
        initial_training_step = 0

    total_notes = get_dataset_total(session, args.dataset_url)
    if args.max_notes and args.max_notes > 0:
        total_notes = min(total_notes, args.max_notes)

    log_line(f"dataset_total_notes {total_notes}")

    split_map: Dict[str, str] = {}
    if args.split_strategy == "stratified":
        split_map = build_stratified_split_map(
            session=session,
            dataset_url=args.dataset_url,
            total_notes=total_notes,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
    else:
        log_line("split_strategy hash (streaming assignment)")

    timings = TimingStats(rewrite_sec=[], icd10_sec=[], reward_sec=[], train_cycle_sec=[], note_total_sec=[])
    stats = RunStats()
    train_cycle_failures: List[str] = []

    val_rewards: List[float] = []
    test_rewards: List[float] = []
    val_micro = {"tp": 0, "fp": 0, "fn": 0}
    test_micro = {"tp": 0, "fp": 0, "fn": 0}
    val_macro = {"precision_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0, "exact_match": 0.0, "count": 0.0}
    test_macro = {"precision_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0, "exact_match": 0.0, "count": 0.0}

    start_wall = time.time()
    deadline_wall: Optional[float] = None
    if float(args.max_run_minutes) > 0:
        deadline_wall = start_wall + (float(args.max_run_minutes) * 60.0)
        log_line(
            "timebox_enabled "
            f"max_run_minutes={float(args.max_run_minutes):.2f} "
            f"deadline_utc={datetime.fromtimestamp(deadline_wall, tz=timezone.utc).isoformat().replace('+00:00', 'Z')}"
        )

    checkpoint_interval_sec = max(0.0, float(args.checkpoint_every_minutes) * 60.0)
    next_checkpoint_wall: Optional[float] = None
    if checkpoint_interval_sec > 0:
        next_checkpoint_wall = start_wall + checkpoint_interval_sec
        log_line(
            "periodic_checkpoint_enabled "
            f"every_minutes={float(args.checkpoint_every_minutes):.2f}"
        )

    note_index = 0
    offset = 0
    train_sample_index = 0

    rewriter_executor = ThreadPoolExecutor(max_workers=1)
    train_executor = ThreadPoolExecutor(max_workers=1)
    pending_train_triggers = 0
    train_future = None
    completed_batch_counter = 0
    trajectory_rollout_buffer: List[Dict] = []

    def maybe_start_train() -> None:
        nonlocal train_future, pending_train_triggers
        if pending_train_triggers <= 0:
            return
        if train_future is not None and not train_future.done():
            return

        pending_train_triggers -= 1

        def _run_one_cycle() -> Tuple[bool, str, float]:
            local = _get_thread_session()
            _RL_IO.log_input(action="train_cycle_trigger")
            t0 = now()
            try:
                qst = flush_and_wait_reward_queue(local, args.reward_url)
                st = run_train_cycle(local, args.rl_url)
                _RL_IO.log_output(
                    action="train_cycle_trigger",
                    status="ok",
                    training_step=st.get("training_step"),
                    rollouts_loaded=st.get("rollouts_loaded"),
                )
                return True, (
                    "train_cycle_done "
                    f"queue_pending={qst.get('pending_count')} "
                    f"training_step={st.get('training_step')} "
                    f"rollouts_loaded={st.get('rollouts_loaded')}"
                ), now() - t0
            except Exception as exc:
                _RL_IO.log_output(
                    action="train_cycle_trigger",
                    status="failed",
                    error=str(exc),
                )
                return False, f"train_cycle_failed err={exc}", now() - t0

        stats.train_cycle_attempts += 1
        train_future = train_executor.submit(_run_one_cycle)

    prev_rewriter_future = None
    prev_rewriter_items: List[PipelineItem] = []
    stop_requested = False

    def maybe_emit_periodic_checkpoint(reason: str) -> None:
        nonlocal next_checkpoint_wall
        if next_checkpoint_wall is None:
            return
        if time.time() < next_checkpoint_wall:
            return

        checkpoint_payload: Dict = {}
        try:
            ck = session.get(f"{args.rl_url}/checkpoint", timeout=30)
            ck.raise_for_status()
            checkpoint_payload = ck.json()
        except Exception as exc:
            checkpoint_payload = {"error": str(exc)}

        snapshot = build_runtime_snapshot(
            run_id=run_id,
            total_notes=total_notes,
            stats=stats,
            timings=timings,
            start_wall=start_wall,
            deadline_wall=deadline_wall,
            pending_train_triggers=pending_train_triggers,
            train_future_active=(train_future is not None and not train_future.done()),
            checkpoint_payload=checkpoint_payload,
        )
        snapshot["reason"] = reason
        snapshot_name = f"checkpoint_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        snapshot_path = checkpoints_dir / snapshot_name
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_line(
            "checkpoint_saved "
            f"reason={reason} path={snapshot_path} processed={stats.processed}/{total_notes}"
        )

        while next_checkpoint_wall is not None and time.time() >= next_checkpoint_wall:
            next_checkpoint_wall += checkpoint_interval_sec

    with open(notes_path, "a", encoding="utf-8") as notes_f:
        while offset < total_notes:
            if deadline_wall is not None and time.time() >= deadline_wall:
                stop_requested = True
                log_line(
                    "timebox_reached "
                    f"processed={stats.processed}/{total_notes} elapsed_min={(time.time() - start_wall)/60.0:.1f}"
                )
                break

            size = min(args.batch_size, total_notes - offset)
            br = session.get(f"{args.dataset_url}/batch", params={"offset": offset, "size": size}, timeout=60)
            br.raise_for_status()
            batch = br.json().get("batch", [])
            if not batch:
                break
            offset += size

            current_items: List[PipelineItem] = []
            for rec in batch:
                if note_index >= total_notes:
                    break
                note_id = str(rec["note_id"])
                note_text = rec["text"]
                gt_codes = rec.get("gt_codes", [])
                if args.split_strategy == "stratified":
                    split = split_map.get(note_id, "test")
                else:
                    split = split_for_note_id_hash(
                        note_id,
                        train_ratio=args.train_ratio,
                        val_ratio=args.val_ratio,
                        seed=args.split_seed,
                    )
                sample_index = None
                if split == "train":
                    sample_index = train_sample_index
                    train_sample_index += 1
                group_id = (
                    build_group_id(note_id, run_id, sample_index=sample_index, group_size=args.grpo_group_size)
                    if split == "train"
                    else build_group_id(note_id, run_id)
                )

                note_record: Dict = {
                    "run_id": run_id,
                    "ts_utc": utc_now_iso(),
                    "note_index": note_index,
                    "note_id": note_id,
                    "split": split,
                    "group_id": group_id,
                    "inputs": {
                        "clinical_note": {"sha1": sha1_text(note_text), "len": len(note_text)},
                        "gt_codes": list(gt_codes or []),
                    },
                    "rewriter": {"request_meta": {}, "response_meta": {}, "timing_sec": 0.0},
                    "icd10": {"request_meta": {}, "response": {}, "timing_sec": 0.0},
                    "reward": {"called": False, "request": None, "response": None, "timing_sec": 0.0},
                    "metrics": {},
                    "status": "unknown",
                    "error": None,
                    "timing": {"note_total_sec": 0.0},
                }
                current_items.append(
                    PipelineItem(
                        note_id=note_id,
                        note_text=note_text,
                        gt_codes=list(gt_codes or []),
                        split=split,
                        group_id=group_id,
                        note_index=note_index,
                        note_started=now(),
                        note_record=note_record,
                    )
                )
                note_index += 1

            if args.stream_note_wise:
                completed_items_stream: List[PipelineItem] = []
                for item in current_items:
                    try:
                        rewritten_items = run_rewriter_stage(args, [item], timings)
                        icd_items = run_icd_stage(args, run_id, rewritten_items, timings)
                        completed_items = finalize_stage(
                            args,
                            run_id,
                            icd_items,
                            stats,
                            timings,
                            val_rewards,
                            test_rewards,
                            val_micro,
                            test_micro,
                            val_macro,
                            test_macro,
                            rollout_buffer=trajectory_rollout_buffer,
                        )
                        for done_item in completed_items:
                            note_total_sec = now() - done_item.note_started
                            timings.note_total_sec.append(note_total_sec)
                            done_item.note_record["timing"]["note_total_sec"] = note_total_sec
                            notes_f.write(json.dumps(done_item.note_record, ensure_ascii=False) + "\n")
                            stats.processed += 1
                        completed_items_stream.extend(completed_items)
                    except Exception as exc:
                        stats.failures += 1
                        log_line(f"error stage_pipeline_stream err={exc}")

                if completed_items_stream:
                    completed_batch_counter += 1
                    emit_pipeline_batch_logs(
                        completed_items_stream,
                        pipeline_run_id=pipeline_logs_run_id,
                        mode1=mode1_label,
                        mode2=mode2_label,
                        batch_num=completed_batch_counter,
                    )

                if args.train_every > 0:
                    expected_triggers = stats.train_seen // args.train_every
                    while (
                        (stats.train_cycle_attempts + pending_train_triggers) < expected_triggers
                        and pending_train_triggers < max(1, args.max_pending_train_triggers)
                    ):
                        pending_train_triggers += 1
                    maybe_start_train()

                if train_future is not None and train_future.done():
                    ok, message, elapsed = train_future.result()
                    timings.train_cycle_sec.append(elapsed)
                    if ok:
                        stats.train_cycle_success_count += 1
                    else:
                        train_cycle_failures.append(message)
                    log_line(message)
                    train_future = None
                    maybe_start_train()

                progress_every = max(1, args.progress_every)
                if stats.processed % progress_every == 0 or stats.processed == total_notes:
                    elapsed = time.time() - start_wall
                    eta = estimate_remaining_seconds(
                        processed=stats.processed,
                        total=total_notes,
                        train_every=args.train_every,
                        timings=timings,
                        train_ratio=args.train_ratio,
                    )
                    log_line(
                        "progress "
                        f"processed={stats.processed}/{total_notes} "
                        f"elapsed_min={elapsed/60:.1f} "
                        f"eta_hr={eta/3600:.2f} "
                        f"failures={stats.failures}"
                    )
                    rewriter_snapshot = _rewriter_metrics_snapshot()
                    log_line(f"rewriter_balance {_format_rewriter_split_for_log(rewriter_snapshot)}")

                maybe_emit_periodic_checkpoint(reason="interval")
                continue

            current_rewriter_future = rewriter_executor.submit(run_rewriter_stage, args, current_items, timings)

            # Process previous rewriter output while next rewriter batch is running.
            if prev_rewriter_future is not None:
                try:
                    queue_rewriter_output = list(prev_rewriter_future.result())
                    queue_icd_output = run_icd_stage(args, run_id, queue_rewriter_output, timings)
                    completed_items = finalize_stage(
                        args,
                        run_id,
                        queue_icd_output,
                        stats,
                        timings,
                        val_rewards,
                        test_rewards,
                        val_micro,
                        test_micro,
                        val_macro,
                        test_macro,
                        rollout_buffer=None,
                    )
                    for item in completed_items:
                        note_total_sec = now() - item.note_started
                        timings.note_total_sec.append(note_total_sec)
                        item.note_record["timing"]["note_total_sec"] = note_total_sec
                        notes_f.write(json.dumps(item.note_record, ensure_ascii=False) + "\n")
                        stats.processed += 1

                    completed_batch_counter += 1
                    emit_pipeline_batch_logs(
                        completed_items,
                        pipeline_run_id=pipeline_logs_run_id,
                        mode1=mode1_label,
                        mode2=mode2_label,
                        batch_num=completed_batch_counter,
                    )

                    if args.train_every > 0:
                        expected_triggers = stats.train_seen // args.train_every
                        while (
                            (stats.train_cycle_attempts + pending_train_triggers) < expected_triggers
                            and pending_train_triggers < max(1, args.max_pending_train_triggers)
                        ):
                            pending_train_triggers += 1
                        maybe_start_train()

                    if train_future is not None and train_future.done():
                        ok, message, elapsed = train_future.result()
                        timings.train_cycle_sec.append(elapsed)
                        if ok:
                            stats.train_cycle_success_count += 1
                        else:
                            train_cycle_failures.append(message)
                        log_line(message)
                        train_future = None
                        maybe_start_train()

                except Exception as exc:
                    stats.failures += len(prev_rewriter_items) if prev_rewriter_items else 1
                    log_line(f"error stage_pipeline err={exc}")

                progress_every = max(1, args.progress_every)
                if stats.processed % progress_every == 0 or stats.processed == total_notes:
                    elapsed = time.time() - start_wall
                    eta = estimate_remaining_seconds(
                        processed=stats.processed,
                        total=total_notes,
                        train_every=args.train_every,
                        timings=timings,
                        train_ratio=args.train_ratio,
                    )
                    log_line(
                        "progress "
                        f"processed={stats.processed}/{total_notes} "
                        f"elapsed_min={elapsed/60:.1f} "
                        f"eta_hr={eta/3600:.2f} "
                        f"failures={stats.failures}"
                    )
                    rewriter_snapshot = _rewriter_metrics_snapshot()
                    log_line(f"rewriter_balance {_format_rewriter_split_for_log(rewriter_snapshot)}")

                maybe_emit_periodic_checkpoint(reason="interval")

            prev_rewriter_future = current_rewriter_future
            prev_rewriter_items = current_items

        # Drain final buffered batch.
        if prev_rewriter_future is not None:
            try:
                queue_rewriter_output = list(prev_rewriter_future.result())
                queue_icd_output = run_icd_stage(args, run_id, queue_rewriter_output, timings)
                completed_items = finalize_stage(
                    args,
                    run_id,
                    queue_icd_output,
                    stats,
                    timings,
                    val_rewards,
                    test_rewards,
                    val_micro,
                    test_micro,
                    val_macro,
                    test_macro,
                    rollout_buffer=None,
                )
                for item in completed_items:
                    note_total_sec = now() - item.note_started
                    timings.note_total_sec.append(note_total_sec)
                    item.note_record["timing"]["note_total_sec"] = note_total_sec
                    notes_f.write(json.dumps(item.note_record, ensure_ascii=False) + "\n")
                    stats.processed += 1

                completed_batch_counter += 1
                emit_pipeline_batch_logs(
                    completed_items,
                    pipeline_run_id=pipeline_logs_run_id,
                    mode1=mode1_label,
                    mode2=mode2_label,
                    batch_num=completed_batch_counter,
                )

                if args.train_every > 0:
                    expected_triggers = stats.train_seen // args.train_every
                    while (
                        (stats.train_cycle_attempts + pending_train_triggers) < expected_triggers
                        and pending_train_triggers < max(1, args.max_pending_train_triggers)
                    ):
                        pending_train_triggers += 1
                    maybe_start_train()

                maybe_emit_periodic_checkpoint(reason="post_drain")

            except Exception as exc:
                stats.failures += len(prev_rewriter_items) if prev_rewriter_items else 1
                log_line(f"error final_pipeline err={exc}")

        # Final train trigger for remaining train rollouts.
        if stats.train_seen > 0 and args.train_every > 0 and (stats.train_seen % args.train_every) != 0:
            pending_train_triggers += 1
            maybe_start_train()

        while (pending_train_triggers > 0) or (train_future is not None):
            if train_future is None:
                maybe_start_train()
                time.sleep(0.2)
                continue
            if train_future.done():
                ok, message, elapsed = train_future.result()
                timings.train_cycle_sec.append(elapsed)
                if ok:
                    stats.train_cycle_success_count += 1
                else:
                    train_cycle_failures.append(message)
                log_line(message)
                train_future = None
                maybe_start_train()
            else:
                time.sleep(0.5)

        maybe_emit_periodic_checkpoint(reason="final_before_summary")

    if trajectory_rollout_buffer:
        try:
            flush_rollout_buffer(args, run_id, trajectory_rollout_buffer)
        except Exception as exc:
            log_line(f"error trajectory_buffer_flush_final err={exc}")

    rewriter_executor.shutdown(wait=True)
    train_executor.shutdown(wait=True)

    ck = session.get(f"{args.rl_url}/checkpoint", timeout=30)
    ck.raise_for_status()

    final_training_step = initial_training_step
    try:
        final_training_step = int(get_rl_status(session, args.rl_url).get("training_step", initial_training_step))
    except Exception:
        final_training_step = initial_training_step

    reward_observability: Dict = {}
    icd_observability: Dict = {}
    try:
        obs_reward = session.get(f"{args.reward_url}/observability", timeout=30)
        obs_reward.raise_for_status()
        reward_observability = obs_reward.json()
    except Exception as exc:
        log_line(f"warning reward_observability_unavailable err={exc}")

    try:
        obs_icd = session.get(f"{args.icd10_url}/observability", timeout=30)
        obs_icd.raise_for_status()
        icd_observability = obs_icd.json()
    except Exception as exc:
        log_line(f"warning icd_observability_unavailable err={exc}")

    val_precision, val_recall, val_f1 = micro_prf(val_micro)
    test_precision, test_recall, test_f1 = micro_prf(test_micro)
    val_macro_precision, val_macro_recall, val_macro_f1, val_exact_match = macro_prf_exact(val_macro)
    test_macro_precision, test_macro_recall, test_macro_f1, test_exact_match = macro_prf_exact(test_macro)

    elapsed_total = time.time() - start_wall
    throughput_notes_per_sec = (stats.processed / elapsed_total) if elapsed_total > 0 else 0.0

    summary = build_runtime_snapshot(
        run_id=run_id,
        total_notes=total_notes,
        stats=stats,
        timings=timings,
        start_wall=start_wall,
        deadline_wall=deadline_wall,
        pending_train_triggers=pending_train_triggers,
        train_future_active=False,
        checkpoint_payload=ck.json(),
    )
    summary.update(
        {
            "train_comparability_failure_count": stats.train_comparability_failure_count,
            "train_cycle_failure_count": len(train_cycle_failures),
            "train_cycle_failures": train_cycle_failures,
            "val_reward_mean": statistics.mean(val_rewards) if val_rewards else 0.0,
            "test_reward_mean": statistics.mean(test_rewards) if test_rewards else 0.0,
            "val_micro_precision": val_precision,
            "val_micro_recall": val_recall,
            "val_micro_f1": val_f1,
            "val_macro_precision": val_macro_precision,
            "val_macro_recall": val_macro_recall,
            "val_macro_f1": val_macro_f1,
            "val_exact_match_rate": val_exact_match,
            "test_micro_precision": test_precision,
            "test_micro_recall": test_recall,
            "test_micro_f1": test_f1,
            "test_macro_precision": test_macro_precision,
            "test_macro_recall": test_macro_recall,
            "test_macro_f1": test_macro_f1,
            "test_exact_match_rate": test_exact_match,
            "rewrite_sec_p99": percentile_or_zero(timings.rewrite_sec, 0.99),
            "icd10_sec_p99": percentile_or_zero(timings.icd10_sec, 0.99),
            "reward_sec_p99": percentile_or_zero(timings.reward_sec, 0.99),
            "train_cycle_sec_p95": percentile_or_zero(timings.train_cycle_sec, 0.95),
            "train_cycle_sec_p99": percentile_or_zero(timings.train_cycle_sec, 0.99),
            "note_total_sec_p99": percentile_or_zero(timings.note_total_sec, 0.99),
            "rollout_drop_rate": float(reward_observability.get("rollout_drop_rate", 0.0)),
            "queue_lag_seconds": float(reward_observability.get("queue_lag_seconds", 0.0)),
            "parse_failure_taxonomy": icd_observability.get("parse_failure_taxonomy", {}),
            "reward_observability": reward_observability,
            "icd_observability": icd_observability,
            "sbmi_status": "success" if stats.train_cycle_success_count > 0 else "failed",
            "batches_processed": completed_batch_counter,
            "training_step_delta": int(final_training_step - initial_training_step),
            "stopped_due_to_timebox": bool(stop_requested),
        }
    )

    run_guard_failures: List[str] = []
    if float(summary.get("rollout_drop_rate", 0.0)) > args.guard_max_rollout_drop_rate:
        run_guard_failures.append(
            "rollout_drop_rate_exceeded"
            f"({summary.get('rollout_drop_rate'):.4f}>{args.guard_max_rollout_drop_rate:.4f})"
        )
    if float(summary.get("both_parse_success_rate", 0.0)) < args.guard_min_both_parse_rate:
        run_guard_failures.append(
            "both_parse_success_rate_below_threshold"
            f"({summary.get('both_parse_success_rate'):.4f}<{args.guard_min_both_parse_rate:.4f})"
        )
    if float(summary.get("original_parse_failure_rate", 0.0)) > args.guard_max_original_failure_rate:
        run_guard_failures.append(
            "original_parse_failure_rate_exceeded"
            f"({summary.get('original_parse_failure_rate'):.4f}>{args.guard_max_original_failure_rate:.4f})"
        )

    if bool(reward_observability.get("transport_degraded", False)):
        run_guard_failures.append("reward_transport_degraded")

    if train_cycle_failures:
        run_guard_failures.append(f"train_cycle_failures_detected(count={len(train_cycle_failures)})")

    summary["run_guard_failures"] = run_guard_failures
    summary["started_at_utc"] = started_at_utc
    summary["finished_at_utc"] = utc_now_iso()
    summary["artifacts"] = {
        "results_dir": str(results_dir),
        "notes_jsonl": str(notes_path),
        "summary_json": str(summary_path),
    }

    log_line(f"rewriter_balance_final {_format_rewriter_split_for_log(_rewriter_metrics_snapshot())}")
    if pipeline_logs_base and pipeline_logs_run_id and pipeline_logs_dir:
        summary["artifacts"]["pipeline_logs"] = {
            "base_dir": pipeline_logs_base,
            "run_id": pipeline_logs_run_id,
            "run_dir": pipeline_logs_dir,
        }

    log_line(f"summary {summary}")
    with open(summary_path, "w", encoding="utf-8") as summary_f:
        summary_f.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    if run_guard_failures:
        log_line(f"run_guard_failed count={len(run_guard_failures)} failures={run_guard_failures}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
