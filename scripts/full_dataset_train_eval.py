#!/usr/bin/env python3
"""Full-dataset train/test/validate runner for the prompt optimisation pipeline.

Flow per note:
1. Fetch note text from dataset_svc (/batch).
2. Generate rewritten prompt with rewriter_inference_svc (/rewrite_prompt).
3. Generate ICD10 codes with icd10_coding_svc (/generate_codes).
4. Train split: enqueue rollout payloads in reward queue and flush/ack before RL train.
5. Val/Test split: call reward_metrics_svc (/reward) without rollout fields
   so reward is computed but not forwarded to RL (no training leakage).

The script periodically triggers rl_loop_svc (/train) and waits for cycle completion.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import requests


@dataclass
class TimingStats:
    rewrite_sec: List[float]
    icd10_sec: List[float]
    reward_sec: List[float]
    train_cycle_sec: List[float]
    note_total_sec: List[float]


def now() -> float:
    return time.perf_counter()


def log_line(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full dataset train/test/validate pipeline")
    p.add_argument("--dataset-url", default="http://localhost:8003")
    p.add_argument("--rewriter-url", default="http://localhost:8000")
    p.add_argument("--icd10-url", default="http://localhost:8001")
    p.add_argument("--reward-url", default="http://localhost:8002")
    p.add_argument("--rl-url", default="http://localhost:8004")
    p.add_argument("--run-id", default="", help="Optional run identifier for rollout isolation")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--grpo-group-size", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--request-timeout", type=int, default=1800)
    p.add_argument("--max-notes", type=int, default=0, help="0 means full dataset")
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
    def wait_ready(name: str, url: str, attempts: int = 120, sleep_sec: int = 2) -> None:
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

    checks = [
        ("dataset", f"{args.dataset_url}/health"),
        ("rewriter", f"{args.rewriter_url}/health"),
        ("icd10", f"{args.icd10_url}/health"),
        ("reward", f"{args.reward_url}/health"),
        ("rl", f"{args.rl_url}/status"),
    ]
    for name, url in checks:
        wait_ready(name, url)


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


def split_for_note_id(note_id: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{note_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / float(2 ** 64)

    train_end = train_ratio
    val_end = train_ratio + val_ratio
    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "val"
    return "test"


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
    sample_index: int | None = None,
    group_size: int = 3,
) -> str:
    if sample_index is not None and group_size > 1:
        bucket = int(sample_index) // int(group_size)
        digest = hashlib.sha1(f"{run_id}:group:{bucket}".encode("utf-8")).hexdigest()
        return f"g_{digest[:16]}"

    digest = hashlib.sha1(f"{run_id}:{note_id}".encode("utf-8")).hexdigest()
    return f"g_{digest[:16]}"


def poll_train_idle(
    session: requests.Session,
    rl_url: str,
    previous_finished_at: str | None,
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


def get_rl_status(session: requests.Session, rl_url: str) -> Dict:
    r = session.get(f"{rl_url}/status", timeout=30)
    r.raise_for_status()
    return r.json()


def run_train_cycle(
    session: requests.Session,
    rl_url: str,
    timeout_sec: int = 7200,
) -> Dict:
    pre_status = get_rl_status(session, rl_url)
    pre_step = int(pre_status.get("training_step", 0))
    pre_finished_at = pre_status.get("last_train_finished_at")

    tr = session.post(f"{rl_url}/train", timeout=30)
    tr.raise_for_status()
    train_response = tr.json()
    if not bool(train_response.get("triggered", False)):
        raise RuntimeError(f"training cycle rejected: {train_response.get('message', 'trainer busy')}")

    ok, status = poll_train_idle(
        session,
        rl_url,
        previous_finished_at=pre_finished_at,
        timeout_sec=timeout_sec,
    )
    if not ok:
        raise RuntimeError(f"training cycle timeout, last_status={status}")

    if status.get("last_train_success") is False:
        raise RuntimeError(f"training cycle failed: {status.get('last_train_error', 'unknown error')}")

    post_step = int(status.get("training_step", pre_step))
    if post_step <= pre_step:
        raise RuntimeError(
            "training step did not advance after cycle "
            f"(pre_step={pre_step} post_step={post_step})"
        )

    return status

def flush_and_wait_reward_queue(
    session: requests.Session,
    reward_url: str,
    timeout_sec: int = 120,
) -> Dict:
    """No-op: reward_metrics_svc delivers rollouts synchronously via POST,
    so there is no queue to flush.  Returns a stub status dict so call-site
    logging (e.g. ``qst.get('pending_count')``) keeps working."""
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
    train_seen: int,
    train_every: int,
    timings: TimingStats,
    train_ratio: float,
) -> float:
    if processed <= 0:
        return 0.0

    # Per-note estimate from measured call timings.
    per_note = median_or_zero(timings.rewrite_sec) + median_or_zero(timings.icd10_sec)
    # Val/test also compute reward synchronously.
    per_reward = median_or_zero(timings.reward_sec)

    remaining = max(total - processed, 0)
    remaining_valtest = int(remaining * max(0.0, 1.0 - train_ratio))
    remaining_train = remaining - remaining_valtest

    estimate_notes = (remaining_train * per_note) + (remaining_valtest * (per_note + per_reward))

    # Add RL cycle time estimate for remaining train notes.
    cycle_sec = median_or_zero(timings.train_cycle_sec)
    if train_every > 0 and cycle_sec > 0:
        cycles_remaining = math.ceil(max(remaining_train, 0) / train_every)
        estimate_notes += cycles_remaining * cycle_sec

    return estimate_notes


def main() -> int:
    args = parse_args()

    run_id = args.run_id.strip() if args.run_id else ""
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")

    if args.train_ratio < 0 or args.val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative")
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    session = requests.Session()

    ensure_services(session, args)
    if args.reset_reward_observability:
        reset_reward_observability(session, args.reward_url)

    total_notes = get_dataset_total(session, args.dataset_url)
    if args.max_notes and args.max_notes > 0:
        total_notes = min(total_notes, args.max_notes)

    log_line(f"dataset_total_notes {total_notes}")

    split_map = build_stratified_split_map(
        session=session,
        dataset_url=args.dataset_url,
        total_notes=total_notes,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    timings = TimingStats(rewrite_sec=[], icd10_sec=[], reward_sec=[], train_cycle_sec=[], note_total_sec=[])

    processed = 0
    train_seen = 0
    train_success = 0
    val_success = 0
    test_success = 0
    failures = 0
    parse_success_count = 0
    both_parse_success_count = 0
    original_parse_failure_count = 0
    train_comparability_failure_count = 0
    train_cycle_attempts = 0
    train_cycle_success_count = 0
    train_cycle_failures: List[str] = []
    val_rewards: List[float] = []
    test_rewards: List[float] = []
    val_micro = {"tp": 0, "fp": 0, "fn": 0}
    test_micro = {"tp": 0, "fp": 0, "fn": 0}
    val_macro = {"precision_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0, "exact_match": 0.0, "count": 0.0}
    test_macro = {"precision_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0, "exact_match": 0.0, "count": 0.0}

    start_wall = time.time()

    offset = 0
    while offset < total_notes:
        size = min(args.batch_size, total_notes - offset)
        br = session.get(
            f"{args.dataset_url}/batch",
            params={"offset": offset, "size": size},
            timeout=60,
        )
        br.raise_for_status()
        batch = br.json().get("batch", [])
        if not batch:
            break

        for rec in batch:
            if processed >= total_notes:
                break

            note_id = str(rec["note_id"])
            note_text = rec["text"]
            gt_codes = rec.get("gt_codes", [])
            split = split_map.get(note_id, "test")
            note_started = now()

            try:
                t0 = now()
                rw = session.post(
                    f"{args.rewriter_url}/rewrite_prompt",
                    json={"note_id": note_id, "clinical_note": note_text},
                    timeout=args.request_timeout,
                )
                rw.raise_for_status()
                rwj = rw.json()
                timings.rewrite_sec.append(now() - t0)

                t1 = now()
                group_id = (
                    build_group_id(
                        note_id,
                        run_id,
                        sample_index=train_seen,
                        group_size=args.grpo_group_size,
                    )
                    if split == "train"
                    else build_group_id(note_id, run_id)
                )
                ic = session.post(
                    f"{args.icd10_url}/generate_codes",
                    json={
                        "note_id": note_id,
                        "run_id": run_id,
                        "group_id": group_id,
                        "original_prompt": build_original_prompt(note_text),
                        "rewritten_prompt": rwj["rewritten_prompt"],
                        "generation_source": rwj.get("generation_source"),
                        "log_prob_old": rwj.get("log_prob_old"),
                        "value_estimate": rwj.get("value_estimate"),
                        "skip_reward_forward": split != "train",
                    },
                    timeout=args.request_timeout,
                )
                ic.raise_for_status()
                icj = ic.json()
                timings.icd10_sec.append(now() - t1)

                enh_parse_ok = bool(icj.get("enh_parse_ok", bool(icj.get("enh_codes", []))))
                org_parse_ok = bool(icj.get("org_parse_ok", bool(icj.get("org_codes", []))))
                both_parse_success = bool(
                    icj.get("both_parse_success", bool(enh_parse_ok and org_parse_ok))
                )

                parse_success_count += 1 if icj.get("parsing_success") else 0
                both_parse_success_count += 1 if both_parse_success else 0
                original_parse_failure_count += 0 if org_parse_ok else 1

                if split == "train":
                    train_seen += 1
                    train_success += 1
                    if not both_parse_success:
                        train_comparability_failure_count += 1
                else:
                    t2 = now()
                    rr = session.post(
                        f"{args.reward_url}/compute_reward",
                        json={
                            "note_id": note_id,
                            "gt_codes": gt_codes,
                            "enh_codes": icj.get("enh_codes", []),
                            "org_codes": icj.get("org_codes", []),
                            "parsing_success": icj.get("parsing_success", True),
                        },
                        timeout=60,
                    )
                    rr.raise_for_status()
                    rv = float(rr.json()["reward"])
                    timings.reward_sec.append(now() - t2)

                    gt_set = canonicalized_code_set(gt_codes)
                    pred_set = canonicalized_code_set(icj.get("enh_codes", []))
                    if split == "val":
                        val_success += 1
                        val_rewards.append(rv)
                        update_micro_counts(gt_set, pred_set, val_micro)
                        update_macro_counts(gt_set, pred_set, val_macro)
                    else:
                        test_success += 1
                        test_rewards.append(rv)
                        update_micro_counts(gt_set, pred_set, test_micro)
                        update_macro_counts(gt_set, pred_set, test_macro)

            except Exception as exc:
                failures += 1
                print(f"error note_id={note_id} split={split} err={exc}", file=sys.stderr, flush=True)
            finally:
                timings.note_total_sec.append(now() - note_started)

            processed += 1

            # Periodic RL train trigger.
            if split == "train" and args.train_every > 0 and train_seen % args.train_every == 0:
                log_line(f"train_cycle_start train_seen={train_seen} processed={processed}")
                qst = flush_and_wait_reward_queue(session, args.reward_url)
                cycle_start = now()
                train_cycle_attempts += 1
                try:
                    st = run_train_cycle(session, args.rl_url)
                    timings.train_cycle_sec.append(now() - cycle_start)
                    train_cycle_success_count += 1
                    log_line(
                        "train_cycle_done "
                        f"train_seen={train_seen} "
                        f"queue_pending={qst.get('pending_count')} "
                        f"training_step={st.get('training_step')} "
                        f"rollouts_loaded={st.get('rollouts_loaded')}"
                    )
                except Exception as exc:
                    timings.train_cycle_sec.append(now() - cycle_start)
                    failure = (
                        "train_cycle_failed "
                        f"train_seen={train_seen} "
                        f"processed={processed} "
                        f"queue_pending={qst.get('pending_count')} "
                        f"err={exc}"
                    )
                    train_cycle_failures.append(failure)
                    log_line(failure)

            # Progress heartbeat with configurable cadence.
            progress_every = max(1, args.progress_every)
            if processed % progress_every == 0 or processed == total_notes:
                elapsed = time.time() - start_wall
                eta = estimate_remaining_seconds(
                    processed=processed,
                    total=total_notes,
                    train_seen=train_seen,
                    train_every=args.train_every,
                    timings=timings,
                    train_ratio=args.train_ratio,
                )
                log_line(
                    "progress "
                    f"processed={processed}/{total_notes} "
                    f"elapsed_min={elapsed/60:.1f} "
                    f"eta_hr={eta/3600:.2f} "
                    f"failures={failures}"
                )

        offset += size

    # Final train flush.
    if train_seen > 0:
        log_line(f"final_train_cycle_start train_seen={train_seen} processed={processed}")
        qst = flush_and_wait_reward_queue(session, args.reward_url)
        cycle_start = now()
        train_cycle_attempts += 1
        try:
            st = run_train_cycle(session, args.rl_url)
            timings.train_cycle_sec.append(now() - cycle_start)
            train_cycle_success_count += 1
            log_line(
                "final_train_cycle_done "
                f"queue_pending={qst.get('pending_count')} "
                f"training_step={st.get('training_step')} "
                f"rollouts_loaded={st.get('rollouts_loaded')}"
            )
        except Exception as exc:
            timings.train_cycle_sec.append(now() - cycle_start)
            failure = (
                "final_train_cycle_failed "
                f"queue_pending={qst.get('pending_count')} "
                f"err={exc}"
            )
            train_cycle_failures.append(failure)
            log_line(failure)

    ck = session.get(f"{args.rl_url}/checkpoint", timeout=30)
    ck.raise_for_status()

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

    summary = {
        "run_id": run_id,
        "total_processed": processed,
        "total_target": total_notes,
        "failures": failures,
        "train_success": train_success,
        "val_success": val_success,
        "test_success": test_success,
        "parse_success_rate": (parse_success_count / processed) if processed else 0.0,
        "both_parse_success_rate": (both_parse_success_count / processed) if processed else 0.0,
        "original_parse_failure_rate": (original_parse_failure_count / processed) if processed else 0.0,
        "train_comparability_failure_count": train_comparability_failure_count,
        "train_cycle_attempts": train_cycle_attempts,
        "train_cycle_success_count": train_cycle_success_count,
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
        "rewrite_sec_median": median_or_zero(timings.rewrite_sec),
        "rewrite_sec_p95": percentile_or_zero(timings.rewrite_sec, 0.95),
        "rewrite_sec_p99": percentile_or_zero(timings.rewrite_sec, 0.99),
        "icd10_sec_median": median_or_zero(timings.icd10_sec),
        "icd10_sec_p95": percentile_or_zero(timings.icd10_sec, 0.95),
        "icd10_sec_p99": percentile_or_zero(timings.icd10_sec, 0.99),
        "reward_sec_median": median_or_zero(timings.reward_sec),
        "reward_sec_p95": percentile_or_zero(timings.reward_sec, 0.95),
        "reward_sec_p99": percentile_or_zero(timings.reward_sec, 0.99),
        "train_cycle_sec_median": median_or_zero(timings.train_cycle_sec),
        "train_cycle_sec_p95": percentile_or_zero(timings.train_cycle_sec, 0.95),
        "train_cycle_sec_p99": percentile_or_zero(timings.train_cycle_sec, 0.99),
        "note_total_sec_median": median_or_zero(timings.note_total_sec),
        "note_total_sec_p95": percentile_or_zero(timings.note_total_sec, 0.95),
        "note_total_sec_p99": percentile_or_zero(timings.note_total_sec, 0.99),
        "rollout_drop_rate": float(reward_observability.get("rollout_drop_rate", 0.0)),
        "queue_lag_seconds": float(reward_observability.get("queue_lag_seconds", 0.0)),
        "parse_failure_taxonomy": icd_observability.get("parse_failure_taxonomy", {}),
        "reward_observability": reward_observability,
        "icd_observability": icd_observability,
        "elapsed_hr": (time.time() - start_wall) / 3600.0,
        "checkpoint": ck.json(),
    }

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
        run_guard_failures.append(
            f"train_cycle_failures_detected(count={len(train_cycle_failures)})"
        )

    summary["run_guard_failures"] = run_guard_failures

    log_line(f"summary {summary}")
    if run_guard_failures:
        log_line(f"run_guard_failed count={len(run_guard_failures)} failures={run_guard_failures}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
