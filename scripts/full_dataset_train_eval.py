#!/usr/bin/env python3
"""Full-dataset train/test/validate runner for the prompt optimisation pipeline.

Flow per note:
1. Fetch note text from dataset_svc (/batch).
2. Generate rewritten prompt with rewriter_inference_svc (/rewrite_prompt).
3. Generate ICD10 codes with icd10_coding_svc (/generate_codes).
4. Train split: rely on async icd10 -> reward -> rl /rollout forwarding.
5. Val/Test split: call reward_metrics_svc (/reward) without rollout fields
   so reward is computed but not forwarded to RL (no training leakage).

The script periodically triggers rl_loop_svc (/train) and waits for cycle completion.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests


@dataclass
class TimingStats:
    rewrite_sec: List[float]
    icd10_sec: List[float]
    reward_sec: List[float]
    train_cycle_sec: List[float]


def now() -> float:
    return time.perf_counter()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full dataset train/test/validate pipeline")
    p.add_argument("--dataset-url", default="http://localhost:8003")
    p.add_argument("--rewriter-url", default="http://localhost:8000")
    p.add_argument("--icd10-url", default="http://localhost:8001")
    p.add_argument("--reward-url", default="http://localhost:8002")
    p.add_argument("--rl-url", default="http://localhost:8004")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--request-timeout", type=int, default=1800)
    p.add_argument("--max-notes", type=int, default=0, help="0 means full dataset")
    return p.parse_args()


def ensure_services(session: requests.Session, args: argparse.Namespace) -> None:
    checks = [
        ("dataset", f"{args.dataset_url}/health"),
        ("icd10", f"{args.icd10_url}/health"),
        ("reward", f"{args.reward_url}/health"),
        ("rl", f"{args.rl_url}/status"),
    ]
    for name, url in checks:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        print(f"service_ok {name} {url}")


def get_dataset_total(session: requests.Session, dataset_url: str) -> int:
    r = session.get(f"{dataset_url}/health", timeout=30)
    r.raise_for_status()
    payload = r.json()
    return int(payload["total_notes"])


def split_for_index(i: int, n_total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)
    if i < train_end:
        return "train"
    if i < val_end:
        return "val"
    return "test"


def build_original_prompt(clinical_note: str) -> str:
    return (
        "You are a clinical coding expert. "
        "Extract all ICD-10-CM diagnosis codes from the clinical note below. "
        "Output the codes as a JSON list of strings only. "
        "Do not include any explanation or other text.\n\n"
        f"Clinical note:\n{clinical_note}"
    )


def poll_train_idle(
    session: requests.Session,
    rl_url: str,
    timeout_sec: int = 7200,
) -> Tuple[bool, Dict]:
    t0 = time.time()
    while True:
        r = session.get(f"{rl_url}/status", timeout=30)
        r.raise_for_status()
        st = r.json()
        if st.get("trainer_state") == "IDLE":
            return True, st
        if time.time() - t0 > timeout_sec:
            return False, st
        time.sleep(2)


def median_or_zero(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def estimate_remaining_seconds(
    processed: int,
    total: int,
    train_seen: int,
    train_every: int,
    timings: TimingStats,
) -> float:
    if processed <= 0:
        return 0.0

    # Per-note estimate from measured call timings.
    per_note = median_or_zero(timings.rewrite_sec) + median_or_zero(timings.icd10_sec)
    # Val/test also compute reward synchronously.
    per_reward = median_or_zero(timings.reward_sec)

    remaining = max(total - processed, 0)
    # Approximate remaining val/test fraction as 20% by default split.
    remaining_valtest = int(remaining * 0.2)
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
    session = requests.Session()

    ensure_services(session, args)

    total_notes = get_dataset_total(session, args.dataset_url)
    if args.max_notes and args.max_notes > 0:
        total_notes = min(total_notes, args.max_notes)

    print(f"dataset_total_notes {total_notes}")

    timings = TimingStats(rewrite_sec=[], icd10_sec=[], reward_sec=[], train_cycle_sec=[])

    processed = 0
    train_seen = 0
    train_success = 0
    val_success = 0
    test_success = 0
    failures = 0
    parse_success_count = 0
    val_rewards: List[float] = []
    test_rewards: List[float] = []

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
            split = split_for_index(processed, total_notes, args.train_ratio, args.val_ratio)

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
                ic = session.post(
                    f"{args.icd10_url}/generate_codes",
                    json={
                        "note_id": note_id,
                        "original_prompt": build_original_prompt(note_text),
                        "rewritten_prompt": rwj["rewritten_prompt"],
                        "log_prob_old": rwj.get("log_prob_old"),
                        "value_estimate": rwj.get("value_estimate"),
                    },
                    timeout=args.request_timeout,
                )
                ic.raise_for_status()
                icj = ic.json()
                timings.icd10_sec.append(now() - t1)

                parse_success_count += 1 if icj.get("parsing_success") else 0

                if split == "train":
                    train_seen += 1
                    train_success += 1
                else:
                    t2 = now()
                    rr = session.post(
                        f"{args.reward_url}/reward",
                        json={
                            "gt_codes": gt_codes,
                            "enh_codes": icj.get("enh_codes", []),
                            "org_codes": icj.get("org_codes", []),
                        },
                        timeout=60,
                    )
                    rr.raise_for_status()
                    rv = float(rr.json()["reward"])
                    timings.reward_sec.append(now() - t2)
                    if split == "val":
                        val_success += 1
                        val_rewards.append(rv)
                    else:
                        test_success += 1
                        test_rewards.append(rv)

            except Exception as exc:
                failures += 1
                print(f"error note_id={note_id} split={split} err={exc}", file=sys.stderr)

            processed += 1

            # Periodic RL train trigger.
            if split == "train" and args.train_every > 0 and train_seen % args.train_every == 0:
                cycle_start = now()
                tr = session.post(f"{args.rl_url}/train", timeout=30)
                tr.raise_for_status()
                ok, st = poll_train_idle(session, args.rl_url)
                timings.train_cycle_sec.append(now() - cycle_start)
                if not ok:
                    raise RuntimeError(f"training cycle timeout, last_status={st}")

            # Progress heartbeat every 25 notes.
            if processed % 25 == 0 or processed == total_notes:
                elapsed = time.time() - start_wall
                eta = estimate_remaining_seconds(
                    processed=processed,
                    total=total_notes,
                    train_seen=train_seen,
                    train_every=args.train_every,
                    timings=timings,
                )
                print(
                    "progress "
                    f"processed={processed}/{total_notes} "
                    f"elapsed_min={elapsed/60:.1f} "
                    f"eta_hr={eta/3600:.2f} "
                    f"failures={failures}"
                )

        offset += size

    # Final train flush.
    if train_seen > 0:
        cycle_start = now()
        tr = session.post(f"{args.rl_url}/train", timeout=30)
        tr.raise_for_status()
        ok, st = poll_train_idle(session, args.rl_url)
        timings.train_cycle_sec.append(now() - cycle_start)
        if not ok:
            raise RuntimeError(f"final training cycle timeout, last_status={st}")

    ck = session.get(f"{args.rl_url}/checkpoint", timeout=30)
    ck.raise_for_status()

    summary = {
        "total_processed": processed,
        "total_target": total_notes,
        "failures": failures,
        "train_success": train_success,
        "val_success": val_success,
        "test_success": test_success,
        "parse_success_rate": (parse_success_count / processed) if processed else 0.0,
        "val_reward_mean": statistics.mean(val_rewards) if val_rewards else 0.0,
        "test_reward_mean": statistics.mean(test_rewards) if test_rewards else 0.0,
        "rewrite_sec_median": median_or_zero(timings.rewrite_sec),
        "icd10_sec_median": median_or_zero(timings.icd10_sec),
        "reward_sec_median": median_or_zero(timings.reward_sec),
        "train_cycle_sec_median": median_or_zero(timings.train_cycle_sec),
        "elapsed_hr": (time.time() - start_wall) / 3600.0,
        "checkpoint": ck.json(),
    }

    print("summary", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
