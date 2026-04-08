"""
simulate.py – Standalone simulation that exercises reward_metrics_svc
without needing an HTTP server.

Runs all 14 spec §19 scenarios directly against the core Python functions,
emitting full DEBUG logs to stdout so you can inspect Wu-Palmer values,
LCA pairs, penalty breakdowns, and aggregate stats.

All spec codes (J45.0, I10, etc.) are out-of-tree for the bundled
icd10_tree.json; the similarity layer uses string-heuristic depth as
the fallback (spec §8.3, §19).

Usage
-----
  cd reward_metrics_svc/
  python simulate.py                          # DEBUG to stdout (default)
  python simulate.py 2>&1 | tee run.log       # save logs to file

To exercise the HTTP API instead, start the server first:
  uvicorn app:app --host 0.0.0.0 --port 8002 --log-level debug

Then use the curl examples printed at the end of this script.
"""

import logging
import os
import sys

# ── make reward_metrics_svc/ importable when running from any directory ───────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── configure logging BEFORE any project import (captures tree-build logs) ───
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── project imports (after logging setup) ─────────────────────────────────────
from log_utils import log_coverage_debug, log_extra_debug, update_aggregate_stats
from reward import compute_reward

logger = logging.getLogger("simulate")

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios  (spec §19.1 – §19.14)
# Each entry: (label, gt, enh, org, invalid_codes, duplicate_codes,
#              parsing_success, expected_notes)
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # §19.1 Improvement (Tree Signal)
    (
        "§19.1  Improvement (Tree Signal)",
        ["J45.0"],
        ["J45"],
        ["I10"],
        [],
        [],
        True,
        "D_enh < D_org, R_tree > 0, R_exact < 1, R_structure ≈ 1, reward > 0",
    ),
    # §19.2 Perfect Match (Exact Reward Dominance)
    (
        "§19.2  Perfect Match (Exact Reward Dominance)",
        ["J45.0"],
        ["J45.0"],
        ["J45"],
        [],
        [],
        True,
        "D_enh ≈ 0, R_exact = 1, R_structure = 1, reward ≈ 1",
    ),
    # §19.3 No Improvement (Neutral Case)
    (
        "§19.3  No Improvement (Neutral Case)",
        ["J45.0"],
        ["I10"],
        ["I10"],
        [],
        [],
        True,
        "D_enh == D_org, R_tree ≈ 0, R_exact ≈ -1, reward ≈ 0 or slightly neg",
    ),
    # §19.4 Worse Prediction
    (
        "§19.4  Worse Prediction",
        ["J45.0"],
        ["I10"],
        ["J45"],
        [],
        [],
        True,
        "D_enh > D_org, R_tree < 0, reward < 0",
    ),
    # §19.5 Overprediction (Extra Codes Penalty)
    (
        "§19.5  Overprediction (Extra Codes Penalty)",
        ["J45.0"],
        ["J45.0", "I10"],
        ["J45"],
        [],
        [],
        True,
        "P_extra active, R_exact < 1, reward slightly lower than perfect match",
    ),
    # §19.6 Underprediction (Missing Codes)
    (
        "§19.6  Underprediction (Missing Codes)",
        ["J45.0", "I10"],
        ["J45.0"],
        ["J45.0"],
        [],
        [],
        True,
        "Coverage penalty active, R_exact < 1, reward reduced",
    ),
    # §19.7 Duplicate Codes (Structure Penalty)
    (
        "§19.7  Duplicate Codes (Structure Penalty)",
        ["J45.0"],
        ["J45.0"],  # enh after dedup
        ["J45"],
        [],
        ["J45.0"],
        True,  # J45.0 given as a duplicate
        "duplicate_codes detected, R_structure < 1, reward slightly penalised",
    ),
    # §19.8 Invalid Codes (Structure Penalty)
    (
        "§19.8  Invalid Codes (Structure Penalty)",
        ["J45.0"],
        ["INVALID_CODE"],
        ["J45"],
        ["INVALID_CODE"],
        [],
        True,
        "invalid_codes detected, R_structure significantly reduced, reward < 0",
    ),
    # §19.9 Parsing Failure (Hard Penalty)
    (
        "§19.9  Parsing Failure (Hard Penalty)",
        ["J45.0"],
        [],
        ["J45"],
        [],
        [],
        False,
        "R_structure = -1, reward strongly negative",
    ),
    # §19.10 Hierarchical Similarity Check
    (
        "§19.10 Hierarchical Similarity Check",
        ["J45.0"],
        ["J45"],
        ["J40-J47"],
        [],
        [],
        True,
        "sim(J45.0, J45) > sim(J45.0, J40-J47), D_enh < D_org, reward > 0",
    ),
    # §19.11 Completely Unrelated Codes
    (
        "§19.11 Completely Unrelated Codes",
        ["J45.0"],
        ["A01.1"],
        ["I10"],
        [],
        [],
        True,
        "low similarity, R_exact ≈ -1, reward near or below 0",
    ),
    # §19.12 Mixed Quality Prediction
    (
        "§19.12 Mixed Quality Prediction",
        ["J45.0", "I10"],
        ["J45", "INVALID_CODE"],
        ["J45"],
        ["INVALID_CODE"],
        [],
        True,
        "partial coverage, invalid penalty, moderate reward degradation",
    ),
    # §19.13 Cardinality Stress Test
    (
        "§19.13 Cardinality Stress Test",
        ["J45.0"],
        ["J45.0", "I10", "A01.1", "B20"],
        ["J45"],
        [],
        [],
        True,
        "high P_card penalty, reward reduced significantly",
    ),
    # §19.14 Symmetry Check (enh ↔ org swapped vs §19.1)
    (
        "§19.14 Symmetry Check (enh↔org swap)",
        ["J45.0"],
        ["I10"],  # swapped: was org in §19.1
        ["J45"],  # swapped: was enh in §19.1
        [],
        [],
        True,
        "reward should flip sign vs §19.1 (enh/org exchanged)",
    ),
]

# Extra: determinism check (same input run 3×)
DETERMINISM_INPUT = (["J45.0"], ["J45"], ["I10"], [], [], True)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def run() -> None:
    SEP = "=" * 76

    logger.info(SEP)
    logger.info("reward_metrics_svc  ·  STANDALONE SIMULATION  (spec §19)")
    logger.info("All J45.x / I10 codes are out-of-tree → heuristic depth fallback")
    logger.info(SEP)

    results = []

    for scenario in SCENARIOS:
        label, gt, enh, org, invalid, dupes, parsing_ok, notes = scenario

        logger.info("-" * 76)
        logger.info("SCENARIO : %s", label)
        logger.info("  gt      = %s", gt)
        logger.info("  enh     = %s", enh)
        logger.info("  org     = %s", org)
        logger.info(
            "  invalid = %s  dupes = %s  parsing_ok = %s", invalid, dupes, parsing_ok
        )
        logger.info("  expect  : %s", notes)

        reward, metrics = compute_reward(
            gt,
            enh,
            org,
            invalid_codes=invalid,
            duplicate_codes=dupes,
            parsing_success=parsing_ok,
        )

        # Critical debug logs
        log_coverage_debug(gt, enh, label="enh")
        log_coverage_debug(gt, org, label="org")
        log_extra_debug(enh, gt, label="enh")
        log_extra_debug(org, gt, label="org")

        # Aggregate stats
        update_aggregate_stats(reward, metrics["delta_D"])

        rc = metrics["reward_components"]
        sign = (
            "↑ BETTER"
            if reward > 0.05
            else ("↓ WORSE" if reward < -0.05 else "= NEUTRAL")
        )
        logger.info(
            "RESULT   D_enh=%.4f  D_org=%.4f  ΔD=%+.4f"
            "  R_tree=%+.4f  R_exact=%+.4f  R_structure=%+.4f"
            "  reward=%+.6f  %s",
            metrics["D_enh"],
            metrics["D_org"],
            metrics["delta_D"],
            rc["R_tree"],
            rc["R_exact"],
            rc["R_structure"],
            reward,
            sign,
        )
        results.append((label, reward, rc))

    # ── §19.15 Determinism check ──────────────────────────────────────────────
    logger.info("-" * 76)
    logger.info("SCENARIO : §19.15 Determinism Check (3 identical runs)")
    det_rewards = []
    for _ in range(3):
        r, _ = compute_reward(*DETERMINISM_INPUT)
        det_rewards.append(r)
    all_same = len(set(det_rewards)) == 1
    logger.info("  rewards = %s  →  deterministic=%s", det_rewards, all_same)
    results.append(("§19.15 Determinism Check", det_rewards[0], {}))

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info("SIMULATION SUMMARY")
    logger.info(SEP)
    for label, reward, rc in results:
        bar_len = int(abs(reward) * 20)
        bar = ("+" * bar_len) if reward > 0 else ("-" * bar_len)
        logger.info("  %-45s  reward=%+.4f  |%s", label, reward, bar)
    logger.info(SEP)

    # ── HTTP curl examples ────────────────────────────────────────────────────
    logger.info("")
    logger.info("TO TEST VIA HTTP:")
    logger.info("  1. Start the server:")
    logger.info("       cd reward_metrics_svc/")
    logger.info("       source .venv/bin/activate")
    logger.info("       uvicorn app:app --host 0.0.0.0 --port 8002 --log-level debug")
    logger.info("")
    logger.info("  2. Health check:")
    logger.info("       curl http://localhost:8002/health")
    logger.info("")
    logger.info("  3. §19.1 Improvement scenario:")
    logger.info(
        "       curl -s -X POST http://localhost:8002/compute_reward \\\n"
        '         -H "Content-Type: application/json" \\\n'
        '         -d \'{"note_id":"sim_001","gt_codes":["J45.0"],"enh_codes":["J45"],'
        '"org_codes":["I10"],"invalid_codes":[],"duplicate_codes":[],'
        '"state":"s","action":"a","log_prob_old":0.1,"value_estimate":0.5}\' '
        "| python3 -m json.tool"
    )
    logger.info("")
    logger.info("  4. §19.9 Parsing failure scenario:")
    logger.info(
        "       curl -s -X POST http://localhost:8002/compute_reward \\\n"
        '         -H "Content-Type: application/json" \\\n'
        '         -d \'{"note_id":"sim_009","gt_codes":["J45.0"],"enh_codes":[],'
        '"org_codes":["J45"],"invalid_codes":[],"duplicate_codes":[],'
        '"parsing_success":false,"state":"s","action":"a",'
        '"log_prob_old":0.0,"value_estimate":0.0}\' | python3 -m json.tool'
    )
    logger.info("")
    logger.info("  5. §19.8 Invalid code scenario:")
    logger.info(
        "       curl -s -X POST http://localhost:8002/compute_reward \\\n"
        '         -H "Content-Type: application/json" \\\n'
        '         -d \'{"note_id":"sim_008","gt_codes":["J45.0"],'
        '"enh_codes":["INVALID_CODE"],"org_codes":["J45"],'
        '"invalid_codes":["INVALID_CODE"],"duplicate_codes":[],'
        '"state":"s","action":"a","log_prob_old":0.0,"value_estimate":0.0}\' '
        "| python3 -m json.tool"
    )


if __name__ == "__main__":
    run()
