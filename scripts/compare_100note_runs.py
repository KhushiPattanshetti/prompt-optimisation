#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict


def _load_summary(log_path: Path) -> Dict[str, Any]:
    summary_line = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("summary "):
            summary_line = line[len("summary ") :].strip()

    if summary_line is None:
        raise RuntimeError(f"No summary line found in {log_path}")

    parsed = ast.literal_eval(summary_line)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Summary payload in {log_path} is not a dict")
    return parsed


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and distributed 100-note runs")
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--distributed-log", required=True)
    parser.add_argument("--baseline-world-size", type=int, default=1)
    parser.add_argument("--distributed-world-size", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    baseline = _load_summary(Path(args.baseline_log))
    distributed = _load_summary(Path(args.distributed_log))

    baseline_note_median = float(baseline.get("note_total_sec_median", 0.0))
    distributed_note_median = float(distributed.get("note_total_sec_median", 0.0))
    baseline_elapsed_hr = float(baseline.get("elapsed_hr", 0.0))
    distributed_elapsed_hr = float(distributed.get("elapsed_hr", 0.0))

    baseline_processed = int(baseline.get("total_processed", 0))
    distributed_processed = int(distributed.get("total_processed", 0))

    baseline_service_counts = {
        "dataset_svc_notes": baseline_processed,
        "rewriter_svc_notes": baseline_processed,
        "icd10_svc_notes": baseline_processed,
        "reward_svc_notes": baseline_processed,
        "rl_loop_svc_rollouts_train_split": int(baseline.get("train_success", 0)),
    }

    distributed_service_counts = {
        "dataset_svc_notes": distributed_processed,
        "rewriter_svc_notes": distributed_processed,
        "icd10_svc_notes": distributed_processed,
        "reward_svc_notes": distributed_processed,
        "rl_loop_svc_rollouts_train_split": int(distributed.get("train_success", 0)),
    }

    baseline_parallel = {
        "service_container_instances": {
            "dataset_svc": 1,
            "rewriter_svc": 1,
            "icd10_svc": 1,
            "reward_svc": 1,
            "rl_loop_svc": 1,
        },
        "model_instances": {
            "rewriter_phi3_actor": 1,
            "icd_med42_frozen": 1,
            "rl_policy_model": 1,
            "rl_reference_model": 1,
            "rl_value_head": 1,
        },
        "rl_training_parallel_workers": args.baseline_world_size,
    }

    distributed_parallel = {
        "service_container_instances": {
            "dataset_svc": 1,
            "rewriter_svc": 1,
            "icd10_svc": 1,
            "reward_svc": 1,
            "rl_loop_svc": 1,
        },
        "model_instances": {
            "rewriter_phi3_actor": 1,
            "icd_med42_frozen": 1,
            "rl_policy_model": args.distributed_world_size,
            "rl_reference_model": args.distributed_world_size,
            "rl_value_head": args.distributed_world_size,
        },
        "rl_training_parallel_workers": args.distributed_world_size,
    }

    comparison = {
        "baseline": {
            "summary": baseline,
            "parallelism": baseline_parallel,
            "notes_per_component": baseline_service_counts,
        },
        "distributed": {
            "summary": distributed,
            "parallelism": distributed_parallel,
            "notes_per_component": distributed_service_counts,
        },
        "latency_diff": {
            "note_total_sec_median_baseline": baseline_note_median,
            "note_total_sec_median_distributed": distributed_note_median,
            "note_total_sec_median_delta": distributed_note_median - baseline_note_median,
            "note_total_sec_median_speedup_x": _safe_ratio(
                baseline_note_median,
                distributed_note_median,
            ),
            "elapsed_hr_baseline": baseline_elapsed_hr,
            "elapsed_hr_distributed": distributed_elapsed_hr,
            "elapsed_hr_delta": distributed_elapsed_hr - baseline_elapsed_hr,
            "elapsed_hr_speedup_x": _safe_ratio(
                baseline_elapsed_hr,
                distributed_elapsed_hr,
            ),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    md = []
    md.append("# 100-Note Baseline vs Distributed Comparison")
    md.append("")
    md.append("## Latency")
    md.append(
        f"- Note median latency (s): baseline={baseline_note_median:.4f}, distributed={distributed_note_median:.4f}, speedup={comparison['latency_diff']['note_total_sec_median_speedup_x']:.3f}x"
    )
    md.append(
        f"- End-to-end elapsed (hr): baseline={baseline_elapsed_hr:.4f}, distributed={distributed_elapsed_hr:.4f}, speedup={comparison['latency_diff']['elapsed_hr_speedup_x']:.3f}x"
    )
    md.append("")
    md.append("## Parallel Instances")
    md.append(f"- Baseline RL workers: {args.baseline_world_size}")
    md.append(f"- Distributed RL workers: {args.distributed_world_size}")
    md.append("- Service container instances in both phases: dataset=1, rewriter=1, icd10=1, reward=1, rl_loop=1")
    md.append("")
    md.append("## Notes Through Components")
    md.append(
        f"- Baseline: dataset={baseline_service_counts['dataset_svc_notes']}, rewriter={baseline_service_counts['rewriter_svc_notes']}, icd10={baseline_service_counts['icd10_svc_notes']}, reward={baseline_service_counts['reward_svc_notes']}, rl_train_rollouts={baseline_service_counts['rl_loop_svc_rollouts_train_split']}"
    )
    md.append(
        f"- Distributed: dataset={distributed_service_counts['dataset_svc_notes']}, rewriter={distributed_service_counts['rewriter_svc_notes']}, icd10={distributed_service_counts['icd10_svc_notes']}, reward={distributed_service_counts['reward_svc_notes']}, rl_train_rollouts={distributed_service_counts['rl_loop_svc_rollouts_train_split']}"
    )

    output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
