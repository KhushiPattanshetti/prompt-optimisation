#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_suite() {
  python -m pytest -m "not e2e and not stress" "$@"
}

run_suite dataset_svc/tests/unit dataset_svc/tests/integration dataset_svc/tests/e2e "$@"
run_suite reward_metrics_svc/test "$@"
run_suite rewriter_inference_svc/tests/unit rewriter_inference_svc/tests/integration "$@"
run_suite icd10_coding_svc/tests/unit icd10_coding_svc/tests/integration "$@"
run_suite rl_loop_svc/tests/unit rl_loop_svc/tests/integration rl_loop_svc/tests/e2e "$@"
