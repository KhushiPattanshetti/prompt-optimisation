#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/raid/adityasd314/temp/prompt-optimisation-merged"
TARGET_SCRIPT="$REPO_DIR/scripts/run_100note_baseline_vs_distributed_tmux.sh"
SESSION_NAME="${SESSION_NAME:-bench100_validation}"
AUTO_ATTACH="${AUTO_ATTACH:-1}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/manual_validation}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 1
fi

if [[ ! -x "$TARGET_SCRIPT" ]]; then
  echo "[ERROR] target script is not executable: $TARGET_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
launcher_log="$LOG_DIR/bench100_launcher_${timestamp}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] tmux session already exists: $SESSION_NAME"
else
  tmux new-session -d -s "$SESSION_NAME" \
    "cd \"$REPO_DIR\" && BENCH100_SKIP_TMUX_LAUNCH=1 \"$TARGET_SCRIPT\" 2>&1 | tee -a \"$launcher_log\""
  echo "[INFO] started session: $SESSION_NAME"
  echo "[INFO] launcher log: $launcher_log"
fi

if [[ "$AUTO_ATTACH" == "1" ]]; then
  exec tmux attach -t "$SESSION_NAME"
fi

echo "[INFO] attach with: tmux attach -t $SESSION_NAME"
