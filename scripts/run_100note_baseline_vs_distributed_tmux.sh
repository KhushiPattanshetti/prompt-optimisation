#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/raid/adityasd314/temp/prompt-optimisation-merged"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
VENV_PYTHON="/raid/adityasd314/temp/prompt-optimisation/.venv/bin/python"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-bench100_validation}"
TMUX_AUTO_ATTACH="${TMUX_AUTO_ATTACH:-1}"
TMUX_BOOTSTRAP_LOG_DIR="${TMUX_BOOTSTRAP_LOG_DIR:-$REPO_DIR/logs/manual_validation}"

if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN="$(command -v python3)"
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

bootstrap_tmux_session() {
  if [[ "${BENCH100_SKIP_TMUX_LAUNCH:-0}" == "1" ]]; then
    return 0
  fi

  if [[ -n "${TMUX:-}" ]]; then
    return 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    log "error tmux_not_found install_tmux_or_run_with_BENCH100_SKIP_TMUX_LAUNCH=1"
    exit 1
  fi

  mkdir -p "$TMUX_BOOTSTRAP_LOG_DIR"
  local launch_ts session_log
  launch_ts="$(date -u +%Y%m%dT%H%M%SZ)"
  session_log="$TMUX_BOOTSTRAP_LOG_DIR/bench100_tmux_${launch_ts}.log"

  if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
    log "tmux_session_exists session=$TMUX_SESSION_NAME"
    if [[ "$TMUX_AUTO_ATTACH" == "1" ]]; then
      tmux attach -t "$TMUX_SESSION_NAME"
    else
      log "tmux_attach_hint cmd='tmux attach -t $TMUX_SESSION_NAME'"
    fi
    exit 0
  fi

  tmux new-session -d -s "$TMUX_SESSION_NAME" \
    "cd \"$REPO_DIR\" && BENCH100_SKIP_TMUX_LAUNCH=1 \"$SCRIPT_PATH\" 2>&1 | tee -a \"$session_log\""
  log "tmux_session_started session=$TMUX_SESSION_NAME log=$session_log"

  if [[ "$TMUX_AUTO_ATTACH" == "1" ]]; then
    tmux attach -t "$TMUX_SESSION_NAME"
  else
    log "tmux_attach_hint cmd='tmux attach -t $TMUX_SESSION_NAME'"
  fi
  exit 0
}

bootstrap_tmux_session

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-240}"
  local sleep_sec="${4:-3}"

  for i in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "health_ok name=$name url=$url attempts=$i"
      return 0
    fi
    if (( i % 20 == 0 )); then
      log "health_wait name=$name attempt=$i"
    fi
    sleep "$sleep_sec"
  done

  log "health_fail name=$name url=$url"
  return 1
}

wait_all_services() {
  wait_http "dataset" "http://localhost:8003/health"
  wait_http "rewriter" "http://localhost:8000/health"
  wait_http "icd10" "http://localhost:8001/health"
  wait_http "reward" "http://localhost:8002/health"
  wait_http "rl" "http://localhost:8004/status"
}

visible_devices_for_ws() {
  local ws="$1"
  seq 0 $((ws - 1)) | paste -sd, -
}

probe_world_size() {
  local ws="$1"
  local visible="$2"
  local probe_run_id="probe_ws${ws}_$(date -u +%Y%m%dT%H%M%SZ)"

  log "probe_begin world_size=$ws visible=$visible"
  DIST_WORLD_SIZE="$ws" DIST_VISIBLE_DEVICES="$visible" \
    docker compose -f docker-compose.yml -f docker-compose.rl-distributed-max.yml \
    up -d --build rl_loop_svc >/dev/null

  wait_http "rl_probe" "http://localhost:8004/status"

  local step_before
  step_before="$(curl -fsS http://localhost:8004/status | jq -r '.training_step')"

  for i in 1 2; do
    local payload
    payload="$(jq -n \
      --arg run_id "$probe_run_id" \
      --arg original_prompt "Probe clinical note $i" \
      --arg rewritten_prompt "Extract all ICD-10-CM diagnosis codes and return only a JSON list of strings." \
      --argjson reward 0.2 \
      --argjson log_prob_old -0.7 \
      --argjson value_estimate 0.3 \
      '{run_id:$run_id, original_prompt:$original_prompt, rewritten_prompt:$rewritten_prompt, reward:$reward, log_prob_old:$log_prob_old, value_estimate:$value_estimate}')"
    curl -fsS -X POST http://localhost:8004/rollout \
      -H "Content-Type: application/json" \
      -d "$payload" >/dev/null
  done

  curl -fsS -X POST http://localhost:8004/train >/dev/null

  for i in $(seq 1 240); do
    local status state step
    status="$(curl -fsS http://localhost:8004/status)"
    state="$(echo "$status" | jq -r '.trainer_state')"
    step="$(echo "$status" | jq -r '.training_step')"

    if [[ "$state" == "IDLE" && "$step" -gt "$step_before" ]]; then
      log "probe_success world_size=$ws step_before=$step_before step_after=$step"
      return 0
    fi

    if (( i % 20 == 0 )); then
      log "probe_wait world_size=$ws poll=$i state=$state step=$step"
    fi
    sleep 3
  done

  log "probe_fail world_size=$ws"
  return 1
}

cd "$REPO_DIR"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$REPO_DIR/logs/manual_validation/bench100_${timestamp}"
mkdir -p "$run_root"

baseline_run_id="baseline100_${timestamp}"
distributed_run_id="distributed100_${timestamp}"

log "run_root=$run_root"
log "python_bin=$PYTHON_BIN"

log "phase=baseline bring_up"
docker compose -f docker-compose.yml -f docker-compose.rl-baseline.yml \
  up -d --build dataset_svc rewriter_svc icd10_svc reward_svc rl_loop_svc
wait_all_services

docker compose ps > "$run_root/baseline_compose_ps.txt"
nvidia-smi > "$run_root/baseline_preflight_nvidia.txt"
free -h > "$run_root/baseline_preflight_free.txt"

baseline_start_epoch="$(date +%s)"
"$PYTHON_BIN" scripts/full_dataset_train_eval.py \
  --max-notes 100 \
  --run-id "$baseline_run_id" \
  --batch-size 8 \
  --train-every 8 \
  --progress-every 5 \
  --request-timeout 1800 \
  | tee "$run_root/baseline_100_pipeline.log"
baseline_end_epoch="$(date +%s)"

curl -fsS http://localhost:8004/status > "$run_root/baseline_rl_status.json"
curl -fsS http://localhost:8004/checkpoint > "$run_root/baseline_rl_checkpoint.json"
curl -fsS http://localhost:8002/observability > "$run_root/baseline_reward_observability.json"

log "phase=baseline complete elapsed_sec=$((baseline_end_epoch - baseline_start_epoch))"

log "phase=distributed select_max_world_size"
gpu_count="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ "$gpu_count" -lt 2 ]]; then
  log "error insufficient_gpu_count=$gpu_count"
  exit 1
fi

selected_world_size=0
selected_visible=""
for ws in $(seq "$gpu_count" -1 2); do
  visible="$(visible_devices_for_ws "$ws")"
  if probe_world_size "$ws" "$visible"; then
    selected_world_size="$ws"
    selected_visible="$visible"
    break
  fi
done

if [[ "$selected_world_size" -le 1 ]]; then
  log "error no_distributed_world_size_succeeded"
  exit 1
fi

log "phase=distributed selected_world_size=$selected_world_size selected_visible=$selected_visible"
DIST_WORLD_SIZE="$selected_world_size" DIST_VISIBLE_DEVICES="$selected_visible" \
  docker compose -f docker-compose.yml -f docker-compose.rl-distributed-max.yml \
  up -d --build rl_loop_svc
wait_all_services

docker compose ps > "$run_root/distributed_compose_ps.txt"
nvidia-smi > "$run_root/distributed_preflight_nvidia.txt"
free -h > "$run_root/distributed_preflight_free.txt"

distributed_start_epoch="$(date +%s)"
"$PYTHON_BIN" scripts/full_dataset_train_eval.py \
  --max-notes 100 \
  --run-id "$distributed_run_id" \
  --batch-size 8 \
  --train-every 8 \
  --progress-every 5 \
  --request-timeout 1800 \
  | tee "$run_root/distributed_100_pipeline.log"
distributed_end_epoch="$(date +%s)"

curl -fsS http://localhost:8004/status > "$run_root/distributed_rl_status.json"
curl -fsS http://localhost:8004/checkpoint > "$run_root/distributed_rl_checkpoint.json"
curl -fsS http://localhost:8002/observability > "$run_root/distributed_reward_observability.json"

"$PYTHON_BIN" scripts/compare_100note_runs.py \
  --baseline-log "$run_root/baseline_100_pipeline.log" \
  --distributed-log "$run_root/distributed_100_pipeline.log" \
  --baseline-world-size 1 \
  --distributed-world-size "$selected_world_size" \
  --output-json "$run_root/benchmark_comparison.json" \
  --output-md "$run_root/benchmark_comparison.md"

cat > "$run_root/run_manifest.txt" <<EOF
run_root=$run_root
baseline_run_id=$baseline_run_id
distributed_run_id=$distributed_run_id
baseline_elapsed_sec=$((baseline_end_epoch - baseline_start_epoch))
distributed_elapsed_sec=$((distributed_end_epoch - distributed_start_epoch))
selected_world_size=$selected_world_size
selected_visible_devices=$selected_visible
EOF

log "phase=distributed complete elapsed_sec=$((distributed_end_epoch - distributed_start_epoch))"
log "done run_root=$run_root report_json=$run_root/benchmark_comparison.json report_md=$run_root/benchmark_comparison.md"
