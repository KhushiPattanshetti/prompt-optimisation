#!/usr/bin/env bash
# =============================================================================
# run_schedule_t2s2_t3r1.sh
#
# Autonomous 3-phase execution schedule:
#   Phase 1  →  Tier 2 Session 2  (resume from step=804 checkpoint)
#   Gate     →  System health check + checkpoint validation
#   Phase 2  →  Tier 3 Run 1 prerequisites (distributed overlay)
#   Phase 3  →  Tier 3 Run 1 (first 120-min session)
#
# Run ONLY inside a tmux session from the project root:
#   cd /raid/adityasd314/temp/prompt-opt-v3/prompt-optimisation
#   source .venv/bin/activate
#   bash scripts/run_schedule_t2s2_t3r1.sh 2>&1 | tee run_results/schedule_$(date -u +%Y%m%dT%H%M%SZ).log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() { log "FATAL: $*" >&2; exit 1; }

check_healthy() {
    local label="$1"; local port="$2"; local path="${3:-/health}"
    local response
    response=$(curl -sf --max-time 10 "http://localhost:${port}${path}") \
        || { log "UNHEALTHY: ${label} (:${port}${path})"; return 1; }
    log "HEALTHY: ${label} (:${port}) — $(echo "$response" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status","ok"))' 2>/dev/null || echo 'ok')";
    return 0
}

wait_healthy() {
    local label="$1"; local port="$2"; local path="${3:-/health}"; local timeout_sec="${4:-300}"
    log "Waiting for ${label} (:${port}) to become healthy (timeout ${timeout_sec}s)..."
    local elapsed=0
    while ! check_healthy "$label" "$port" "$path" 2>/dev/null; do
        sleep 10; elapsed=$((elapsed + 10))
        [[ $elapsed -ge $timeout_sec ]] && fail "${label} did not become healthy within ${timeout_sec}s"
    done
}

gpu_snap() {
    log "GPU snapshot:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits \
        | while IFS=',' read -r idx used total util; do
            printf "  GPU %s: %s/%s MiB  util=%s%%\n" "$idx" "$(echo "$used" | tr -d ' ')" "$(echo "$total" | tr -d ' ')" "$(echo "$util" | tr -d ' ')"
          done
}

validate_ckpt() {
    # Returns 0 (ok), 1 (warning — continue), exits script on KL>3.0 (abort)
    local ckpt_dir="${PROJECT_ROOT}/rl_checkpoints"
    local latest
    latest=$(ls -t "${ckpt_dir}"/*.json 2>/dev/null | head -1 || true)

    if [[ -z "$latest" ]]; then
        log "WARNING: No checkpoint found in ${ckpt_dir} — will start from step 0"
        return 0
    fi

    python3 - "$latest" <<'PYEOF'
import sys, json, math
path = sys.argv[1]
d = json.load(open(path))
step = d.get("training_step", "?")
kl   = d.get("kl_divergence", 0.0)
loss = d.get("last_loss", float("nan"))
log_str = f"  Checkpoint: step={step}  kl={kl:.4f}  loss={loss:.4f}"
print(log_str)
if math.isnan(kl) or math.isinf(kl) or kl > 3.0:
    print(f"ABORT: kl={kl:.4f} exceeds 3.0 (policy collapsed). Wipe rl_checkpoints before retrying.")
    sys.exit(2)
if kl > 1.0:
    print(f"WARNING: kl={kl:.4f} is in the warning zone (1.0–3.0). Continuing with T3 config which has KL_ceil=10.")
if math.isnan(loss) or math.isinf(loss):
    print(f"WARNING: loss={loss} is non-finite — monitor closely")
PYEOF
    local ret=$?
    if [[ $ret -eq 2 ]]; then
        fail "KL abort gate triggered. Do not proceed to Tier 3 until checkpoint is healthy."
    fi
    return 0
}

backup_ckpts() {
    local suffix="$1"
    local src="${PROJECT_ROOT}/rl_checkpoints"
    local dst="${PROJECT_ROOT}/rl_checkpoints_bak_${suffix}_$(date +%Y%m%d_%H%M%S)"
    if [[ -d "$src" ]] && [[ -n "$(ls -A "$src" 2>/dev/null)" ]]; then
        cp -r "$src" "$dst"
        log "Checkpoint backup: ${dst}"
    else
        log "No checkpoints to back up (empty dir)"
    fi
}

# ---------------------------------------------------------------------------
# PRE-FLIGHT: verify all services are up
# ---------------------------------------------------------------------------

log "============================================================"
log " SCHEDULE: T2S2 → System Check → T3 Prerequisites → T3R1"
log "============================================================"
log ""
log "--- PRE-FLIGHT: Service health check ---"

for spec in "dataset_svc:9003:/health" "rewriter_svc:9000:/health" "rewriter_svc_b:9005:/health" \
            "icd10_svc:9001:/health" "reward_svc:9002:/health" "rl_loop_svc:9004:/status"; do
    IFS=':' read -r svc port path <<< "$spec"
    check_healthy "$svc" "$port" "$path" || fail "Service ${svc} is not healthy. Fix before running this schedule."
done

gpu_snap

# ---------------------------------------------------------------------------
# PHASE 1: Tier 2 Session 2
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log " PHASE 1: Tier 2 Session 2"
log "============================================================"
log "  LR=3e-6, KL_ceil=1.0, max-notes=500, timebox=360min"
log "  Resume from step=804 checkpoint (REWRITER_LOAD_LOCAL_CHECKPOINTS=true)"
log "  NOTE: max-notes=500 is intentional — pipeline has no offset/resume;"
log "        notes 0-149 will be re-processed with the improved policy."
log ""

# Export env vars picked up by docker-compose for rl_loop_svc and rewriters
export RL_LEARNING_RATE=3e-6
export RL_MAX_ABS_KL_FOR_UPDATE=1.0
export RL_GRPO_MIN_GROUP_SIZE=3
export RL_BATCH_SIZE=2
export RL_GRADIENT_ACCUMULATION_STEPS=8
export RL_SBMI_EPOCHS=4
export RL_PPO_DEBUG_DISABLE_ROLLOUT_FILTERS=false
export RL_DISTRIBUTED_ENABLED=false
export REWRITER_LOAD_LOCAL_CHECKPOINTS=true   # load LoRA from rl_checkpoints/

log "Recreating rl_loop_svc + rewriter_svc + rewriter_svc_b with updated config..."
docker compose up --force-recreate -d rl_loop_svc rewriter_svc rewriter_svc_b

log "Waiting for services to become healthy after recreation..."
wait_healthy "rl_loop_svc"    9004 "/status"  360
wait_healthy "rewriter_svc"   9000 "/health"  400
wait_healthy "rewriter_svc_b" 9005 "/health"  400
gpu_snap

T2S2_RUN_ID="tier_2_run_6_session2_$(date -u +%Y%m%dT%H%M%SZ)"
log "Starting Tier 2 Session 2: run_id=${T2S2_RUN_ID}"

python -u scripts/full_dataset_train_eval.py \
    --dataset-url  http://localhost:9003 \
    --rewriter-urls http://localhost:9000,http://localhost:9005 \
    --icd10-url    http://localhost:9001 \
    --reward-url   http://localhost:9002 \
    --rl-url       http://localhost:9004 \
    --run-id       "${T2S2_RUN_ID}" \
    --results-dir  "run_results/tier_2_session2" \
    --max-notes    500 \
    --train-ratio  0.8 \
    --val-ratio    0.1 \
    --train-every  16 \
    --rewrites-per-note 6 \
    --grpo-group-size 3 \
    --rewriter-workers 4 \
    --icd-workers  4 \
    --stream-note-wise \
    --trajectory-flush-size 6 \
    --guard-min-both-parse-rate 0.2 \
    --guard-max-rollout-drop-rate 0.3 \
    --max-run-minutes 360 \
    --checkpoint-every-minutes 30 \
    2>&1 | tee "run_results/${T2S2_RUN_ID}.log"

log "Tier 2 Session 2 completed."

# ---------------------------------------------------------------------------
# SYSTEM CHECK GATE
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log " SYSTEM CHECK GATE"
log "============================================================"

log "--- Checkpoint health ---"
validate_ckpt
# validate_ckpt exits the script (via fail) if KL > 3.0

log "--- Disk space ---"
df -h "${PROJECT_ROOT}" || true

log "--- Backing up checkpoints before T3 ---"
backup_ckpts "pre_t3r1"

log "--- GPU state ---"
gpu_snap

# ---------------------------------------------------------------------------
# PHASE 2: Tier 3 Prerequisites — Recreate rl_loop_svc with distributed overlay
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log " PHASE 2: Tier 3 Prerequisites"
log "============================================================"
log "  Applying docker-compose.rl-distributed-max.yml overlay"
log "  Sets NVIDIA_VISIBLE_DEVICES=0,1,2,3 and RL_DISTRIBUTED_ENABLED=true"
log "  rewriter_svc, icd10_svc, reward_svc, dataset_svc remain running"
log "  (no model reload; saves ~15 min)"
log ""

export RL_LEARNING_RATE=2e-5
export RL_MAX_ABS_KL_FOR_UPDATE=10
export RL_GRADIENT_ACCUMULATION_STEPS=16
export RL_BATCH_SIZE=2
export RL_SBMI_EPOCHS=4
export RL_GRPO_MIN_GROUP_SIZE=3
export RL_DISTRIBUTED_ENABLED=true
export REWRITER_LOAD_LOCAL_CHECKPOINTS=true

log "Recreating rl_loop_svc with distributed overlay..."
docker compose \
    -f docker-compose.yml \
    -f docker-compose.rl-distributed-max.yml \
    up --force-recreate -d rl_loop_svc

# Distributed mode takes longer to initialise (torch.distributed rendezvous)
wait_healthy "rl_loop_svc" 9004 "/status" 600

log "Validating checkpoint health after restart..."
validate_ckpt

gpu_snap

# ---------------------------------------------------------------------------
# PHASE 3: Tier 3 Run 1 — first 120-min session
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log " PHASE 3: Tier 3 Run 1 (Session 1)"
log "============================================================"
log "  LR=2e-5, KL_ceil=10, grad_accum=16, distributed=true"
log "  --max-notes=500, --timebox=120min (timebox first session)"
log "  Additional sessions: re-run Phase 3 only with updated run-id"
log ""

T3R1_RUN_ID="tier_3_run_1_session1_$(date -u +%Y%m%dT%H%M%SZ)"
log "Starting Tier 3 Run 1 Session 1: run_id=${T3R1_RUN_ID}"

python -u scripts/full_dataset_train_eval.py \
    --dataset-url  http://localhost:9003 \
    --rewriter-urls http://localhost:9000,http://localhost:9005 \
    --icd10-url    http://localhost:9001 \
    --reward-url   http://localhost:9002 \
    --rl-url       http://localhost:9004 \
    --run-id       "${T3R1_RUN_ID}" \
    --results-dir  "run_results/tier_3_run_1" \
    --max-notes    500 \
    --train-ratio  0.8 \
    --val-ratio    0.1 \
    --train-every  32 \
    --rewrites-per-note 6 \
    --grpo-group-size 3 \
    --rewriter-workers 8 \
    --icd-workers  4 \
    --stream-note-wise \
    --trajectory-flush-size 6 \
    --guard-min-both-parse-rate 0.3 \
    --guard-max-rollout-drop-rate 0.3 \
    --max-run-minutes 120 \
    --checkpoint-every-minutes 30 \
    2>&1 | tee "run_results/${T3R1_RUN_ID}.log"

log "Tier 3 Run 1 Session 1 completed."

# ---------------------------------------------------------------------------
# POST-RUN
# ---------------------------------------------------------------------------

log ""
log "============================================================"
log " SCHEDULE COMPLETE"
log "============================================================"

log "--- Final checkpoint health ---"
validate_ckpt || true   # don't abort post-run; just report

log "--- Final GPU state ---"
gpu_snap

log ""
log "To run subsequent Tier 3 sessions:"
log "  export REWRITER_LOAD_LOCAL_CHECKPOINTS=true"
log "  export RL_LEARNING_RATE=2e-5"
log "  export RL_MAX_ABS_KL_FOR_UPDATE=10"
log "  export RL_GRADIENT_ACCUMULATION_STEPS=16"
log "  docker compose -f docker-compose.yml -f docker-compose.rl-distributed-max.yml \\"
log "    up --force-recreate -d rl_loop_svc"
log "  python -u scripts/full_dataset_train_eval.py \\"
log "    --rewriter-urls http://localhost:9000,http://localhost:9005 \\"
log "    --max-notes 500 --max-run-minutes 120 --train-every 32 \\"
log "    --run-id tier_3_run_1_sessionN_\$(date -u +%Y%m%dT%H%M%SZ) \\"
log "    --results-dir run_results/tier_3_run_1"
log ""
log "Done."
