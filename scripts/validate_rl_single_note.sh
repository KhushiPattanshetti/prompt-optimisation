#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 ]]; then
  echo "args_ignored=true"
fi

for dep in curl jq docker; do
  if ! command -v "$dep" >/dev/null 2>&1; then
    echo "missing_dependency=$dep"
    exit 1
  fi
done

wait_http_ok() {
  local service_name="$1"
  local url="$2"
  local max_attempts="${3:-120}"
  local delay_seconds="${4:-2}"

  for attempt in $(seq 1 "$max_attempts"); do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)"
    if [[ "$code" =~ ^2[0-9][0-9]$ ]]; then
      echo "${service_name}_ready=true"
      return 0
    fi
    sleep "$delay_seconds"
  done

  echo "${service_name}_ready=false"
  return 1
}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="logs/manual_validation"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/rl_recovery_single_note_${TS}.log"

exec > >(tee "$LOG") 2>&1

echo "[0] Ensure required services are running"
docker compose up -d dataset_svc rewriter_svc icd10_svc reward_svc rl_loop_svc >/dev/null

wait_http_ok "dataset_svc" "http://localhost:8003/health"
wait_http_ok "rewriter_svc" "http://localhost:8000/docs"
wait_http_ok "icd10_svc" "http://localhost:8001/health"
wait_http_ok "reward_svc" "http://localhost:8002/health"
wait_http_ok "rl_loop_svc" "http://localhost:8004/status"

echo "timestamp=$TS"
echo "log_file=$LOG"

echo "[1] RL status before"
curl -sS http://localhost:8004/status | tee "$OUTDIR/rl_status_before_${TS}.json"
STEP_BEFORE="$(jq -r '.training_step' "$OUTDIR/rl_status_before_${TS}.json")"
ROLLOUTS_BEFORE_STATUS="$(jq -r '.rollouts_loaded' "$OUTDIR/rl_status_before_${TS}.json")"
echo "training_step_before=$STEP_BEFORE"
echo "rollouts_loaded_before_status=$ROLLOUTS_BEFORE_STATUS"

COUNT_BEFORE="$(find rl_loop_svc/rollouts -maxdepth 1 \( -name 'rollout_batch_*.json' -o -name 'rollout_segment_*.jsonl' \) | wc -l | tr -d ' ')"
echo "rollout_artifacts_before=$COUNT_BEFORE"

echo "[2] Pull one note from dataset"
curl -sS "http://localhost:8003/batch?offset=0&size=1" \
  | tee "$OUTDIR/dataset_batch_${TS}.json" >/dev/null

NOTE_ID="$(jq -r '.batch[0].note_id // empty' "$OUTDIR/dataset_batch_${TS}.json")"
NOTE_TEXT="$(jq -r '.batch[0].text // empty' "$OUTDIR/dataset_batch_${TS}.json")"
if [[ -z "$NOTE_ID" || -z "$NOTE_TEXT" ]]; then
  echo "dataset_batch_missing_note"
  exit 1
fi
echo "note_id=$NOTE_ID"

echo "[3] Rewriter inference"
jq -n --arg note_id "$NOTE_ID" --arg clinical_note "$NOTE_TEXT" \
  '{note_id:$note_id, clinical_note:$clinical_note}' \
  > "$OUTDIR/rewriter_request_${TS}.json"

curl -sS -X POST http://localhost:8000/rewrite_prompt \
  -H 'Content-Type: application/json' \
  -d @"$OUTDIR/rewriter_request_${TS}.json" \
  | tee "$OUTDIR/rewriter_response_${TS}.json" >/dev/null

ORIG_PROMPT="$(jq -r '.original_prompt // empty' "$OUTDIR/rewriter_response_${TS}.json")"
REWRITTEN_PROMPT="$(jq -r '.rewritten_prompt // empty' "$OUTDIR/rewriter_response_${TS}.json")"
LOG_PROB_OLD="$(jq -r '.log_prob_old // empty' "$OUTDIR/rewriter_response_${TS}.json")"
VALUE_ESTIMATE="$(jq -r '.value_estimate // empty' "$OUTDIR/rewriter_response_${TS}.json")"
if [[ -z "$ORIG_PROMPT" ]]; then
  ORIG_PROMPT="$NOTE_TEXT"
fi

if [[ -z "$REWRITTEN_PROMPT" || -z "$LOG_PROB_OLD" || -z "$VALUE_ESTIMATE" ]]; then
  echo "rewriter_missing_fields"
  exit 1
fi
echo "rewriter_ok=true"
echo "rewritten_prompt_chars=$(printf '%s' "$REWRITTEN_PROMPT" | wc -c | tr -d ' ')"
echo "log_prob_old=$LOG_PROB_OLD"
echo "value_estimate=$VALUE_ESTIMATE"

echo "[4] ICD10 inference (triggers queued reward->rollout forwarding)"
jq -n \
  --arg note_id "$NOTE_ID" \
  --arg original_prompt "$ORIG_PROMPT" \
  --arg rewritten_prompt "$REWRITTEN_PROMPT" \
  --argjson log_prob_old "$LOG_PROB_OLD" \
  --argjson value_estimate "$VALUE_ESTIMATE" \
  '{note_id:$note_id, original_prompt:$original_prompt, rewritten_prompt:$rewritten_prompt, log_prob_old:$log_prob_old, value_estimate:$value_estimate}' \
  > "$OUTDIR/icd_request_${TS}.json"

curl -sS -X POST http://localhost:8001/generate_codes \
  -H 'Content-Type: application/json' \
  -d @"$OUTDIR/icd_request_${TS}.json" \
  | tee "$OUTDIR/icd_response_${TS}.json" >/dev/null

echo "icd_parsing_success=$(jq -r '.parsing_success' "$OUTDIR/icd_response_${TS}.json")"
echo "icd_enh_count=$(jq -r '.enh_codes | length' "$OUTDIR/icd_response_${TS}.json")"
echo "icd_org_count=$(jq -r '.org_codes | length' "$OUTDIR/icd_response_${TS}.json")"
echo "icd_gt_count=$(jq -r '.gt_codes | length' "$OUTDIR/icd_response_${TS}.json")"

echo "[5] Wait for queued reward->rl forwarding"
sleep 6

COUNT_AFTER="$(find rl_loop_svc/rollouts -maxdepth 1 \( -name 'rollout_batch_*.json' -o -name 'rollout_segment_*.jsonl' \) | wc -l | tr -d ' ')"
echo "rollout_artifacts_after=$COUNT_AFTER"
if (( COUNT_AFTER > COUNT_BEFORE )); then
  echo "rollout_forwarding_result=accepted"
else
  echo "rollout_forwarding_result=not_observed"
fi

LATEST_ROLLOUT="$(ls -1t rl_loop_svc/rollouts/rollout_batch_*.json rl_loop_svc/rollouts/rollout_segment_*.jsonl 2>/dev/null | head -n1 || true)"
if [[ -n "$LATEST_ROLLOUT" ]]; then
  echo "latest_rollout_file=$LATEST_ROLLOUT"
fi

echo "[6] Trigger training"
curl -sS -X POST http://localhost:8004/train | tee "$OUTDIR/rl_train_trigger_${TS}.json"

echo "[7] Poll status for training step change"
STEP_AFTER="$STEP_BEFORE"
for i in 1 2 3 4 5 6 7 8; do
  sleep 5
  curl -sS http://localhost:8004/status | tee "$OUTDIR/rl_status_poll_${TS}_${i}.json" >/dev/null
  STEP_AFTER="$(jq -r '.training_step' "$OUTDIR/rl_status_poll_${TS}_${i}.json")"
  ROLLOUTS_AFTER_STATUS="$(jq -r '.rollouts_loaded' "$OUTDIR/rl_status_poll_${TS}_${i}.json")"
  STATE_NOW="$(jq -r '.trainer_state' "$OUTDIR/rl_status_poll_${TS}_${i}.json")"
  echo "poll=$i trainer_state=$STATE_NOW training_step=$STEP_AFTER rollouts_loaded=$ROLLOUTS_AFTER_STATUS"
  if (( STEP_AFTER > STEP_BEFORE )); then
    break
  fi
done

echo "training_step_after=$STEP_AFTER"
if (( STEP_AFTER > STEP_BEFORE )); then
  echo "training_step_incremented=true"
else
  echo "training_step_incremented=false"
fi

echo "[8] Final status snapshot"
curl -sS http://localhost:8004/status | tee "$OUTDIR/rl_status_after_${TS}.json"

echo "transcript_complete=true"
