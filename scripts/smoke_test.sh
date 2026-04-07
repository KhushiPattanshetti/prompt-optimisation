#!/bin/bash
set -e

check() {
  local name=$1
  local url=$2
  local expected=$3
  response=$(curl -sf "$url" 2>/dev/null || echo "FAILED")
  if echo "$response" | grep -q "$expected"; then
    echo "PASS: $name"
  else
    echo "FAIL: $name — response: $response"
    exit 1
  fi
}

check "dataset_svc"  "http://localhost:8003/health"  '"ok"'
check "rewriter_svc" "http://localhost:8000/docs"     '"openapi"'
check "icd10_svc"    "http://localhost:8001/health"   '"ok"'
check "reward_svc"   "http://localhost:8002/health"   '"ok"'
check "rl_loop_svc"  "http://localhost:8004/status"   '"trainer_state"'

echo "All services healthy."
