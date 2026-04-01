#!/usr/bin/env bash
# setup_env.sh — create and prepare the rl_loop_svc virtual environment
# Usage: bash scripts/setup_env.sh  (run from rl_loop_svc/ root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVC_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SVC_ROOT/venv"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt"
pip install -r "$SVC_ROOT/requirements.txt"

echo ""
echo "✅  Environment ready."
echo "    Activate with:  source $VENV_DIR/bin/activate"
echo "    Start service:  uvicorn app.main:app --reload"
