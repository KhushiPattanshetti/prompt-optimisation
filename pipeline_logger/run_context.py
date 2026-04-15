"""
pipeline_logger/run_context.py

Manages the single shared run_id and the on-disk directory tree.

The run_id is:
    YYYY-MM-DD_HHMMSS_<mode1>_<mode2>
e.g. "2026-04-15_143022_Hybrid_SBMi"

Directory layout (under PIPELINE_LOGS_DIR):
    pipeline_logs/
    └── <run_id>/
        ├── system_exec.csv
        ├── batch_summaries/
        └── per_service/

init_run() is called ONCE by rl_loop_svc at startup.
Every other service calls get_run_context() to learn (base_dir, run_id).
They derive their log paths from PIPELINE_LOGS_DIR + PIPELINE_RUN_ID env vars.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


_LOGS_DIR_ENV = "PIPELINE_LOGS_DIR"
_RUN_ID_ENV = "PIPELINE_RUN_ID"

# Module-level cache so rl_loop_svc can call get_run_context() without re-reading env
_cached_base: Path | None = None
_cached_run_id: str | None = None


def init_run(
    mode1: str, mode2: str, base_dir: str | Path | None = None
) -> tuple[Path, str]:
    """
    Create the directory tree for a new pipeline run and return (base, run_id).

    Args:
        mode1:    RL algorithm label — "PPO", "GRPO", or "Hybrid".
        mode2:    Batch strategy label — "SBMi" (SingleBatch-MultiIter)
                  or "MBSi" (MultiBatch-SingleIter).
        base_dir: Override for PIPELINE_LOGS_DIR.  Falls back to env var,
                  then to "./pipeline_logs".
    """
    global _cached_base, _cached_run_id

    base = Path(base_dir or os.environ.get(_LOGS_DIR_ENV, "./pipeline_logs"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{ts}_{mode1}_{mode2}"

    run_dir = base / run_id
    (run_dir / "batch_summaries").mkdir(parents=True, exist_ok=True)
    (run_dir / "per_service").mkdir(parents=True, exist_ok=True)

    _cached_base = base
    _cached_run_id = run_id

    # Write env vars so child processes and sub-imports see them
    os.environ[_LOGS_DIR_ENV] = str(base)
    os.environ[_RUN_ID_ENV] = run_id

    return base, run_id


def get_run_context() -> tuple[Path | None, str | None]:
    """
    Return (base_dir, run_id) from module cache or environment variables.
    Returns (None, None) when no run has been initialised — callers should
    skip pretty-logging gracefully in that case.
    """
    global _cached_base, _cached_run_id

    if _cached_base is not None and _cached_run_id is not None:
        return _cached_base, _cached_run_id

    base_str = os.environ.get(_LOGS_DIR_ENV)
    run_id = os.environ.get(_RUN_ID_ENV)
    if base_str and run_id:
        _cached_base = Path(base_str)
        _cached_run_id = run_id
        return _cached_base, run_id

    return None, None


def get_run_dir() -> Path | None:
    """Convenience: return the full run directory path or None."""
    base, run_id = get_run_context()
    if base is None or run_id is None:
        return None
    return base / run_id
