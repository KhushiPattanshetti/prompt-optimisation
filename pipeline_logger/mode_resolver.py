"""
pipeline_logger/mode_resolver.py

Derives the human-readable mode labels from rl_loop_svc settings.

Mode 1 (RL algorithm):
    Hybrid only — requires both GRPO and value estimates.

Mode 2 (batch strategy):
    SBMi (SingleBatch-MultiIter) — ppo_epochs > 1
    MBSi (MultiBatch-SingleIter) — ppo_epochs == 1
"""

from __future__ import annotations

import os


def resolve_mode1(grpo_enabled: bool, has_value_estimates: bool) -> str:
    if not grpo_enabled:
        raise RuntimeError("Hybrid mode enforcement failed: grpo_enabled must be true")
    if not has_value_estimates:
        raise RuntimeError(
            "Hybrid mode enforcement failed: value estimates must be enabled"
        )
    return "Hybrid"


def resolve_mode2(ppo_epochs: int) -> str:
    mbmi_enabled = str(os.environ.get("RL_MBMI_ENABLED", "false")).strip().lower()
    if mbmi_enabled in {"1", "true", "yes", "on"}:
        return "MBMi"
    return "SBMi" if ppo_epochs > 1 else "MBSi"
