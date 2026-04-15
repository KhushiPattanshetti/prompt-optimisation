"""
pipeline_logger/mode_resolver.py

Derives the human-readable mode labels from rl_loop_svc settings.

Mode 1 (RL algorithm):
    PPO    — grpo_enabled=False
    GRPO   — grpo_enabled=True, no value estimates
    Hybrid — grpo_enabled=True, value estimates present

Mode 2 (batch strategy):
    SBMi (SingleBatch-MultiIter) — ppo_epochs > 1
    MBSi (MultiBatch-SingleIter) — ppo_epochs == 1
"""

from __future__ import annotations


def resolve_mode1(grpo_enabled: bool, has_value_estimates: bool) -> str:
    if grpo_enabled and has_value_estimates:
        return "Hybrid"
    if grpo_enabled:
        return "GRPO"
    return "PPO"


def resolve_mode2(ppo_epochs: int) -> str:
    return "SBMi" if ppo_epochs > 1 else "MBSi"
