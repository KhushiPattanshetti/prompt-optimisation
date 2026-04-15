"""
config.py — Central configuration for trajectory_store_svc.

All runtime behaviour is driven by this module.  Values can be overridden
by environment variables (prefixed TRAJ_) so the service is
twelve-factor-app friendly without adding third-party deps.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(f"TRAJ_{key}", default)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(f"TRAJ_{key}")
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(f"TRAJ_{key}", default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(f"TRAJ_{key}", default))


@dataclass
class Config:
    # ── Filesystem paths ──────────────────────────────────────────────────────
    base_dir: Path = field(
        default_factory=lambda: Path(_env("BASE_DIR", str(Path(__file__).parent)))
    )

    @property
    def rollouts_dir(self) -> Path:
        p = self.base_dir / _env("ROLLOUTS_DIR", "rollouts_store")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def prepared_batches_dir(self) -> Path:
        p = self.base_dir / _env("BATCHES_DIR", "prepared_batches")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seen_files_path(self) -> Path:
        return self.rollouts_dir / "seen_files.json"

    @property
    def segment_offsets_path(self) -> Path:
        return self.rollouts_dir / "segment_offsets.json"

    # ── Batch parameters ──────────────────────────────────────────────────────
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 8))

    # ── Reward handling ───────────────────────────────────────────────────────
    normalize_rewards: bool = field(
        default_factory=lambda: _env_bool("NORMALIZE_REWARDS", False)
    )
    reward_already_normalized: bool = field(
        default_factory=lambda: _env_bool("REWARD_ALREADY_NORMALIZED", True)
    )

    # ── GRPO / advantage ──────────────────────────────────────────────────────
    advantage_mode: str = field(
        default_factory=lambda: _env("ADVANTAGE_MODE", "grpo")
    )  # "standard" | "grpo" | "hybrid"

    grpo_enabled: bool = field(default_factory=lambda: _env_bool("GRPO_ENABLED", True))
    grpo_group_size: int = field(default_factory=lambda: _env_int("GRPO_GROUP_SIZE", 4))
    grpo_min_group_size: int = field(
        default_factory=lambda: _env_int("GRPO_MIN_GROUP_SIZE", 2)
    )
    hybrid_advantage_lambda: float = field(
        default_factory=lambda: _env_float("HYBRID_LAMBDA", 0.5)
    )

    # ── Batch reuse ───────────────────────────────────────────────────────────
    batch_reuse_mode: str = field(
        default_factory=lambda: _env("BATCH_REUSE_MODE", "single_pass")
    )  # "repeat" | "single_pass"
    max_batch_reuse_count: int = field(
        default_factory=lambda: _env_int("MAX_BATCH_REUSE_COUNT", 3)
    )

    # ── API ───────────────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8200))

    # ── Simulation defaults ───────────────────────────────────────────────────
    sim_num_rollouts: int = field(
        default_factory=lambda: _env_int("SIM_NUM_ROLLOUTS", 32)
    )
    sim_reward_mean: float = field(
        default_factory=lambda: _env_float("SIM_REWARD_MEAN", 0.5)
    )
    sim_reward_std: float = field(
        default_factory=lambda: _env_float("SIM_REWARD_STD", 0.2)
    )


# Singleton used across the service
cfg = Config()
