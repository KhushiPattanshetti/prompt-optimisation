from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RL_", case_sensitive=False)

    rollouts_dir: Path = Path(__file__).resolve().parents[1] / "rollouts"
    checkpoints_dir: Path = Path(
        os.environ.get(
            "RL_CHECKPOINT_DIR",
            str(Path(__file__).resolve().parents[2] / "rl_checkpoints"),
        )
    )

    gamma: float = 0.99
    lam: float = 0.95
    epsilon: float = 0.2
    value_coef: float = 0.5
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    beta: float = 0.01
    max_abs_kl_for_update: float = 100.0
    normalize_rewards: bool = True

    concept_reward_alpha: float = 0.5
    final_reward_beta: float = 0.5
    hybrid_advantage_lambda: float = 0.6
    grpo_enabled: bool = True
    grpo_min_group_size: int = 3
    grpo_group_size: int = 3
    hybrid_max_abs_kl_for_update: float = 5.0

    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    ppo_min_effective_batch_size: int = 8
    ppo_epochs: int = 3
    learning_rate: float = 3e-5
    lr_warmup_ratio: float = 0.1
    lr_min_ratio: float = 0.1
    max_checkpoints: int = 5
    ratio_clip_max: float = 10.0

    model_name: str = "ishanmane/phi3-rewriter-sft"
    hidden_size: int = 3072
    policy_cuda_device: int = 0
    reference_cuda_device: int = 0
    startup_log_gpu_inventory: bool = True

    distributed_enabled: bool = False
    distributed_world_size: int = 2
    distributed_min_free_ram_gb: float = 64.0
    distributed_min_free_vram_gb_per_gpu: float = 8.0
    distributed_launch_timeout_seconds: int = 7200

    ppo_advantage_zero_epsilon: float = 1e-8
    ppo_debug_mode: bool = False
    ppo_debug_disable_kl_skip: bool = False
    ppo_debug_disable_invalid_span_skip: bool = False
    ppo_debug_disable_reward_normalization: bool = False
    ppo_debug_clamp_advantages: bool = False
    ppo_debug_disable_rollout_filters: bool = False

    poll_interval_seconds: float = 5.0

    # trajectory_store_svc integration
    # When set, the RolloutLoader will also poll this directory for rollout JSONL
    # files written by trajectory_store_svc (i.e. its rollouts_store/ output).
    trajectory_store_rollouts_dir: Optional[Path] = None
    # Base URL of the trajectory_store_svc HTTP service.  Used by the status
    # endpoint to surface trajectory store health alongside training state.
    trajectory_store_url: str = "http://localhost:8200"


settings = Settings()
