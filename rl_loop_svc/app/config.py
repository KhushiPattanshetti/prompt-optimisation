"""
Centralised configuration for the RL training microservice.

All values can be overridden using environment variables thanks to
Pydantic's BaseSettings.  Prefix every env-var with RL_ to avoid
collisions with other services.

Example:
    RL_GAMMA=0.98 uvicorn app.main:app --reload
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RL_", case_sensitive=False)

    # ── Paths ──────────────────────────────────────────────────────────────
    rollouts_dir: Path = Path(__file__).resolve().parents[1] / "rollouts"
    checkpoints_dir: Path = Path(__file__).resolve().parents[2] / "rl_checkpoints"

    # ── PPO hyper-parameters ───────────────────────────────────────────────
    gamma: float = 0.99  # discount factor
    lam: float = 0.95  # GAE lambda
    epsilon: float = 0.2  # PPO clip ratio
    value_coef: float = 0.5  # value-loss coefficient
    entropy_coef: float = 0.01  # entropy bonus coefficient

    # ── KL regularisation ──────────────────────────────────────────────────
    beta: float = 0.01  # KL penalty coefficient

    # ── Training schedule ──────────────────────────────────────────────────
    batch_size: int = 16
    ppo_epochs: int = 3  # number of PPO update epochs per rollout batch
    learning_rate: float = 3e-5
    max_checkpoints: int = 5  # keep only the N most recent checkpoints

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "gpt2"  # HuggingFace model identifier
    hidden_size: int = 768  # dimension of the policy hidden state

    # ── Lifecycle ──────────────────────────────────────────────────────────
    poll_interval_seconds: float = 5.0  # how often IDLE re-checks for new rollouts


# Singleton used across the application
settings = Settings()
