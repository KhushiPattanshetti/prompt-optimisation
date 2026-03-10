"""
run_training.py — manual training runner script.

Loads rollouts from the rollouts/ directory and runs one PPO cycle.

Usage:
    python scripts/run_training.py [--model gpt2] [--steps 1]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from models.policy_model import PolicyModel
from models.reference_model import ReferenceModel
from models.value_head import ValueHead
from rl.training_loop import TrainingLoop
from storage.checkpoint_manager import CheckpointManager
from storage.rollout_loader import RolloutLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually run one PPO training cycle")
    parser.add_argument(
        "--model", default=settings.model_name, help="HuggingFace model name"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of collect→train→ckpt cycles"
    )
    args = parser.parse_args()

    settings.rollouts_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising models …")
    policy_model = PolicyModel(model_name=args.model)
    reference_model = ReferenceModel(model_name=args.model)
    value_head = ValueHead(hidden_size=settings.hidden_size).to(policy_model.device)

    training_loop = TrainingLoop(
        rollout_loader=RolloutLoader(rollouts_dir=settings.rollouts_dir),
        checkpoint_manager=CheckpointManager(
            checkpoints_dir=settings.checkpoints_dir,
            max_checkpoints=settings.max_checkpoints,
        ),
        policy_model=policy_model,
        reference_model=reference_model,
        value_head=value_head,
    )

    for step in range(args.steps):
        logger.info("=== Cycle %d / %d ===", step + 1, args.steps)
        ran = training_loop.run_once()
        if not ran:
            logger.info("No new rollouts found. Stopping.")
            break

    logger.info("Done. Training step: %d", training_loop.training_step)


if __name__ == "__main__":
    main()
