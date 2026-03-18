"""
Main training loop: orchestrates the full RL cycle.

IDLE → COLLECT → TRAIN → CHECKPOINT → IDLE → ...

FIX SUMMARY (from review):
  - checkpoint_manager.save() signature changed: now takes policy_model
    (instance) instead of policy_state_dict (raw dict), because LoRA
    adapters must be saved via PEFT's save_pretrained(), not torch.save().
  - gradient_accumulation_steps wired into PPO epoch loop.
  - batch_size reduced to 4 in config; accumulation makes effective
    batch = 16 without OOM.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from app.config import settings
from models.policy_model import PolicyModel
from models.reference_model import ReferenceModel
from models.value_head import ValueHead
from rl.advantage import compute_gae
from rl.kl_controller import KLController
from rl.lifecycle_manager import LifecycleManager, TrainerState
from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBuffer
from schemas.rollout_schema import RolloutEntry
from storage.checkpoint_manager import CheckpointManager
from storage.rollout_loader import RolloutLoader

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Ties together all RL components and exposes run_once() which
    executes one full collect → train → checkpoint cycle.
    """

    def __init__(
        self,
        rollout_loader:     RolloutLoader,
        checkpoint_manager: CheckpointManager,
        policy_model:       PolicyModel,
        reference_model:    ReferenceModel,
        value_head:         ValueHead,
    ) -> None:
        self.rollout_loader     = rollout_loader
        self.checkpoint_manager = checkpoint_manager
        self.policy_model       = policy_model
        self.reference_model    = reference_model
        self.value_head         = value_head

        self.lifecycle      = LifecycleManager()
        self.kl_controller  = KLController(beta=settings.beta)
        self.buffer         = RolloutBuffer(device=str(policy_model.device))

        self.optimizer = torch.optim.AdamW(
            list(policy_model.parameters()) + list(value_head.parameters()),
            lr=settings.learning_rate,
        )
        self.ppo_trainer = PPOTrainer(
            policy_model=policy_model,
            value_head=value_head,
            optimizer=self.optimizer,
            epsilon=settings.epsilon,
            value_coef=settings.value_coef,
            entropy_coef=settings.entropy_coef,
        )

        self.training_step:  int   = 0
        self.last_loss:      float = 0.0
        self.rollouts_loaded: int  = 0

    def run_once(self) -> bool:
        """Execute one full RL cycle if new rollouts are available.

        Returns:
            True if training was performed, False if no new rollouts.
        """
        # ── COLLECT ───────────────────────────────────────────────────────
        self.lifecycle.transition(TrainerState.COLLECT)
        entries = self.rollout_loader.load_new()

        if not entries:
            self.lifecycle.transition(TrainerState.IDLE)
            return False

        self.rollouts_loaded += len(entries)
        self._fill_buffer(entries)
        logger.info("Collected %d new rollout entries", len(entries))

        # ── TRAIN ─────────────────────────────────────────────────────────
        self.lifecycle.transition(TrainerState.TRAIN)
        self._run_ppo_epochs()

        # ── CHECKPOINT ────────────────────────────────────────────────────
        self.lifecycle.transition(TrainerState.CHECKPOINT)
        self._save_checkpoint()

        # ── IDLE ──────────────────────────────────────────────────────────
        self.lifecycle.transition(TrainerState.IDLE)
        self.buffer.clear()
        return True

    # ── Internal helpers ──────────────────────────────────────────────────

    def _fill_buffer(self, entries: list) -> None:
        self.buffer.clear()
        for entry in entries:
            self.buffer.store(
                reward=entry.reward,
                log_prob_old=entry.log_prob_old,
                value_estimate=entry.value_estimate,
                original_prompt=entry.original_prompt,
                rewritten_prompt=entry.rewritten_prompt,
            )

    def _run_ppo_epochs(self) -> None:
        """Compute GAE and run PPO update epochs with gradient accumulation."""
        rewards    = torch.tensor(list(self.buffer._rewards),    dtype=torch.float32)
        values     = torch.tensor(list(self.buffer._values),     dtype=torch.float32)
        advantages = compute_gae(rewards, values, gamma=settings.gamma, lam=settings.lam)
        batch      = self.buffer.build(advantages)

        accum_steps = settings.gradient_accumulation_steps

        for epoch in range(settings.ppo_epochs):
            all_texts = batch.original_prompts
            tokenised = self.policy_model.tokenize(all_texts)
            input_ids      = tokenised["input_ids"]
            attention_mask = tokenised.get("attention_mask")

            # Process in mini-batches of settings.batch_size
            n = len(all_texts)
            self.optimizer.zero_grad()
            accum_count = 0

            for start in range(0, n, settings.batch_size):
                end      = min(start + settings.batch_size, n)
                mb_ids   = input_ids[start:end]
                mb_mask  = attention_mask[start:end] if attention_mask is not None else None

                # Current policy log-probs and hidden states
                log_probs_new, hidden = self.policy_model(mb_ids, mb_mask)
                seq_log_prob_new      = log_probs_new.sum(dim=-1)

                # Value estimates
                values_new = self.value_head(hidden)

                # Reference log-probs (frozen)
                ref_log_prob = self.reference_model.get_sequence_log_prob(
                    mb_ids, mb_mask
                )

                # KL penalty
                kl_penalty = self.kl_controller.compute_kl(
                    seq_log_prob_new, ref_log_prob
                )

                # Entropy: mean of negative log-prob
                entropy = -log_probs_new.mean()

                # Build mini-batch slice
                from rl.rollout_buffer import RolloutBatch
                import dataclasses
                mb_batch = RolloutBatch(
                    rewards=batch.rewards[start:end],
                    log_probs_old=batch.log_probs_old[start:end],
                    values=batch.values[start:end],
                    advantages=batch.advantages[start:end],
                    returns=batch.returns[start:end],
                    original_prompts=batch.original_prompts[start:end],
                    rewritten_prompts=batch.rewritten_prompts[start:end],
                )

                components = self.ppo_trainer.update(
                    batch=mb_batch,
                    log_probs_new=seq_log_prob_new,
                    values_new=values_new,
                    entropy=entropy,
                    kl_penalty=kl_penalty,
                )
                self.last_loss   = components.total_loss
                self.training_step += 1
                accum_count += 1

                # Gradient accumulation step
                if accum_count % accum_steps == 0 or end == n:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters())
                        + list(self.value_head.parameters()),
                        max_norm=1.0,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            logger.info(
                "Epoch %d/%d | step=%d | loss=%.4f | KL=%.4f",
                epoch + 1, settings.ppo_epochs,
                self.training_step,
                self.last_loss,
                self.kl_controller.last_kl,
            )

    def _save_checkpoint(self) -> None:
        """Serialise model weights to the checkpoint directory.

        FIX: Uses updated CheckpointManager.save() which accepts
        policy_model instance and saves LoRA adapter via PEFT format.
        """
        self.checkpoint_manager.save(
            policy_model=self.policy_model,
            value_head_state_dict=self.value_head.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            training_step=self.training_step,
            extra_meta={
                "last_loss":       self.last_loss,
                "kl_divergence":   self.kl_controller.last_kl,
                "rollouts_loaded": self.rollouts_loaded,
            },
        )

# """
# Main training loop: orchestrates the full RL cycle.

# IDLE → COLLECT → TRAIN → CHECKPOINT → IDLE → ...
# """

# import logging
# from pathlib import Path
# from typing import Optional

# import torch

# from app.config import settings
# from models.policy_model import PolicyModel
# from models.reference_model import ReferenceModel
# from models.value_head import ValueHead
# from rl.advantage import compute_gae
# from rl.kl_controller import KLController
# from rl.lifecycle_manager import LifecycleManager, TrainerState
# from rl.ppo_trainer import PPOTrainer
# from rl.rollout_buffer import RolloutBuffer
# from schemas.rollout_schema import RolloutEntry
# from storage.checkpoint_manager import CheckpointManager
# from storage.rollout_loader import RolloutLoader

# logger = logging.getLogger(__name__)


# class TrainingLoop:
#     """
#     Ties together all RL components and exposes a run_once() method
#     that executes one full collect → train → checkpoint cycle.

#     Args:
#         rollout_loader:     Reads rollout JSON files.
#         checkpoint_manager: Persists model weights.
#         policy_model:       Trainable language model.
#         reference_model:    Frozen baseline language model.
#         value_head:         Trainable value estimator.
#     """

#     def __init__(
#         self,
#         rollout_loader: RolloutLoader,
#         checkpoint_manager: CheckpointManager,
#         policy_model: PolicyModel,
#         reference_model: ReferenceModel,
#         value_head: ValueHead,
#     ) -> None:
#         self.rollout_loader = rollout_loader
#         self.checkpoint_manager = checkpoint_manager
#         self.policy_model = policy_model
#         self.reference_model = reference_model
#         self.value_head = value_head

#         self.lifecycle = LifecycleManager()
#         self.kl_controller = KLController(beta=settings.beta)
#         self.buffer = RolloutBuffer(device=str(policy_model.device))

#         self.optimizer = torch.optim.AdamW(
#             list(policy_model.parameters()) + list(value_head.parameters()),
#             lr=settings.learning_rate,
#         )
#         self.ppo_trainer = PPOTrainer(
#             policy_model=policy_model,
#             value_head=value_head,
#             optimizer=self.optimizer,
#             epsilon=settings.epsilon,
#             value_coef=settings.value_coef,
#             entropy_coef=settings.entropy_coef,
#         )

#         self.training_step: int = 0
#         self.last_loss: float = 0.0
#         self.rollouts_loaded: int = 0

#     def run_once(self) -> bool:
#         """
#         Execute one full RL cycle if new rollouts are available.

#         Returns:
#             True if training was performed, False if no new rollouts found.
#         """
#         # ── COLLECT ───────────────────────────────────────────────────────
#         self.lifecycle.transition(TrainerState.COLLECT)
#         entries = self.rollout_loader.load_new()

#         if not entries:
#             self.lifecycle.transition(TrainerState.IDLE)
#             return False

#         self.rollouts_loaded += len(entries)
#         self._fill_buffer(entries)
#         logger.info("Collected %d new rollout entries", len(entries))

#         # ── TRAIN ─────────────────────────────────────────────────────────
#         self.lifecycle.transition(TrainerState.TRAIN)
#         self._run_ppo_epochs()

#         # ── CHECKPOINT ────────────────────────────────────────────────────
#         self.lifecycle.transition(TrainerState.CHECKPOINT)
#         self._save_checkpoint()

#         # ── IDLE ──────────────────────────────────────────────────────────
#         self.lifecycle.transition(TrainerState.IDLE)
#         self.buffer.clear()
#         return True

#     # ── Internal helpers ──────────────────────────────────────────────────

#     def _fill_buffer(self, entries: list[RolloutEntry]) -> None:
#         """Populate the rollout buffer from loaded entries."""
#         self.buffer.clear()
#         for entry in entries:
#             self.buffer.store(
#                 reward=entry.reward,
#                 log_prob_old=entry.log_prob_old,
#                 value_estimate=entry.value_estimate,
#                 original_prompt=entry.original_prompt,
#                 rewritten_prompt=entry.rewritten_prompt,
#             )

#     def _run_ppo_epochs(self) -> None:
#         """Compute GAE and run PPO update epochs."""
#         rewards = torch.tensor([e for e in self.buffer._rewards], dtype=torch.float32)
#         values = torch.tensor([e for e in self.buffer._values], dtype=torch.float32)
#         advantages = compute_gae(
#             rewards, values, gamma=settings.gamma, lam=settings.lam
#         )
#         batch = self.buffer.build(advantages)

#         for epoch in range(settings.ppo_epochs):
#             all_texts = batch.original_prompts
#             tokenised = self.policy_model.tokenize(all_texts)
#             input_ids = tokenised["input_ids"]
#             attention_mask = tokenised.get("attention_mask")

#             # Current policy log-probs and hidden states
#             log_probs_new, hidden = self.policy_model(input_ids, attention_mask)
#             seq_log_prob_new = log_probs_new.sum(dim=-1)

#             # Value estimates from value head
#             values_new = self.value_head(hidden)

#             # Reference log-probs (frozen)
#             ref_log_prob = self.reference_model.get_sequence_log_prob(
#                 input_ids, attention_mask
#             )

#             # KL-adjusted rewards
#             kl_penalty = self.kl_controller.compute_kl(seq_log_prob_new, ref_log_prob)

#             # Entropy approximation: mean of negative log-probs
#             entropy = -log_probs_new.mean()

#             components = self.ppo_trainer.update(
#                 batch=batch,
#                 log_probs_new=seq_log_prob_new,
#                 values_new=values_new,
#                 entropy=entropy,
#                 kl_penalty=kl_penalty,
#             )
#             self.last_loss = components.total_loss
#             self.training_step += 1
#             logger.info(
#                 "Epoch %d/%d | step=%d | loss=%.4f | KL=%.4f",
#                 epoch + 1,
#                 settings.ppo_epochs,
#                 self.training_step,
#                 components.total_loss,
#                 self.kl_controller.last_kl,
#             )

#     def _save_checkpoint(self) -> None:
#         """Serialise model weights to the checkpoint directory."""
#         self.checkpoint_manager.save(
#             policy_state_dict=self.policy_model.model.state_dict(),
#             value_head_state_dict=self.value_head.state_dict(),
#             optimizer_state_dict=self.optimizer.state_dict(),
#             training_step=self.training_step,
#             extra_meta={
#                 "last_loss": self.last_loss,
#                 "kl_divergence": self.kl_controller.last_kl,
#                 "rollouts_loaded": self.rollouts_loaded,
#             },
#         )
