import logging
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F

from ..app.config import settings
from ..models.policy_model import PolicyModel
from ..models.reference_model import ReferenceModel
from ..models.value_head import ValueHead
from .advantage import compute_gae
from .grpo_utils import (
    build_grpo_fallback_mask,
    build_repeated_index,
    compute_grpo_relative_rewards,
    group_reward_std_mean,
    resolve_grpo_group_ids,
    select_rollout_batch,
)
from .kl_controller import KLController
from .lifecycle_manager import LifecycleManager, TrainerState
from .rollout_buffer import RolloutBatch, RolloutBuffer
from ..schemas.rollout_schema import RolloutEntry
from ..storage.checkpoint_manager import CheckpointManager
from ..storage.rollout_loader import RolloutLoader

try:
    _pkg_root = str(Path(__file__).resolve().parents[3])
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from pipeline_logger.system_logger import write_csv_rows, write_batch_summary
    from pipeline_logger.hash_utils import prompt_hash
    from pipeline_logger.mode_resolver import resolve_mode1, resolve_mode2
    from pipeline_logger.run_context import get_run_context
    from pipeline_logger import ServiceIOLogger

    _rl_io = ServiceIOLogger("rl_loop_svc")
    _PRETTY_LOG = True
except Exception:
    _PRETTY_LOG = False

logger = logging.getLogger(__name__)


def _select_optional_tensor(
    tensor: Optional[torch.Tensor],
    keep_idx: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor[keep_idx.to(device=tensor.device)]


def _select_tensor(
    tensor: torch.Tensor,
    keep_idx: torch.Tensor,
) -> torch.Tensor:
    return tensor[keep_idx.to(device=tensor.device)]


def _fuse_action_attention_mask(
    action_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Combine action_mask with attention_mask so padding tokens are excluded."""
    if attention_mask is None:
        return action_mask
    return action_mask * attention_mask[:, 1:].to(dtype=action_mask.dtype)


class TrainingLoop:
    def __init__(
        self,
        rollout_loader: RolloutLoader,
        checkpoint_manager: CheckpointManager,
        policy_model: Optional[PolicyModel],
        reference_model: Optional[ReferenceModel],
        value_head: Optional[ValueHead],
    ) -> None:
        self.rollout_loader = rollout_loader
        self.checkpoint_manager = checkpoint_manager
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.value_head = value_head
        self._distributed_mode = (
            settings.distributed_enabled and settings.distributed_world_size > 1
        )

        if not self._distributed_mode:
            if policy_model is None or reference_model is None:
                raise ValueError(
                    "Local GRPO mode requires policy_model and reference_model"
                )

        self.lifecycle = LifecycleManager()
        self.kl_controller = KLController(beta=settings.beta)

        buffer_device = str(policy_model.device) if policy_model is not None else "cpu"
        self.buffer = RolloutBuffer(device=buffer_device)

        if not self._distributed_mode:
            assert policy_model is not None
            self.optimizer = torch.optim.AdamW(
                list(policy_model.parameters()),
                lr=settings.learning_rate,
            )
        else:
            self.optimizer = None

        self.scheduler = None
        self.training_step = 0
        self.last_loss = 0.0
        self.rollouts_loaded = 0
        self.batch_counter = 0
        self.last_train_success: Optional[bool] = None
        self.last_train_error: Optional[str] = None
        self.last_train_started_at: Optional[str] = None
        self.last_train_finished_at: Optional[str] = None
        self._mbmi_group_buffer: List[Tuple[str, List[RolloutEntry]]] = []
        self._sbmi_validation_passed: bool = False

    @staticmethod
    def _is_finite(value: float) -> bool:
        return math.isfinite(float(value))

    @staticmethod
    def _is_valid_log_prob_old(value: float) -> bool:
        # Log-probabilities are commonly non-positive; only require finite and non-zero.
        return math.isfinite(float(value)) and abs(float(value)) > 1e-12

    def _assert_hybrid_runtime_ready(self, entries: List[RolloutEntry]) -> None:
        if not settings.hybrid_mode_enforced:
            return

        if not settings.grpo_enabled:
            raise RuntimeError("Hybrid enforcement failed: RL_GRPO_ENABLED must be true")
        if not settings.value_head_enabled:
            raise RuntimeError("Hybrid enforcement failed: RL_VALUE_HEAD_ENABLED must be true")
        if self.value_head is None:
            raise RuntimeError("Hybrid enforcement failed: value head is not initialized")

        for idx, entry in enumerate(entries):
            if entry.value_estimate is None:
                raise RuntimeError(
                    f"Hybrid enforcement failed: value_estimate missing at entry {idx}"
                )
            if not str(entry.group_id or "").strip():
                raise RuntimeError(
                    f"Hybrid enforcement failed: group_id missing at entry {idx}"
                )
            if not self._is_valid_log_prob_old(float(entry.log_prob_old)):
                raise RuntimeError(
                    f"Hybrid enforcement failed: invalid log_prob_old at entry {idx}"
                )

    def _validate_group_guardrails(
        self, group_id: str, entries: List[RolloutEntry]
    ) -> Tuple[bool, Dict[str, float]]:
        if not entries:
            return False, {
                "rollout_count": 0,
                "unique_action_count": 0,
                "dropped_count": 0,
                "group_validity": False,
            }

        state_set = {str(e.original_prompt) for e in entries}
        action_set = {str(e.rewritten_prompt) for e in entries}
        finite_reward = [self._is_finite(float(e.reward)) for e in entries]
        valid_logp = [self._is_valid_log_prob_old(float(e.log_prob_old)) for e in entries]
        has_group = [str(e.group_id or "").strip() != "" for e in entries]

        group_validity = (
            len(state_set) == 1
            and len(action_set) >= 2
            and all(finite_reward)
            and all(valid_logp)
            and all(has_group)
        )
        dropped_count = int(len(entries) - len(action_set))
        diag = {
            "rollout_count": float(len(entries)),
            "unique_action_count": float(len(action_set)),
            "dropped_count": float(max(dropped_count, 0)),
            "group_validity": 1.0 if group_validity else 0.0,
        }
        logger.info(
            "group_guardrail_diag | group_id=%s | rollout_count=%d | unique_action_count=%d | dropped_count=%d | group_validity=%s",
            group_id,
            len(entries),
            len(action_set),
            max(dropped_count, 0),
            group_validity,
        )
        return group_validity, diag

    def _group_entries(self, entries: List[RolloutEntry]) -> List[Tuple[str, List[RolloutEntry]]]:
        grouped: Dict[str, List[RolloutEntry]] = {}
        for entry in entries:
            key = str(entry.group_id or "").strip()
            if not key:
                key = f"missing_group:{entry.run_id or 'default'}"
            grouped.setdefault(key, []).append(entry)
        return sorted(grouped.items(), key=lambda item: item[0])

    def run_once(self) -> bool:
        self.lifecycle.transition(TrainerState.COLLECT)
        entries = self.rollout_loader.load_new()

        if not entries:
            self.lifecycle.transition(TrainerState.IDLE)
            return False

        self._assert_hybrid_runtime_ready(entries)

        self.rollouts_loaded += len(entries)

        if _PRETTY_LOG:
            _rl_io.log_input(
                n_entries=len(entries),
                mode2=resolve_mode2(settings.ppo_epochs),
            )

        if self._distributed_mode:
            self.lifecycle.transition(TrainerState.TRAIN)
            result = self._run_distributed_ppo(entries)
            self.training_step = int(result.get("training_step", self.training_step))
            self.last_loss = float(result.get("last_loss", self.last_loss))
            self.kl_controller.last_kl = float(
                result.get("kl_divergence", self.kl_controller.last_kl)
            )

            self.lifecycle.transition(TrainerState.CHECKPOINT)
            self._notify_rewriter_reload()
            self.lifecycle.transition(TrainerState.IDLE)
            self.buffer.clear()
            return True

        grouped_entries = self._group_entries(entries)
        valid_groups: List[Tuple[str, List[RolloutEntry]]] = []
        for group_id, group_items in grouped_entries:
            is_valid, _ = self._validate_group_guardrails(group_id, group_items)
            if not is_valid:
                logger.warning(
                    "group_guardrail_drop | group_id=%s | reason=validation_failed",
                    group_id,
                )
                continue
            valid_groups.append((group_id, group_items))

        if not valid_groups:
            self.lifecycle.transition(TrainerState.IDLE)
            return False

        self.lifecycle.transition(TrainerState.TRAIN)

        if settings.mbmi_enabled:
            if not self._sbmi_validation_passed:
                raise RuntimeError(
                    "MBMi enabled before SBMi validation passed"
                )
            self._mbmi_group_buffer.extend(valid_groups)
            if len(self._mbmi_group_buffer) < max(settings.mbmi_training_batch_size, 1):
                logger.info(
                    "mbmi_buffering | buffered_groups=%d | required_groups=%d",
                    len(self._mbmi_group_buffer),
                    max(settings.mbmi_training_batch_size, 1),
                )
                self.lifecycle.transition(TrainerState.IDLE)
                return False

            consume_n = max(settings.mbmi_training_batch_size, 1)
            groups_to_train = self._mbmi_group_buffer[:consume_n]
            self._mbmi_group_buffer = self._mbmi_group_buffer[consume_n:]

            for mbmi_epoch in range(max(settings.mbmi_epochs, 1)):
                for group_id, group_items in groups_to_train:
                    self.batch_counter += 1
                    self._fill_buffer(group_items)
                    pre_step = int(self.training_step)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._run_ppo_epochs(
                        sbmi_iteration_index=mbmi_epoch + 1,
                        sbmi_iteration_total=max(settings.mbmi_epochs, 1),
                        group_id=group_id,
                    )
                    if int(self.training_step) <= pre_step:
                        raise RuntimeError(
                            f"MBMi effectiveness guard failed for group_id={group_id}"
                        )
                    if not math.isfinite(float(self.last_loss)):
                        raise RuntimeError(
                            f"MBMi finite-loss guard failed for group_id={group_id}"
                        )
                    self.buffer.clear()
        else:
            for group_id, group_items in valid_groups:
                self.batch_counter += 1
                self._fill_buffer(group_items)
                pre_step = int(self.training_step)
                sbmi_iters = max(settings.sbmi_epochs, 1) if settings.sbmi_enabled else 1
                for sbmi_iter in range(sbmi_iters):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._run_ppo_epochs(
                        sbmi_iteration_index=sbmi_iter + 1,
                        sbmi_iteration_total=sbmi_iters,
                        group_id=group_id,
                    )
                if int(self.training_step) <= pre_step:
                    raise RuntimeError(
                        f"SBMi effectiveness guard failed for group_id={group_id}"
                    )
                if not math.isfinite(float(self.last_loss)):
                    raise RuntimeError(
                        f"SBMi finite-loss guard failed for group_id={group_id}"
                    )
                self._sbmi_validation_passed = True
                self.buffer.clear()

        self.lifecycle.transition(TrainerState.CHECKPOINT)
        self._save_checkpoint()
        self._notify_rewriter_reload()

        self.lifecycle.transition(TrainerState.IDLE)
        return True

    # ── Buffer fill with ValueHead inference ────────────────────────────────

    def _recompute_log_probs_old(self, entries: List[RolloutEntry]) -> List[float]:
        """Recompute log_prob_old from the frozen reference model.

        The rewriter service stores log_prob_old ≈ -1e-6 (an SFT dummy that is
        not a real log-probability).  Using this dummy makes the PPO IS ratio
        degenerate — exp(lp_policy - (-1e-6/action_len)) is anchored to the
        wrong baseline.  By computing log_prob_old fresh from the reference model
        we get a correct per-token baseline that tracks the SFT initialisation
        and makes the importance-sampling ratio well-defined from the first cycle.
        """
        if self.reference_model is None or self.policy_model is None:
            return [float(e.log_prob_old) for e in entries]

        originals = [e.original_prompt for e in entries]
        rewrittens = [e.rewritten_prompt for e in entries]
        log_probs: List[float] = []

        self.reference_model.eval() if hasattr(self.reference_model, "eval") else None
        with torch.no_grad():
            for start in range(0, len(entries), settings.batch_size):
                batch_orig = originals[start : start + settings.batch_size]
                batch_rew = rewrittens[start : start + settings.batch_size]
                tokenized = self.policy_model.tokenize_with_action_mask(
                    batch_orig, batch_rew
                )
                mb_ids = tokenized["input_ids"]
                mb_mask = tokenized.get("attention_mask")
                mb_action = tokenized["action_mask"]
                mb_fused = _fuse_action_attention_mask(mb_action, mb_mask)
                ref_lp = self.reference_model.get_sequence_log_prob(
                    mb_ids, mb_mask, mb_fused
                )
                log_probs.extend(ref_lp.detach().cpu().tolist())

        return log_probs

    def _fill_buffer(self, entries: List[RolloutEntry]) -> None:
        self.buffer.clear()

        originals = [e.original_prompt for e in entries]
        rewrittens = [e.rewritten_prompt for e in entries]

        value_estimates = self._compute_value_estimates(originals)
        # Recompute log_prob_old from the frozen reference model so the PPO
        # importance-sampling ratio is correctly anchored to the SFT baseline,
        # not the dummy -1e-6 stored by the rewriter service.
        log_probs_old = self._recompute_log_probs_old(entries)

        for idx, entry in enumerate(entries):
            self.buffer.store(
                reward=entry.reward,
                log_prob_old=log_probs_old[idx],
                value_estimate=value_estimates[idx],
                original_prompt=entry.original_prompt,
                rewritten_prompt=entry.rewritten_prompt,
                group_id=str(entry.group_id or f"group_{idx}"),
                sample_weight=float(entry.sample_weight if entry.sample_weight is not None else 1.0),
                rollout_id=str(entry.rollout_id or ""),
                og_codes=list(entry.og_codes or []),
                enh_codes=list(entry.enh_codes or []),
                gt_codes=list(entry.gt_codes or []),
            )

    def _compute_value_estimates(self, prompts: List[str]) -> List[float]:
        """Run ValueHead on original prompts to produce V(s) estimates."""
        if self.value_head is None or self.policy_model is None:
            return [0.0] * len(prompts)

        estimates: List[float] = []
        self.value_head.eval()
        with torch.no_grad():
            for start in range(0, len(prompts), settings.batch_size):
                batch_texts = prompts[start : start + settings.batch_size]
                encoded = self.policy_model.tokenize(batch_texts)
                _, hidden_states, _ = self.policy_model(
                    encoded["input_ids"],
                    encoded.get("attention_mask"),
                )
                values = self.value_head(hidden_states)
                estimates.extend(values.detach().cpu().tolist())
        self.value_head.train()
        return estimates

    # ── Core GRPO/PPO training ──────────────────────────────────────────────

    def _run_ppo_epochs(
        self,
        sbmi_iteration_index: int = 1,
        sbmi_iteration_total: int = 1,
        group_id: str = "",
    ) -> None:
        if self.policy_model is None or self.reference_model is None:
            raise RuntimeError(
                "Local GRPO run requested without initialized model components"
            )
        if self.optimizer is None:
            raise RuntimeError("Local GRPO run requested without optimizer")

        rewards = torch.tensor(self.buffer._rewards, dtype=torch.float32)

        effective_group_ids = resolve_grpo_group_ids(
            self.buffer._group_ids,
            settings.grpo_min_group_size,
            settings.grpo_group_size,
        )
        if effective_group_ids != self.buffer._group_ids:
            logger.info(
                "grpo_group_fallback_applied | source_groups=%d | fallback_group_size=%d",
                len(set(self.buffer._group_ids)),
                settings.grpo_group_size,
            )

        # GRPO within-group advantages (no global normalization to preserve group signal)
        grpo_advantages = compute_grpo_relative_rewards(
            rewards,
            effective_group_ids,
            settings.grpo_min_group_size,
        )

        # Fallback for samples in groups too small for GRPO
        fallback_advantages = rewards.clone()
        if fallback_advantages.numel() > 1:
            fallback_advantages = (fallback_advantages - fallback_advantages.mean()) / (
                fallback_advantages.std(unbiased=False) + 1e-6
            )

        fallback_mask_base = build_grpo_fallback_mask(
            effective_group_ids,
            settings.grpo_min_group_size,
            device=grpo_advantages.device,
        )
        pure_reward_advantages = torch.where(
            fallback_mask_base, fallback_advantages, grpo_advantages
        )

        # Blend with GAE advantages from ValueHead when available
        values_tensor = torch.tensor(self.buffer._values, dtype=torch.float32)
        has_value_estimates = values_tensor.abs().sum().item() > 0
        if has_value_estimates:
            gae_advantages = compute_gae(
                rewards,
                values_tensor,
                gamma=settings.gamma,
                lam=settings.lam,
                normalize=True,
            )
            lam_h = settings.hybrid_advantage_lambda
            advantages = lam_h * gae_advantages + (1.0 - lam_h) * pure_reward_advantages
        else:
            advantages = pure_reward_advantages

        group_reward_std = group_reward_std_mean(
            rewards,
            effective_group_ids,
            settings.grpo_min_group_size,
        )
        logger.info(
            "grpo_advantage_diag | rewards_mean=%.6f | group_reward_std=%.6f | advantage_mean=%.6f | advantage_std=%.6f | fallback_samples=%d | gae_blended=%s",
            float(rewards.mean().item()) if rewards.numel() else 0.0,
            group_reward_std,
            float(advantages.mean().item()) if advantages.numel() else 0.0,
            (
                float(advantages.std(unbiased=False).item())
                if advantages.numel() > 1
                else 0.0
            ),
            int(fallback_mask_base.sum().item()),
            has_value_estimates,
        )

        batch = self.buffer.build(advantages)
        loaded_batch_size = len(batch.rewritten_prompts)
        if loaded_batch_size == 0:
            return

        min_effective_batch_size = max(int(settings.ppo_min_effective_batch_size), 1)

        # ── Hoist: tokenize once ────────────────────────────────────────────
        tokenized = self.policy_model.tokenize_with_action_mask(
            batch.original_prompts,
            batch.rewritten_prompts,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        action_mask = tokenized["action_mask"]
        valid_action = tokenized.get("valid_action")

        # Fuse action_mask with attention_mask once for consistent use
        fused_action_mask = _fuse_action_attention_mask(action_mask, attention_mask)

        # ── Hoist: filter once ──────────────────────────────────────────────
        current_batch = batch
        current_fallback_mask = fallback_mask_base.clone()
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        current_fused_mask = fused_action_mask

        filter_diag = {
            "invalid_span_count": 0,
            "skipped_due_to_nan_reward": 0,
            "skipped_due_to_non_finite_advantage": 0,
        }

        invalid_span_mask = fused_action_mask.sum(dim=-1) <= 0
        if valid_action is not None:
            invalid_span_mask = invalid_span_mask | (~valid_action)
        invalid_span_count = int(invalid_span_mask.sum().item())
        filter_diag["invalid_span_count"] = invalid_span_count

        if invalid_span_count > 0:
            keep_idx = torch.nonzero(~invalid_span_mask, as_tuple=False).squeeze(-1)
            if keep_idx.numel() == 0:
                logger.warning(
                    "grpo_skip | reason=no_valid_action_spans_after_tokenization"
                )
                return
            current_batch = select_rollout_batch(current_batch, keep_idx)
            current_input_ids = _select_tensor(current_input_ids, keep_idx)
            current_attention_mask = _select_optional_tensor(current_attention_mask, keep_idx)
            current_fused_mask = _select_tensor(current_fused_mask, keep_idx)
            current_fallback_mask = _select_tensor(current_fallback_mask, keep_idx)

        finite_reward_mask = torch.isfinite(current_batch.rewards)
        dropped_nan = int((~finite_reward_mask).sum().item())
        filter_diag["skipped_due_to_nan_reward"] = dropped_nan
        if dropped_nan > 0:
            keep_idx = torch.nonzero(finite_reward_mask, as_tuple=False).squeeze(-1)
            if keep_idx.numel() == 0:
                logger.warning("grpo_skip | reason=all_rewards_non_finite")
                return
            current_batch = select_rollout_batch(current_batch, keep_idx)
            current_input_ids = _select_tensor(current_input_ids, keep_idx)
            current_attention_mask = _select_optional_tensor(current_attention_mask, keep_idx)
            current_fused_mask = _select_tensor(current_fused_mask, keep_idx)
            current_fallback_mask = _select_tensor(current_fallback_mask, keep_idx)

        finite_adv_mask = torch.isfinite(current_batch.advantages)
        dropped_adv = int((~finite_adv_mask).sum().item())
        filter_diag["skipped_due_to_non_finite_advantage"] = dropped_adv
        if dropped_adv > 0:
            keep_idx = torch.nonzero(finite_adv_mask, as_tuple=False).squeeze(-1)
            if keep_idx.numel() == 0:
                logger.warning("grpo_skip | reason=all_advantages_non_finite")
                return
            current_batch = select_rollout_batch(current_batch, keep_idx)
            current_input_ids = _select_tensor(current_input_ids, keep_idx)
            current_attention_mask = _select_optional_tensor(current_attention_mask, keep_idx)
            current_fused_mask = _select_tensor(current_fused_mask, keep_idx)
            current_fallback_mask = _select_tensor(current_fallback_mask, keep_idx)

        n = len(current_batch.rewritten_prompts)
        if n == 0:
            return

        if n < min_effective_batch_size:
            expand_idx = build_repeated_index(
                size=n,
                target_size=min_effective_batch_size,
                device=current_input_ids.device,
            )
            current_batch = select_rollout_batch(current_batch, expand_idx)
            current_input_ids = _select_tensor(current_input_ids, expand_idx)
            current_attention_mask = _select_optional_tensor(current_attention_mask, expand_idx)
            current_fused_mask = _select_tensor(current_fused_mask, expand_idx)
            current_fallback_mask = _select_tensor(current_fallback_mask, expand_idx)
            n = len(current_batch.rewritten_prompts)

        # ── Hoist: compute reference log-probs once ─────────────────────────
        ref_chunks_cpu = []
        with torch.no_grad():
            for start in range(0, n, settings.batch_size):
                end = min(start + settings.batch_size, n)
                mb_ids = current_input_ids[start:end]
                mb_mask = (
                    current_attention_mask[start:end]
                    if current_attention_mask is not None
                    else None
                )
                mb_fused = current_fused_mask[start:end]
                ref_lp = self.reference_model.get_sequence_log_prob(
                    mb_ids,
                    mb_mask,
                    mb_fused,
                )
                ref_chunks_cpu.append(ref_lp.detach().cpu())
        ref_log_probs = torch.cat(ref_chunks_cpu, dim=0).to(self.policy_model.device)

        # ── LR scheduler for this training cycle ────────────────────────────
        # Reset the scheduler each cycle so that warmup_steps and total_steps
        # reflect the actual batch size for THIS cycle, not the first one.
        # Keeping a stale scheduler from cycle 1 (e.g. 9 rollouts) through cycle 5
        # (e.g. 182 rollouts) means the LR was computed against wrong total_steps,
        # causing either premature decay or no warmup on later cycles.
        total_steps = settings.ppo_epochs * max(
            1,
            math.ceil(n / settings.batch_size) // settings.gradient_accumulation_steps,
        )
        warmup_steps = max(1, int(total_steps * settings.lr_warmup_ratio))
        if total_steps > 1:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=self._build_warmup_cosine_lambda(warmup_steps, total_steps),
            )
        else:
            self.scheduler = None

        # ── Epoch loop (only gradient updates, everything else precomputed) ─
        min_epochs = max(1, int(settings.dynamic_min_epochs))
        max_epochs = max(min_epochs, int(settings.ppo_epochs))
        if not settings.dynamic_iteration_enabled:
            min_epochs = max_epochs

        prev_epoch_avg_loss: Optional[float] = None
        low_improvement_streak = 0
        low_grad_streak = 0

        for epoch in range(max_epochs):
            grad_norm: float = 0.0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            epoch_max_grad_norm = 0.0

            # ── Pretty system log (CSV + ASCII table) ───────────────────────
            if _PRETTY_LOG:
                _mode1 = resolve_mode1(settings.grpo_enabled, has_value_estimates)
                _mode2 = resolve_mode2(max_epochs)
                _, _run_id = get_run_context()
                _run_id = _run_id or "unknown"
                _ts = datetime.now(timezone.utc).isoformat()
                _log_entries = [
                    {
                        "ts": _ts,
                        "run_id": _run_id,
                        "mode1": _mode1,
                        "mode2": _mode2,
                        "batch_num": self.batch_counter,
                        "iter_num": epoch + 1,
                        "batch_size": loaded_batch_size,
                        "max_iters": max_epochs,
                        "rollout_id": (
                            current_batch.rollout_ids[i]
                            if i < len(current_batch.rollout_ids)
                            else ""
                        ),
                        "prompt_hash": prompt_hash(current_batch.original_prompts[i]),
                        "rewritten_hash": prompt_hash(
                            current_batch.rewritten_prompts[i]
                        ),
                        "og_codes": (
                            current_batch.og_codes[i]
                            if i < len(current_batch.og_codes)
                            else []
                        ),
                        "enh_codes": (
                            current_batch.enh_codes[i]
                            if i < len(current_batch.enh_codes)
                            else []
                        ),
                        "gt_codes": (
                            current_batch.gt_codes[i]
                            if i < len(current_batch.gt_codes)
                            else []
                        ),
                        "reward": float(current_batch.rewards[i].item()),
                    }
                    for i in range(len(current_batch.original_prompts))
                ]
                write_csv_rows(_log_entries)
                write_batch_summary(
                    run_id=_run_id,
                    mode1=_mode1,
                    mode2=_mode2,
                    batch_num=self.batch_counter,
                    iter_num=epoch + 1,
                    max_iters=max_epochs,
                    batch_size=loaded_batch_size,
                    entries=_log_entries,
                )
                _rl_io.log_output(
                    batch_num=self.batch_counter,
                    iter_num=epoch + 1,
                    mode1=_mode1,
                    mode2=_mode2,
                    n_prompts=len(_log_entries),
                    mean_reward=(
                        f"{sum(e['reward'] for e in _log_entries)/len(_log_entries):.4f}"
                        if _log_entries
                        else "n/a"
                    ),
                )

            epoch_diag = {
                "batch_size_loaded": loaded_batch_size,
                "batch_size_after_filtering": n,
                "invalid_span_count": filter_diag["invalid_span_count"],
                "skipped_due_to_nan_reward": filter_diag["skipped_due_to_nan_reward"],
                "skipped_due_to_non_finite_advantage": filter_diag[
                    "skipped_due_to_non_finite_advantage"
                ],
                "fallback_samples": int(current_fallback_mask.sum().item()),
                "optimizer_steps": 0,
            }

            self.optimizer.zero_grad()
            accum_counter = 0

            for start in range(0, n, settings.batch_size):
                end = min(start + settings.batch_size, n)
                mb_ids = current_input_ids[start:end]
                mb_mask = (
                    current_attention_mask[start:end]
                    if current_attention_mask is not None
                    else None
                )
                mb_fused = current_fused_mask[start:end]

                token_log_probs_new, hidden_states, logits = self.policy_model(
                    mb_ids, mb_mask
                )
                seq_log_prob_new = (token_log_probs_new * mb_fused).sum(dim=-1)

                # Per-token action length — used to normalise entropy, KL, and GRPO
                # loss so that long and short notes contribute equally regardless of
                # sequence length.  Clamp at 1 to avoid division-by-zero on empty
                # action spans (those are filtered above, but be defensive).
                action_len = mb_fused.sum(dim=-1).clamp(min=1.0)          # (B,)
                seq_log_prob_new_per_tok = seq_log_prob_new / action_len   # (B,)

                # Entropy regularization: average over action tokens, not sum.
                # Summing over 150 tokens with vocab ~32K yields ~1500 nats/sample,
                # making the entropy term dominate the loss and drive it deeply
                # negative.  Per-token average keeps it in the ~2-10 nat range.
                token_probs = F.softmax(logits, dim=-1)
                token_entropy = -(token_probs * torch.log(token_probs + 1e-10)).sum(
                    dim=-1
                )
                seq_entropy = (token_entropy * mb_fused).sum(dim=-1) / action_len  # per-token avg
                mb_entropy = seq_entropy.mean()

                # Value head loss
                value_loss = torch.tensor(0.0, device=seq_log_prob_new.device)
                if self.value_head is not None and has_value_estimates:
                    values_new = self.value_head(hidden_states)
                    values_old = current_batch.values[start:end].to(values_new.device)
                    returns_mb = current_batch.returns[start:end].to(values_new.device)
                    value_pred_clipped = values_old + (values_new - values_old).clamp(
                        -settings.value_clip,
                        settings.value_clip,
                    )
                    vl_unclipped = (values_new - returns_mb).pow(2)
                    vl_clipped = (value_pred_clipped - returns_mb).pow(2)
                    value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                old_log_prob_mb = current_batch.log_probs_old[start:end].to(seq_log_prob_new.device)
                if not torch.isfinite(seq_log_prob_new).all() or not torch.isfinite(old_log_prob_mb).all():
                    logger.warning(
                        "grpo_minibatch_skip | epoch=%d | start=%d | end=%d | reason=non_finite_log_prob",
                        epoch + 1,
                        start,
                        end,
                    )
                    continue

                # PPO importance-sampling ratio: normalise both log-probs by
                # action_len before computing the ratio.  The SFT rewriter stores
                # log_prob_old ≈ -1e-6 (a near-zero dummy); without normalisation the
                # ratio collapses to exp(seq_log_prob_new) ≈ exp(-75) ≈ 0, making
                # the PPO surrogate degenerate.  Per-token normalisation ensures the
                # ratio stays ~O(1) and the PPO path contributes meaningful gradients
                # once the policy drifts from the SFT initialisation.
                old_log_prob_mb_per_tok = old_log_prob_mb / action_len
                ratio = torch.exp(seq_log_prob_new_per_tok - old_log_prob_mb_per_tok)
                ratio = torch.clamp(ratio, 0.0, settings.ratio_clip_max)

                # KL: computed on per-token log-probs so that the divergence is
                # sequence-length-independent and comparable to the
                # RL_MAX_ABS_KL_FOR_UPDATE threshold (which is expressed per-token).
                ref_mb = ref_log_probs[start:end].to(seq_log_prob_new.device)
                ref_mb_per_tok = ref_mb / action_len
                kl_penalty = self.kl_controller.compute_kl(seq_log_prob_new_per_tok, ref_mb_per_tok)

                adv_mb = current_batch.advantages[start:end].to(seq_log_prob_new.device)
                fallback_mb = current_fallback_mask[start:end].to(seq_log_prob_new.device)
                sample_weights = torch.clamp(
                    current_batch.sample_weights[start:end].to(seq_log_prob_new.device),
                    min=0.0,
                )
                normalizer = torch.clamp(sample_weights.sum(), min=1e-8)

                # GRPO policy gradient: use per-token log-prob so gradient magnitude
                # is note-length-independent (long notes no longer dominate).
                grpo_loss_per_sample = -(adv_mb * seq_log_prob_new_per_tok)
                ppo_surr1 = ratio * adv_mb
                ppo_surr2 = (
                    torch.clamp(ratio, 1.0 - settings.epsilon, 1.0 + settings.epsilon)
                    * adv_mb
                )
                ppo_loss_per_sample = -torch.min(ppo_surr1, ppo_surr2)

                policy_loss_per_sample = torch.where(
                    fallback_mb,
                    ppo_loss_per_sample,
                    grpo_loss_per_sample,
                )
                policy_loss = (
                    policy_loss_per_sample * sample_weights
                ).sum() / normalizer
                kl_loss = (kl_penalty * sample_weights).sum() / normalizer
                total_loss = (
                    policy_loss
                    + settings.beta * kl_loss
                    + settings.value_coef * value_loss
                    - settings.entropy_coef * mb_entropy
                )

                if not torch.isfinite(total_loss):
                    logger.warning(
                        "grpo_minibatch_skip | epoch=%d | start=%d | end=%d | reason=non_finite_total_loss",
                        epoch + 1,
                        start,
                        end,
                    )
                    self.optimizer.zero_grad()
                    continue

                (total_loss / settings.gradient_accumulation_steps).backward()
                accum_counter += 1

                if (
                    accum_counter % settings.gradient_accumulation_steps == 0
                    or end == n
                ):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters()),
                        max_norm=settings.grad_clip_max_norm,
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    epoch_diag["optimizer_steps"] += 1
                    self.training_step += 1
                    grad_norm_float = float(grad_norm)
                    if math.isfinite(grad_norm_float):
                        epoch_max_grad_norm = max(epoch_max_grad_norm, grad_norm_float)

                self.last_loss = float(total_loss.detach().item())
                epoch_loss_sum += self.last_loss
                epoch_loss_count += 1
                logger.info(
                    "grpo_loss_diag | epoch=%d | start=%d | end=%d | policy=%.6f | kl=%.6f | value=%.6f | entropy=%.6f | total=%.6f | grad_norm=%.4f | ratio_max=%.4f | fallback=%d/%d",
                    epoch + 1,
                    start,
                    end,
                    float(policy_loss.detach().item()),
                    float(kl_loss.detach().item()),
                    float(value_loss.detach().item()),
                    float(mb_entropy.detach().item()),
                    self.last_loss,
                    (
                        float(grad_norm)
                        if torch.isfinite(torch.tensor(grad_norm))
                        else -1.0
                    ),
                    float(ratio.max().detach().item()),
                    int(fallback_mb.sum().item()),
                    int(fallback_mb.numel()),
                )

            logger.info(
                "GRPO Epoch %d/%d | sbmi_iter=%d/%d | group_id=%s | step=%d | loss=%.4f | KL=%.4f | reward_mean=%.4f | group_reward_std=%.4f | advantage_mean=%.4f | advantage_std=%.4f | batch=%d | filtered=%d | fallback=%d | opt_steps=%d",
                epoch + 1,
                settings.ppo_epochs,
                sbmi_iteration_index,
                sbmi_iteration_total,
                group_id or "n/a",
                self.training_step,
                self.last_loss,
                self.kl_controller.last_kl,
                float(current_batch.rewards.mean().item()) if len(current_batch.rewards) else 0.0,
                group_reward_std,
                (
                    float(current_batch.advantages.mean().item())
                    if len(current_batch.advantages)
                    else 0.0
                ),
                (
                    float(current_batch.advantages.std(unbiased=False).item())
                    if len(current_batch.advantages) > 1
                    else 0.0
                ),
                loaded_batch_size,
                n,
                epoch_diag["fallback_samples"],
                epoch_diag["optimizer_steps"],
            )

            epoch_avg_loss = (
                float(epoch_loss_sum / epoch_loss_count)
                if epoch_loss_count > 0
                else None
            )
            # Store epoch-mean loss rather than the last minibatch loss so that
            # checkpoint health metrics (last_loss) reflect the full epoch average.
            if epoch_avg_loss is not None:
                self.last_loss = epoch_avg_loss

            # KL early stopping
            if self.kl_controller.last_kl > settings.max_abs_kl_for_update:
                logger.warning(
                    "grpo_kl_early_stop | epoch=%d | kl=%.6f | threshold=%.6f",
                    epoch + 1,
                    self.kl_controller.last_kl,
                    settings.max_abs_kl_for_update,
                )
                break

            if not settings.dynamic_iteration_enabled:
                continue

            if (epoch + 1) < min_epochs:
                if epoch_avg_loss is not None:
                    prev_epoch_avg_loss = epoch_avg_loss
                continue

            if epoch_diag["optimizer_steps"] <= 0:
                logger.warning(
                    "grpo_dynamic_stop | reason=no_optimizer_steps | epoch=%d | min_epochs=%d",
                    epoch + 1,
                    min_epochs,
                )
                break

            if epoch_avg_loss is None:
                logger.warning(
                    "grpo_dynamic_stop | reason=no_finite_epoch_loss | epoch=%d",
                    epoch + 1,
                )
                break

            if prev_epoch_avg_loss is not None:
                loss_improvement = abs(prev_epoch_avg_loss - epoch_avg_loss)
                if loss_improvement < settings.dynamic_loss_improvement_threshold:
                    low_improvement_streak += 1
                else:
                    low_improvement_streak = 0
            else:
                loss_improvement = float("inf")

            if epoch_max_grad_norm < settings.dynamic_grad_norm_floor:
                low_grad_streak += 1
            else:
                low_grad_streak = 0

            logger.info(
                "grpo_dynamic_diag | epoch=%d/%d | min_epochs=%d | epoch_avg_loss=%.6f | loss_improvement=%.6f | max_grad_norm=%.6f | low_improvement_streak=%d | low_grad_streak=%d | patience=%d",
                epoch + 1,
                max_epochs,
                min_epochs,
                epoch_avg_loss,
                loss_improvement,
                epoch_max_grad_norm,
                low_improvement_streak,
                low_grad_streak,
                settings.dynamic_patience,
            )

            prev_epoch_avg_loss = epoch_avg_loss

            if low_improvement_streak >= settings.dynamic_patience:
                logger.info(
                    "grpo_dynamic_stop | reason=loss_plateau | epoch=%d | streak=%d | threshold=%.6f",
                    epoch + 1,
                    low_improvement_streak,
                    settings.dynamic_loss_improvement_threshold,
                )
                break

            if low_grad_streak >= settings.dynamic_patience:
                logger.info(
                    "grpo_dynamic_stop | reason=low_grad_norm | epoch=%d | streak=%d | floor=%.6f",
                    epoch + 1,
                    low_grad_streak,
                    settings.dynamic_grad_norm_floor,
                )
                break

    @staticmethod
    def _build_warmup_cosine_lambda(warmup_steps: int, total_steps: int):
        min_lr_ratio = settings.lr_min_ratio

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return max(
                    float(current_step) / float(max(warmup_steps, 1)), min_lr_ratio
                )
            progress = float(current_step - warmup_steps) / float(
                max(total_steps - warmup_steps, 1)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)

        return lr_lambda

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        if (
            self.policy_model is None
            or self.value_head is None
            or self.optimizer is None
        ):
            raise RuntimeError(
                "Cannot save local checkpoint without initialized model state"
            )

        self.checkpoint_manager.save(
            policy_model=self.policy_model,
            value_head_state_dict=self.value_head.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            training_step=self.training_step,
            extra_meta={
                "last_loss": self.last_loss,
                "kl_divergence": self.kl_controller.last_kl,
                "rollouts_loaded": self.rollouts_loaded,
            },
        )

    def _notify_rewriter_reload(self) -> None:
        # Notify all rewriter replicas (primary + secondary) so every replica
        # loads the latest checkpoint.  REWRITER_SERVICE_URLS is a
        # comma-separated list; fall back to REWRITER_SERVICE_URL (singular)
        # for backward compatibility, then to the hardcoded default.
        urls_env = os.environ.get("REWRITER_SERVICE_URLS", "")
        if not urls_env:
            urls_env = os.environ.get("REWRITER_SERVICE_URL", "http://localhost:8000")
        rewriter_urls = [u.strip() for u in urls_env.split(",") if u.strip()]

        def _fire(url: str) -> None:
            endpoint = f"{url}/reload_checkpoint"
            try:
                requests.post(endpoint, timeout=30)
            except requests.RequestException as exc:
                logger.warning(
                    "Failed to notify rewriter checkpoint reload at %s: %s", url, exc
                )

        # Fire-and-forget: do not block the CHECKPOINT→IDLE transition while
        # rewriter services are GPU-busy (each reload call can take up to 30s).
        for url in rewriter_urls:
            t = threading.Thread(target=_fire, args=(url,), daemon=True)
            t.start()

    # ── Distributed training ────────────────────────────────────────────────

    def _run_distributed_ppo(self, entries: List[RolloutEntry]) -> dict:
        self._ensure_distributed_memory_headroom()

        project_root = Path(__file__).resolve().parents[2]
        script_path = (
            project_root / "rl_loop_svc" / "scripts" / "distributed_train_once.py"
        )
        if not script_path.exists():
            raise RuntimeError(f"Distributed training script not found: {script_path}")

        with tempfile.TemporaryDirectory(prefix="rl_dist_train_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            entries_path = tmp_dir_path / "entries.json"
            result_path = tmp_dir_path / "result.json"
            torchrun_log_dir = tmp_dir_path / "torchrun_logs"

            entries_payload = [entry.model_dump(exclude_none=True) for entry in entries]
            entries_path.write_text(json.dumps(entries_payload), encoding="utf-8")

            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={settings.distributed_world_size}",
                "--log-dir",
                str(torchrun_log_dir),
                "--redirects",
                "3",
                "--tee",
                "3",
                "--module",
                "rl_loop_svc.scripts.distributed_train_once",
                "--entries-file",
                str(entries_path),
                "--result-file",
                str(result_path),
                "--checkpoints-dir",
                str(settings.checkpoints_dir),
            ]

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            root_str = str(project_root)
            env["PYTHONPATH"] = (
                f"{root_str}:{existing_pythonpath}" if existing_pythonpath else root_str
            )
            env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

            logger.info(
                "distributed_train_launch | world_size=%d | command=%s",
                settings.distributed_world_size,
                " ".join(command),
            )

            completed = subprocess.run(
                command,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=settings.distributed_launch_timeout_seconds,
                check=False,
            )

            if completed.returncode != 0:
                artifacts_dir, copied_error_files = (
                    self._persist_distributed_failure_artifacts(
                        tmp_dir_path,
                        completed.stdout or "",
                        completed.stderr or "",
                    )
                )
                stdout_tail = (completed.stdout or "")[-4000:]
                stderr_tail = (completed.stderr or "")[-4000:]
                raise RuntimeError(
                    "Distributed PPO run failed with exit code "
                    f"{completed.returncode}.\n"
                    f"Failure artifacts: {artifacts_dir}\n"
                    f"Captured elastic error files: {copied_error_files}\n"
                    f"STDOUT:\n{stdout_tail}\nSTDERR:\n{stderr_tail}"
                )

            if not result_path.exists():
                raise RuntimeError("Distributed PPO completed without result payload")

            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok", False):
                raise RuntimeError(
                    f"Distributed PPO failed: {result.get('error', 'unknown error')}"
                )

            logger.info(
                "distributed_train_complete | step=%s | loss=%s | kl=%s",
                result.get("training_step"),
                result.get("last_loss"),
                result.get("kl_divergence"),
            )
            return result

    @staticmethod
    def _extract_torchelastic_error_files(text: str) -> List[Path]:
        pattern = re.compile(r"error_file:\s*([^\s]+)")
        found: List[Path] = []
        seen: set[str] = set()
        for match in pattern.findall(text):
            candidate = str(match).strip().strip("\"'")
            if not candidate or candidate == "<N/A>":
                continue
            normalized = candidate.rstrip(",")
            if normalized in seen:
                continue
            seen.add(normalized)
            found.append(Path(normalized))
        return found

    def _persist_distributed_failure_artifacts(
        self,
        tmp_dir_path: Path,
        stdout_text: str,
        stderr_text: str,
    ) -> tuple[str, List[str]]:
        artifact_root = settings.checkpoints_dir / "distributed_failures"
        artifact_root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        artifact_dir = artifact_root / f"dist_fail_{stamp}_{os.getpid()}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        (artifact_dir / "launcher_stdout.log").write_text(stdout_text, encoding="utf-8")
        (artifact_dir / "launcher_stderr.log").write_text(stderr_text, encoding="utf-8")

        combined_text = f"{stdout_text}\n{stderr_text}"
        error_files = self._extract_torchelastic_error_files(combined_text)
        copied_error_files: List[str] = []
        for idx, source in enumerate(error_files):
            if not source.exists():
                continue
            destination = artifact_dir / f"rank_error_{idx}_{source.name}"
            try:
                shutil.copy2(source, destination)
                copied_error_files.append(str(destination))
            except Exception:
                continue

        torchrun_log_dir = tmp_dir_path / "torchrun_logs"
        if torchrun_log_dir.exists():
            shutil.copytree(
                torchrun_log_dir, artifact_dir / "torchrun_logs", dirs_exist_ok=True
            )
            for idx, source in enumerate(sorted(torchrun_log_dir.rglob("error.json"))):
                destination = artifact_dir / f"torchrun_error_{idx}_{source.name}"
                try:
                    shutil.copy2(source, destination)
                    copied_error_files.append(str(destination))
                except Exception:
                    continue

        self._make_world_readable(artifact_dir)

        metadata = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "error_files_reported": [str(path) for path in error_files],
            "error_files_copied": copied_error_files,
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(artifact_dir), copied_error_files

    @staticmethod
    def _make_world_readable(path: Path) -> None:
        for root, dirnames, filenames in os.walk(path):
            root_path = Path(root)
            try:
                os.chmod(root_path, 0o755)
            except OSError:
                pass

            for dirname in dirnames:
                try:
                    os.chmod(root_path / dirname, 0o755)
                except OSError:
                    continue

            for filename in filenames:
                try:
                    os.chmod(root_path / filename, 0o644)
                except OSError:
                    continue

    def _ensure_distributed_memory_headroom(self) -> None:
        required = settings.distributed_world_size
        if required <= 1:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Distributed PPO requested but CUDA is unavailable")

        visible = torch.cuda.device_count()
        if visible < required:
            raise RuntimeError(
                f"Distributed PPO requested world_size={required} but only {visible} CUDA devices are visible"
            )

        mem_available_gb = self._read_mem_available_gb()
        if mem_available_gb < settings.distributed_min_free_ram_gb:
            raise RuntimeError(
                "Insufficient host RAM safety buffer for distributed PPO: "
                f"available={mem_available_gb:.2f}GB, required>={settings.distributed_min_free_ram_gb:.2f}GB"
            )

        per_gpu_free = []
        for index in range(required):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            per_gpu_free.append((index, free_gb, total_gb))
            if free_gb < settings.distributed_min_free_vram_gb_per_gpu:
                raise RuntimeError(
                    "Insufficient VRAM safety buffer on visible cuda:%d: free=%.2fGB total=%.2fGB required>=%.2fGB"
                    % (
                        index,
                        free_gb,
                        total_gb,
                        settings.distributed_min_free_vram_gb_per_gpu,
                    )
                )

        logger.info(
            "distributed_safety_guard_passed | free_ram_gb=%.2f | min_vram_free_gb=%s",
            mem_available_gb,
            ",".join(f"cuda:{idx}={free:.2f}" for idx, free, _ in per_gpu_free),
        )

    @staticmethod
    def _read_mem_available_gb() -> float:
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            return 0.0

        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                break
            try:
                kib = float(parts[1])
            except ValueError:
                break
            return kib / (1024**2)

        return 0.0
