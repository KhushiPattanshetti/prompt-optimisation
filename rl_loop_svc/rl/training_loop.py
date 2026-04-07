import logging
import json
import os
import hashlib
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import List, Optional

import requests
import torch

from app.config import settings
from models.policy_model import PolicyModel
from models.reference_model import ReferenceModel
from models.value_head import ValueHead
from rl.advantage import compute_gae
from rl.kl_controller import KLController
from rl.lifecycle_manager import LifecycleManager, TrainerState
from rl.ppo_trainer import PPOTrainer
from rl.rollout_buffer import RolloutBatch, RolloutBuffer
from schemas.rollout_schema import RolloutEntry
from storage.checkpoint_manager import CheckpointManager
from storage.rollout_loader import RolloutLoader

logger = logging.getLogger(__name__)


def _resolve_group_id(entry: RolloutEntry) -> str:
    if entry.group_id:
        return str(entry.group_id)
    base = str(entry.original_prompt or "")
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"g_{digest}"


def _resolve_sample_weight(entry: RolloutEntry) -> float:
    if entry.sample_weight is None:
        return 1.0

    try:
        weight = float(entry.sample_weight)
    except (TypeError, ValueError):
        return 1.0

    if not math.isfinite(weight):
        return 1.0

    return float(min(max(weight, 0.0), 1.0))


def _compute_grpo_relative_rewards(
    rewards: torch.Tensor,
    group_ids: List[str],
    min_group_size: int,
) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards

    relative = torch.zeros_like(rewards)
    grouped: dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)

    for indices in grouped.values():
        if len(indices) < max(min_group_size, 2):
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_rewards = rewards[idx_tensor]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False) + 1e-6
        relative[idx_tensor] = (group_rewards - group_mean) / group_std

    return relative


def _resolve_grpo_group_ids(
    group_ids: List[str],
    min_group_size: int,
    fallback_group_size: int,
) -> List[str]:
    """Ensure GRPO has usable groups by backfilling deterministic K-sized groups when needed."""
    normalized = [str(group_id) for group_id in group_ids]
    if not normalized:
        return normalized

    threshold = max(int(min_group_size), 2)
    counts = Counter(normalized)
    if any(size >= threshold for size in counts.values()):
        return normalized

    group_size = max(int(fallback_group_size), threshold)
    return [f"grpo_auto_{idx // group_size}" for idx in range(len(normalized))]


def _build_grpo_fallback_mask(
    group_ids: List[str],
    min_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    threshold = max(int(min_group_size), 2)
    counts = Counter(str(group_id) for group_id in group_ids)
    mask = [counts.get(str(group_id), 0) < threshold for group_id in group_ids]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _group_reward_std_mean(
    rewards: torch.Tensor,
    group_ids: List[str],
    min_group_size: int,
) -> float:
    threshold = max(int(min_group_size), 2)
    grouped: dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)

    std_values: List[float] = []
    for indices in grouped.values():
        if len(indices) < threshold:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_std = rewards[idx_tensor].std(unbiased=False)
        std_values.append(float(group_std.item()))

    if not std_values:
        return 0.0
    return float(sum(std_values) / len(std_values))


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
        self._distributed_mode = settings.distributed_enabled and settings.distributed_world_size > 1

        if not self._distributed_mode:
            if policy_model is None or reference_model is None:
                raise ValueError("Local GRPO mode requires policy_model and reference_model")

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

        self.training_step = 0
        self.last_loss = 0.0
        self.rollouts_loaded = 0
        self.last_train_success: Optional[bool] = None
        self.last_train_error: Optional[str] = None
        self.last_train_started_at: Optional[str] = None
        self.last_train_finished_at: Optional[str] = None

    def run_once(self) -> bool:
        self.lifecycle.transition(TrainerState.COLLECT)
        entries = self.rollout_loader.load_new()

        if not entries:
            self.lifecycle.transition(TrainerState.IDLE)
            return False

        self.rollouts_loaded += len(entries)

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

        self._fill_buffer(entries)

        self.lifecycle.transition(TrainerState.TRAIN)
        self._run_ppo_epochs()

        self.lifecycle.transition(TrainerState.CHECKPOINT)
        self._save_checkpoint()
        self._notify_rewriter_reload()

        self.lifecycle.transition(TrainerState.IDLE)
        self.buffer.clear()
        return True

    def _run_distributed_ppo(self, entries: List[RolloutEntry]) -> dict:
        self._ensure_distributed_memory_headroom()

        project_root = Path(__file__).resolve().parents[2]
        script_path = project_root / "rl_loop_svc" / "scripts" / "distributed_train_once.py"
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
                str(script_path),
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
                artifacts_dir, copied_error_files = self._persist_distributed_failure_artifacts(
                    tmp_dir_path,
                    completed.stdout or "",
                    completed.stderr or "",
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
                raise RuntimeError(f"Distributed PPO failed: {result.get('error', 'unknown error')}")

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
            shutil.copytree(torchrun_log_dir, artifact_dir / "torchrun_logs", dirs_exist_ok=True)
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
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
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
            return kib / (1024 ** 2)

        return 0.0

    def _fill_buffer(self, entries: List[RolloutEntry]) -> None:
        self.buffer.clear()
        for entry in entries:
            concept_reward = float(entry.concept_reward) if entry.concept_reward is not None else float(entry.reward)
            group_id = _resolve_group_id(entry)
            sample_weight = _resolve_sample_weight(entry)
            self.buffer.store(
                reward=entry.reward,
                log_prob_old=entry.log_prob_old,
                value_estimate=0.0,
                original_prompt=entry.original_prompt,
                rewritten_prompt=entry.rewritten_prompt,
                concept_reward=concept_reward,
                group_id=group_id,
                sample_weight=sample_weight,
            )

    @staticmethod
    def _select_optional_tensor(
        tensor: Optional[torch.Tensor],
        keep_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor[keep_idx]

    @staticmethod
    def _select_rollout_batch(batch: RolloutBatch, keep_idx: torch.Tensor) -> RolloutBatch:
        idx_list = [int(v) for v in keep_idx.detach().cpu().tolist()]
        return RolloutBatch(
            rewards=batch.rewards[keep_idx],
            log_probs_old=batch.log_probs_old[keep_idx],
            values=batch.values[keep_idx],
            advantages=batch.advantages[keep_idx],
            returns=batch.returns[keep_idx],
            sample_weights=batch.sample_weights[keep_idx],
            original_prompts=[batch.original_prompts[i] for i in idx_list],
            rewritten_prompts=[batch.rewritten_prompts[i] for i in idx_list],
            concept_rewards=batch.concept_rewards[keep_idx],
            group_ids=[batch.group_ids[i] for i in idx_list],
        )

    @staticmethod
    def _build_repeated_index(size: int, target_size: int, device: torch.device) -> torch.Tensor:
        if size <= 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        if size >= target_size:
            return torch.arange(size, dtype=torch.long, device=device)

        repeats = int(math.ceil(target_size / float(size)))
        base = torch.arange(size, dtype=torch.long, device=device)
        return base.repeat(repeats)[:target_size]

    def _run_ppo_epochs(self) -> None:
        if self.policy_model is None or self.reference_model is None:
            raise RuntimeError("Local GRPO run requested without initialized model components")
        if self.optimizer is None:
            raise RuntimeError("Local GRPO run requested without optimizer")

        final_rewards = torch.tensor(self.buffer._rewards, dtype=torch.float32)
        concept_rewards = torch.tensor(self.buffer._concept_rewards, dtype=torch.float32)
        rewards = (
            settings.final_reward_beta * final_rewards
            + settings.concept_reward_alpha * concept_rewards
        )
        if settings.normalize_rewards and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        effective_group_ids = _resolve_grpo_group_ids(
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

        fallback_advantages = rewards.clone()
        if fallback_advantages.numel() > 1:
            fallback_advantages = (
                fallback_advantages - fallback_advantages.mean()
            ) / (fallback_advantages.std(unbiased=False) + 1e-6)

        grpo_advantages = _compute_grpo_relative_rewards(
            rewards,
            effective_group_ids,
            settings.grpo_min_group_size,
        )
        fallback_mask_base = _build_grpo_fallback_mask(
            effective_group_ids,
            settings.grpo_min_group_size,
            device=grpo_advantages.device,
        )
        advantages = torch.where(fallback_mask_base, fallback_advantages, grpo_advantages)

        group_reward_std = _group_reward_std_mean(
            rewards,
            effective_group_ids,
            settings.grpo_min_group_size,
        )
        logger.info(
            "grpo_advantage_diag | rewards_mean=%.6f | group_reward_std=%.6f | advantage_mean=%.6f | advantage_std=%.6f | fallback_samples=%d",
            float(rewards.mean().item()) if rewards.numel() else 0.0,
            group_reward_std,
            float(advantages.mean().item()) if advantages.numel() else 0.0,
            float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
            int(fallback_mask_base.sum().item()),
        )

        batch = self.buffer.build(advantages)
        loaded_batch_size = len(batch.rewritten_prompts)
        if loaded_batch_size == 0:
            return

        min_effective_batch_size = max(int(settings.ppo_min_effective_batch_size), 1)

        for epoch in range(settings.ppo_epochs):
            epoch_diag = {
                "batch_size_loaded": loaded_batch_size,
                "batch_size_after_filtering": loaded_batch_size,
                "batch_size_after_stability_pad": loaded_batch_size,
                "invalid_span_count": 0,
                "skipped_due_to_mask_count": 0,
                "skipped_due_to_nan_reward": 0,
                "skipped_due_to_advantage_zero": 0,
                "fallback_samples": int(fallback_mask_base.sum().item()),
                "optimizer_steps": 0,
            }

            current_batch = batch
            fallback_mask = fallback_mask_base.clone()

            tokenized = self.policy_model.tokenize_with_action_mask(
                batch.original_prompts,
                batch.rewritten_prompts,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask")
            action_mask = tokenized["action_mask"]
            valid_action = tokenized.get("valid_action")

            invalid_span_mask = action_mask.sum(dim=-1) <= 0
            if valid_action is not None:
                invalid_span_mask = invalid_span_mask | (~valid_action)

            invalid_span_count = int(invalid_span_mask.sum().item())
            epoch_diag["invalid_span_count"] = invalid_span_count
            if invalid_span_count > 0:
                epoch_diag["skipped_due_to_mask_count"] = invalid_span_count
                keep_idx = torch.nonzero(~invalid_span_mask, as_tuple=False).squeeze(-1)
                if keep_idx.numel() == 0:
                    logger.warning(
                        "grpo_epoch_skip | epoch=%d | reason=no_valid_action_spans",
                        epoch + 1,
                    )
                    continue
                current_batch = self._select_rollout_batch(current_batch, keep_idx)
                input_ids = input_ids[keep_idx]
                attention_mask = self._select_optional_tensor(attention_mask, keep_idx)
                action_mask = action_mask[keep_idx]
                fallback_mask = fallback_mask[keep_idx]

            finite_reward_mask = torch.isfinite(current_batch.rewards)
            dropped_nan_reward = int((~finite_reward_mask).sum().item())
            if dropped_nan_reward > 0:
                epoch_diag["skipped_due_to_nan_reward"] += dropped_nan_reward
                keep_idx = torch.nonzero(finite_reward_mask, as_tuple=False).squeeze(-1)
                if keep_idx.numel() == 0:
                    logger.warning(
                        "grpo_epoch_skip | epoch=%d | reason=all_rewards_non_finite",
                        epoch + 1,
                    )
                    continue
                current_batch = self._select_rollout_batch(current_batch, keep_idx)
                input_ids = input_ids[keep_idx]
                attention_mask = self._select_optional_tensor(attention_mask, keep_idx)
                action_mask = action_mask[keep_idx]
                fallback_mask = fallback_mask[keep_idx]

            advantage_keep_mask = torch.isfinite(current_batch.advantages)
            dropped_non_finite_adv = int((~advantage_keep_mask).sum().item())
            if dropped_non_finite_adv > 0:
                epoch_diag["skipped_due_to_advantage_zero"] += dropped_non_finite_adv
                keep_idx = torch.nonzero(advantage_keep_mask, as_tuple=False).squeeze(-1)
                if keep_idx.numel() == 0:
                    logger.warning(
                        "grpo_epoch_skip | epoch=%d | reason=all_advantages_non_finite",
                        epoch + 1,
                    )
                    continue
                current_batch = self._select_rollout_batch(current_batch, keep_idx)
                input_ids = input_ids[keep_idx]
                attention_mask = self._select_optional_tensor(attention_mask, keep_idx)
                action_mask = action_mask[keep_idx]
                fallback_mask = fallback_mask[keep_idx]

            n = len(current_batch.rewritten_prompts)
            epoch_diag["batch_size_after_filtering"] = n
            if n == 0:
                continue

            if n < min_effective_batch_size:
                expand_idx = self._build_repeated_index(
                    size=n,
                    target_size=min_effective_batch_size,
                    device=input_ids.device,
                )
                current_batch = self._select_rollout_batch(current_batch, expand_idx)
                input_ids = input_ids[expand_idx]
                attention_mask = self._select_optional_tensor(attention_mask, expand_idx)
                action_mask = action_mask[expand_idx]
                fallback_mask = fallback_mask[expand_idx]
                n = len(current_batch.rewritten_prompts)

            epoch_diag["batch_size_after_stability_pad"] = n
            epoch_diag["fallback_samples"] = int(fallback_mask.sum().item())

            ref_chunks_cpu = []
            with torch.no_grad():
                for start in range(0, n, settings.batch_size):
                    end = min(start + settings.batch_size, n)
                    mb_ids = input_ids[start:end]
                    mb_mask = attention_mask[start:end] if attention_mask is not None else None
                    mb_action_mask = action_mask[start:end]
                    ref_lp = self.reference_model.get_sequence_log_prob(
                        mb_ids,
                        mb_mask,
                        mb_action_mask,
                    )
                    ref_chunks_cpu.append(ref_lp.detach().cpu())

            ref_log_probs = torch.cat(ref_chunks_cpu, dim=0).to(self.policy_model.device)

            self.optimizer.zero_grad()
            accum_counter = 0

            for start in range(0, n, settings.batch_size):
                end = min(start + settings.batch_size, n)
                mb_ids = input_ids[start:end]
                mb_mask = attention_mask[start:end] if attention_mask is not None else None
                mb_action_mask = action_mask[start:end]

                token_log_probs_new, _ = self.policy_model(mb_ids, mb_mask)
                if mb_mask is not None:
                    mb_action_mask = mb_action_mask * mb_mask[:, 1:].to(dtype=mb_action_mask.dtype)
                seq_log_prob_new = (token_log_probs_new * mb_action_mask).sum(dim=-1)

                old_log_prob_mb = current_batch.log_probs_old[start:end]
                if not torch.isfinite(seq_log_prob_new).all() or not torch.isfinite(old_log_prob_mb).all():
                    logger.warning(
                        "grpo_minibatch_skip | epoch=%d | start=%d | end=%d | reason=non_finite_log_prob",
                        epoch + 1,
                        start,
                        end,
                    )
                    continue

                ratio = torch.exp(seq_log_prob_new - old_log_prob_mb)
                ref_mb = ref_log_probs[start:end].to(seq_log_prob_new.device)
                kl_penalty = self.kl_controller.compute_kl(seq_log_prob_new, ref_mb)

                adv_mb = current_batch.advantages[start:end]
                fallback_mb = fallback_mask[start:end]
                sample_weights = torch.clamp(
                    current_batch.sample_weights[start:end].to(seq_log_prob_new.device),
                    min=0.0,
                )
                normalizer = torch.clamp(sample_weights.sum(), min=1e-8)

                grpo_loss_per_sample = -(adv_mb * seq_log_prob_new)
                ppo_surr1 = ratio * adv_mb
                ppo_surr2 = torch.clamp(ratio, 1.0 - settings.epsilon, 1.0 + settings.epsilon) * adv_mb
                ppo_loss_per_sample = -torch.min(ppo_surr1, ppo_surr2)

                policy_loss_per_sample = torch.where(
                    fallback_mb,
                    ppo_loss_per_sample,
                    grpo_loss_per_sample,
                )
                policy_loss = (policy_loss_per_sample * sample_weights).sum() / normalizer
                kl_loss = (torch.clamp(kl_penalty, min=0.0) * sample_weights).sum() / normalizer
                total_loss = policy_loss + (settings.beta * kl_loss)

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

                if accum_counter % settings.gradient_accumulation_steps == 0 or end == n:
                    torch.nn.utils.clip_grad_norm_(list(self.policy_model.parameters()), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_diag["optimizer_steps"] += 1
                    self.training_step += 1

                self.last_loss = float(total_loss.detach().item())
                logger.info(
                    "grpo_loss_diag | epoch=%d | start=%d | end=%d | policy_loss=%.6f | kl=%.6f | total_loss=%.6f | fallback=%d/%d",
                    epoch + 1,
                    start,
                    end,
                    float(policy_loss.detach().item()),
                    float(kl_loss.detach().item()),
                    self.last_loss,
                    int(fallback_mb.sum().item()),
                    int(fallback_mb.numel()),
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info(
                "GRPO Epoch %d/%d | step=%d | loss=%.4f | KL=%.4f | group_reward_std=%.4f | advantage_mean=%.4f | advantage_std=%.4f | batch_size_loaded=%d | batch_size_after_filtering=%d | batch_size_after_stability_pad=%d | fallback_samples=%d | optimizer_steps=%d",
                epoch + 1,
                settings.ppo_epochs,
                self.training_step,
                self.last_loss,
                self.kl_controller.last_kl,
                group_reward_std,
                float(current_batch.advantages.mean().item()) if len(current_batch.advantages) else 0.0,
                float(current_batch.advantages.std(unbiased=False).item()) if len(current_batch.advantages) > 1 else 0.0,
                epoch_diag["batch_size_loaded"],
                epoch_diag["batch_size_after_filtering"],
                epoch_diag["batch_size_after_stability_pad"],
                epoch_diag["fallback_samples"],
                epoch_diag["optimizer_steps"],
            )

    def _save_checkpoint(self) -> None:
        if self.policy_model is None or self.value_head is None or self.optimizer is None:
            raise RuntimeError("Cannot save local checkpoint without initialized model state")

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
        rewriter_service_url = os.environ.get("REWRITER_SERVICE_URL", "http://localhost:8000")
        endpoint = f"{rewriter_service_url}/reload_checkpoint"
        try:
            requests.post(endpoint, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Failed to notify rewriter checkpoint reload: %s", exc)
