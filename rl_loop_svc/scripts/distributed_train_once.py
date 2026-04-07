import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from models.policy_model import PolicyModel
from models.reference_model import ReferenceModel
from models.value_head import ValueHead
from rl.kl_controller import KLController
from rl.rollout_buffer import RolloutBatch, RolloutBuffer
from schemas.rollout_schema import RolloutEntry
from storage.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distributed_train_once")


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
        std = group_rewards.std(unbiased=False)
        relative[idx_tensor] = (group_rewards - group_rewards.mean()) / (std + 1e-6)

    return relative


def _build_grpo_fallback_mask(
    group_ids: List[str],
    min_group_size: int,
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(str(group_id) for group_id in group_ids)
    threshold = max(int(min_group_size), 2)
    return torch.tensor(
        [counts.get(str(group_id), 0) < threshold for group_id in group_ids],
        dtype=torch.bool,
        device=device,
    )


def _resolve_grpo_group_ids(
    group_ids: List[str],
    min_group_size: int,
    fallback_group_size: int,
) -> List[str]:
    normalized = [str(group_id) for group_id in group_ids]
    if not normalized:
        return normalized

    threshold = max(int(min_group_size), 2)
    counts = Counter(normalized)
    if any(size >= threshold for size in counts.values()):
        return normalized

    group_size = max(int(fallback_group_size), threshold)
    return [f"grpo_auto_{idx // group_size}" for idx in range(len(normalized))]


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


def _build_repeated_index(size: int, target_size: int, device: torch.device) -> torch.Tensor:
    if size <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if size >= target_size:
        return torch.arange(size, dtype=torch.long, device=device)

    repeats = int(math.ceil(target_size / float(size)))
    base = torch.arange(size, dtype=torch.long, device=device)
    return base.repeat(repeats)[:target_size]


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None

    candidates = [
        path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _all_reduce_gradients(parameters: Iterable[torch.nn.Parameter], world_size: int) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size


def _all_reduce_scalar(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return float(tensor.item())


def _build_rank_slice(
    rank: int,
    world_size: int,
    global_start: int,
    local_batch_size: int,
    total_items: int,
) -> tuple[int, int]:
    start = global_start + rank * local_batch_size
    end = min(start + local_batch_size, total_items)
    return start, end


def _load_entries(entries_file: Path) -> List[RolloutEntry]:
    payload = json.loads(entries_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("entries payload must be a JSON list")
    return [RolloutEntry.model_validate(row) for row in payload]


@record
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one distributed PPO training cycle")
    parser.add_argument("--entries-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    args = parser.parse_args()

    entries_file = Path(args.entries_file)
    result_file = Path(args.result_file)
    checkpoints_dir = Path(args.checkpoints_dir)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    elastic_error_file = os.environ.get("TORCHELASTIC_ERROR_FILE", "")

    logger.info(
        "distributed_rank_bootstrap | rank=%d | local_rank=%d | world_size=%d | elastic_error_file=%s",
        rank,
        local_rank,
        world_size,
        elastic_error_file or "<unset>",
    )

    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed GRPO requires CUDA")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )

    try:
        entries = _load_entries(entries_file)
        n_entries = len(entries)
        if n_entries == 0:
            if rank == 0:
                result = {
                    "ok": True,
                    "training_step": 0,
                    "last_loss": 0.0,
                    "kl_divergence": 0.0,
                    "entries": 0,
                }
                result_file.write_text(json.dumps(result), encoding="utf-8")
            dist.barrier()
            return

        latest_ckpt = _find_latest_checkpoint(checkpoints_dir)
        checkpoint_path = None
        if latest_ckpt is not None:
            lora_dir = latest_ckpt / "lora_adapter"
            if lora_dir.exists():
                checkpoint_path = str(lora_dir)

        policy_model = PolicyModel(
            model_name=settings.model_name,
            checkpoint_path=checkpoint_path,
            device=str(device),
        )
        reference_model = ReferenceModel(
            model_name=settings.model_name,
            device=str(device),
        )
        trainable_params = list(policy_model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=settings.learning_rate)
        kl_controller = KLController(beta=settings.beta)

        checkpoint_manager = CheckpointManager(
            checkpoints_dir=checkpoints_dir,
            max_checkpoints=settings.max_checkpoints,
        )
        latest_meta = checkpoint_manager.load_latest_meta() or {}
        training_step = int(latest_meta.get("training_step", 0))
        last_loss = float(latest_meta.get("last_loss", 0.0))
        last_kl = float(latest_meta.get("kl_divergence", 0.0))

        value_head = ValueHead(hidden_size=settings.hidden_size)
        if latest_ckpt is not None:
            value_head_path = latest_ckpt / "value_head.pt"
            if value_head_path.exists():
                try:
                    state = torch.load(value_head_path, map_location="cpu")
                    if isinstance(state, dict) and state:
                        value_head.load_state_dict(state, strict=True)
                except Exception:
                    logger.warning(
                        "value_head_load_failed | path=%s | using_fresh_head=true",
                        value_head_path,
                        exc_info=True,
                    )

        buffer = RolloutBuffer(device=str(device))
        for entry in entries:
            concept_reward = float(entry.concept_reward) if entry.concept_reward is not None else float(entry.reward)
            group_id = _resolve_group_id(entry)
            sample_weight = _resolve_sample_weight(entry)
            buffer.store(
                reward=entry.reward,
                log_prob_old=entry.log_prob_old,
                value_estimate=0.0,
                original_prompt=entry.original_prompt,
                rewritten_prompt=entry.rewritten_prompt,
                concept_reward=concept_reward,
                group_id=group_id,
                sample_weight=sample_weight,
            )

        final_rewards = torch.tensor(buffer._rewards, dtype=torch.float32, device=device)
        concept_rewards = torch.tensor(buffer._concept_rewards, dtype=torch.float32, device=device)
        rewards = (
            settings.final_reward_beta * final_rewards
            + settings.concept_reward_alpha * concept_rewards
        )
        if settings.normalize_rewards and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        effective_group_ids = _resolve_grpo_group_ids(
            buffer._group_ids,
            settings.grpo_min_group_size,
            settings.grpo_group_size,
        )
        if rank == 0 and effective_group_ids != buffer._group_ids:
            logger.info(
                "distributed_grpo_group_fallback_applied | source_groups=%d | fallback_group_size=%d",
                len(set(buffer._group_ids)),
                settings.grpo_group_size,
            )

        grpo_relative = _compute_grpo_relative_rewards(
            rewards,
            effective_group_ids,
            settings.grpo_min_group_size,
        )

        fallback_advantages = rewards.clone()
        if fallback_advantages.numel() > 1:
            fallback_advantages = (
                fallback_advantages - fallback_advantages.mean()
            ) / (fallback_advantages.std(unbiased=False) + 1e-6)
        fallback_mask = _build_grpo_fallback_mask(
            effective_group_ids,
            settings.grpo_min_group_size,
            device=grpo_relative.device,
        )
        advantages = torch.where(fallback_mask, fallback_advantages, grpo_relative)

        batch = buffer.build(advantages)

        n = len(batch.rewritten_prompts)
        if n == 0:
            if rank == 0:
                result = {
                    "ok": True,
                    "training_step": training_step,
                    "last_loss": last_loss,
                    "kl_divergence": last_kl,
                    "entries": 0,
                }
                result_file.write_text(json.dumps(result), encoding="utf-8")
            dist.barrier()
            return

        min_effective_batch_size = max(int(settings.ppo_min_effective_batch_size), 1)
        if n < min_effective_batch_size:
            original_n = n
            expand_idx = _build_repeated_index(
                size=n,
                target_size=min_effective_batch_size,
                device=batch.rewards.device,
            )
            batch = _select_rollout_batch(batch, expand_idx)
            n = len(batch.rewritten_prompts)
            if rank == 0:
                logger.info(
                    "distributed_batch_stability_pad | before=%d | after=%d | min_effective_batch_size=%d",
                    original_n,
                    n,
                    min_effective_batch_size,
                )

        global_batch_size = settings.batch_size * world_size

        for epoch in range(settings.ppo_epochs):
            optimizer.zero_grad()
            accum_counter = 0

            for global_start in range(0, n, global_batch_size):
                local_start, local_end = _build_rank_slice(
                    rank=rank,
                    world_size=world_size,
                    global_start=global_start,
                    local_batch_size=settings.batch_size,
                    total_items=n,
                )

                if local_start < local_end:
                    local_original = batch.original_prompts[local_start:local_end]
                    local_rewritten = batch.rewritten_prompts[local_start:local_end]
                    tokenized = policy_model.tokenize_with_action_mask(local_original, local_rewritten)

                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized.get("attention_mask")
                    action_mask = tokenized["action_mask"]
                    valid_action = tokenized.get("valid_action")

                    local_rewards = batch.rewards[local_start:local_end]
                    local_log_probs_old = batch.log_probs_old[local_start:local_end]
                    local_advantages = batch.advantages[local_start:local_end]
                    local_concept_rewards = batch.concept_rewards[local_start:local_end]
                    local_sample_weights = batch.sample_weights[local_start:local_end]
                    local_group_ids = batch.group_ids[local_start:local_end]
                    local_fallback = fallback_mask[local_start:local_end]

                    if valid_action is not None:
                        valid_idx = torch.nonzero(valid_action, as_tuple=False).squeeze(-1)
                        if valid_idx.numel() == 0:
                            dummy = torch.zeros((), dtype=torch.float32, device=device)
                            for parameter in trainable_params:
                                dummy = dummy + (parameter.float().sum() * 0.0)
                            (dummy / settings.gradient_accumulation_steps).backward()
                            local_total_loss = 0.0
                            local_kl = 0.0
                            accum_counter += 1
                            should_step = (
                                accum_counter % settings.gradient_accumulation_steps == 0
                                or global_start + global_batch_size >= n
                            )
                            if should_step:
                                _all_reduce_gradients(trainable_params, world_size)
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad()

                                training_step += 1
                                last_loss = _all_reduce_scalar(local_total_loss, device, world_size)
                                last_kl = _all_reduce_scalar(local_kl, device, world_size)
                            continue

                        idx_list = [int(v) for v in valid_idx.detach().cpu().tolist()]
                        input_ids = input_ids[valid_idx]
                        if attention_mask is not None:
                            attention_mask = attention_mask[valid_idx]
                        action_mask = action_mask[valid_idx]
                        local_rewards = local_rewards[valid_idx]
                        local_log_probs_old = local_log_probs_old[valid_idx]
                        local_advantages = local_advantages[valid_idx]
                        local_concept_rewards = local_concept_rewards[valid_idx]
                        local_sample_weights = local_sample_weights[valid_idx]
                        local_fallback = local_fallback[valid_idx]
                        local_group_ids = [local_group_ids[i] for i in idx_list]
                        local_original = [local_original[i] for i in idx_list]
                        local_rewritten = [local_rewritten[i] for i in idx_list]

                    with torch.no_grad():
                        try:
                            ref_log_probs = reference_model.get_sequence_log_prob(
                                input_ids,
                                attention_mask,
                                action_mask,
                            )
                        except TypeError:
                            ref_log_probs = reference_model.get_sequence_log_prob(input_ids, attention_mask)

                    token_log_probs_new, hidden_states = policy_model(input_ids, attention_mask)
                    seq_log_prob_new = (token_log_probs_new * action_mask).sum(dim=-1)
                    _ = hidden_states

                    kl_penalty = kl_controller.compute_kl(seq_log_prob_new, ref_log_probs)
                    ratio = torch.exp(seq_log_prob_new - local_log_probs_old)
                    sample_weights = torch.clamp(local_sample_weights, min=0.0)
                    normalizer = torch.clamp(sample_weights.sum(), min=1e-8)

                    grpo_loss_per_sample = -(local_advantages * seq_log_prob_new)
                    ppo_surr1 = ratio * local_advantages
                    ppo_surr2 = torch.clamp(ratio, 1.0 - settings.epsilon, 1.0 + settings.epsilon) * local_advantages
                    ppo_loss_per_sample = -torch.min(ppo_surr1, ppo_surr2)

                    policy_loss_per_sample = torch.where(local_fallback, ppo_loss_per_sample, grpo_loss_per_sample)
                    policy_loss = (policy_loss_per_sample * sample_weights).sum() / normalizer
                    kl_loss = (torch.clamp(kl_penalty, min=0.0) * sample_weights).sum() / normalizer
                    total_loss = policy_loss + (settings.beta * kl_loss)

                    (total_loss / settings.gradient_accumulation_steps).backward()

                    local_total_loss = float(total_loss.detach().item())
                    local_kl = float(kl_loss.detach().item())
                else:
                    dummy = torch.zeros((), dtype=torch.float32, device=device)
                    for parameter in trainable_params:
                        dummy = dummy + (parameter.float().sum() * 0.0)
                    (dummy / settings.gradient_accumulation_steps).backward()
                    local_total_loss = 0.0
                    local_kl = 0.0

                accum_counter += 1
                should_step = (
                    accum_counter % settings.gradient_accumulation_steps == 0
                    or global_start + global_batch_size >= n
                )

                if not should_step:
                    continue

                _all_reduce_gradients(trainable_params, world_size)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                training_step += 1
                last_loss = _all_reduce_scalar(local_total_loss, device, world_size)
                last_kl = _all_reduce_scalar(local_kl, device, world_size)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if rank == 0:
                logger.info(
                    "distributed_epoch_complete | epoch=%d/%d | step=%d | loss=%.4f | kl=%.4f",
                    epoch + 1,
                    settings.ppo_epochs,
                    training_step,
                    last_loss,
                    last_kl,
                )

        if rank == 0:
            checkpoint_manager.save(
                policy_model=policy_model,
                value_head_state_dict=value_head.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                training_step=training_step,
                extra_meta={
                    "last_loss": last_loss,
                    "kl_divergence": last_kl,
                    "rollouts_loaded": n_entries,
                    "distributed_world_size": world_size,
                },
            )

        dist.barrier()

        if rank == 0:
            result = {
                "ok": True,
                "training_step": training_step,
                "last_loss": last_loss,
                "kl_divergence": last_kl,
                "entries": n_entries,
                "distributed_world_size": world_size,
            }
            result_file.write_text(json.dumps(result), encoding="utf-8")

    except Exception as exc:
        logger.exception(
            "distributed_train_failed | rank=%d | local_rank=%d | elastic_error_file=%s",
            rank,
            local_rank,
            elastic_error_file or "<unset>",
        )
        if rank == 0:
            result = {
                "ok": False,
                "error": str(exc),
            }
            result_file.write_text(json.dumps(result), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
