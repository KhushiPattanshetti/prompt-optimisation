import argparse
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from ..app.config import settings
from ..models.policy_model import PolicyModel
from ..models.reference_model import ReferenceModel
from ..models.value_head import ValueHead
from ..rl.advantage import compute_gae
from ..rl.grpo_utils import (
    build_grpo_fallback_mask,
    build_repeated_index,
    compute_grpo_relative_rewards,
    resolve_grpo_group_ids,
    resolve_group_id,
    resolve_sample_weight,
    select_rollout_batch,
)
from ..rl.kl_controller import KLController
from ..rl.rollout_buffer import RolloutBuffer
from ..schemas.rollout_schema import RolloutEntry
from ..storage.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distributed_train_once")


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None

    candidates = [
        path for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint_")
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


def _fuse_action_attention_mask(
    action_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return action_mask
    return action_mask * attention_mask[:, 1:].to(dtype=action_mask.dtype)


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
        rank, local_rank, world_size, elastic_error_file or "<unset>",
    )

    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed GRPO requires CUDA")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

    try:
        entries = _load_entries(entries_file)
        n_entries = len(entries)
        if n_entries == 0:
            if rank == 0:
                result = {
                    "ok": True, "training_step": 0, "last_loss": 0.0,
                    "kl_divergence": 0.0, "entries": 0,
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

        value_head = ValueHead(hidden_size=settings.hidden_size).to(device)
        if latest_ckpt is not None:
            value_head_path = latest_ckpt / "value_head.pt"
            if value_head_path.exists():
                try:
                    state = torch.load(value_head_path, map_location=device)
                    if isinstance(state, dict) and state:
                        value_head.load_state_dict(state, strict=True)
                except Exception:
                    logger.warning(
                        "value_head_load_failed | path=%s | using_fresh_head=true",
                        value_head_path, exc_info=True,
                    )

        value_head.eval()
        value_estimates: List[float] = []
        with torch.no_grad():
            for vs_start in range(0, len(entries), settings.batch_size):
                vs_batch = [e.original_prompt for e in entries[vs_start:vs_start + settings.batch_size]]
                encoded = policy_model.tokenize(vs_batch)
                _, hidden_states, _ = policy_model(
                    encoded["input_ids"],
                    encoded.get("attention_mask"),
                )
                vals = value_head(hidden_states)
                value_estimates.extend(float(v) for v in vals.detach().cpu().tolist())

        buffer = RolloutBuffer(device=str(device))
        for idx, entry in enumerate(entries):
            concept_reward = float(entry.concept_reward) if entry.concept_reward is not None else float(entry.reward)
            buffer.store(
                reward=entry.reward,
                log_prob_old=entry.log_prob_old,
                value_estimate=value_estimates[idx],
                original_prompt=entry.original_prompt,
                rewritten_prompt=entry.rewritten_prompt,
                concept_reward=concept_reward,
                group_id=resolve_group_id(entry),
                sample_weight=resolve_sample_weight(entry),
            )

        final_rewards = torch.tensor(buffer._rewards, dtype=torch.float32, device=device)
        concept_rewards = torch.tensor(buffer._concept_rewards, dtype=torch.float32, device=device)
        rewards = (
            settings.final_reward_beta * final_rewards
            + settings.concept_reward_alpha * concept_rewards
        )

        effective_group_ids = resolve_grpo_group_ids(
            buffer._group_ids,
            settings.grpo_min_group_size,
            settings.grpo_group_size,
        )
        if rank == 0 and effective_group_ids != buffer._group_ids:
            logger.info(
                "distributed_grpo_group_fallback_applied | source_groups=%d | fallback_group_size=%d",
                len(set(buffer._group_ids)), settings.grpo_group_size,
            )

        grpo_relative = compute_grpo_relative_rewards(
            rewards, effective_group_ids, settings.grpo_min_group_size,
        )

        fallback_advantages = rewards.clone()
        if fallback_advantages.numel() > 1:
            fallback_advantages = (
                fallback_advantages - fallback_advantages.mean()
            ) / (fallback_advantages.std(unbiased=False) + 1e-6)
        fallback_mask = build_grpo_fallback_mask(
            effective_group_ids, settings.grpo_min_group_size, device=grpo_relative.device,
        )
        pure_reward_advantages = torch.where(fallback_mask, fallback_advantages, grpo_relative)

        values_tensor = torch.tensor(buffer._values, dtype=torch.float32, device=device)
        has_value_estimates = values_tensor.abs().sum().item() > 0
        if has_value_estimates:
            gae_advantages = compute_gae(
                rewards, values_tensor,
                gamma=settings.gamma, lam=settings.lam, normalize=True,
            )
            lam_h = settings.hybrid_advantage_lambda
            advantages = lam_h * gae_advantages + (1.0 - lam_h) * pure_reward_advantages
        else:
            advantages = pure_reward_advantages

        batch = buffer.build(advantages)

        n = len(batch.rewritten_prompts)
        if n == 0:
            if rank == 0:
                result = {
                    "ok": True, "training_step": training_step,
                    "last_loss": last_loss, "kl_divergence": last_kl, "entries": 0,
                }
                result_file.write_text(json.dumps(result), encoding="utf-8")
            dist.barrier()
            return

        min_effective_batch_size = max(int(settings.ppo_min_effective_batch_size), 1)
        if n < min_effective_batch_size:
            original_n = n
            expand_idx = build_repeated_index(n, min_effective_batch_size, batch.rewards.device)
            batch = select_rollout_batch(batch, expand_idx)
            fallback_mask = fallback_mask[expand_idx]
            n = len(batch.rewritten_prompts)
            if rank == 0:
                logger.info(
                    "distributed_batch_stability_pad | before=%d | after=%d",
                    original_n, n,
                )

        global_batch_size = settings.batch_size * world_size

        for epoch in range(settings.ppo_epochs):
            optimizer.zero_grad()
            accum_counter = 0

            for global_start in range(0, n, global_batch_size):
                local_start, local_end = _build_rank_slice(
                    rank, world_size, global_start, settings.batch_size, n,
                )

                if local_start < local_end:
                    local_original = batch.original_prompts[local_start:local_end]
                    local_rewritten = batch.rewritten_prompts[local_start:local_end]
                    tokenized = policy_model.tokenize_with_action_mask(local_original, local_rewritten)

                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized.get("attention_mask")
                    action_mask = tokenized["action_mask"]
                    valid_action = tokenized.get("valid_action")

                    fused_mask = _fuse_action_attention_mask(action_mask, attention_mask)

                    local_log_probs_old = batch.log_probs_old[local_start:local_end]
                    local_advantages = batch.advantages[local_start:local_end]
                    local_sample_weights = batch.sample_weights[local_start:local_end]
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
                        fused_mask = fused_mask[valid_idx]
                        local_log_probs_old = local_log_probs_old[valid_idx]
                        local_advantages = local_advantages[valid_idx]
                        local_sample_weights = local_sample_weights[valid_idx]
                        local_fallback = local_fallback[valid_idx]

                    with torch.no_grad():
                        try:
                            ref_log_probs = reference_model.get_sequence_log_prob(
                                input_ids, attention_mask, fused_mask,
                            )
                        except TypeError:
                            ref_log_probs = reference_model.get_sequence_log_prob(input_ids, attention_mask)

                    token_log_probs_new, _, _ = policy_model(input_ids, attention_mask)
                    seq_log_prob_new = (token_log_probs_new * fused_mask).sum(dim=-1)

                    kl_penalty = kl_controller.compute_kl(seq_log_prob_new, ref_log_probs)
                    ratio = torch.exp(seq_log_prob_new - local_log_probs_old)
                    ratio = torch.clamp(ratio, 0.0, settings.ratio_clip_max)
                    sample_weights = torch.clamp(local_sample_weights, min=0.0)
                    normalizer = torch.clamp(sample_weights.sum(), min=1e-8)

                    grpo_loss_per_sample = -(local_advantages * seq_log_prob_new)
                    ppo_surr1 = ratio * local_advantages
                    ppo_surr2 = torch.clamp(
                        ratio, 1.0 - settings.epsilon, 1.0 + settings.epsilon,
                    ) * local_advantages
                    ppo_loss_per_sample = -torch.min(ppo_surr1, ppo_surr2)

                    policy_loss_per_sample = torch.where(
                        local_fallback, ppo_loss_per_sample, grpo_loss_per_sample,
                    )
                    policy_loss = (policy_loss_per_sample * sample_weights).sum() / normalizer
                    kl_loss = (kl_penalty * sample_weights).sum() / normalizer
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

            if rank == 0:
                logger.info(
                    "distributed_epoch_complete | epoch=%d/%d | step=%d | loss=%.4f | kl=%.4f",
                    epoch + 1, settings.ppo_epochs, training_step, last_loss, last_kl,
                )

            if kl_controller.last_kl > settings.max_abs_kl_for_update:
                if rank == 0:
                    logger.warning(
                        "distributed_kl_early_stop | epoch=%d | kl=%.6f | threshold=%.6f",
                        epoch + 1, kl_controller.last_kl, settings.max_abs_kl_for_update,
                    )
                break

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
            rank, local_rank, elastic_error_file or "<unset>",
        )
        if rank == 0:
            result = {"ok": False, "error": str(exc)}
            result_file.write_text(json.dumps(result), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
