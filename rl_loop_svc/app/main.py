import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api_routes import router, set_training_loop
from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _use_distributed_training() -> bool:
    return settings.distributed_enabled and settings.distributed_world_size > 1


def _resolve_device_label(configured_index: int, role: str) -> str:
    if not torch.cuda.is_available():
        logger.info("%s_device_fallback_cpu | reason=no_cuda", role)
        return "cpu"

    available = torch.cuda.device_count()
    if available <= 0:
        logger.info("%s_device_fallback_cpu | reason=no_visible_cuda_devices", role)
        return "cpu"

    if configured_index < 0 or configured_index >= available:
        logger.warning(
            "%s_device_index_out_of_range | requested=%d | visible=%d | fallback=0",
            role,
            configured_index,
            available,
        )
        return "cuda:0"

    return f"cuda:{configured_index}"


def _log_gpu_inventory() -> None:
    if not torch.cuda.is_available():
        logger.info("gpu_inventory | cuda_available=false")
        return

    count = torch.cuda.device_count()
    logger.info("gpu_inventory | cuda_available=true | device_count=%d", count)
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        logger.info(
            "gpu_device | index=%d | name=%s | total_mem_gb=%.2f",
            idx,
            props.name,
            total_gb,
        )


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None

    candidates = [
        p for p in checkpoints_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup_t0 = time.perf_counter()
    logger.info(
        "rl_startup_begin | model=%s | distributed=%s | world_size=%d",
        settings.model_name,
        _use_distributed_training(),
        settings.distributed_world_size,
    )

    try:
        settings.rollouts_dir.mkdir(parents=True, exist_ok=True)
        settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "startup_stage | stage=dirs_ready | rollouts_dir=%s | checkpoints_dir=%s",
            settings.rollouts_dir,
            settings.checkpoints_dir,
        )

        if settings.startup_log_gpu_inventory:
            _log_gpu_inventory()

        from rl.training_loop import TrainingLoop
        from storage.checkpoint_manager import CheckpointManager
        from storage.rollout_loader import RolloutLoader

        stage_t0 = time.perf_counter()
        latest_ckpt = _find_latest_checkpoint(settings.checkpoints_dir)
        load_checkpoint_weights = not settings.ppo_debug_mode
        checkpoint_path = None
        if latest_ckpt is not None and load_checkpoint_weights:
            lora_dir = latest_ckpt / "lora_adapter"
            if lora_dir.exists():
                checkpoint_path = str(lora_dir)
        elif latest_ckpt is not None and not load_checkpoint_weights:
            logger.warning(
                "startup_debug_mode_checkpoint_load_skipped | checkpoint=%s",
                latest_ckpt,
            )
        logger.info(
            "startup_stage | stage=checkpoint_scan_done | elapsed_s=%.2f | latest_ckpt=%s | lora_path=%s",
            time.perf_counter() - stage_t0,
            latest_ckpt,
            checkpoint_path,
        )

        stage_t0 = time.perf_counter()
        rollout_loader = RolloutLoader(settings.rollouts_dir)
        checkpoint_manager = CheckpointManager(
            checkpoints_dir=settings.checkpoints_dir,
            max_checkpoints=settings.max_checkpoints,
        )
        if _use_distributed_training():
            training_loop = TrainingLoop(
                rollout_loader=rollout_loader,
                checkpoint_manager=checkpoint_manager,
                policy_model=None,
                reference_model=None,
                value_head=None,
            )
            logger.info("startup_stage | stage=distributed_proxy_ready")
            policy_device = "distributed"
            reference_device = "distributed"
        else:
            from models.policy_model import PolicyModel
            from models.reference_model import ReferenceModel
            from models.value_head import ValueHead

            policy_device = _resolve_device_label(settings.policy_cuda_device, "policy")
            reference_device = _resolve_device_label(settings.reference_cuda_device, "reference")

            model_t0 = time.perf_counter()
            policy_model = PolicyModel(
                model_name=settings.model_name,
                checkpoint_path=checkpoint_path,
                device=policy_device,
            )
            logger.info(
                "startup_stage | stage=policy_model_loaded | elapsed_s=%.2f | device=%s",
                time.perf_counter() - model_t0,
                policy_model.device,
            )

            model_t0 = time.perf_counter()
            reference_model = ReferenceModel(
                model_name=settings.model_name,
                device=reference_device,
            )
            logger.info(
                "startup_stage | stage=reference_model_loaded | elapsed_s=%.2f | device=%s",
                time.perf_counter() - model_t0,
                reference_model.device,
            )

            model_t0 = time.perf_counter()
            value_head = ValueHead(hidden_size=settings.hidden_size).to(policy_model.device)
            if latest_ckpt is not None and load_checkpoint_weights:
                value_head_path = latest_ckpt / "value_head.pt"
                if value_head_path.exists():
                    state = torch.load(value_head_path, map_location=policy_model.device)
                    value_head.load_state_dict(state)
            logger.info(
                "startup_stage | stage=value_head_ready | elapsed_s=%.2f",
                time.perf_counter() - model_t0,
            )

            training_loop = TrainingLoop(
                rollout_loader=rollout_loader,
                checkpoint_manager=checkpoint_manager,
                policy_model=policy_model,
                reference_model=reference_model,
                value_head=value_head,
            )

        set_training_loop(training_loop, settings.rollouts_dir)
        logger.info(
            "startup_stage | stage=training_loop_bound | elapsed_s=%.2f",
            time.perf_counter() - stage_t0,
        )

        logger.info(
            "rl_startup_complete | total_elapsed_s=%.2f | policy_device=%s | reference_device=%s",
            time.perf_counter() - startup_t0,
            policy_device,
            reference_device,
        )

        yield
    except Exception:
        logger.exception("rl_startup_failed")
        raise
    finally:
        logger.info("rl_shutdown")


app = FastAPI(
    title="RL Training Microservice",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
