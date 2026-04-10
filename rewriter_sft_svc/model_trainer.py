"""
model_trainer.py — rewriter_sft_svc

Loads Phi-3 Mini with QLoRA adapters and fine-tunes using TRL SFTTrainer.
Compatible with TRL versions that expect `tokenizer=` instead of `processing_class=`.
"""

import logging
import os
from typing import List, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from config import (
    BASE_MODEL_NAME,
    CHECKPOINT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    SAVE_STEPS,
    LOGGING_STEPS,
    WARMUP_RATIO,
    LR_SCHEDULER,
    TRUST_REMOTE_CODE,
)

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on hardware."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_name: str = BASE_MODEL_NAME) -> AutoTokenizer:
    """Load and configure the Phi-3 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded: {model_name}")
    return tokenizer


def load_model_for_training(
    model_name: str = BASE_MODEL_NAME,
    device: str = "cpu",
) -> AutoModelForCausalLM:
    """
    Load Phi-3 Mini.

    On CUDA:
        use 4-bit QLoRA with BitsAndBytes.
    On MPS/CPU:
        load without quantization.
    """
    if device == "cuda":
        logger.info("Loading model with 4-bit QLoRA (BitsAndBytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info(f"Loading model on {device} without quantization.")
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=dtype,
            device_map={"": device},
        )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    logger.info("Base model loaded.")
    return model


def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Wrap model with LoRA adapters."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA adapters applied.")
    return model


def records_to_hf_dataset(text_records: List[Dict[str, str]]) -> Dataset:
    """Convert list of {'text': '...'} dicts to a Hugging Face Dataset."""
    return Dataset.from_list(text_records)


def train(
    train_texts: List[Dict[str, str]],
    val_texts: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    checkpoint_dir: str = CHECKPOINT_DIR,
    num_epochs: int = TRAIN_EPOCHS,
) -> str:
    """
    Main training function.

    Args:
        train_texts: List of {"text": "..."} dicts.
        val_texts:   List of {"text": "..."} dicts.
        tokenizer:   Pre-loaded tokenizer.
        checkpoint_dir: Where to save the adapter checkpoint.
        num_epochs: Number of training epochs.

    Returns:
        Path to the saved checkpoint.
    """
    device = detect_device()
    logger.info(f"Training device: {device}")

    model = load_model_for_training(BASE_MODEL_NAME, device)
    model = apply_lora(model)

    train_dataset = records_to_hf_dataset(train_texts)
    val_dataset = records_to_hf_dataset(val_texts)

    optimiser = "paged_adamw_8bit" if device == "cuda" else "adamw_torch"

    os.makedirs(checkpoint_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=optimiser,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=(device == "cuda"),
        dataset_text_field="text",
        report_to="none",
        overwrite_output_dir=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT training …")
    trainer.train()
    logger.info("Training complete.")

    trainer.model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

    return checkpoint_dir