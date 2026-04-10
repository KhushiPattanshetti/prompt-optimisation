"""
train_sft.py — rewriter_sft_svc

Main entry point for the SFT training pipeline.

Usage:
    python train_sft.py
    python train_sft.py --test-only
    python train_sft.py --epochs 3
    python train_sft.py --dataset /workspace/data/structured_notes.csv --epochs 3
"""

import os

# Set cache + tokenizer env BEFORE importing transformers-related modules
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CHECKPOINT_DIR,
    DATASET_PATH,
    LOG_LEVEL,
    TRAIN_EPOCHS,
    TESTING_SUMMARY_PATH,
)
from dataset_loader import get_datasets
from model_trainer import load_tokenizer, train
from preprocessing import preprocess_for_training
from test_runner import run_all_tests, generate_testing_summary

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="rewriter_sft_svc — Phi-3 Mini SFT Trainer")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and only run tests + generate testing_summary.md",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {TRAIN_EPOCHS})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help=f"Path to training dataset file (CSV or JSON) (default: {DATASET_PATH})",
    )
    return parser.parse_args()


def run_training_pipeline(dataset_path: str, num_epochs: int):
    """Execute the full SFT training pipeline."""
    logger.info("━" * 60)
    logger.info("  rewriter_sft_svc — SFT Training Pipeline")
    logger.info("━" * 60)

    logger.info("[1/10] Loading dataset …")
    train_records, val_records, test_records = get_datasets(dataset_path)

    logger.info(
        f"[2/10] Dataset validated — train={len(train_records)}, "
        f"val={len(val_records)}, test={len(test_records)}"
    )

    logger.info("[3/10] Loading tokenizer …")
    tokenizer = load_tokenizer()

    logger.info("[4/10] Preprocessing dataset into chat format …")
    train_texts = preprocess_for_training(train_records, tokenizer)
    val_texts = preprocess_for_training(val_records, tokenizer)
    _ = preprocess_for_training(test_records, tokenizer)  # optional precheck

    logger.info("[5/10] Loading base model …")
    logger.info("[6/10] Applying QLoRA adapters …")
    logger.info("[7/10] Starting SFT training …")
    start = time.time()

    checkpoint_path = train(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        checkpoint_dir=CHECKPOINT_DIR,
        num_epochs=num_epochs,
    )

    elapsed = round(time.time() - start, 1)
    logger.info(f"[8/10] Training complete in {elapsed}s — checkpoint: {checkpoint_path}")

    logger.info("[9/10] Running schema-based evaluation on test set …")
    from preprocessing import validate_output_schema
    from config import REQUIRED_FIELDS

    passed = sum(
        1
        for r in test_records
        if validate_output_schema(r["structured clinical note"], REQUIRED_FIELDS)["valid"]
    )
    logger.info(f"        Test set schema pass rate: {passed}/{len(test_records)}")

    logger.info("[10/10] Running test suite and generating testing_summary.md …")


def main():
    args = parse_args()
    num_epochs = args.epochs if args.epochs else TRAIN_EPOCHS

    os.makedirs("/workspace/.cache/huggingface/transformers", exist_ok=True)
    os.makedirs("/workspace/.cache/huggingface/hub", exist_ok=True)

    if not args.test_only:
        if not os.path.exists(args.dataset):
            logger.error(
                f"Dataset not found: {args.dataset}\n"
                "Please provide a valid dataset path, for example:\n"
                "    python train_sft.py --dataset /workspace/data/structured_notes.csv"
            )
            sys.exit(1)
        run_training_pipeline(args.dataset, num_epochs)

    logger.info("\nRunning comprehensive test suite …\n")
    results = run_all_tests()
    generate_testing_summary(results, TESTING_SUMMARY_PATH)

    failed = sum(len(r.failed) + len(r.errors) for r in results)
    passed = sum(len(r.passed) for r in results)

    logger.info("\n" + "━" * 60)
    logger.info(f"  Test Suite Complete — {passed} passed / {failed} failed")
    logger.info(f"  testing_summary.md → {TESTING_SUMMARY_PATH}")
    logger.info("━" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()