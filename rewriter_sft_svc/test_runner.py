"""
test_runner.py — rewriter_sft_svc

Comprehensive test suite for the rewriter_sft_svc microservice.

Test categories:
  1. Unit Tests
  2. Test-Data Validation Tests
  3. Inference Tests on test_data/
  4. End-to-End Tests
  5. Stress Tests
  6. Edge Case Tests
  7. Regression Tests

Run directly:
    python test_runner.py
or via:
    python train_sft.py --test-only
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

# Allow imports from the service directory
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CHECKPOINT_DIR,
    DATASET_KEYS,
    REQUIRED_FIELDS,
    SYSTEM_INSTRUCTION,
    TEST_DATA_DIR,
    TESTING_SUMMARY_PATH,
)
from dataset_loader import load_raw_dataset, validate_schema, split_dataset
from preprocessing import convert_dataset, record_to_messages, validate_output_schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.errors: List[str] = []
        self.start = time.time()
        self.duration: float = 0.0

    def ok(self, msg: str):
        self.passed.append(msg)

    def fail(self, msg: str):
        self.failed.append(msg)

    def finish(self) -> "TestResult":
        self.duration = round(time.time() - self.start, 2)
        return self

    @property
    def status(self) -> str:
        return "PASS" if not self.failed and not self.errors else "FAIL"


def _load_json_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> TestResult:
    r = TestResult("Unit Tests")

    # 1.1 – Dataset schema keys
    try:
        assert "clinical note" in DATASET_KEYS
        assert "structured clinical note" in DATASET_KEYS
        r.ok("DATASET_KEYS contains required keys")
    except AssertionError as e:
        r.fail(f"DATASET_KEYS missing keys: {e}")

    # 1.2 – System instruction is non-empty
    try:
        assert isinstance(SYSTEM_INSTRUCTION, str) and len(SYSTEM_INSTRUCTION) > 20
        r.ok("SYSTEM_INSTRUCTION is non-empty string")
    except AssertionError:
        r.fail("SYSTEM_INSTRUCTION is empty or too short")

    # 1.3 – REQUIRED_FIELDS ordering
    try:
        assert REQUIRED_FIELDS[0] == "Patient Demographics:"
        assert REQUIRED_FIELDS[-1] == "Complications:"
        assert len(REQUIRED_FIELDS) == 12
        r.ok("REQUIRED_FIELDS contains 12 fields in correct order")
    except AssertionError as e:
        r.fail(f"REQUIRED_FIELDS ordering issue: {e}")

    # 1.4 – record_to_messages structure
    try:
        sample = {
            "clinical note": "65M HTN DM2 chest pain",
            "structured clinical note": "Patient Demographics:\nAge:\n65\nGender:\nMale\nPrimary Diagnosis:\nNot specified\nSecondary Diagnoses:\nHypertension\nType 2 Diabetes Mellitus\nSymptoms:\nChest pain\nDuration:\nNot specified\nInvestigations:\nNot specified\nProcedures:\nNot specified\nComorbidities:\nHypertension\nType 2 Diabetes Mellitus\nRisk Factors:\nNot specified\nComplications:\nNot specified",
        }
        msg = record_to_messages(sample)
        assert msg["messages"][0]["role"] == "system"
        assert msg["messages"][0]["content"] == SYSTEM_INSTRUCTION
        assert msg["messages"][1]["role"] == "user"
        assert msg["messages"][1]["content"] == sample["clinical note"]
        assert msg["messages"][2]["role"] == "assistant"
        assert msg["messages"][2]["content"] == sample["structured clinical note"]
        r.ok("record_to_messages produces correct 3-turn message structure")
    except AssertionError as e:
        r.fail(f"record_to_messages structure incorrect: {e}")

    # 1.5 – convert_dataset on a list
    try:
        records = [
            {
                "clinical note": "fever x3d",
                "structured clinical note": "Patient Demographics:\nAge:\nNot specified\nGender:\nNot specified\nPrimary Diagnosis:\nNot specified\nSecondary Diagnoses:\nNot specified\nSymptoms:\nFever\nDuration:\n3 days\nInvestigations:\nNot specified\nProcedures:\nNot specified\nComorbidities:\nNot specified\nRisk Factors:\nNot specified\nComplications:\nNot specified",
            }
        ]
        converted = convert_dataset(records)
        assert len(converted) == 1
        assert "messages" in converted[0]
        r.ok("convert_dataset processes list correctly")
    except AssertionError as e:
        r.fail(f"convert_dataset failed: {e}")

    # 1.6 – validate_schema removes malformed records
    try:
        bad_records = [
            {"clinical note": "ok note", "structured clinical note": "ok structured"},
            {"clinical note": "missing key"},
            {"clinical note": "", "structured clinical note": "structured"},
        ]
        valid = validate_schema(bad_records)
        assert len(valid) == 1
        r.ok("validate_schema correctly filters malformed records")
    except AssertionError as e:
        r.fail(f"validate_schema filtering incorrect: {e}")

    # 1.7 – validate_output_schema detects missing fields
    try:
        incomplete_output = "Patient Demographics:\nAge:\n65"
        result = validate_output_schema(incomplete_output, REQUIRED_FIELDS)
        assert not result["valid"]
        assert len(result["missing_fields"]) > 0
        r.ok("validate_output_schema detects missing fields")
    except AssertionError as e:
        r.fail(f"validate_output_schema missed missing fields: {e}")

    # 1.8 – validate_output_schema detects forbidden text
    try:
        leaked_output = "Patient Demographics:\nAge:\nDo not add anything.\nGender:\n"
        result = validate_output_schema(leaked_output, REQUIRED_FIELDS)
        assert result["extra_text_detected"]
        r.ok("validate_output_schema detects leaked forbidden phrases")
    except AssertionError as e:
        r.fail(f"validate_output_schema missed forbidden phrases: {e}")

    # 1.9 – split_dataset proportions
    try:
        recs = [{"clinical note": f"note {i}", "structured clinical note": f"structured {i}"} for i in range(100)]
        train_s, val_s, test_s = split_dataset(recs, val_split=0.1, test_split=0.1)
        assert len(train_s) + len(val_s) + len(test_s) == 100
        assert len(test_s) >= 1
        assert len(val_s) >= 1
        r.ok("split_dataset maintains total count and creates all splits")
    except AssertionError as e:
        r.fail(f"split_dataset proportions wrong: {e}")

    # 1.10 – Checkpoint directory config
    try:
        assert CHECKPOINT_DIR.endswith("phi3_rewriter_sft") or "sft_checkpoints" in CHECKPOINT_DIR
        r.ok("CHECKPOINT_DIR configured correctly")
    except AssertionError as e:
        r.fail(f"CHECKPOINT_DIR misconfigured: {e}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Test-Data Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_test_data_validation() -> TestResult:
    r = TestResult("Test-Data Validation")

    files = {
        "sample_test.json": os.path.join(TEST_DATA_DIR, "sample_test.json"),
        "edge_cases.json":  os.path.join(TEST_DATA_DIR, "edge_cases.json"),
        "stress_cases.json": os.path.join(TEST_DATA_DIR, "stress_cases.json"),
        "e2e_cases.json":   os.path.join(TEST_DATA_DIR, "e2e_cases.json"),
    }

    for fname, fpath in files.items():
        # Existence
        if not os.path.exists(fpath):
            r.fail(f"{fname}: file not found at {fpath}")
            continue
        r.ok(f"{fname}: file exists")

        # Parseable JSON
        try:
            records = _load_json_file(fpath)
            r.ok(f"{fname}: valid JSON ({len(records)} records)")
        except json.JSONDecodeError as e:
            r.fail(f"{fname}: JSON parse error — {e}")
            continue

        # Schema compliance
        valid = validate_schema(records)
        if len(valid) == len(records):
            r.ok(f"{fname}: all {len(records)} records pass schema validation")
        else:
            r.fail(
                f"{fname}: {len(records) - len(valid)} / {len(records)} records failed schema validation"
            )

        # Non-empty
        if len(records) == 0:
            r.fail(f"{fname}: file is empty (0 records)")
        else:
            r.ok(f"{fname}: non-empty ({len(records)} records)")

        # Expected output field presence
        for i, rec in enumerate(records):
            if "structured clinical note" in rec:
                missing = [f for f in REQUIRED_FIELDS if f not in rec["structured clinical note"]]
                if missing:
                    r.fail(f"{fname} record {i}: expected output missing fields: {missing}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Inference Tests (schema-based, no model loading)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_schema_tests() -> TestResult:
    """
    Validate that the expected outputs in test_data/ satisfy the required schema.
    This is a schema-only check: no model is loaded.
    Full inference tests require a trained checkpoint.
    """
    r = TestResult("Inference Schema Tests")

    files = ["sample_test.json", "edge_cases.json", "stress_cases.json", "e2e_cases.json"]

    for fname in files:
        fpath = os.path.join(TEST_DATA_DIR, fname)
        if not os.path.exists(fpath):
            r.fail(f"{fname}: not found")
            continue

        records = _load_json_file(fpath)
        for i, rec in enumerate(records):
            expected = rec.get("structured clinical note", "")
            result = validate_output_schema(expected, REQUIRED_FIELDS)
            if result["valid"]:
                r.ok(f"{fname}[{i}]: expected output passes schema")
            else:
                details = []
                if result["missing_fields"]:
                    details.append(f"missing={result['missing_fields']}")
                if result["wrong_order"]:
                    details.append("wrong_order=True")
                if result["extra_text_detected"]:
                    details.append("extra_text=True")
                r.fail(f"{fname}[{i}]: schema fail — {'; '.join(details)}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Edge Case Tests (schema-only)
# ══════════════════════════════════════════════════════════════════════════════

def run_edge_case_tests() -> TestResult:
    r = TestResult("Edge Case Tests")

    fpath = os.path.join(TEST_DATA_DIR, "edge_cases.json")
    if not os.path.exists(fpath):
        r.fail("edge_cases.json not found")
        return r.finish()

    records = _load_json_file(fpath)
    r.ok(f"Loaded {len(records)} edge case records")

    for i, rec in enumerate(records):
        note = rec.get("clinical note", "")
        expected = rec.get("structured clinical note", "")

        # Schema check on expected output
        result = validate_output_schema(expected, REQUIRED_FIELDS)
        if result["valid"]:
            r.ok(f"Edge case {i}: expected output schema valid")
        else:
            r.fail(f"Edge case {i}: schema issues — {result}")

        # Heuristic: empty / near-empty notes should have mostly 'Not specified'
        if len(note.strip()) < 20:
            not_specified_count = expected.count("Not specified")
            if not_specified_count >= 6:
                r.ok(f"Edge case {i} (minimal note): ≥6 'Not specified' fields — OK")
            else:
                r.fail(
                    f"Edge case {i} (minimal note, len={len(note)}): only {not_specified_count} 'Not specified' — may be hallucinating"
                )

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stress Tests (preprocessing only)
# ══════════════════════════════════════════════════════════════════════════════

def run_stress_tests() -> TestResult:
    r = TestResult("Stress Tests")

    fpath = os.path.join(TEST_DATA_DIR, "stress_cases.json")
    if not os.path.exists(fpath):
        r.fail("stress_cases.json not found")
        return r.finish()

    records = _load_json_file(fpath)
    r.ok(f"Loaded {len(records)} stress records")

    # Preprocessing stability: convert entire stress set
    try:
        converted = convert_dataset(records)
        assert len(converted) == len(records)
        r.ok("Stress set: convert_dataset stable on all records")
    except Exception as e:
        r.fail(f"Stress set: convert_dataset failed — {e}")

    # Schema check on expected outputs
    for i, rec in enumerate(records):
        expected = rec.get("structured clinical note", "")
        result = validate_output_schema(expected, REQUIRED_FIELDS)
        if result["valid"]:
            r.ok(f"Stress case {i}: expected output schema valid")
        else:
            r.fail(f"Stress case {i}: schema issues — {result}")

    # Check that long notes do not break message construction
    for i, rec in enumerate(records):
        try:
            msg = record_to_messages(rec)
            assert msg["messages"][1]["role"] == "user"
            r.ok(f"Stress case {i}: message construction OK (note len={len(rec['clinical note'])})")
        except Exception as e:
            r.fail(f"Stress case {i}: message construction failed — {e}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 6. End-to-End Test (preprocessing pipeline, no model)
# ══════════════════════════════════════════════════════════════════════════════

def run_e2e_tests() -> TestResult:
    r = TestResult("End-to-End Tests")

    fpath = os.path.join(TEST_DATA_DIR, "e2e_cases.json")
    if not os.path.exists(fpath):
        r.fail("e2e_cases.json not found")
        return r.finish()

    records = _load_json_file(fpath)
    r.ok(f"Loaded {len(records)} e2e records")

    # Step 1 – Validate schema
    valid = validate_schema(records)
    if len(valid) == len(records):
        r.ok("E2E: all records pass schema validation")
    else:
        r.fail(f"E2E: {len(records) - len(valid)} records failed schema validation")

    # Step 2 – Convert to chat format
    try:
        converted = convert_dataset(valid)
        for c in converted:
            assert "messages" in c
            assert len(c["messages"]) == 3
        r.ok(f"E2E: {len(converted)} records converted to chat format")
    except Exception as e:
        r.fail(f"E2E: chat conversion failed — {e}")
        return r.finish()

    # Step 3 – Verify system instruction in every sample
    try:
        for i, c in enumerate(converted):
            assert c["messages"][0]["content"] == SYSTEM_INSTRUCTION, \
                f"System instruction mismatch in record {i}"
        r.ok("E2E: system instruction correctly embedded in all samples")
    except AssertionError as e:
        r.fail(str(e))

    # Step 4 – Checkpoint existence check  
    if os.path.exists(CHECKPOINT_DIR):
        contents = os.listdir(CHECKPOINT_DIR)
        if contents:
            r.ok(f"E2E: checkpoint directory exists and contains {len(contents)} files")
        else:
            r.fail("E2E: checkpoint directory exists but is empty — training may not have completed")
    else:
        r.fail(f"E2E: checkpoint directory not found at {CHECKPOINT_DIR}")

    # Step 5 – Dataset split integrity
    try:
        train_s, val_s, test_s = split_dataset(valid)
        total = len(train_s) + len(val_s) + len(test_s)
        assert total == len(valid)
        r.ok(f"E2E: dataset split integrity OK (train={len(train_s)}, val={len(val_s)}, test={len(test_s)})")
    except AssertionError as e:
        r.fail(f"E2E: dataset split integrity failed — {e}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Regression Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_regression_tests() -> TestResult:
    r = TestResult("Regression Tests")

    # R1 – Field count is exactly 12
    try:
        assert len(REQUIRED_FIELDS) == 12
        r.ok("Regression: REQUIRED_FIELDS count == 12")
    except AssertionError:
        r.fail(f"Regression: REQUIRED_FIELDS count changed — got {len(REQUIRED_FIELDS)}")

    # R2 – Field names unchanged
    expected_names = [
        "Patient Demographics:", "Age:", "Gender:",
        "Primary Diagnosis:", "Secondary Diagnoses:",
        "Symptoms:", "Duration:", "Investigations:",
        "Procedures:", "Comorbidities:", "Risk Factors:", "Complications:",
    ]
    try:
        assert REQUIRED_FIELDS == expected_names
        r.ok("Regression: REQUIRED_FIELDS names and order unchanged")
    except AssertionError:
        r.fail(f"Regression: REQUIRED_FIELDS changed — got {REQUIRED_FIELDS}")

    # R3 – Config paths are consistent
    try:
        from config import ROOT_DIR, SVC_DIR
        assert os.path.isdir(ROOT_DIR), f"ROOT_DIR not found: {ROOT_DIR}"
        assert os.path.isdir(SVC_DIR),  f"SVC_DIR not found: {SVC_DIR}"
        r.ok("Regression: ROOT_DIR and SVC_DIR exist")
    except AssertionError as e:
        r.fail(str(e))

    # R4 – System instruction unchanged (length check as proxy)
    try:
        assert len(SYSTEM_INSTRUCTION) > 100
        r.ok("Regression: SYSTEM_INSTRUCTION length consistent (>100 chars)")
    except AssertionError:
        r.fail("Regression: SYSTEM_INSTRUCTION suspiciously short")

    # R5 – validate_output_schema accepts a well-formed output
    try:
        perfect = (
            "Patient Demographics:\nAge:\n45\nGender:\nFemale\n"
            "Primary Diagnosis:\nPneumonia\nSecondary Diagnoses:\nHypertension\n"
            "Symptoms:\nCough\nFever\nDuration:\n5 days\n"
            "Investigations:\nChest X-ray\nProcedures:\nNot specified\n"
            "Comorbidities:\nHypertension\nRisk Factors:\nNot specified\n"
            "Complications:\nNot specified"
        )
        result = validate_output_schema(perfect, REQUIRED_FIELDS)
        assert result["valid"]
        r.ok("Regression: validate_output_schema accepts well-formed output")
    except AssertionError as e:
        r.fail(f"Regression: validate_output_schema rejected valid output — {e}")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Checkpoint Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_checkpoint_validation() -> TestResult:
    r = TestResult("Checkpoint Validation")

    if not os.path.exists(CHECKPOINT_DIR):
        r.fail(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return r.finish()

    r.ok(f"Checkpoint directory exists: {CHECKPOINT_DIR}")

    # Expected artifact files (at minimum, adapter config and tokenizer)
    adapter_files = ["adapter_config.json"]
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    weight_files   = ["adapter_model.bin", "adapter_model.safetensors"]

    contents = set(os.listdir(CHECKPOINT_DIR))

    for f in adapter_files:
        if f in contents:
            r.ok(f"Checkpoint artifact found: {f}")
        else:
            r.fail(f"Missing checkpoint artifact: {f}")

    for f in tokenizer_files:
        if f in contents:
            r.ok(f"Tokenizer artifact found: {f}")
        else:
            r.fail(f"Missing tokenizer artifact: {f}")

    # At least one weight file must exist
    if any(f in contents for f in weight_files):
        r.ok("LoRA adapter weights found (bin or safetensors)")
    else:
        r.fail("No adapter weight file found (expected .bin or .safetensors)")

    # Config.json should be present
    if "config.json" in contents:
        r.ok("config.json found in checkpoint")
    else:
        r.fail("config.json not found in checkpoint")

    return r.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Main runner + summary generation
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests() -> List[TestResult]:
    logger.info("=" * 60)
    logger.info("rewriter_sft_svc — Full Test Suite")
    logger.info("=" * 60)

    suite = [
        run_unit_tests,
        run_test_data_validation,
        run_inference_schema_tests,
        run_edge_case_tests,
        run_stress_tests,
        run_e2e_tests,
        run_regression_tests,
        run_checkpoint_validation,
    ]

    results: List[TestResult] = []
    for fn in suite:
        logger.info(f"\n▶  Running: {fn.__name__}")
        try:
            res = fn()
        except Exception as exc:
            res = TestResult(fn.__name__)
            res.errors.append(traceback.format_exc())
            res.finish()
        results.append(res)
        _print_result(res)

    return results


def _print_result(r: TestResult):
    status_icon = "✓" if r.status == "PASS" else "✗"
    logger.info(f"{status_icon} [{r.status}] {r.name} ({r.duration}s)")
    for msg in r.passed:
        logger.info(f"    PASS  {msg}")
    for msg in r.failed:
        logger.warning(f"    FAIL  {msg}")
    for msg in r.errors:
        logger.error(f"    ERROR {msg}")


def generate_testing_summary(results: List[TestResult], output_path: str = TESTING_SUMMARY_PATH):
    """Generate the testing_summary.md file from test results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_passed = sum(len(r.passed) for r in results)
    total_failed = sum(len(r.failed) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    overall_status = "PASS" if total_failed == 0 and total_errors == 0 else "PARTIAL" if total_passed > 0 else "FAIL"

    lines = [
        "# Testing Summary — rewriter_sft_svc",
        "",
        f"> Generated: {now}",
        f"> Overall Status: **{overall_status}**",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total test categories | {len(results)} |",
        f"| Assertions passed | {total_passed} |",
        f"| Assertions failed | {total_failed} |",
        f"| Errors | {total_errors} |",
        "",
        "---",
        "",
    ]

    for r in results:
        status_badge = "✅ PASS" if r.status == "PASS" else "❌ FAIL"
        lines += [
            f"## {r.name}",
            "",
            f"**Status:** {status_badge}  |  **Duration:** {r.duration}s  |  "
            f"**Passed:** {len(r.passed)}  |  **Failed:** {len(r.failed)}",
            "",
        ]
        if r.passed:
            lines.append("### Passed")
            for msg in r.passed:
                lines.append(f"- {msg}")
            lines.append("")
        if r.failed:
            lines.append("### Failed")
            for msg in r.failed:
                lines.append(f"- {msg}")
            lines.append("")
        if r.errors:
            lines.append("### Errors")
            for msg in r.errors:
                lines.append(f"```\n{msg}\n```")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        "## Checkpoint Validation",
        "",
        f"Checkpoint directory: `{CHECKPOINT_DIR}`",
        "",
        "Existence check performed as part of 'Checkpoint Validation' category above.",
        "",
        "---",
        "",
        "## Known Issues",
        "",
        "- Full model inference tests require a trained checkpoint to be present.",
        "- Training on CPU/MPS is significantly slower than on CUDA; use GPU for production runs.",
        "- BitsAndBytes 4-bit quantisation is only available on CUDA; MPS/CPU use float16/float32.",
        "",
        "---",
        "",
        "## Final Status",
        "",
        f"**{overall_status}** — {total_passed} assertions passed, "
        f"{total_failed} failed, {total_errors} errors.",
        "",
        "---",
        "",
        "_Report auto-generated by `test_runner.py`_",
    ]

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"testing_summary.md written to {output_path}")
    return content


if __name__ == "__main__":
    results = run_all_tests()
    generate_testing_summary(results)
    failed = sum(len(r.failed) + len(r.errors) for r in results)
    sys.exit(0 if failed == 0 else 1)
