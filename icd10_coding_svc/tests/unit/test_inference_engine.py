import json
from unittest.mock import MagicMock, patch

import pytest
import torch

import icd10_coding_svc.inference_engine as inference_engine
from icd10_coding_svc.inference_engine import run_inference


@pytest.fixture(autouse=True)
def patch_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_engine, "OUTPUT_PATH", str(tmp_path))
    import icd10_coding_svc.config as config
    monkeypatch.setattr(config, "OUTPUT_PATH", str(tmp_path))
    return tmp_path


class MockEncoding(dict):
    """Mimics transformers BatchEncoding: dict-like with .to() method."""
    def to(self, device):
        return self


@pytest.fixture
def mock_model_and_tokenizer():
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100, 101, 102]])

    mock_tokenizer = MagicMock()
    encoding = MockEncoding({
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    })
    mock_tokenizer.return_value = encoding
    mock_tokenizer.decode.return_value = '["R07.9", "I20.9"]'

    return mock_model, mock_tokenizer


class TestRunInferenceReturnsCorrectKeys:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9", "I20.9", "R06.0"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_returns_all_required_keys(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer

        result = run_inference("12345", "original prompt", "rewritten prompt")

        required_keys = {"note_id", "enh_codes", "org_codes", "gt_codes", "enh_raw_output", "org_raw_output", "parsing_success"}
        assert required_keys.issubset(result.keys())


class TestRunInferenceParsingSuccess:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_parsing_success_true(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer
        result = run_inference("12345", "original prompt", "rewritten prompt")
        assert result["parsing_success"] is True

    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_parsing_success_false(self, mock_load, mock_gt, mock_fwd, tmp_path):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100]])

        mock_tokenizer = MagicMock()
        encoding = MockEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.return_value = encoding
        # Return empty string → no codes → parsing_success = False
        mock_tokenizer.decode.return_value = "No codes here."

        mock_load.return_value = (mock_model, mock_tokenizer)
        result = run_inference("12345", "original prompt", "rewritten prompt")
        assert result["parsing_success"] is False


class TestOutputFileWritten:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_output_file_created(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer

        run_inference("12345", "original prompt", "rewritten prompt")

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["note_id"] == "12345"
        assert "timestamp" in data
        assert "enh_codes" in data
        assert "org_codes" in data
        assert "gt_codes" in data
        assert "parsing_success" in data


class TestInferenceLockAcquired:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_lock_used(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer

        lock_mock = MagicMock()
        with patch.object(inference_engine, "_inference_lock", lock_mock):
            run_inference("12345", "original prompt", "rewritten prompt")

        lock_mock.__enter__.assert_called_once()
        lock_mock.__exit__.assert_called_once()


class TestObservabilityCounters:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_both_parse_failure_updates_taxonomy(self, mock_load, _mock_gt, _mock_fwd, tmp_path):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100]])

        mock_tokenizer = MagicMock()
        encoding = MockEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.return_value = encoding
        mock_tokenizer.decode.return_value = "No ICD code generated"

        mock_load.return_value = (mock_model, mock_tokenizer)

        before = inference_engine.get_observability_snapshot()
        before_both_failed = before["parse_failure_taxonomy"]["joint_failure_modes"].get("both_failed", 0)

        run_inference("12345", "original prompt", "rewritten prompt")

        after = inference_engine.get_observability_snapshot()
        after_both_failed = after["parse_failure_taxonomy"]["joint_failure_modes"].get("both_failed", 0)

        assert after["total_requests"] == before["total_requests"] + 1
        assert after["both_parse_failures"] >= before["both_parse_failures"] + 1
        assert after_both_failed >= before_both_failed + 1


class TestParseRecoveryAndGroupPropagation:
    @patch("icd10_coding_svc.inference_engine._forward_to_reward_service")
    @patch("icd10_coding_svc.inference_engine.gt_fetcher.get_gt_codes", return_value=["I10"])
    @patch("icd10_coding_svc.inference_engine.model_loader.load_model")
    def test_original_parse_recovery_and_group_id_forwarded(
        self,
        mock_load,
        _mock_gt,
        mock_fwd,
        tmp_path,
    ):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100]])

        mock_tokenizer = MagicMock()
        encoding = MockEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.return_value = encoding
        mock_tokenizer.decode.side_effect = [
            '["R07.9"]',
            "No codes in this narrative response",
            '["I10"]',
        ]
        mock_load.return_value = (mock_model, mock_tokenizer)

        result = run_inference(
            "12345",
            "Clinical note:\nPatient with chronic hypertension.",
            "rewritten prompt",
            run_id="run-test",
            group_id="g-test-1",
        )

        assert result["org_parse_ok"] is True
        assert result["both_parse_success"] is True

        forwarded_payload = mock_fwd.call_args.args[0]
        assert forwarded_payload["group_id"] == "g-test-1"
        assert forwarded_payload["org_parse_ok"] is True

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        saved = json.loads(files[0].read_text())
        assert saved["group_id"] == "g-test-1"
        assert saved["org_recovery_used"] is True
