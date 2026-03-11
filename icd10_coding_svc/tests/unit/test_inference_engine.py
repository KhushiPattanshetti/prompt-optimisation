import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import inference_engine
from inference_engine import run_inference


@pytest.fixture(autouse=True)
def patch_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_engine, "OUTPUT_PATH", str(tmp_path))
    import config
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
    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9", "I20.9", "R06.0"])
    @patch("inference_engine.model_loader.load_model")
    def test_returns_all_required_keys(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer

        result = run_inference("12345", "original prompt", "rewritten prompt")

        required_keys = {"note_id", "enh_codes", "org_codes", "gt_codes", "enh_raw_output", "org_raw_output", "parsing_success"}
        assert required_keys.issubset(result.keys())


class TestRunInferenceParsingSuccess:
    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("inference_engine.model_loader.load_model")
    def test_parsing_success_true(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer
        result = run_inference("12345", "original prompt", "rewritten prompt")
        assert result["parsing_success"] is True

    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("inference_engine.model_loader.load_model")
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
    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("inference_engine.model_loader.load_model")
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
    @patch("inference_engine._forward_to_reward_service")
    @patch("inference_engine.gt_fetcher.get_gt_codes", return_value=["R07.9"])
    @patch("inference_engine.model_loader.load_model")
    def test_lock_used(self, mock_load, mock_gt, mock_fwd, mock_model_and_tokenizer, tmp_path):
        mock_load.return_value = mock_model_and_tokenizer

        lock_mock = MagicMock()
        with patch.object(inference_engine, "_inference_lock", lock_mock):
            run_inference("12345", "original prompt", "rewritten prompt")

        lock_mock.__enter__.assert_called_once()
        lock_mock.__exit__.assert_called_once()
