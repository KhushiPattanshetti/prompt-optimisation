"""Unit tests for gt_fetcher HTTP + cache behavior."""

import json
from unittest.mock import MagicMock, patch

import icd10_coding_svc.gt_fetcher as gt_fetcher


def test_returns_cached_codes_without_http_call(tmp_path, monkeypatch):
    monkeypatch.setattr(gt_fetcher, "GT_CODES_PATH", str(tmp_path))

    cache_file = tmp_path / "12345.json"
    cache_file.write_text(json.dumps({"note_id": "12345", "gt_codes": ["i209", "E11.9"]}))

    with patch("icd10_coding_svc.gt_fetcher.requests.get") as mock_get:
        result = gt_fetcher.get_gt_codes("12345")

    mock_get.assert_not_called()
    assert result == ["I20.9", "E11.9"]


def test_fetches_from_dataset_and_persists_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gt_fetcher, "GT_CODES_PATH", str(tmp_path))
    monkeypatch.setattr(gt_fetcher, "DATASET_SVC_URL", "http://dataset")

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"gt_codes": ["I21", "e119", "Z79.4"]}

    with patch("icd10_coding_svc.gt_fetcher.requests.get", return_value=response) as mock_get:
        result = gt_fetcher.get_gt_codes("777")

    mock_get.assert_called_once_with("http://dataset/gt_codes/777", timeout=10)
    assert result == ["I21", "E11.9", "Z79.4"]

    cached = json.loads((tmp_path / "777.json").read_text())
    assert cached["gt_codes"] == ["I21", "E11.9", "Z79.4"]
