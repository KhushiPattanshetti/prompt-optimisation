import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import gt_fetcher


@pytest.fixture(autouse=True)
def patch_gt_codes_path(tmp_path, monkeypatch):
    """Redirect GT_CODES_PATH to a temp directory for every test."""
    monkeypatch.setattr(gt_fetcher, "GT_CODES_PATH", str(tmp_path))
    # Also patch the config import used inside gt_fetcher at module-level
    import config
    monkeypatch.setattr(config, "GT_CODES_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_diagnoses_df():
    return pd.DataFrame({
        "note_id": ["12345", "12345", "12345", "99999"],
        "seq_num": [1, 2, 3, 1],
        "icd_code": ["I21.9", "E11.9", "Z79.4", "J45.20"],
    })


class TestGetGtCodesFromDisk:
    def test_returns_codes_from_disk(self, tmp_path, monkeypatch):
        # Pre-create cached file
        cache_file = tmp_path / "12345.json"
        cache_file.write_text(json.dumps({
            "note_id": "12345",
            "gt_codes": ["I21.9", "E11.9"],
        }))

        # Patch _diagnoses_df to a sentinel that would fail if accessed
        monkeypatch.setattr(gt_fetcher, "_diagnoses_df", None)

        result = gt_fetcher.get_gt_codes("12345")
        assert result == ["I21.9", "E11.9"]


class TestGetGtCodesFromDataset:
    def test_returns_codes_from_dataframe(self, tmp_path, monkeypatch, sample_diagnoses_df):
        monkeypatch.setattr(gt_fetcher, "_diagnoses_df", sample_diagnoses_df)

        result = gt_fetcher.get_gt_codes("12345")
        assert result == ["I21.9", "E11.9", "Z79.4"]

        # Verify disk persistence
        cache_file = tmp_path / "12345.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["gt_codes"] == ["I21.9", "E11.9", "Z79.4"]


class TestGetGtCodesUnknownNote:
    def test_returns_empty_for_unknown_note(self, tmp_path, monkeypatch, sample_diagnoses_df):
        monkeypatch.setattr(gt_fetcher, "_diagnoses_df", sample_diagnoses_df)

        result = gt_fetcher.get_gt_codes("00000")
        assert result == []

        # File should still be written (empty gt_codes)
        cache_file = tmp_path / "00000.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["gt_codes"] == []
