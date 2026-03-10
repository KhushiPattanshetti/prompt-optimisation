"""Unit tests for the Pydantic schemas.

Tests:
- Request validation
- Response validation
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from schemas import RewriteRequest, RewriteResponse


class TestRewriteRequest:
    """Tests for the RewriteRequest schema."""

    def test_valid_request(self) -> None:
        """A valid clinical note should pass validation."""
        req = RewriteRequest(clinical_note="Patient presents with chest pain.")
        assert req.clinical_note == "Patient presents with chest pain."

    def test_missing_clinical_note(self) -> None:
        """Missing clinical_note should raise a validation error."""
        with pytest.raises(Exception):
            RewriteRequest()  # type: ignore[call-arg]

    def test_empty_clinical_note(self) -> None:
        """An empty string should fail validation (min_length=1)."""
        with pytest.raises(Exception):
            RewriteRequest(clinical_note="")


class TestRewriteResponse:
    """Tests for the RewriteResponse schema."""

    def test_valid_response(self) -> None:
        """A valid response with all fields should pass validation."""
        resp = RewriteResponse(
            rewritten_prompt="Rewritten text",
            log_prob_old=-10.23,
            value_estimate=0.41,
        )
        assert resp.rewritten_prompt == "Rewritten text"
        assert resp.log_prob_old == pytest.approx(-10.23)
        assert resp.value_estimate == pytest.approx(0.41)

    def test_missing_fields(self) -> None:
        """Missing required fields should raise a validation error."""
        with pytest.raises(Exception):
            RewriteResponse(rewritten_prompt="text")  # type: ignore[call-arg]

    def test_invalid_log_prob_type(self) -> None:
        """Non-numeric log_prob_old should fail validation."""
        with pytest.raises(Exception):
            RewriteResponse(
                rewritten_prompt="text",
                log_prob_old="not_a_number",  # type: ignore[arg-type]
                value_estimate=0.5,
            )

    def test_response_serialisation(self) -> None:
        """Response should serialise to a dict with correct keys."""
        resp = RewriteResponse(
            rewritten_prompt="Rewritten",
            log_prob_old=-5.0,
            value_estimate=0.2,
        )
        data = resp.model_dump()
        assert set(data.keys()) == {"rewritten_prompt", "log_prob_old", "value_estimate"}
