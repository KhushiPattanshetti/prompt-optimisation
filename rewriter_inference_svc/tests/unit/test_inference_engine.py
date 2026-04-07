"""Unit tests for rewriter inference engine helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import rewriter_inference_svc.inference_engine as inference_engine

from rewriter_inference_svc.inference_engine import (
    build_optimized_prompt,
    compress_clinical_text,
    _compute_log_prob,
    _compute_value_estimate,
    _is_valid_rewrite,
    extract_high_signal_sections,
    extract_semantic_diagnosis_descriptors,
    reorder_for_coding_priority,
    run_inference,
)


def _build_tokenizer_stub(input_len: int = 4):
    class TokenizerStub:
        eos_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": torch.randint(0, 20, (1, input_len)),
                "attention_mask": torch.ones(1, input_len, dtype=torch.long),
            }

        def decode(self, _ids, skip_special_tokens=True):
            return (
                "Extract all clinically relevant ICD-10 diagnosis codes from the note. "
                "Identify conditions, comorbidities, and complications, then output only a "
                "JSON list of diagnosis code strings."
            )

    return TokenizerStub()


def _build_model_stub(vocab_size: int = 32, seq_len: int = 8, hidden_dim: int = 16):
    model = MagicMock()
    input_ids = torch.randint(0, vocab_size, (1, 4))
    generated = torch.cat([input_ids, torch.randint(0, vocab_size, (1, 4))], dim=1)
    model.generate.return_value = generated

    logits = torch.randn(1, seq_len, vocab_size)
    hidden = torch.randn(1, seq_len, hidden_dim)
    model.side_effect = lambda *a, **k: SimpleNamespace(logits=logits, hidden_states=[hidden])

    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    return model


class TestComputeLogProb:
    def test_returns_float(self) -> None:
        logits = torch.randn(1, 6, 20)
        mock_model = MagicMock(return_value=SimpleNamespace(logits=logits))
        full_ids = torch.randint(0, 20, (1, 6))

        result = _compute_log_prob(mock_model, full_ids, input_length=2)
        assert isinstance(result, float)


class TestComputeValueEstimate:
    def test_returns_float(self) -> None:
        hidden_dim = 16
        hidden_states = [torch.randn(1, 5, hidden_dim)]
        mock_model = MagicMock(return_value=SimpleNamespace(hidden_states=hidden_states))
        value_head = torch.nn.Linear(hidden_dim, 1, bias=False)

        result = _compute_value_estimate(
            mock_model,
            value_head,
            torch.randint(0, 30, (1, 5)),
        )
        assert isinstance(result, float)


class TestRuleBasedPreprocessing:
    def test_extract_high_signal_sections_filters_low_signal_headers(self) -> None:
        note = (
            "Chief Complaint:\n"
            "Chest pain and dyspnea for two days with radiation to the left arm and nausea.\n"
            "Pain worsens with exertion and improves partially at rest.\n\n"
            "Medications on discharge:\n"
            "Aspirin 81mg daily.\n\n"
            "Assessment and Plan:\n"
            "NSTEMI with hypertension and diabetes. Continue telemetry, serial troponins, and"
            " cardiology follow-up for ischemic workup and secondary prevention planning."
        )

        extracted = extract_high_signal_sections(note)
        assert "Chief Complaint" in extracted
        assert "Assessment and Plan" in extracted
        assert "Medications on discharge" not in extracted

    def test_extract_high_signal_sections_drops_demographic_fields(self) -> None:
        note = (
            "Name: Jane Doe\n"
            "Unit No: 12345\n"
            "Admission Date: 2026-01-01\n"
            "Sex: F\n\n"
            "Chief Complaint: Worsening abdominal distension and pain\n\n"
            "Discharge Diagnosis: Ascites from portal hypertension"
        )

        extracted = extract_high_signal_sections(note)
        assert "Name: Jane Doe" not in extracted
        assert "Unit No" not in extracted
        assert "Chief Complaint" in extracted
        assert "Ascites from portal hypertension" in extracted

    def test_compress_clinical_text_removes_noise_patterns(self) -> None:
        text = (
            "Assessment:\n"
            "MRN: 123456\n"
            "Dictated by: Resident A\n"
            "Call 415-555-1212\n"
            "Sepsis likely due to pneumonia."
        )

        compressed = compress_clinical_text(text)
        assert "MRN: 123456" not in compressed
        assert "Dictated by" not in compressed
        assert "415-555-1212" not in compressed
        assert "Sepsis likely" in compressed

    def test_reorder_for_coding_priority_moves_diagnosis_earlier(self) -> None:
        text = (
            "History of Present Illness:\n"
            "Progressive cough.\n\n"
            "Final Diagnosis:\n"
            "Community acquired pneumonia."
        )

        reordered = reorder_for_coding_priority(text)
        assert reordered.index("Final Diagnosis") < reordered.index("History of Present Illness")

    def test_build_optimized_prompt_has_expected_format(self) -> None:
        prompt = build_optimized_prompt("Assessment: Acute CHF exacerbation.")
        assert "CLINICAL INFORMATION" in prompt
        assert "JSON array" in prompt

    def test_semantic_descriptor_extraction_filters_medication_noise(self) -> None:
        text = (
            "Diagnosis:\n"
            "HCV cirrhosis with ascites, HIV disease\n\n"
            "Assessment:\n"
            "known chronic obstructive pulmonary disease with exacerbation\n\n"
            "Medications on discharge:\n"
            "Furosemide 40 mg PO daily"
        )

        descriptors = extract_semantic_diagnosis_descriptors(text)
        descriptor_text = " | ".join(descriptors).lower()
        assert "cirrhosis" in descriptor_text
        assert "hiv" in descriptor_text
        assert "furosemide" not in descriptor_text

    def test_build_optimized_prompt_includes_semantic_descriptor_block(self) -> None:
        prompt = build_optimized_prompt(
            "Assessment: decompensated cirrhosis with ascites.",
            semantic_descriptors=["decompensated cirrhosis", "portal hypertension with ascites"],
        )
        assert "POSSIBLE DIAGNOSTIC DESCRIPTORS" in prompt
        assert "decompensated cirrhosis" in prompt

    def test_semantic_descriptors_prefer_clean_terms_over_fragments(self) -> None:
        text = (
            "Assessment:\n"
            "REASON FOR CONSULT: Femur fracture\n"
            "status post mechanical fall with immediate pain\n\n"
            "Diagnosis:\n"
            "mitral valve prolapse, osteoporosis with fracture risk\n"
        )

        descriptors = extract_semantic_diagnosis_descriptors(text)
        lowered = [item.lower() for item in descriptors]

        assert any("femur fracture" in item for item in lowered)
        assert any("mitral valve prolapse" in item for item in lowered)
        assert any("osteoporosis" in item for item in lowered)
        assert not any("immediate pain" in item for item in lowered)

    def test_semantic_descriptors_require_minimum_signal(self) -> None:
        text = "Chief Complaint:\nNot feeling well"
        descriptors = extract_semantic_diagnosis_descriptors(text)
        assert descriptors == []

    def test_semantic_descriptors_reject_fragments_and_limit_phrase_length(self) -> None:
        text = (
            "Assessment and Plan:\n"
            "History of hypertension with poor outpatient control and prolonged recent medication nonadherence.\n"
            "Known atrial fibrillation with rapid ventricular response overnight.\n"
            "with a\n"
            "and a\n"
        )

        descriptors = extract_semantic_diagnosis_descriptors(text)
        lowered = [item.lower() for item in descriptors]

        assert descriptors
        assert any("hypertension" in item for item in lowered)
        assert any("atrial fibrillation" in item for item in lowered)
        assert "with a" not in lowered
        assert "and a" not in lowered
        assert all(2 <= len(item.split()) <= 5 for item in descriptors)

    def test_semantic_descriptors_select_clean_fragment_from_noisy_clause(self) -> None:
        text = (
            "Assessment and Plan:\n"
            "history of disease, dyslipidemia, and a\n"
            "with a COPD exacerbation while admitted for monitoring.\n"
            "Past Medical History:\n"
            "Atrial fibrillation\n"
        )

        descriptors = extract_semantic_diagnosis_descriptors(text)
        lowered = [item.lower() for item in descriptors]

        assert descriptors
        assert not any("dyslipidemia" == item for item in lowered)
        assert any("copd exacerbation" == item for item in lowered)
        assert not any("disease dyslipidemia" in item for item in lowered)
        assert not any(item in {"and a", "with a", "disease"} for item in lowered)


class TestRewriteValidator:
    def test_rejects_generic_template(self) -> None:
        candidate = (
            "Extract all ICD-10-CM diagnosis codes from the clinical information below. "
            "Output only a JSON array of code strings. Include diagnoses, complications, and relevant co-morbidities."
        )
        assert not _is_valid_rewrite(
            candidate,
            "Rule prompt baseline",
            semantic_descriptors=["atrial fibrillation", "hypertension"],
        )

    def test_rejects_rewrite_missing_clinical_focus(self) -> None:
        candidate = (
            "Extract diagnosis coding instructions and output a JSON list of code strings "
            "using concise formatting for the downstream model."
        )
        assert not _is_valid_rewrite(
            candidate,
            "Rule prompt baseline for atrial fibrillation and hypertension",
            semantic_descriptors=["atrial fibrillation", "hypertension"],
        )

    def test_rejects_generic_template_even_with_focus_word(self) -> None:
        candidate = (
            "Extract all ICD-10-CM diagnosis codes from the clinical information below for cardio focus. "
            "Output only a JSON array of code strings and include diagnoses, complications, and relevant co-morbidities."
        )
        assert not _is_valid_rewrite(
            candidate,
            "Rule prompt baseline for heart failure and coronary artery disease",
            semantic_descriptors=["heart failure", "coronary artery disease"],
        )

    def test_rejects_rewrite_without_descriptor_reference(self) -> None:
        candidate = (
            "Generate ICD-10-CM coding guidance focused on neurologic and cardiology domains "
            "and output a JSON list of diagnosis code strings."
        )
        assert not _is_valid_rewrite(
            candidate,
            "Rule prompt baseline for atrial fibrillation and hypertension",
            semantic_descriptors=["atrial fibrillation", "hypertension"],
        )

    def test_rejects_rewrite_that_is_too_long_or_repetitive(self) -> None:
        candidate = (
            "Extract ICD-10-CM diagnosis coding focus for dysphagia and GERD with complications and output only "
            "a JSON list of diagnosis code strings for dysphagia and GERD with complications and output only a "
            "JSON list of diagnosis code strings for dysphagia and GERD with complications and output only a "
            "JSON list of diagnosis code strings for dysphagia and GERD with complications and output only a "
            "JSON list of diagnosis code strings for dysphagia and GERD with complications and output only a "
            "JSON list of diagnosis code strings for dysphagia and GERD with complications and output only a "
            "JSON list of diagnosis code strings."
        )
        assert not _is_valid_rewrite(
            candidate,
            "Rule prompt baseline for dysphagia and gerd",
            semantic_descriptors=["dysphagia", "gerd"],
        )

    def test_accepts_note_specific_rewrite_with_focus_terms(self) -> None:
        candidate = (
            "Extract ICD-10-CM diagnosis codes for atrial fibrillation, COPD exacerbation, "
            "and hypertension with active complications, then output only a JSON list of code strings."
        )
        assert _is_valid_rewrite(
            candidate,
            "Rule prompt baseline",
            semantic_descriptors=["atrial fibrillation", "copd exacerbation", "hypertension"],
        )


class TestRunInference:
    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_returns_required_fields(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        model = _build_model_stub()
        tokenizer = _build_tokenizer_stub()
        value_head = torch.nn.Linear(16, 1, bias=False)

        mock_load_model.return_value = (model, tokenizer, value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        result = run_inference("Patient has chest pain and dyspnea.", note_id="n1")

        assert set(result.keys()) == {
            "rewritten_prompt",
            "log_prob_old",
            "value_estimate",
            "generation_source",
        }
        assert isinstance(result["rewritten_prompt"], str)
        assert isinstance(result["log_prob_old"], float)
        assert isinstance(result["value_estimate"], float)
        assert isinstance(result["generation_source"], str)
        assert result["log_prob_old"] <= 0.0

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_exact_healthcheck_literal_skips_model_load(
        self,
        mock_load_model: MagicMock,
        _mock_save_output: MagicMock,
    ) -> None:
        result = run_inference("healthcheck", note_id="n1")
        mock_load_model.assert_not_called()
        assert result["log_prob_old"] == -1e-6

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_real_note_containing_test_does_not_skip_model_load(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        model = _build_model_stub()
        tokenizer = _build_tokenizer_stub()
        value_head = torch.nn.Linear(16, 1, bias=False)
        mock_load_model.return_value = (model, tokenizer, value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        run_inference("Patient has an outpatient pharmacological stress test planned.")
        mock_load_model.assert_called_once()

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_invalid_model_rewrites_fall_back_to_rule_prompt(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        model = _build_model_stub()

        class InvalidTokenizer:
            eos_token_id = 0

            def __call__(self, *_args, **_kwargs):
                return {
                    "input_ids": torch.randint(0, 20, (1, 4)),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }

            def decode(self, _ids, skip_special_tokens=True):
                return "The codes are I20.9 and E11.9."

        value_head = torch.nn.Linear(16, 1, bias=False)
        mock_load_model.return_value = (model, InvalidTokenizer(), value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        result = run_inference(
            "Final Diagnosis:\nAcute myocardial infarction.\n\n"
            "Medications on discharge:\nAspirin daily.",
            note_id="n_invalid",
        )

        assert result["generation_source"] in {"guided_fallback", "rule_fallback"}
        assert "JSON list of code strings" in result["rewritten_prompt"]

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_dynamic_base_mode_uses_filtered_input(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_model_stub()
        tokenizer = _build_tokenizer_stub()
        value_head = torch.nn.Linear(16, 1, bias=False)

        mock_load_model.return_value = (model, tokenizer, value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        monkeypatch.setattr(inference_engine, "REWRITER_INPUT_MODE", "dynamic")
        monkeypatch.setattr(inference_engine, "REWRITER_MODEL_VARIANT", "base")

        run_inference("Chief Complaint:\nChest pain\n\nDischarge Diagnosis:\nNSTEMI")

        called_input = model.generate.call_args.kwargs["input_ids"]
        assert called_input is not None

        saved_payload = mock_save_output.call_args.args[0]
        assert saved_payload["model_input_source"] == "filtered"

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_dynamic_sft_mode_enforces_semantic_guard(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_model_stub()

        class GenericTokenizer:
            eos_token_id = 0

            def __call__(self, *_args, **_kwargs):
                return {
                    "input_ids": torch.randint(0, 20, (1, 4)),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }

            def decode(self, _ids, skip_special_tokens=True):
                return (
                    "Extract all ICD-10 diagnosis codes and output a JSON array. "
                    "Use concise coding instructions only."
                )

        value_head = torch.nn.Linear(16, 1, bias=False)
        mock_load_model.return_value = (model, GenericTokenizer(), value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        monkeypatch.setattr(inference_engine, "REWRITER_INPUT_MODE", "dynamic")
        monkeypatch.setattr(inference_engine, "REWRITER_MODEL_VARIANT", "sft")
        monkeypatch.setattr(inference_engine, "REWRITER_ENABLE_POSTFILTER_GUARD", True)
        monkeypatch.setattr(inference_engine, "REWRITER_GUARD_MIN_KEYWORD_HITS", 2)

        result = run_inference(
            "Chief Complaint:\nWorsening abdominal distension and pain\n\n"
            "Past Medical History:\nHCV cirrhosis\nHIV disease\nCOPD\n\n"
            "Discharge Diagnosis:\nAscites from Portal HTN"
        )

        # Guard rejects generic rewrite in SFT/raw mode; guided rewrite remains structured and compact.
        assert result["generation_source"] in {"guided_fallback", "rule_fallback"}
        assert "JSON list of code strings" in result["rewritten_prompt"]

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_generate_failure_falls_back_without_raising(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        model = _build_model_stub()
        model.generate.side_effect = RuntimeError(
            "probability tensor contains either inf, nan or element < 0"
        )
        tokenizer = _build_tokenizer_stub()
        value_head = torch.nn.Linear(16, 1, bias=False)

        mock_load_model.return_value = (model, tokenizer, value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        result = run_inference("Final Diagnosis:\nAcute myocardial infarction.", note_id="n_gen_fail")

        assert isinstance(result["rewritten_prompt"], str)
        assert result["generation_source"] in {"guided_fallback", "rule_fallback"}

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    def test_model_load_failure_returns_fallback(
        self,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        mock_load_model.side_effect = RuntimeError("adapter load failed")
        mock_save_output.return_value = Path("/tmp/dummy.json")

        result = run_inference(
            "Final Diagnosis:\nAcute myocardial infarction with heart failure.",
            note_id="n_load_fail",
        )

        assert isinstance(result["rewritten_prompt"], str)
        assert result["generation_source"] in {
            "guided_fallback_model_load_error",
            "rule_fallback_model_load_error",
        }

    @patch("rewriter_inference_svc.inference_engine._save_output")
    @patch("rewriter_inference_svc.inference_engine.load_model")
    @patch("rewriter_inference_svc.inference_engine._generate_rewrite_with_base_adapter_disabled")
    @patch("rewriter_inference_svc.inference_engine._generate_rewrite")
    def test_base_adapter_disabled_rescue_before_guided_fallback(
        self,
        mock_generate_rewrite: MagicMock,
        mock_base_retry: MagicMock,
        mock_load_model: MagicMock,
        mock_save_output: MagicMock,
    ) -> None:
        model = _build_model_stub()
        tokenizer = _build_tokenizer_stub()
        value_head = torch.nn.Linear(16, 1, bias=False)

        mock_load_model.return_value = (model, tokenizer, value_head)
        mock_save_output.return_value = Path("/tmp/dummy.json")

        prompt_ids = torch.randint(0, 20, (1, 4), dtype=torch.long)
        full_ids = torch.cat([prompt_ids, torch.zeros((1, 2), dtype=torch.long)], dim=1)

        # First and second model attempts fail validation (empty rewrite),
        # then adapter-disabled retry provides a valid instruction prompt.
        mock_generate_rewrite.side_effect = [
            ("", full_ids, 4, prompt_ids),
            ("", full_ids, 4, prompt_ids),
        ]
        mock_base_retry.return_value = (
            "Extract ICD-10-CM diagnosis codes for acute myocardial infarction and related complications, then output only a JSON list of code strings.",
            full_ids,
            4,
            prompt_ids,
        )

        result = run_inference("Final Diagnosis:\nAcute myocardial infarction.", note_id="n_base_rescue")

        assert result["generation_source"] == "model_base_adapter_disabled"
        assert "acute myocardial infarction" in result["rewritten_prompt"].lower()
        mock_base_retry.assert_called_once()
