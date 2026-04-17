import types
from contextlib import nullcontext

import pytest
import torch

from rewriter_inference_svc import inference_engine
from rewriter_inference_svc.policy import PromptPolicy


class DummyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):  # noqa: ANN001
        del messages, add_generation_prompt, return_tensors
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True):  # noqa: ANN001
        del token_ids, skip_special_tokens
        return (
            "Generate ICD-10 diagnosis extraction output in JSON list format with focus on "
            "cardiovascular and renal comorbidities plus documented complications only"
        )


class DummyValueHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        return self.bias.view(1, 1)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):  # noqa: ANN001
        del attention_mask, max_new_tokens, kwargs
        continuation = torch.tensor([[5, 6, 7, 8]], device=input_ids.device)
        return torch.cat([input_ids, continuation], dim=1)

    def forward(self, input_ids, output_hidden_states=False):  # noqa: ANN001
        batch, seq = input_ids.shape
        vocab = 16
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32, device=input_ids.device)
        if output_hidden_states:
            hidden = torch.zeros((batch, seq, 4), dtype=torch.float32, device=input_ids.device)
            return types.SimpleNamespace(logits=logits, hidden_states=[hidden, hidden])
        return types.SimpleNamespace(logits=logits)

    def disable_adapter(self):
        return nullcontext()


def _install_model_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    value_head = DummyValueHead()

    monkeypatch.setattr(inference_engine, "load_model", lambda: (model, tokenizer, value_head))
    monkeypatch.setattr(inference_engine, "_save_output", lambda payload: None)
    monkeypatch.setattr(inference_engine, "_compute_log_prob", lambda *args, **kwargs: -3.14)
    monkeypatch.setattr(inference_engine, "_compute_value_estimate", lambda *args, **kwargs: 0.25)


class TestPolicySampling:
    def test_policy_candidates_include_minimum_diversity(self) -> None:
        candidates = inference_engine._select_policy_candidates(
            note_text="Patient with stroke and CHF.",
            note_id="note-1",
            count=3,
        )

        strategy_names = {item.template_name for item in candidates}
        assert len(candidates) >= 2
        assert len(strategy_names) >= 2

    def test_policy_action_string_encodes_strategy_and_modifiers(self) -> None:
        policy = PromptPolicy(
            template_name="balanced",
            modifiers={"strict_precision": True, "expand_secondary": False},
        )
        action = inference_engine._policy_action_string(policy)
        assert action.startswith("balanced|")
        assert "strict_precision" in action


class TestInferenceFlow:
    def test_run_inference_preserves_response_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_model_stubs(monkeypatch)

        result = inference_engine.run_inference(
            clinical_note=(
                "Elderly patient with acute decompensated heart failure, CKD stage 3, and pulmonary edema requiring "
                "inpatient management and medication adjustment."
            ),
            note_id="note-42",
        )

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
        assert result["log_prob_old"] != 0.0

    def test_healthcheck_payload_uses_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_model_stubs(monkeypatch)

        result = inference_engine.run_inference(clinical_note="healthcheck", note_id="probe")
        assert result["generation_source"] == "healthcheck_fallback"
        assert result["log_prob_old"] == -1e-6

    def test_model_load_failure_path_returns_valid_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(inference_engine, "load_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(inference_engine, "_save_output", lambda payload: None)

        result = inference_engine.run_inference(
            clinical_note="Patient with COPD exacerbation and chronic hypoxemia.",
            note_id="note-err",
        )

        assert result["generation_source"] in {
            "guided_fallback_model_load_error",
            "rule_fallback_model_load_error",
        }
        assert isinstance(result["rewritten_prompt"], str)

