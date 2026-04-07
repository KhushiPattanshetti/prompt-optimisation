import pytest

from reward_metrics_svc.main import calculate_reward_components, get_descriptions


def _ensure_description(code: str) -> str:
    descriptions = get_descriptions([code])
    if not descriptions:
        pytest.skip(f"No ICD description available for code {code}")
    return descriptions[0]


def test_concept_reward_missing_descriptors_returns_soft_negative():
    components = calculate_reward_components(
        gt_codes=["A01.1"],
        enh_codes=["A01.1"],
        org_codes=["A01.2"],
        semantic_descriptors=[],
        rewritten_prompt="",
    )
    assert components["concept_reward"] == -0.2


def test_concept_reward_uses_semantic_descriptors():
    description = _ensure_description("A01.1")

    components = calculate_reward_components(
        gt_codes=["A01.1"],
        enh_codes=["A01.1"],
        org_codes=["A01.2"],
        semantic_descriptors=[description],
        rewritten_prompt="",
    )

    assert -1.0 <= components["concept_reward"] <= 1.0
    assert components["concept_reward"] >= 0.0


def test_concept_reward_penalizes_noise_and_duplicates():
    description = _ensure_description("A01.1")

    baseline = calculate_reward_components(
        gt_codes=["A01.1"],
        enh_codes=["A01.1"],
        org_codes=["A01.2"],
        semantic_descriptors=[description],
        rewritten_prompt="",
    )["concept_reward"]

    penalized = calculate_reward_components(
        gt_codes=["A01.1"],
        enh_codes=["A01.1"],
        org_codes=["A01.2"],
        semantic_descriptors=[description, description, "for", "the", "unknown term"],
        rewritten_prompt="",
    )["concept_reward"]

    assert penalized <= baseline
