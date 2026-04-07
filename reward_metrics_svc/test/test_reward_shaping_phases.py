import pytest

from reward_metrics_svc.main import (
    _resolve_curriculum_weights,
    calculate_reward_components,
    get_descriptions,
)


def _ensure_description(code: str) -> str:
    descriptions = get_descriptions([code])
    if not descriptions:
        pytest.skip(f"No ICD description available for code {code}")
    return descriptions[0]


def test_curriculum_weight_schedule_thresholds() -> None:
    early = _resolve_curriculum_weights(10)
    mid = _resolve_curriculum_weights(120)
    late = _resolve_curriculum_weights(300)

    assert early == mid == late
    assert early["exact"] == 0.5
    assert early["semantic"] == 0.25
    assert early["concept"] == 0.15
    assert early["struct"] == 0.05
    assert early["delta"] == 0.05


def test_alignment_bonus_applies_when_semantic_and_concept_positive() -> None:
    desc = _ensure_description("A01.1")
    components = calculate_reward_components(
        gt_codes=["A01.1"],
        enh_codes=["A01.1"],
        org_codes=["A01.2"],
        semantic_descriptors=[desc],
        rewritten_prompt=desc,
        training_step=12,
    )

    assert components["semantic_reward"] > 0.0
    assert components["alignment_bonus"] == components["consistency_bonus"]
    if components["semantic_score"] > 0.6 and components["concept_f1"] > 0.5:
        assert components["consistency_bonus"] == 0.2
    else:
        assert components["consistency_bonus"] == 0.0


def test_semantic_reward_shaping_stays_bounded() -> None:
    components = calculate_reward_components(
        gt_codes=["A01.1", "A01.2"],
        enh_codes=["A01.1", "A01.2"],
        org_codes=["A09.0"],
        semantic_descriptors=["enteric fever", "salmonella infection"],
        rewritten_prompt="focus on enteric fever and salmonella infection",
        training_step=5,
    )
    assert -1.0 <= components["semantic_reward"] <= 1.0
