"""Policy object and sampling helpers for instruction strategy selection."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .strategy_templates import VALID_STRATEGIES

VALID_MODIFIER_KEYS = frozenset(
    {
        "enforce_fall_detection",
        "enforce_z_codes",
        "strict_precision",
        "expand_secondary",
        "prioritize_primary",
        "strict_exclusion",
        "expand_risk_factors",
    }
)


@dataclass(frozen=True)
class PromptPolicy:
    """Discrete action representation for prompt-policy routing."""

    template_name: str
    modifiers: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.template_name not in VALID_STRATEGIES:
            raise ValueError(f"Unknown template_name: {self.template_name}")

        invalid_keys = [key for key in self.modifiers.keys() if key not in VALID_MODIFIER_KEYS]
        if invalid_keys:
            raise ValueError(f"Unknown modifier keys: {', '.join(sorted(invalid_keys))}")

        normalized = {str(key): bool(value) for key, value in self.modifiers.items()}
        object.__setattr__(self, "modifiers", normalized)


def sample_policy_candidates(
    note_text: str,
    note_id: Optional[str],
    count: int = 3,
    temperature: float = 1.0,
) -> List[PromptPolicy]:
    """Return diverse policy candidates with lightweight stochasticity.

    Guarantees at least two distinct strategies when at least two are requested.
    """

    strategies = list(VALID_STRATEGIES)
    if not strategies:
        raise ValueError("No valid strategies configured")

    normalized_temperature = max(float(temperature), 0.2)
    seed_material = f"{note_id or ''}|{note_text[:128]}|{random.random():.12f}"
    seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    rng.shuffle(strategies)

    target = max(1, min(int(count), len(strategies)))
    selected = strategies[:target]

    if target >= 2 and len(set(selected)) < 2:
        alt = [s for s in strategies if s != selected[0]]
        if alt:
            selected[-1] = rng.choice(alt)

    candidates: List[PromptPolicy] = []
    for name in selected:
        p_true = min(0.45 * normalized_temperature, 0.65)
        modifiers = {
            "enforce_fall_detection": rng.random() < p_true * 0.40,
            "enforce_z_codes": rng.random() < p_true * 0.40,
            "strict_precision": rng.random() < p_true * 0.55,
            "expand_secondary": rng.random() < p_true * 0.55,
            "prioritize_primary": rng.random() < p_true * 0.50,
            "strict_exclusion": rng.random() < p_true * 0.45,
            "expand_risk_factors": rng.random() < p_true * 0.45,
        }
        candidates.append(PromptPolicy(template_name=name, modifiers=modifiers))

    return candidates
