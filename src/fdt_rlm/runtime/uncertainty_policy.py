from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class UncertaintyFeatures:
    anchor_entropy: float = 0.0
    normalized_anchor_entropy: float = 0.0
    effective_anchor_count: float = 0.0
    top1_membership: float = 0.0
    top1_top2_margin: float = 0.0
    token_entropy: float = 0.0
    mean_logprob: float = 0.0
    state_conflict: float = 0.0
    confidence: float = 0.0
    remaining_budget_ratio: float = 1.0
    depth: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


class UncertaintyPolicy:
    """Small auditable controller; coefficients must be calibrated before claims."""

    def __init__(self, coefficients: Dict[str, float] | None = None, bias: float = 0.0):
        self.coefficients = coefficients or {
            "token_entropy": 0.45,
            "mean_logprob": -0.25,
            "normalized_anchor_entropy": 0.20,
            "top1_membership": -0.15,
            "top1_top2_margin": -0.15,
            "state_conflict": 0.25,
            "confidence": -0.35,
            "remaining_budget_ratio": 0.10,
            "depth": -0.20,
        }
        self.bias = bias

    def need_more_context_probability(self, features: UncertaintyFeatures) -> float:
        values = features.to_dict()
        score = self.bias + sum(
            weight * float(values.get(name, 0.0))
            for name, weight in self.coefficients.items()
        )
        return sigmoid(score)

    def action_hint(self, features: UncertaintyFeatures) -> str:
        probability = self.need_more_context_probability(features)
        if probability < 0.35:
            return "STOP"
        if features.depth < 1 and features.state_conflict > 0.35:
            return "COMPARE"
        if probability > 0.75 and features.depth < 1:
            return "CALL"
        return "SEARCH"
