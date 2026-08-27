from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping


TOP_LEVEL_MIX = {
    "fineweb_edu": 0.60,
    "fineweb": 0.15,
    "code": 0.25,
}

CODE_MIX = {
    "python": 0.45,
    "javascript_typescript": 0.20,
    "c_cpp": 0.10,
    "java": 0.10,
    "rust_go": 0.10,
    "structured_docs": 0.05,
}


@dataclass(frozen=True)
class TokenPlan:
    total_tokens: int
    quotas: Dict[str, int]

    @property
    def code_tokens(self) -> int:
        return sum(v for k, v in self.quotas.items() if k.startswith("code_"))


def _round_alloc(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    raw = {key: float(total) * ratio for key, ratio in ratios.items()}
    base = {key: int(value) for key, value in raw.items()}
    remainder = int(total) - sum(base.values())
    order = sorted(raw, key=lambda key: raw[key] - base[key], reverse=True)
    for key in order[:remainder]:
        base[key] += 1
    return base


def build_token_plan(
    total_tokens: int,
    target_mix: Mapping[str, float] | None = None,
    code_mix: Mapping[str, float] | None = None,
) -> TokenPlan:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    target_mix = {key: float(value) for key, value in (target_mix or TOP_LEVEL_MIX).items()}
    if not target_mix or any(value < 0 for value in target_mix.values()):
        raise ValueError("target_mix must contain non-negative source ratios")
    if not math.isclose(sum(target_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("target_mix ratios must sum to 1.0")
    if "code" not in target_mix:
        return TokenPlan(total_tokens=total_tokens, quotas=_round_alloc(total_tokens, target_mix))

    unknown = set(target_mix) - set(TOP_LEVEL_MIX)
    if unknown:
        raise ValueError(f"code-expanded mixes cannot include unknown buckets: {sorted(unknown)}")
    code_mix = {key: float(value) for key, value in (code_mix or CODE_MIX).items() if key in CODE_MIX}
    top = _round_alloc(total_tokens, target_mix)
    code_total = top.pop("code")
    code = _round_alloc(code_total, code_mix)
    quotas = {
        "fineweb_edu": top["fineweb_edu"],
        "fineweb": top["fineweb"],
    }
    quotas.update({f"code_{key}": value for key, value in code.items()})
    return TokenPlan(total_tokens=total_tokens, quotas=quotas)


def ratio_report(actual: Mapping[str, int], plan: TokenPlan) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    total_actual = max(sum(actual.get(key, 0) for key in plan.quotas), 1)
    for key, expected in plan.quotas.items():
        got = int(actual.get(key, 0))
        report[key] = {
            "expected_tokens": float(expected),
            "actual_tokens": float(got),
            "expected_ratio": expected / max(plan.total_tokens, 1),
            "actual_ratio": got / total_actual,
            "absolute_error": abs(got - expected),
        }
    return report


def assert_within_tolerance(actual: Mapping[str, int], plan: TokenPlan, tolerance: float = 0.001) -> None:
    total_actual = max(sum(actual.get(key, 0) for key in plan.quotas), 1)
    failures = []
    for key, expected in plan.quotas.items():
        expected_ratio = expected / max(plan.total_tokens, 1)
        actual_ratio = int(actual.get(key, 0)) / total_actual
        if abs(actual_ratio - expected_ratio) > tolerance:
            failures.append(f"{key}: expected {expected_ratio:.6f}, got {actual_ratio:.6f}")
    if failures:
        raise ValueError("Dataset ratio validation failed: " + "; ".join(failures))
