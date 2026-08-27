from __future__ import annotations

from typing import Iterable


def brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    pairs = list(zip(probabilities, labels))
    return sum((float(prob) - int(label)) ** 2 for prob, label in pairs) / max(len(pairs), 1)


def expected_calibration_error(probabilities: Iterable[float], labels: Iterable[int], bins: int = 10) -> float:
    pairs = list(zip(probabilities, labels))
    total = max(len(pairs), 1)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [(p, y) for p, y in pairs if low <= p < high or (index == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(y for _, y in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - accuracy)
    return error
