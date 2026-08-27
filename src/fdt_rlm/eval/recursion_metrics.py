from __future__ import annotations

from typing import Iterable, Mapping


def recursion_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    rows = list(rows)
    tp = sum(bool(row.get("call_made")) and bool(row.get("call_helpful")) for row in rows)
    fp = sum(bool(row.get("call_made")) and not bool(row.get("call_helpful")) for row in rows)
    fn = sum(not bool(row.get("call_made")) and bool(row.get("call_helpful")) for row in rows)
    tn = sum(not bool(row.get("call_made")) and not bool(row.get("call_helpful")) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "unnecessary_call_rate": fp / max(len(rows), 1),
    }
