from __future__ import annotations

from typing import Iterable, Mapping


def action_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    rows = list(rows)
    total = max(len(rows), 1)
    return {
        "valid_json_rate": sum(bool(row.get("valid_json")) for row in rows) / total,
        "valid_action_rate": sum(bool(row.get("valid_action")) for row in rows) / total,
        "action_accuracy": sum(bool(row.get("action_correct")) for row in rows) / total,
        "argument_accuracy": sum(bool(row.get("arguments_correct")) for row in rows) / total,
        "repairable_parse_rate": sum(bool(row.get("repairable")) for row in rows) / total,
    }
