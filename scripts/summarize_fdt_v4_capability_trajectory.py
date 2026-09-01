from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def exact_summary(report: dict[str, Any]) -> dict[str, Any]:
    matrix = report["exact_memory"]["matrix"]
    if isinstance(matrix, list):
        matrix = matrix[0]
    cells = [cell for by_distance in matrix.values() for cell in by_distance.values()]
    retrieve = [cell["modes"]["retrieve"] for cell in cells]
    copy = [cell["modes"]["copy"] for cell in cells]
    recalled = [
        {
            "length": int(length),
            "distance": int(distance),
        }
        for length, by_distance in matrix.items()
        for distance, cell in by_distance.items()
        if cell["modes"]["retrieve"].get("candidate_recall")
    ]
    gates = [float(cell.get("mix_gate") or 0.0) for cell in copy]
    return {
        "cells": len(cells),
        "proposal_recall_count": sum(bool(cell.get("candidate_recall")) for cell in retrieve),
        "proposal_recall_rate": mean(bool(cell.get("candidate_recall")) for cell in retrieve),
        "recalled_cells": recalled,
        "whole_string_exact_count": sum(bool(cell.get("free_exact")) for cell in copy),
        "whole_string_exact_rate": mean(bool(cell.get("free_exact")) for cell in copy),
        "mean_token_accuracy": mean(float(cell.get("token_accuracy") or 0.0) for cell in copy),
        "first_token_divergence_count": sum(cell.get("first_divergence") == 0 for cell in copy),
        "mean_copy_gate": mean(gates),
        "min_copy_gate": min(gates),
        "max_copy_gate": max(gates),
        "full_scan_fallback_count": sum(bool(cell.get("used_full_scan_fallback")) for cell in copy),
    }


def point_summary(path: Path, token_override: int | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = report["checkpoint"]
    tokens = checkpoint.get("tokens_seen")
    if tokens is None:
        tokens = token_override
    repetition = report["independent_repetition"]["summary"]
    paired = report["paired_bootstrap"]["candidate_minus_comparator"]
    qualitative: dict[str, Any] = {}
    for category in ("natural", "factual", "retrieval"):
        row = report["free_generation"]["categories"][category]["rows"][0]
        qualitative[category] = {
            "prompt": row["prompt"],
            "target": row["target"],
            "completion": row["completion"],
            "loop": row["loop"],
        }
    return {
        "tokens_seen": int(tokens or 0),
        "checkpoint": checkpoint,
        "evaluation": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        },
        "teacher_forced": {
            "nll": report["teacher_forced"]["nll"],
            "top1_rate": report["teacher_forced"]["top1"]["rate"],
            "paired_nll_delta_vs_v20": paired["mean_nll"],
            "paired_nll_delta_ci95": paired["mean_nll_bootstrap_95"],
            "paired_top1_delta_vs_v20": paired["mean_top1"],
            "paired_top1_delta_ci95": paired["mean_top1_bootstrap_95"],
        },
        "penalty_off_repetition": {
            "rows": repetition["count"],
            "loop_free_count": repetition["loop_free"]["count"],
            "loop_free_rate": repetition["loop_free"]["rate"],
            "loop_free_ci95": repetition["loop_free_bootstrap_95"],
            "mean_repetition_rate": repetition["mean_repetition_rate"],
            "mean_repetition_ci95": repetition["mean_repetition_bootstrap_95"],
        },
        "exact_memory": exact_summary(report),
        "qualitative_first_rows": qualitative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-contract FDT v4 trajectory evaluations")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--official-200m", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = [
        point_summary(args.trajectory_dir / "fdt_v4_000000000_official_fp32_eval.json", token_override=0),
        point_summary(args.trajectory_dir / "fdt_v4_009831728_official_fp32_eval.json", token_override=9_831_728),
        point_summary(args.trajectory_dir / "fdt_v4_036370475_official_fp32_eval.json", token_override=36_370_475),
        point_summary(args.trajectory_dir / "fdt_v4_100005238_official_fp32_eval.json", token_override=100_005_238),
        point_summary(args.official_200m, token_override=200_102_827),
    ]
    points.sort(key=lambda point: point["tokens_seen"])

    intervals = []
    for previous, current in zip(points, points[1:]):
        token_delta_m = (current["tokens_seen"] - previous["tokens_seen"]) / 1_000_000
        nll_delta = current["teacher_forced"]["nll"] - previous["teacher_forced"]["nll"]
        paired_delta = (
            current["teacher_forced"]["paired_nll_delta_vs_v20"]
            - previous["teacher_forced"]["paired_nll_delta_vs_v20"]
        )
        intervals.append(
            {
                "from_tokens": previous["tokens_seen"],
                "to_tokens": current["tokens_seen"],
                "token_delta_m": token_delta_m,
                "nll_change_per_million_tokens": nll_delta / token_delta_m,
                "paired_gap_change_per_million_tokens": paired_delta / token_delta_m,
                "loop_free_rate_change": (
                    current["penalty_off_repetition"]["loop_free_rate"]
                    - previous["penalty_off_repetition"]["loop_free_rate"]
                ),
            }
        )

    last_rate = intervals[-1]["paired_gap_change_per_million_tokens"]
    remaining_gap = points[-1]["teacher_forced"]["paired_nll_delta_vs_v20"]
    projected_additional_m = None
    if last_rate < 0 and remaining_gap > 0:
        projected_additional_m = remaining_gap / -last_rate

    summary = {
        "schema": "fdt_v4_capability_trajectory_v1",
        "evaluation_contract": {
            "device": "cpu",
            "dtype": "float32",
            "quantization": "none",
            "paired_rows": 52,
            "bootstrap_samples": 20000,
            "independent_repetition_rows": 100,
            "repetition_penalty": 1.0,
            "exact_memory_cells": 30,
        },
        "points": points,
        "intervals": intervals,
        "late_linear_projection": {
            "basis": "100M-to-200M paired-gap slope; descriptive, not a forecast",
            "additional_million_tokens_to_close_v20_gap": projected_additional_m,
        },
        "decision": {
            "architecture_failure_proven": False,
            "base_adaptation_observed": True,
            "base_adaptation_complete": False,
            "free_generation_recovery_observed": False,
            "exact_memory_behavioral_learning_observed": False,
            "current_trajectory_safe_to_scale_blindly": False,
            "verdict": "KEEP_PAUSED_REDESIGN_TRANSITION_AND_OBJECTIVES",
        },
        "limitations": [
            "No preserved 50M or 150M model-only checkpoint exists; no values were interpolated.",
            "The 200M point uses the verified 200.103M PAUSED checkpoint evaluation.",
            "The late linear projection assumes an unchanged slope and is descriptive only.",
        ],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    atomic_text(args.output, text)
    atomic_text(args.output.with_name(args.output.name + ".sha256"), sha256_file(args.output) + "\n")


if __name__ == "__main__":
    main()
