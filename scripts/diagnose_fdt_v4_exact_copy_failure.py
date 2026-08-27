from __future__ import annotations

"""Read-only diagnosis of a completed FDT v4 exact-copy audit.

This tool deliberately does not import torch, load a checkpoint, start CUDA,
or change copy semantics. It classifies the first failed stage from immutable
audit rows and the warm-start conversion manifest.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def classify(audit: dict[str, Any], conversion: dict[str, Any]) -> dict[str, Any]:
    cells = audit.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("audit must contain non-empty cells")
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    checkpoint = audit.get("checkpoint") if isinstance(audit.get("checkpoint"), dict) else {}
    exact_provenance = checkpoint.get("exact_weight_provenance", "UNKNOWN")
    random_components = conversion.get("new_random_components", [])
    skipped = conversion.get("skipped_new_or_denied_keys", [])
    first_steps = [
        cell.get("cursor_trace", [])[0]
        for cell in cells
        if isinstance(cell.get("cursor_trace"), list) and cell.get("cursor_trace")
    ]
    cursor_steps = [
        step
        for cell in cells
        for step in (cell.get("cursor_trace") or [])
        if isinstance(step, dict) and step.get("mode") == "cursor"
    ]
    alignment = [cell.get("prompt_target_token_alignment_exact") for cell in cells]
    alignment_status = (
        "PASS" if alignment and all(value is True for value in alignment)
        else "FAIL" if any(value is False for value in alignment)
        else "NOT_MEASURED_IN_THIS_LEGACY_AUDIT"
    )
    all_first_mixed = bool(first_steps) and all(
        isinstance(step, dict) and step.get("mode") == "mixed" for step in first_steps
    )
    retrieval_failed = float(summary.get("retrieval_success_rate", 0.0)) == 0.0
    gate_active = float(summary.get("copy_gate_activation_rate", 0.0)) == 1.0
    untrained = exact_provenance == "UNTRAINED_WARM_START" and bool(random_components)
    if untrained and retrieval_failed and gate_active:
        immediate_cause = "UNTRAINED_EXACT_POINTER_SELECTION_PARAMETERS"
        confidence = "HIGH"
    elif retrieval_failed and gate_active:
        immediate_cause = "RETRIEVAL_OR_CANDIDATE_SELECTION_FAILURE_REQUIRES_TRAINED_CHECKPOINT_DIAGNOSIS"
        confidence = "MEDIUM"
    else:
        immediate_cause = "INCONCLUSIVE"
        confidence = "LOW"
    return {
        "schema": "fdt_v4_exact_copy_failure_diagnosis_v1",
        "gpu_launched": False,
        "training_started": False,
        "audit_summary": {
            "tested_cells": int(summary.get("tested_cells", 0)),
            "whole_string_exact_rate": summary.get("whole_string_exact_rate"),
            "retrieval_success_rate": summary.get("retrieval_success_rate"),
            "copy_gate_activation_rate": summary.get("copy_gate_activation_rate"),
            "max_full_scan_count": summary.get("max_full_scan_count"),
        },
        "classification": {
            "immediate_cause": immediate_cause,
            "confidence": confidence,
            "untrained_exact_parameters": {
                "status": "CONFIRMED" if untrained else "NOT_CONFIRMED",
                "checkpoint_provenance": exact_provenance,
                "new_random_components": random_components,
                "conversion_skipped_components": skipped,
            },
            "evaluator_prompt_alignment": {
                "status": alignment_status,
                "reason": "The completed legacy audit did not record embedded prompt-token alignment. The evaluator now records it for future runs.",
            },
            "retrieval_candidate_selection": {
                "status": "FAILED_BEFORE_CURSOR" if retrieval_failed and gate_active else "INCONCLUSIVE",
                "reason": "All first free-generation steps remained mixed and the expected source key was absent from every recorded top candidate set.",
                "all_first_steps_mixed": all_first_mixed,
            },
            "decode_cursor": {
                "status": "NOT_REACHED" if not cursor_steps else "NOT_CAUSAL_FOR_INITIAL_RETRIEVAL_FAILURE",
                "reason": (
                    "No cursor step occurred."
                    if not cursor_steps
                    else "Cursor steps occurred only after a mixed first step. Every audit cell failed first-step target-key retrieval, so later cursor behavior cannot explain the immediate retrieval failure."
                ),
                "free_generation_cursor_steps": len(cursor_steps),
            },
        },
        "minimal_next_action": "Before any long-scale run, create a trained FDT v4 checkpoint with the exact-pointer loss enabled, then rerun this unchanged 60-cell audit with prompt-token alignment recorded. Do not change exact-copy criteria or architecture semantics.",
    }


def write_atomic(output_dir: Path, report: dict[str, Any]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        report_path = temporary / "fdt_v4_exact_copy_failure_diagnosis.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        hashes = {report_path.name: sha256_file(report_path)}
        (temporary / "sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only FDT v4 exact-copy failure diagnosis")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    manifest_path = args.conversion_manifest.resolve()
    report = classify(read_json(audit_path), read_json(manifest_path))
    report["inputs"] = {
        "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "conversion_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
    }
    report["evaluator"] = {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}
    write_atomic(args.output_dir.resolve(), report)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "immediate_cause": report["classification"]["immediate_cause"], "gpu_launched": False}))


if __name__ == "__main__":
    main()
