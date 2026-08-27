from __future__ import annotations

"""Atomic handoff wrapper for the FDT v4 CPU evaluator.

Terra writes a READY manifest once all inputs are complete.  This wrapper
validates the manifest and input digests, delegates the evaluation, then writes
an independently digestible RESULT manifest atomically.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fdt_v4


MAJOR_CHANGE_CLASSES = {"architecture", "objective", "data_contract", "model_scale"}
LIGHTWEIGHT_CHANGE_CLASSES = {"operational", "runtime_config"}
EVALUATION_HANDOFF_STATUS = "READY"
EVALUATION_HANDOFF_TYPE = "EVALUATION"
INCIDENT_HANDOFF_STATUS = "ABNORMAL"
INCIDENT_HANDOFF_TYPE = "INCIDENT"


def classify_luna_abnormality(handoff: dict[str, Any]) -> dict[str, Any]:
    """Classify without applying a repair or changing the single 426M lineage."""
    event = str(handoff.get("event") or handoff.get("abnormality") or "unknown").lower()
    requested = str(handoff.get("requested_change") or "").lower()
    affected = {str(value).lower() for value in handoff.get("affected_components", [])}
    parameter_target = handoff.get("proposed_parameter_count")
    major_tokens = {
        "architecture": ("architecture", "anchor", "rope", "exact_memory", "layer", "top_k", "router"),
        "objective": ("objective", "loss", "weight", "scheduled", "unlikelihood", "gate"),
        "data_contract": ("dataset", "tokenizer", "manifest", "schema", "source", "license", "overlap", "shard"),
        "model_scale": ("parameter", "116m", "scale", "width", "depth"),
    }
    combined = " ".join((event, requested, " ".join(sorted(affected))))
    change_class = next(
        (kind for kind, tokens in major_tokens.items() if any(token in combined for token in tokens)),
        None,
    )
    if parameter_target is not None:
        try:
            if int(parameter_target) != 426_000_000:
                change_class = "model_scale"
        except (TypeError, ValueError):
            change_class = "model_scale"
    if change_class is not None:
        return {
            "severity": "high" if change_class != "data_contract" else "critical",
            "change_class": change_class,
            "route": "Sol",
            "auto_apply": False,
            "repair_recommendation": "Do not alter the active 426M lineage. Prepare an evidence bundle for Sol review.",
        }
    critical_events = ("nonfinite", "nan", "inf", "checkpoint", "corrupt", "security", "overlap", "tdr", "nvlddmkm")
    if any(token in combined for token in critical_events):
        if any(token in combined for token in ("tdr", "nvlddmkm")):
            return {
                "severity": "high",
                "change_class": "operational",
                "route": "Terra",
                "auto_apply": False,
                "repair_recommendation": "Pause GPU work, preserve the atomic checkpoint and driver evidence, then run a CPU-only integrity check before a bounded resume recommendation.",
            }
        return {
            "severity": "critical",
            "change_class": "operational",
            "route": "Terra",
            "auto_apply": False,
            "repair_recommendation": "Do not resume. Preserve evidence and verify checkpoint, disk, and input digests before proposing a bounded operational repair.",
        }
    runtime_events = ("oom", "memory", "disk", "stale", "pid", "allocator", "throughput", "timeout", "log")
    if any(token in combined for token in runtime_events):
        return {
            "severity": "medium",
            "change_class": "runtime_config",
            "route": "Terra",
            "auto_apply": False,
            "repair_recommendation": "Recommend one bounded runtime/configuration adjustment with rollback, then require a fresh handoff and integrity check. Do not modify architecture, objective, or data contracts.",
        }
    return {
        "severity": "medium",
        "change_class": "unknown",
        "route": "Sol",
        "auto_apply": False,
        "repair_recommendation": "Insufficient evidence for a direct repair. Route the incident and raw evidence to Sol.",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def require_file(value: str, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"handoff {field} is not a file: {path}")
    return path


def require_tokenizer_directory(value: str, field: str) -> tuple[Path, Path]:
    directory = Path(value).expanduser().resolve()
    tokenizer_json = directory / "tokenizer.json"
    if not directory.is_dir() or not tokenizer_json.is_file():
        raise ValueError(f"handoff {field} must be a directory containing tokenizer.json: {directory}")
    return directory, tokenizer_json


def require_tensor_dataset(value: str, field: str) -> Path:
    path = require_file(value, field)
    if path.suffix.lower() in {".json", ".jsonl"}:
        raise ValueError("official Terra evaluation requires a fixed tensor dataset file, not JSON rows")
    return path


def require_digest(manifest: dict[str, Any], field: str, path: Path) -> str:
    expected = manifest.get(field)
    if not expected:
        raise ValueError(f"handoff requires {field}")
    actual = sha256_file(path)
    if actual != str(expected).upper():
        raise ValueError(f"handoff digest mismatch for {field}")
    return actual


def load_ready_handoff(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    if path.suffix == ".tmp" or not path.is_file():
        raise ValueError("handoff must be a completed, non-temporary file")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != EVALUATION_HANDOFF_STATUS or manifest.get("handoff_type") != EVALUATION_HANDOFF_TYPE:
        raise ValueError("evaluation handoff must declare status READY and handoff_type EVALUATION")
    checkpoint = require_file(manifest["checkpoint"], "checkpoint")
    require_digest(manifest, "checkpoint_sha256", checkpoint)
    dataset = require_tensor_dataset(manifest["tensor_dataset"], "tensor_dataset")
    require_digest(manifest, "tensor_dataset_sha256", dataset)
    tokenizer = None
    if manifest.get("tokenizer_dir"):
        tokenizer, tokenizer_json = require_tokenizer_directory(manifest["tokenizer_dir"], "tokenizer_dir")
        require_digest(manifest, "tokenizer_json_sha256", tokenizer_json)
    comparator = None
    if manifest.get("comparator_checkpoint"):
        comparator = require_file(manifest["comparator_checkpoint"], "comparator_checkpoint")
        require_digest(manifest, "comparator_checkpoint_sha256", comparator)
    repetition_dataset = None
    if manifest.get("repetition_tensor_dataset"):
        repetition_dataset = require_tensor_dataset(manifest["repetition_tensor_dataset"], "repetition_tensor_dataset")
        require_digest(manifest, "repetition_tensor_dataset_sha256", repetition_dataset)
    return manifest, {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "comparator": comparator,
        "repetition_dataset": repetition_dataset,
    }


def run_handoff(handoff: Path, result: Path) -> dict[str, Any]:
    handoff = handoff.resolve()
    manifest, paths = load_ready_handoff(handoff)
    evaluation_output = Path(manifest.get("evaluation_output") or result.with_suffix(".evaluation.json")).resolve()
    report = evaluate_fdt_v4.evaluate(
        paths["checkpoint"],
        evaluation_output,
        paths["tokenizer"],
        paths["dataset"],
        int(manifest.get("dataset_limit", 32)),
        bool(manifest.get("allow_python_unit_exec", False)),
        paths["comparator"],
        paths["repetition_dataset"],
        int(manifest.get("bootstrap_samples", 2000)),
    )
    payload = {
        "schema": "terra_fdt_v4_result_v2",
        "status": "RESULT",
        "handoff_type": "EVALUATION_RESULT",
        "handoff": str(handoff),
        "handoff_sha256": sha256_file(handoff),
        "handoff_id": manifest.get("handoff_id"),
        "checkpoint_sha256": report["checkpoint"]["checkpoint_sha256"],
        "evaluation": report["output"],
        "official_evaluation": report["official_evaluation"],
        "integrity_audit": report["integrity_audit"],
        "completed_at_unix": time.time(),
    }
    atomic_json(result.resolve(), payload)
    payload["integrity_digest"] = sha256_file(result.resolve())
    atomic_text(result.resolve().with_name(result.name + ".sha256"), payload["integrity_digest"] + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Terra atomic FDT v4 evaluator handoff")
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run_handoff(args.handoff, args.result)
    print(json.dumps({"result": str(args.result.resolve()), "integrity_digest": result["integrity_digest"]}))


if __name__ == "__main__":
    main()
