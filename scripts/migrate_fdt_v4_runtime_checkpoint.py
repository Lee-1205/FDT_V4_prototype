from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_speed_migration_trainer", TRAINER)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)

RUNTIME_KEYS = {
    "activation_checkpointing_min_sequence_length",
    "lm_loss_checkpointing_min_sequence_length",
    "short_sequence_batch_size",
    "short_sequence_max_length",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--routing-audit", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination_run.resolve()
    config_path = args.config.resolve()
    benchmark_path = args.benchmark.resolve()
    routing_audit_path = args.routing_audit.resolve()
    for path in (source, destination, config_path, benchmark_path, routing_audit_path, TRAINER):
        if path.drive.upper() != "C:":
            raise ValueError("runtime migration is C-only")
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in destination.parents or destination.exists():
        raise FileExistsError("destination run must be a fresh child of runs")
    actual_source_sha = sha256(source)
    if actual_source_sha != args.expected_source_sha256.upper():
        raise ValueError("source recovery hash mismatch")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        benchmark.get("status") != "PASS"
        or benchmark.get("checkpoint_sha256") != actual_source_sha
        or benchmark.get("trainer_sha256") != sha256(TRAINER)
        or not benchmark.get("equivalence_pass")
        or not benchmark.get("all_sample_orders_identical")
        or not benchmark.get("all_source_states_identical")
        or not benchmark.get("long_context_checkpoint_safety_pass")
    ):
        raise ValueError("speed benchmark did not pass all migration gates")
    selected_name = benchmark.get("selected_variant")
    if not selected_name or not benchmark.get("candidate_equivalence", {}).get(selected_name):
        raise ValueError("benchmark did not select an equivalent runtime")
    selected = next(item for item in benchmark["variants"] if item["name"] == selected_name)
    if float(selected["mean_tokens_per_second"]) < 1000.0:
        raise ValueError("selected runtime did not reach the throughput target")
    routing_audit = json.loads(routing_audit_path.read_text(encoding="utf-8"))
    if (
        routing_audit.get("status") != "PASS"
        or routing_audit.get("checkpoint_sha256") != actual_source_sha
        or routing_audit.get("trainer_sha256") != sha256(TRAINER)
        or not routing_audit.get("all_sample_orders_identical")
        or not routing_audit.get("all_source_states_identical")
    ):
        raise ValueError("25-step routing-window audit did not pass")

    config, new_train, _, _ = trainer.model_config_and_settings(config_path, destination)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "train_config",
        "source_states",
        "torch_rng_state",
        "cuda_rng_state_all",
        "python_random_state",
    }
    if (
        payload.get("stage_status") != "PAUSED"
        or not payload.get("optimizer_state_included")
        or not required.issubset(payload)
        or payload.get("model_config") != asdict(config)
    ):
        raise ValueError("source is not an exact-resumable FDT v4 PAUSED checkpoint")
    old_train = dict(payload["train_config"])
    changed = {}
    for key, new_value in new_train.items():
        if key in {"output_dir", *RUNTIME_KEYS}:
            continue
        if key not in old_train or old_train[key] != new_value:
            changed[key] = {"old": old_train.get(key), "new": new_value}
    if changed:
        raise ValueError(f"mathematical/data resume contract changed: {changed}")

    destination.mkdir(parents=True)
    destination_checkpoint = destination / "latest_recovery.pt"
    migrated_train = dict(old_train)
    for key in RUNTIME_KEYS:
        migrated_train[key] = new_train[key]
    migrated_train.update(
        {
            "output_dir": str(destination),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runtime_trainer": str(TRAINER),
            "runtime_trainer_sha256": sha256(TRAINER),
            "runtime_migration_source": str(source),
            "runtime_migration_source_sha256": actual_source_sha,
            "runtime_benchmark": str(benchmark_path),
            "runtime_benchmark_sha256": sha256(benchmark_path),
            "runtime_routing_audit": str(routing_audit_path),
            "runtime_routing_audit_sha256": sha256(routing_audit_path),
            "runtime_selected_variant": selected["name"],
            "runtime_measured_tokens_per_second": selected["mean_tokens_per_second"],
            "runtime_math_and_data_contract_preserved": True,
        }
    )
    payload["train_config"] = migrated_train
    payload["stage_status"] = "PAUSED"
    temporary = destination_checkpoint.with_name(destination_checkpoint.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination_checkpoint)
    destination_sha = sha256(destination_checkpoint)

    verified = torch.load(destination_checkpoint, map_location="meta", weights_only=False)
    verification = {
        "status": "PASS",
        "source": str(source),
        "source_sha256": actual_source_sha,
        "destination": str(destination_checkpoint),
        "destination_sha256": destination_sha,
        "stage_status": verified.get("stage_status"),
        "optimizer_step": int(verified.get("optimizer_step", -1)),
        "tokens_seen": int(verified.get("tokens_seen", -1)),
        "optimizer_state_included": bool(verified.get("optimizer_state_included")),
        "optimizer_state_present": bool(verified["optimizer_state_dict"].get("state")),
        "source_states_equal": verified.get("source_states") == payload.get("source_states"),
        "torch_rng_present": "torch_rng_state" in verified,
        "cuda_rng_present": "cuda_rng_state_all" in verified,
        "python_rng_present": "python_random_state" in verified,
        "model_config_equal": verified.get("model_config") == payload.get("model_config"),
        "selected_variant": selected["name"],
        "measured_tokens_per_second": selected["mean_tokens_per_second"],
        "runtime_changes": {key: migrated_train[key] for key in sorted(RUNTIME_KEYS)},
        "temporary_residue": [path.name for path in destination.glob("*.tmp")],
    }
    if not all(
        (
            verification["stage_status"] == "PAUSED",
            verification["optimizer_state_included"],
            verification["optimizer_state_present"],
            verification["source_states_equal"],
            verification["torch_rng_present"],
            verification["cuda_rng_present"],
            verification["python_rng_present"],
            verification["model_config_equal"],
            not verification["temporary_residue"],
        )
    ):
        raise RuntimeError(f"migrated recovery verification failed: {verification}")
    atomic_json(destination / "runtime_migration.json", verification)
    print(json.dumps(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
