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
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_speed_observable.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_batch4_migration_trainer", TRAINER)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)

RUNTIME_KEYS = {
    "activation_checkpointing_min_sequence_length",
    "lm_loss_checkpointing_min_sequence_length",
    "short_sequence_batch_size",
    "short_sequence_max_length",
    "lm_loss_sequence_chunk_size",
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


def variant(benchmark: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in benchmark["variants"] if item["name"] == name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination_run.resolve()
    config_path = args.config.resolve()
    benchmark_path = args.benchmark.resolve()
    for path in (source, destination, config_path, benchmark_path, TRAINER):
        if path.drive.upper() != "C:":
            raise ValueError("batch4 migration is C-only")
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in destination.parents or destination.exists():
        raise FileExistsError("destination run must be a fresh child of runs")

    source_sha = sha256(source)
    if source_sha != args.expected_source_sha256.upper():
        raise ValueError("source recovery hash mismatch")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selected_name = "no_grad_batch4_chunk256"
    selected = variant(benchmark, selected_name)
    baseline = variant(benchmark, "live_batch1_chunk256")
    update = benchmark["sampled_update_comparisons_vs_live"][selected_name]
    if not all(
        (
            benchmark.get("status") == "PASS",
            benchmark.get("checkpoint_sha256") == source_sha,
            benchmark.get("candidate_trainer_sha256") == sha256(TRAINER),
            benchmark.get("selected_variant") == selected_name,
            benchmark.get("equivalence", {}).get(selected_name) is True,
            benchmark.get("all_sample_orders_identical") is True,
            benchmark.get("all_source_states_identical") is True,
            benchmark.get("all_finite") is True,
            benchmark.get("long_context_safety", {}).get(selected_name) is True,
            float(selected["mean_tokens_per_second"]) >= 1.40 * float(baseline["mean_tokens_per_second"]),
            float(selected["peak_reserved_gib"]) < 15.5,
            float(update["relative_l2"]) <= 1e-6,
            float(update["max_abs"]) <= 1e-5,
            float(benchmark["base_loss_max_abs_deltas_vs_live"][selected_name]) <= 1e-3,
            float(benchmark["exact_loss_max_abs_deltas_vs_live"][selected_name]) <= 1e-3,
        )
    ):
        raise ValueError("batch4 benchmark did not pass the pinned migration gates")

    config, new_train, _, _ = trainer.model_config_and_settings(config_path, destination)
    if int(new_train.get("short_sequence_batch_size", 0)) != 4:
        raise ValueError("destination config is not the verified batch4 runtime")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "model_config", "train_config",
        "source_states", "torch_rng_state", "cuda_rng_state_all", "python_random_state",
    }
    if not all(
        (
            payload.get("stage_status") == "PAUSED",
            payload.get("optimizer_state_included") is True,
            required.issubset(payload),
            payload.get("model_config") == asdict(config),
            bool(payload["optimizer_state_dict"].get("state")),
        )
    ):
        raise ValueError("source is not an exact-resumable FDT v4 checkpoint")

    old_train = dict(payload["train_config"])
    ignored = {"output_dir", "config_path", "config_sha256", "runtime_trainer", "runtime_trainer_sha256", *RUNTIME_KEYS}
    changed = {
        key: {"old": old_train.get(key), "new": value}
        for key, value in new_train.items()
        if key not in ignored and old_train.get(key) != value
    }
    if changed:
        raise ValueError(f"mathematical/data resume contract changed: {changed}")

    destination.mkdir(parents=True)
    checkpoint = destination / "latest_recovery.pt"
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
            "runtime_migration_contract": "batch4_short_sequence_equivalent_v1",
            "runtime_migration_source": str(source),
            "runtime_migration_source_sha256": source_sha,
            "runtime_benchmark": str(benchmark_path),
            "runtime_benchmark_sha256": sha256(benchmark_path),
            "runtime_selected_variant": selected_name,
            "runtime_measured_tokens_per_second": selected["mean_tokens_per_second"],
            "runtime_math_data_optimizer_rng_preserved": True,
        }
    )
    payload["train_config"] = migrated_train
    payload["stage_status"] = "PAUSED"
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    destination_sha = sha256(checkpoint)

    verified = torch.load(checkpoint, map_location="meta", weights_only=False)
    report = {
        "status": "PASS",
        "source": str(source),
        "source_sha256": source_sha,
        "destination": str(checkpoint),
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
        "selected_variant": selected_name,
        "measured_tokens_per_second": selected["mean_tokens_per_second"],
        "temporary_residue": [path.name for path in destination.glob("*.tmp")],
    }
    if not all(
        (
            report["stage_status"] == "PAUSED", report["optimizer_state_included"],
            report["optimizer_state_present"], report["source_states_equal"],
            report["torch_rng_present"], report["cuda_rng_present"],
            report["python_rng_present"], report["model_config_equal"],
            not report["temporary_residue"],
        )
    ):
        raise RuntimeError(f"migrated recovery verification failed: {report}")
    atomic_json(destination / "batch4_migration.json", report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
