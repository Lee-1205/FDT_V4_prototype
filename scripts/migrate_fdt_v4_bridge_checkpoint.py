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
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_bridge.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_bridge_migration_trainer", TRAINER)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


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
    parser = argparse.ArgumentParser(description="Add the audited 2K/4K bridge to an exact FDT v4 resume")
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
            raise ValueError("bridge migration is C-only")
    if (ROOT / "runs").resolve() not in destination.parents or destination.exists():
        raise FileExistsError("destination must be a fresh run path")
    source_sha = sha256(source)
    if source_sha != args.expected_source_sha256.upper():
        raise ValueError("source checkpoint hash mismatch")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not all(
        (
            benchmark.get("status") == "PASS",
            benchmark.get("checkpoint_sha256") == source_sha,
            benchmark.get("config_sha256") == sha256(config_path),
            benchmark.get("trainer_sha256") == sha256(TRAINER),
            {row.get("sequence_length") for row in benchmark.get("rows", [])} == {2048, 4096},
            all(row.get("finite_loss") and row.get("finite_gradients") for row in benchmark.get("rows", [])),
        )
    ):
        raise ValueError("2K/4K safety benchmark did not pass")

    config, new_train, data_cfg, _ = trainer.model_config_and_settings(config_path, destination)
    dataset_dirs = {
        "natural": ROOT / data_cfg["natural_lm_dir"],
        "factual": ROOT / data_cfg["factual_dir"],
        "exact_copy": ROOT / data_cfg["exact_copy_dir"],
        "long_context": ROOT / data_cfg["long_context_dir"],
        "generated_prefix": ROOT / data_cfg["generated_prefix_dir"],
        "bridge_context": ROOT / data_cfg["bridge_context_dir"],
        "validation": ROOT / data_cfg["validation_dir"],
    }
    preflight = trainer.preflight_dataset_contract(dataset_dirs, data_cfg)
    fingerprints = {name: trainer.manifest_hash(path) for name, path in dataset_dirs.items()}
    bridge_manifest = json.loads((dataset_dirs["bridge_context"] / "manifest.json").read_text(encoding="utf-8"))
    if bridge_manifest.get("post_build_exact_overlap") != 0:
        raise ValueError("bridge manifest did not pass overlap audit")

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
        raise ValueError("source is not an exact-resumable PAUSED checkpoint")
    old_states = dict(payload["source_states"])
    if "bridge_context" in old_states:
        raise ValueError("source already contains a bridge cursor")
    bridge_seed = int(new_train.get("seed", 20260823)) + 5 * 1009
    bridge_pool = trainer.RowPool(dataset_dirs["bridge_context"], data_cfg.get("split", "train"), bridge_seed)
    bridge_state = bridge_pool.state()
    del bridge_pool

    intentional = {
        "source_batch_fractions", "require_context_bridge", "bridge_require_2k_bucket",
        "bridge_require_4k_bucket", "activation_checkpointing_min_sequence_length",
        "lm_loss_checkpointing_min_sequence_length",
    }
    old_train = dict(payload["train_config"])
    ignored = {
        "output_dir", "config_path", "config_sha256", "runtime_trainer", "runtime_trainer_sha256",
        "dataset_dirs", "dataset_manifest_sha256", "data_preflight", *intentional,
    }
    changed = {
        key: {"old": old_train.get(key), "new": value}
        for key, value in new_train.items()
        if key not in ignored and old_train.get(key) != value
    }
    if changed:
        raise ValueError(f"unplanned training contract change: {changed}")

    migrated_train = dict(old_train)
    for key in intentional:
        migrated_train[key] = new_train[key]
    migrated_train.update(
        {
            "output_dir": str(destination),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runtime_trainer": str(TRAINER),
            "runtime_trainer_sha256": sha256(TRAINER),
            "dataset_dirs": {name: str(path) for name, path in dataset_dirs.items()},
            "dataset_manifest_sha256": fingerprints,
            "data_preflight": preflight,
            "bridge_transition_contract": "512_2k_4k_8k_16k_v1",
            "bridge_transition_source": str(source),
            "bridge_transition_source_sha256": source_sha,
            "bridge_safety_benchmark": str(benchmark_path),
            "bridge_safety_benchmark_sha256": sha256(benchmark_path),
            "existing_model_optimizer_rng_cursors_preserved": True,
        }
    )
    payload["train_config"] = migrated_train
    payload["source_states"] = {**old_states, "bridge_context": bridge_state}
    payload["stage_status"] = "PAUSED"

    destination.mkdir(parents=True)
    checkpoint = destination / "latest_recovery.pt"
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    destination_sha = sha256(checkpoint)
    verified = torch.load(checkpoint, map_location="meta", weights_only=False)
    report = {
        "status": "PASS",
        "source_sha256": source_sha,
        "destination": str(checkpoint),
        "destination_sha256": destination_sha,
        "optimizer_step": int(verified.get("optimizer_step", -1)),
        "tokens_seen": int(verified.get("tokens_seen", -1)),
        "stage_status": verified.get("stage_status"),
        "optimizer_state_present": bool(verified["optimizer_state_dict"].get("state")),
        "existing_source_states_preserved": all(verified["source_states"].get(name) == state for name, state in old_states.items()),
        "bridge_state": verified["source_states"].get("bridge_context"),
        "rng_present": all(key in verified for key in ("torch_rng_state", "cuda_rng_state_all", "python_random_state")),
        "model_config_equal": verified.get("model_config") == payload.get("model_config"),
        "temporary_residue": [path.name for path in destination.glob("*.tmp")],
    }
    if not all(
        (
            report["stage_status"] == "PAUSED", report["optimizer_state_present"],
            report["existing_source_states_preserved"], bool(report["bridge_state"]),
            report["rng_present"], report["model_config_equal"], not report["temporary_residue"],
        )
    ):
        raise RuntimeError(f"bridge migration verification failed: {report}")
    atomic_json(destination / "bridge_migration.json", report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
