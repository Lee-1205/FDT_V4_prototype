from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
POINTER = ROOT / "src" / "fdt_rlm" / "lexical_pointer.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_syncfix_trainer", TRAINER)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--backend-diagnosis", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination_run.resolve()
    config_path = args.config.resolve()
    backend_diagnosis_path = args.backend_diagnosis.resolve()
    for path in (source, destination, config_path, backend_diagnosis_path, TRAINER, POINTER):
        if path.drive.upper() != "C:":
            raise ValueError("sync-fix migration is C-only")
    if (ROOT / "runs").resolve() not in destination.parents or destination.exists():
        raise FileExistsError("destination must be a fresh child of runs")
    source_sha = sha256(source)
    if source_sha != args.expected_source_sha256.upper():
        raise ValueError("source checkpoint SHA-256 mismatch")
    backend_diagnosis = json.loads(backend_diagnosis_path.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in backend_diagnosis.get("variants", [])
        if item.get("name") == "candidate_fast_backend"
    )
    if (
        backend_diagnosis.get("status") != "PASS"
        or backend_diagnosis.get("checkpoint_sha256") != source_sha
        or not backend_diagnosis.get("equivalence_pass")
        or not backend_diagnosis.get("long_context_contract_pass")
        or float(candidate.get("mean_tokens_per_second", 0.0)) < 1000.0
    ):
        raise ValueError("fast CUDA backend lacks checkpoint-matched equivalence evidence")

    config, _, _, _ = trainer.model_config_and_settings(config_path, destination)
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
        or bool(payload.get("train_config", {}).get("overfit_triggered", False))
    ):
        raise ValueError("source is not a resumable, non-overfit FDT v4 checkpoint")
    if str(payload["train_config"].get("config_sha256", "")).upper() != sha256(config_path):
        raise ValueError("source training configuration does not match")

    destination.mkdir(parents=True)
    destination_checkpoint = destination / "latest_recovery.pt"
    migrated_train = dict(payload["train_config"])
    migrated_train.update(
        {
            "output_dir": str(destination),
            "runtime_trainer": str(TRAINER),
            "runtime_trainer_sha256": sha256(TRAINER),
            "runtime_pointer_sha256": sha256(POINTER),
            "deterministic_algorithms": False,
            "cuda_backend_contract": "verified_fast_equivalent_v1",
            "backend_diagnosis": str(backend_diagnosis_path),
            "backend_diagnosis_sha256": sha256(backend_diagnosis_path),
            "syncfix_source": str(source),
            "syncfix_source_sha256": source_sha,
            "syncfix_contract": "diagnostics_and_runtime_only_v1",
            "syncfix_math_data_optimizer_rng_preserved": True,
            "syncfix_changes": [
                "session-local throughput meter",
                "one finite-loss synchronization per optimizer step",
                "zero-weight prefix graph elision with identical cursor advance",
                "runtime-aligned proposal recall measured only on metric steps",
                "verified equivalent fast CUDA backend",
            ],
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
        "source_sha256": source_sha,
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
        "trainer_sha256": sha256(TRAINER),
        "pointer_sha256": sha256(POINTER),
        "backend_diagnosis_sha256": sha256(backend_diagnosis_path),
        "measured_short_tokens_per_second": candidate["mean_tokens_per_second"],
        "temporary_residue": [path.name for path in destination.glob("*.tmp")],
    }
    checks = (
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
    if not all(checks):
        raise RuntimeError(f"sync-fix migration verification failed: {verification}")
    atomic_json(destination / "syncfix_migration.json", verification)
    print(json.dumps(verification))
    payload = verified = None
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
