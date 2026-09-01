from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_fdt_v4_curriculum_bridge.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_1_trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = trainer.require_c_path(args.source, "source checkpoint")
    config_path = trainer.require_c_path(args.config, "config")
    evidence_path = trainer.require_c_path(args.evidence, "evidence")
    output_dir = trainer.owned_run_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"migration output is not fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_sha = trainer.sha256_file(source)
    if source_sha != args.source_sha256.upper():
        raise ValueError("source checkpoint SHA-256 mismatch")
    evidence_sha = trainer.sha256_file(evidence_path)
    if evidence_sha != args.evidence_sha256.upper():
        raise ValueError("fixed-validation evidence SHA-256 mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS":
        raise ValueError("fixed-validation recheck did not pass")
    rows = {row["name"]: row for row in evidence.get("candidate", [])}
    parent_loss = float(evidence["parent_validation_loss"])
    legacy_loss = float(rows["legacy_controls"]["validation_loss"])
    scheduled_loss = float(rows["current_controls"]["validation_loss"])
    if legacy_loss > parent_loss * 1.01 or scheduled_loss > parent_loss * 1.05:
        raise ValueError("fixed-validation recheck exceeds the migration gates")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    settings_before = dict(payload.get("train_config", {}))
    if payload.get("stage_status") != "SAFETY_STOP":
        raise ValueError("migration requires a SAFETY_STOP checkpoint")
    if not payload.get("optimizer_state_included") or "optimizer_state_dict" not in payload:
        raise ValueError("migration requires full optimizer state")
    if bool(settings_before.get("overfit_triggered", False)):
        raise ValueError("migration refuses a genuine overfit stop")
    if not bool(settings_before.get("transition_regression_triggered", False)):
        raise ValueError("source does not record the invalid validation gate")

    config, train_cfg, data_cfg, _ = trainer.model_config_and_settings(
        config_path, output_dir
    )
    if not trainer.resume_model_configs_compatible(
        payload["model_config"], trainer.asdict(config)
    ):
        raise ValueError("model architecture changed during validation migration")
    validation_seed = int(train_cfg["validation_seed"])
    validation_dir = trainer.require_c_path(
        ROOT / data_cfg["validation_dir"], "validation dataset"
    )
    validation_pool = trainer.RowPool(
        validation_dir,
        data_cfg.get("validation_split", "validation"),
        validation_seed,
    )

    invalidated = {
        "reason": "validation seed was coupled to the training seed, so the run was compared with a parent baseline from different fixed rows",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha,
        "source_training_seed": settings_before.get("seed"),
        "source_validation_seed": int(settings_before.get("seed", 0)) + 7001,
        "pinned_validation_seed": validation_seed,
        "invalid_scheduled_validation_loss": settings_before.get("last_validation_loss"),
        "invalid_gate_validation_loss": settings_before.get("last_gate_validation_loss"),
        "fixed_recheck_evidence": str(evidence_path),
        "fixed_recheck_evidence_sha256": evidence_sha,
        "fixed_legacy_loss": legacy_loss,
        "fixed_scheduled_loss": scheduled_loss,
        "parent_loss": parent_loss,
    }
    settings = dict(settings_before)
    settings.update(train_cfg)
    settings.update(
        {
            "config_path": str(config_path),
            "config_sha256": trainer.sha256_file(config_path),
            "output_dir": str(output_dir),
            "validation_history": [],
            "overfit_triggered": False,
            "transition_regression_triggered": False,
            "validation_nonfinite_triggered": False,
            "last_validation_loss": scheduled_loss,
            "last_gate_validation_loss": legacy_loss,
            "invalidated_validation_seed_gate": invalidated,
            "resume_migration": "coupled_validation_seed_to_pinned_validation_seed_v1",
        }
    )
    source_states = dict(payload["source_states"])
    source_states["__validation__"] = validation_pool.state()
    payload["source_states"] = source_states
    payload["train_config"] = settings
    payload["stage_status"] = "PAUSED"

    destination = output_dir / "latest_recovery.pt"
    trainer.atomic_torch_save(destination, payload)
    destination_sha = trainer.sha256_file(destination)
    verified = torch.load(
        destination, map_location="cpu", weights_only=False, mmap=True
    )
    verification = {
        "status": "PASS",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha,
        "checkpoint": str(destination),
        "checkpoint_sha256": destination_sha,
        "optimizer_step": int(verified["optimizer_step"]),
        "tokens_seen": int(verified["tokens_seen"]),
        "stage_status": verified["stage_status"],
        "optimizer_state_included": bool(verified["optimizer_state_included"]),
        "source_states_present": bool(verified.get("source_states")),
        "torch_rng_present": "torch_rng_state" in verified,
        "cuda_rng_present": verified.get("cuda_rng_state_all") is not None,
        "python_rng_present": "python_random_state" in verified,
        "pinned_validation_seed": validation_seed,
        "fixed_legacy_loss": legacy_loss,
        "fixed_scheduled_loss": scheduled_loss,
        "temp_residue": [path.name for path in output_dir.glob("*.tmp")],
    }
    trainer.atomic_json(output_dir / "validation_seed_migration.json", verification)
    trainer.atomic_json(output_dir / "run_manifest.json", settings)
    del verified, payload
    gc.collect()
    print(json.dumps(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
