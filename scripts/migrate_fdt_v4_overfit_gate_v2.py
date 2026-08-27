from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_speed_trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = trainer.require_c_path(args.source, "source checkpoint")
    config_path = trainer.require_c_path(args.config, "config")
    output_dir = trainer.owned_run_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"migration output is not fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_source_sha = trainer.sha256_file(source)
    if actual_source_sha != args.source_sha256.upper():
        raise ValueError("source checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("stage_status") != "SAFETY_STOP":
        raise ValueError("gate migration requires the preserved SAFETY_STOP checkpoint")
    if not payload.get("optimizer_state_included") or "optimizer_state_dict" not in payload:
        raise ValueError("gate migration requires optimizer state")
    if not bool(payload.get("train_config", {}).get("overfit_triggered")):
        raise ValueError("source does not record the invalidated overfit decision")

    config, train_cfg, _, _ = trainer.model_config_and_settings(config_path, output_dir)
    if payload.get("model_config") != trainer.asdict(config):
        raise ValueError("model architecture changed during overfit-gate migration")
    gate_version = str(train_cfg.get("overfit_gate_version", ""))
    if gate_version != "fixed_validation_v2":
        raise ValueError("migration requires fixed_validation_v2")

    settings = dict(payload["train_config"])
    invalidated = {
        "reason": "moving validation cursor compared different single rows",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": actual_source_sha,
        "gate_version": settings.get("overfit_gate_version", "moving_cursor_v1"),
        "validation_history": list(settings.get("validation_history", [])),
        "last_validation_loss": settings.get("last_validation_loss"),
        "last_validation_step": settings.get("last_validation_step"),
        "overfit_triggered": bool(settings.get("overfit_triggered")),
    }
    settings.update(train_cfg)
    settings.update(
        {
            "config_path": str(config_path),
            "config_sha256": trainer.sha256_file(config_path),
            "output_dir": str(output_dir),
            "overfit_gate_version": gate_version,
            "validation_history": [],
            "overfit_triggered": False,
            "invalidated_moving_cursor_overfit_gate": invalidated,
            "resume_migration": "moving_cursor_v1_to_fixed_validation_v2",
        }
    )
    payload["train_config"] = settings
    payload["stage_status"] = "PAUSED"

    destination = output_dir / "latest_recovery.pt"
    trainer.atomic_torch_save(destination, payload)
    destination_sha = trainer.sha256_file(destination)
    verified = torch.load(destination, map_location="cpu", weights_only=False)
    verification = {
        "status": "PASS",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": actual_source_sha,
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
        "overfit_gate_version": verified["train_config"]["overfit_gate_version"],
        "old_gate_evidence_preserved": "invalidated_moving_cursor_overfit_gate" in verified["train_config"],
        "temp_residue": [path.name for path in output_dir.glob("*.tmp")],
    }
    trainer.atomic_json(output_dir / "overfit_gate_migration.json", verification)
    trainer.atomic_json(output_dir / "run_manifest.json", settings)
    del verified, payload
    gc.collect()
    print(json.dumps(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
