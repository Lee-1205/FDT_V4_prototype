from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_bridge.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_bridge_benchmark_trainer", TRAINER)
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
    parser = argparse.ArgumentParser(description="Bounded 2K/4K FDT v4 bridge safety benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    for path in (checkpoint, config_path, output, TRAINER):
        if path.drive.upper() != "C:":
            raise ValueError("bridge safety benchmark is C-only")
    if output.exists():
        raise FileExistsError(f"Refusing to reuse benchmark output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config, train_cfg, data_cfg, _ = trainer.model_config_and_settings(
        config_path, ROOT / "runs" / "fdt_v4_bridge_safety_benchmark"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = trainer.build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.to("cuda").train()
    bridge_root = ROOT / data_cfg["bridge_context_dir"] / "shards" / "train"
    rows: list[dict[str, Any]] = []
    for length in (2048, 4096):
        shard = next(path for path in sorted(bridge_root.glob("*.pt")) if f"_{length // 1024}k" in path.stem)
        host = torch.load(shard, map_location="cpu", weights_only=False)
        batch = {key: value[:1].to("cuda", dtype=torch.long) for key, value in host.items()}
        del host
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        checkpointing = trainer.set_sequence_gradient_checkpointing(model, length, train_cfg)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_logits=False)
            loss = trainer.chunked_weighted_lm_loss(
                model,
                result["hidden"],
                batch["labels"],
                batch["attention_mask"],
                config.pad_token_id,
                config.eos_token_id,
                float(train_cfg.get("eos_loss_weight", 2.0)),
                int(train_cfg.get("lm_loss_sequence_chunk_size", 256)),
                trainer.lm_loss_checkpointing_enabled(length, train_cfg),
            )
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite_gradients = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
        rows.append(
            {
                "sequence_length": length,
                "loss": float(loss.detach().float().cpu()),
                "finite_loss": bool(torch.isfinite(loss.detach())),
                "finite_gradients": finite_gradients,
                "activation_checkpointing": checkpointing,
                "lm_loss_checkpointing": trainer.lm_loss_checkpointing_enabled(length, train_cfg),
                "elapsed_seconds": elapsed,
                "tokens_per_second": length / max(elapsed, 1e-9),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "shard": str(shard),
                "shard_sha256": sha256(shard),
            }
        )
        del batch, result, loss, gradients
    status = "PASS" if all(
        row["finite_loss"]
        and row["finite_gradients"]
        and row["activation_checkpointing"]
        and row["lm_loss_checkpointing"]
        and row["peak_reserved_gib"] < 15.5
        for row in rows
    ) else "FAIL"
    report = {
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256(TRAINER),
        "rows": rows,
        "contract": "checkpointed_2k_4k_finite_backward_v1",
    }
    atomic_json(output, report)
    print(json.dumps(report))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
