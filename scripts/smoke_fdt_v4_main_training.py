from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from train_fdt_v4_curriculum import (  # noqa: E402
    chunked_weighted_lm_loss,
    optimizer_parameter_groups,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded 424M FDT v4 CUDA train smoke")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.88)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if checkpoint.drive.upper() != "C:" or output_path.drive.upper() != "C:":
        raise ValueError("FDT v4 smoke is C:-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), device=0)
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)

    raw = json.loads(json.dumps(__import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))))
    train_config = raw["train"]
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v4" or config.max_seq_len != 16384:
        raise ValueError("Checkpoint is not the main 16K FDT v4 architecture")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        if ".anchor." in name or ".anchor_norm." in name:
            parameter.requires_grad_(False)
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    groups, base_trainable, exact_trainable = optimizer_parameter_groups(
        model, train_config
    )
    model.to(device="cuda")
    model.train()
    optimizer = torch.optim.AdamW(groups)

    context = int(args.context)
    if not 2 <= context <= config.max_seq_len:
        raise ValueError("Context is outside the FDT v4 contract")
    ids = torch.randint(3, config.vocab_size, (1, context), device="cuda")
    mask = torch.ones_like(ids)
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(ids, attention_mask=mask, return_logits=False)
        loss = chunked_weighted_lm_loss(
            model,
            result["hidden"],
            labels,
            mask,
            config.pad_token_id,
            config.eos_token_id,
            float(train_config.get("eos_loss_weight", 2.0)),
            int(train_config.get("lm_loss_sequence_chunk_size", 256)),
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Main FDT v4 smoke loss is non-finite")
    loss.backward()
    base_norm = torch.nn.utils.clip_grad_norm_(
        base_trainable, float(train_config.get("grad_clip", 0.7))
    )
    exact_norm = torch.nn.utils.clip_grad_norm_(
        exact_trainable, float(train_config.get("exact_pointer_grad_clip", 1.0))
    )
    if not torch.isfinite(base_norm) or not torch.isfinite(exact_norm):
        raise FloatingPointError("Main FDT v4 smoke gradient norm is non-finite")
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite_parameters = all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    report = {
        "status": "PASS" if finite_parameters else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_stage_status": payload.get("stage_status"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "context": context,
        "precision": "bf16_autocast_fp32_master",
        "quantization": "none",
        "loss": float(loss.detach().float().cpu()),
        "base_grad_norm": float(base_norm.detach().float().cpu()),
        "exact_grad_norm": float(exact_norm.detach().float().cpu()),
        "finite_parameters_after_step": finite_parameters,
        "elapsed_seconds": elapsed,
        "tokens_per_second": context / max(elapsed, 1e-9),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "memory_fraction_cap": float(args.memory_fraction),
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(output_path, report)
    print(json.dumps(report))
    if report["status"] != "PASS" or not math.isfinite(report["loss"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
