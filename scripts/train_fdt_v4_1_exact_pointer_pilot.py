from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_fdt_v4_exact_copy import (  # noqa: E402
    DISTRACTORS,
    LENGTHS,
    POSITIONS,
    STRING_KINDS,
    MatrixSpec,
    build_case,
)
from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def exact_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if getattr(model, "exact_pointer", None) is None:
        raise RuntimeError("checkpoint does not contain an Exact Memory pointer")
    parameters = list(model.exact_pointer.parameters())
    if not parameters:
        raise RuntimeError("Exact Memory pointer has no parameters")
    return parameters


def build_specs(count: int, seed: int) -> list[MatrixSpec]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        rows.append(
            MatrixSpec(
                cell_id=index,
                length=LENGTHS[index % len(LENGTHS)],
                position=POSITIONS[(index // len(LENGTHS)) % len(POSITIONS)],
                distractors=DISTRACTORS[
                    (index // (len(LENGTHS) * len(POSITIONS))) % len(DISTRACTORS)
                ],
                string_kind=STRING_KINDS[
                    (index // (len(LENGTHS) * len(POSITIONS) * len(DISTRACTORS)))
                    % len(STRING_KINDS)
                ],
                seed=rng.randrange(1, 2**31 - 1),
            )
        )
    return rows


def encoded_row(tokenizer: Any, spec: MatrixSpec, max_seq_len: int) -> dict[str, Any]:
    case = build_case(spec)
    prompt_ids = list(tokenizer.encode(case["prompt"], add_special_tokens=False))
    target_ids = list(tokenizer.encode(case["target"], add_special_tokens=False))
    source_prefix = case["prompt"][: case["target_char_start"]]
    source_start = len(tokenizer.encode(source_prefix, add_special_tokens=False))
    observed = prompt_ids[source_start : source_start + len(target_ids)]
    if not target_ids or observed != target_ids:
        raise ValueError("prompt target span does not match independent target tokenization")
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"exact pilot row needs {len(input_ids)} tokens but max_seq_len is {max_seq_len}"
        )
    boundary = len(prompt_ids)
    labels = [-100] * boundary + target_ids
    source_positions = [-1] * len(input_ids)
    target_mask = [False] * len(input_ids)
    for offset in range(len(target_ids)):
        target_position = boundary + offset
        source_positions[target_position] = source_start + offset
        target_mask[target_position] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "copy_source_positions": source_positions,
        "copy_target_mask": target_mask,
        "source_boundary": boundary,
        "length_chars": spec.length,
        "target_tokens": len(target_ids),
    }


def build_encoded_rows(
    tokenizer: Any,
    count: int,
    seed: int,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    rejected = 0
    candidates = build_specs(max(count * 12, count + 256), seed)
    for spec in candidates:
        try:
            row = encoded_row(tokenizer, spec, max_seq_len)
        except ValueError:
            rejected += 1
            continue
        rows.append(row)
        if len(rows) == count:
            return rows, rejected
    raise RuntimeError(
        f"only {len(rows)} of {count} aligned exact-copy rows could be built"
    )


def collate(rows: list[dict[str, Any]], pad_token_id: int, device: torch.device):
    length = max(len(row["input_ids"]) for row in rows)
    batch = len(rows)
    input_ids = torch.full(
        (batch, length), pad_token_id, dtype=torch.long, device=device
    )
    labels = torch.full((batch, length), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, length), dtype=torch.long, device=device)
    source_positions = torch.full(
        (batch, length), -1, dtype=torch.long, device=device
    )
    target_mask = torch.zeros((batch, length), dtype=torch.bool, device=device)
    boundaries = torch.zeros(batch, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        count = len(row["input_ids"])
        input_ids[index, :count] = torch.tensor(row["input_ids"], device=device)
        labels[index, :count] = torch.tensor(row["labels"], device=device)
        attention_mask[index, :count] = 1
        source_positions[index, :count] = torch.tensor(
            row["copy_source_positions"], device=device
        )
        target_mask[index, :count] = torch.tensor(
            row["copy_target_mask"], device=device
        )
        boundaries[index] = int(row["source_boundary"])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "copy_source_positions": source_positions,
        "copy_target_mask": target_mask,
        "source_boundary": boundaries,
    }


def evaluate_pointer(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {
        "loss": 0.0,
        "pointer_loss": 0.0,
        "gate_loss": 0.0,
        "commit_loss": 0.0,
        "pointer_accuracy": 0.0,
        "proposal_recall": 0.0,
    }
    batches = 0
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start : start + batch_size], pad_token_id, device)
        with torch.inference_mode():
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_logits=False,
            )
            result = model.exact_memory_loss(
                output["hidden"],
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                copy_source_positions=batch["copy_source_positions"],
                copy_target_mask=batch["copy_target_mask"],
                source_boundary=batch["source_boundary"],
                measure_proposal_recall=True,
            )
        for name in totals:
            value = result.loss if name == "loss" else getattr(result, name)
            totals[name] += float(value.detach().cpu())
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def checkpoint_payload(
    model: torch.nn.Module,
    parent_payload: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    step: int,
    cursor: int,
    training: dict[str, Any],
    stage_status: str,
) -> dict[str, Any]:
    payload = {
        "model_type": "fdt_v4",
        "model_config": dict(vars(model.config)),
        "model_state_dict": model.state_dict(),
        "stage_status": stage_status,
        "optimizer_step": step,
        "sample_cursor": cursor,
        "tokens_seen": int(parent_payload.get("tokens_seen", 0)),
        "optimizer_state_included": optimizer is not None,
        "exact_memory_training": training,
        "parent_checkpoint_sha256": training["parent_sha256"],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["python_rng_state"] = random.getstate()
        payload["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-base FDT v4.1 Exact Memory pointer pilot"
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--validation-rows", type=int, default=240)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--validation-seed", type=int, default=20260901)
    parser.add_argument("--recovery-interval", type=int, default=250)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("exact pilot output directory must be fresh")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if args.seed == args.validation_seed or 20260823 in {
        args.seed,
        args.validation_seed,
    }:
        raise ValueError("train, validation, and sealed audit seeds must be distinct")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    parent = args.parent.resolve()
    parent_sha = sha256_file(parent)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    parent_payload = torch.load(
        parent, map_location="cpu", mmap=True, weights_only=True
    )
    config = ModelConfig(**parent_payload["model_config"])
    config.exact_memory_enabled = True
    config.exact_memory_mode = "copy"
    config.exact_memory_hard_copy = True
    config.exact_memory_hard_copy_gate_threshold = 0.90
    config.exact_memory_hard_copy_pointer_threshold = 0.90
    config.exact_memory_hard_copy_margin_threshold = 1.0
    config.exact_memory_commit_threshold = 0.50
    config.inference_prefix_stable_group_size = 64
    model = build_model(config)
    model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = exact_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), betas=(0.9, 0.99), weight_decay=0.0
    )
    tokenizer = load_tokenizer(args.tokenizer.resolve())
    train_rows, rejected_train_rows = build_encoded_rows(
        tokenizer, args.train_rows, args.seed, config.max_seq_len
    )
    validation_rows, rejected_validation_rows = build_encoded_rows(
        tokenizer, args.validation_rows, args.validation_seed, config.max_seq_len
    )
    shuffle_seed = args.seed ^ 0x5F3759DF
    random.Random(shuffle_seed).shuffle(train_rows)
    training = {
        "schema": "fdt_v4_1_exact_pointer_pilot_v1",
        "parent": str(parent),
        "parent_sha256": parent_sha,
        "base_frozen": True,
        "trained_parameter_names": sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ),
        "trained_parameter_count": sum(parameter.numel() for parameter in parameters),
        "train_seed": args.seed,
        "validation_seed": args.validation_seed,
        "train_shuffle_seed": shuffle_seed,
        "sealed_audit_seed": 20260823,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "rejected_unaligned_train_candidates": rejected_train_rows,
        "rejected_unaligned_validation_candidates": rejected_validation_rows,
        "lengths": list(LENGTHS),
        "positions": list(POSITIONS),
        "distractors": list(DISTRACTORS),
        "string_kinds": list(STRING_KINDS),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hard_copy_thresholds": {
            "gate": config.exact_memory_hard_copy_gate_threshold,
            "pointer": config.exact_memory_hard_copy_pointer_threshold,
            "margin": config.exact_memory_hard_copy_margin_threshold,
            "commit": config.exact_memory_commit_threshold,
        },
    }
    atomic_json(output_dir / "pilot_manifest.json", training)

    metrics_path = output_dir / "training_log.jsonl"
    start_time = time.perf_counter()
    cursor = 0
    latest: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        selected = [
            train_rows[(cursor + offset) % len(train_rows)]
            for offset in range(args.batch_size)
        ]
        cursor = (cursor + args.batch_size) % len(train_rows)
        batch = collate(selected, config.pad_token_id, device)
        model.eval()
        model.exact_pointer.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_logits=False,
            )
        result = model.exact_memory_loss(
            output["hidden"],
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
            copy_source_positions=batch["copy_source_positions"],
            copy_target_mask=batch["copy_target_mask"],
            source_boundary=batch["source_boundary"],
            measure_proposal_recall=True,
        )
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"nonfinite exact loss at step {step}")
        result.loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite exact gradient at step {step}")
        optimizer.step()
        latest = {
            "step": step,
            "loss": float(result.loss.detach().cpu()),
            "pointer_loss": float(result.pointer_loss.cpu()),
            "gate_loss": float(result.gate_loss.cpu()),
            "commit_loss": float(result.commit_loss.cpu()),
            "pointer_accuracy": float(result.pointer_accuracy.cpu()),
            "proposal_recall": float(result.proposal_recall.cpu()),
            "gradient_norm": gradient_norm,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if step == 1 or step % 25 == 0 or step == args.steps:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest, separators=(",", ":")) + "\n")
            print(json.dumps(latest), flush=True)
        if step % args.recovery_interval == 0 and step != args.steps:
            atomic_torch_save(
                output_dir / "latest_recovery.pt",
                checkpoint_payload(
                    model,
                    parent_payload,
                    optimizer,
                    step,
                    cursor,
                    training,
                    "RECOVERY",
                ),
            )

    validation = evaluate_pointer(
        model,
        validation_rows,
        args.batch_size,
        config.pad_token_id,
        device,
    )
    model_payload = checkpoint_payload(
        model,
        parent_payload,
        None,
        args.steps,
        cursor,
        training,
        "COMPLETE",
    )
    atomic_torch_save(output_dir / "latest.pt", model_payload)
    atomic_torch_save(
        output_dir / "latest_recovery.pt",
        checkpoint_payload(
            model,
            parent_payload,
            optimizer,
            args.steps,
            cursor,
            training,
            "COMPLETE_RECOVERY",
        ),
    )
    report = {
        "status": "COMPLETE",
        "latest_training": latest,
        "validation": validation,
        "checkpoint": {
            "path": str(output_dir / "latest.pt"),
            "sha256": sha256_file(output_dir / "latest.pt"),
        },
        "recovery": {
            "path": str(output_dir / "latest_recovery.pt"),
            "sha256": sha256_file(output_dir / "latest_recovery.pt"),
        },
        "temporary_residue": sorted(
            str(path) for path in output_dir.glob("*.tmp")
        ),
    }
    atomic_json(output_dir / "result.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
