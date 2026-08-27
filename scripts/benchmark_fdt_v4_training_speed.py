from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEED_PATH = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_speed_benchmark_trainer", SPEED_PATH)
speed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(speed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def tensor_digest_update(digest, payload: dict[str, torch.Tensor]) -> None:
    batch_size = int(next(iter(payload.values())).size(0))
    for row_index in range(batch_size):
        for key in sorted(payload):
            tensor = payload[key][row_index].detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(tensor.numpy().tobytes())


def sampled_parameters(model) -> dict[str, list[float]]:
    sampled: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().float().reshape(-1)
        count = min(64, flat.numel())
        indices = torch.linspace(
            0, flat.numel() - 1, count, device=flat.device, dtype=torch.float64
        ).long()
        sampled[name] = flat.index_select(0, indices).cpu().tolist()
    return sampled


def compare_samples(reference, candidate) -> dict[str, float]:
    maximum = 0.0
    squared = 0.0
    reference_squared = 0.0
    count = 0
    for name in reference:
        left = torch.tensor(reference[name], dtype=torch.float64)
        right = torch.tensor(candidate[name], dtype=torch.float64)
        delta = right - left
        maximum = max(maximum, float(delta.abs().max()))
        squared += float(delta.square().sum())
        reference_squared += float(left.square().sum())
        count += delta.numel()
    return {
        "sample_count": count,
        "max_abs": maximum,
        "rms": math.sqrt(squared / max(count, 1)),
        "relative_l2": math.sqrt(squared) / max(math.sqrt(reference_squared), 1e-30),
    }


def build_state(config_path: Path, checkpoint_path: Path, device: torch.device):
    config, train_cfg, data_cfg, _ = speed.model_config_and_settings(
        config_path, ROOT / "runs" / "fdt_v4_speed_benchmark_config_view"
    )
    # Reproduce the trainer's backend contract before model construction.
    speed.seed_everything(int(train_cfg["seed"]))
    payload = speed.load_payload(checkpoint_path)
    model = speed.build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if bool(payload.get("train_config", {}).get("warm_started_from_v20", False)):
        for name, parameter in model.named_parameters():
            if ".anchor." in name or ".anchor_norm." in name:
                parameter.requires_grad_(False)
    optimizer_groups, base_trainable, exact_trainable = speed.optimizer_parameter_groups(
        model, train_cfg
    )
    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    dataset_dirs = {
        "natural": ROOT / data_cfg["natural_lm_dir"],
        "factual": ROOT / data_cfg["factual_dir"],
        "exact_copy": ROOT / data_cfg["exact_copy_dir"],
        "long_context": ROOT / data_cfg["long_context_dir"],
        "generated_prefix": ROOT / data_cfg["generated_prefix_dir"],
    }
    seed = int(train_cfg["seed"])
    pools = {
        name: speed.RowPool(
            path,
            data_cfg.get(f"{name}_split", data_cfg.get("split", "train")),
            seed + index * 1009,
            exact=name == "exact_copy",
            generated_prefix=name == "generated_prefix",
        )
        for index, (name, path) in enumerate(dataset_dirs.items())
    }
    for name, state in payload["source_states"].items():
        if name in pools:
            pools[name].restore(state)
    torch.set_rng_state(payload["torch_rng_state"])
    if payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    step = int(payload["optimizer_step"])
    del payload
    gc.collect()
    return config, train_cfg, model, optimizer, base_trainable, exact_trainable, pools, step


def run_variant(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    short_batch_size: int,
    checkpoint_minimum: int,
    measured_steps: int,
    long_smoke: bool,
    deterministic_algorithms: bool = True,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    (
        config,
        train_cfg,
        model,
        optimizer,
        base_trainable,
        exact_trainable,
        pools,
        step,
    ) = build_state(config_path, checkpoint_path, device)
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms), warn_only=bool(deterministic_algorithms)
    )
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    train_cfg = dict(train_cfg)
    train_cfg.update(
        {
            "short_sequence_batch_size": int(short_batch_size),
            "activation_checkpointing_min_sequence_length": int(checkpoint_minimum),
            "lm_loss_checkpointing_min_sequence_length": int(checkpoint_minimum),
        }
    )
    source_cycle = speed.deterministic_source_cycle(
        train_cfg["source_batch_fractions"], slots=100
    )
    effective_samples = int(train_cfg["batch_size"]) * int(train_cfg["grad_accum_steps"])
    model.train()
    step_rows = []
    sample_digest = hashlib.sha256()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    routing_window_masks = None
    routing_window_steps = 0

    for _ in range(int(measured_steps)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        groups = speed.planned_base_forward_groups(
            pools,
            source_cycle,
            step,
            effective_samples,
            short_batch_size,
            int(train_cfg.get("short_sequence_max_length", 512)),
        )
        base_loss_value = 0.0
        batch_tokens = 0
        optimizer_step_masks = None
        started = time.perf_counter()
        for host_batch in groups:
            tensor_digest_update(sample_digest, host_batch)
            batch = speed.move_batch(host_batch, device)
            sequence_length = int(batch["input_ids"].size(1))
            speed.set_sequence_gradient_checkpointing(model, sequence_length, train_cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_logits=False,
                )
                loss = speed.chunked_weighted_lm_loss(
                    model,
                    output["hidden"],
                    batch["labels"],
                    batch["attention_mask"],
                    config.pad_token_id,
                    config.eos_token_id,
                    float(train_cfg["eos_loss_weight"]),
                    int(train_cfg["lm_loss_sequence_chunk_size"]),
                    speed.lm_loss_checkpointing_enabled(sequence_length, train_cfg),
                ) / effective_samples
                route_metrics = speed.routing_diagnostics(output, device_resident=True)
                optimizer_step_masks = speed.merge_active_anchor_masks(
                    optimizer_step_masks, route_metrics["active_anchor_masks"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} base loss is nonfinite")
            loss.backward()
            base_loss_value += float(loss.detach().cpu())
            batch_tokens += int(batch["attention_mask"].sum())
            del output, loss, batch

        routing_window_masks = speed.merge_active_anchor_masks(
            routing_window_masks, optimizer_step_masks or []
        )
        routing_window_steps += 1
        route_dead_tensor = speed.dead_anchor_fraction_from_masks(
            routing_window_masks, device_resident=True
        )
        if route_dead_tensor is None:
            raise RuntimeError(f"{name} route diagnostics are missing")
        route_dead_value = None
        if (step + 1) % int(train_cfg.get("log_every", 10)) == 0:
            route_dead_value = float(route_dead_tensor.cpu())
        if routing_window_steps >= int(train_cfg.get("routing_safety_window_steps", 25)):
            route_dead_value = float(route_dead_tensor.cpu())
            routing_window_masks = None
            routing_window_steps = 0

        exact_host = pools["exact_copy"].next_batch(int(train_cfg["batch_size"]))
        tensor_digest_update(sample_digest, exact_host)
        exact_batch = speed.move_batch(exact_host, device)
        speed.set_sequence_gradient_checkpointing(
            model, int(exact_batch["input_ids"].size(1)), train_cfg
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            exact_loss, exact_result = speed.exact_copy_objective(
                model,
                exact_batch,
                float(train_cfg["exact_copy_weight"]),
                detach_hidden=bool(train_cfg["detach_exact_hidden"]),
            )
        if not torch.isfinite(exact_loss):
            raise FloatingPointError(f"{name} exact loss is nonfinite")
        exact_loss.backward()
        base_grad_norm = torch.nn.utils.clip_grad_norm_(
            base_trainable, float(train_cfg["grad_clip"])
        )
        exact_grad_norm = torch.nn.utils.clip_grad_norm_(
            exact_trainable, float(train_cfg["exact_pointer_grad_clip"])
        )
        if not torch.isfinite(base_grad_norm) or not torch.isfinite(exact_grad_norm):
            raise FloatingPointError(f"{name} gradient is nonfinite")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        step_rows.append(
            {
                "optimizer_step": step + 1,
                "base_forward_groups": len(groups),
                "effective_samples": effective_samples,
                "base_tokens": batch_tokens,
                "base_loss": base_loss_value,
                "exact_loss": float(exact_loss.detach().cpu()),
                "exact_pointer_accuracy": float(exact_result.pointer_accuracy.cpu()),
                "base_grad_norm": float(base_grad_norm.cpu()),
                "exact_grad_norm": float(exact_grad_norm.cpu()),
                "dead_anchor_fraction_when_observed": route_dead_value,
                "elapsed_seconds": elapsed,
                "tokens_per_second": batch_tokens / elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
        )
        step += 1
        del exact_loss, exact_result, exact_batch, exact_host, groups

    long_rows = []
    if long_smoke:
        long_shards = sorted(
            (ROOT / "prepared_data" / "fdt_v4_curriculum_v4" / "long_context" / "shards" / "train").glob("*.pt")
        )
        for shard in long_shards:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            host_batch = {key: value[:1] for key, value in payload.items()}
            sequence_length = int(host_batch["input_ids"].size(1))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = speed.move_batch(host_batch, device)
            checkpointed = speed.set_sequence_gradient_checkpointing(
                model, sequence_length, train_cfg
            )
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_logits=False,
                )
                loss = speed.chunked_weighted_lm_loss(
                    model,
                    output["hidden"],
                    batch["labels"],
                    batch["attention_mask"],
                    config.pad_token_id,
                    config.eos_token_id,
                    float(train_cfg["eos_loss_weight"]),
                    int(train_cfg["lm_loss_sequence_chunk_size"]),
                    True,
                )
            loss.backward()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            long_rows.append(
                {
                    "sequence_length": sequence_length,
                    "checkpointing_enabled": checkpointed,
                    "finite_loss": bool(torch.isfinite(loss)),
                    "loss": float(loss.detach().cpu()),
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": int(batch["attention_mask"].sum()) / elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
            )
            del payload, host_batch, batch, output, loss

    samples = sampled_parameters(model)
    short_peak_allocated = max(row["peak_allocated_gib"] for row in step_rows)
    short_peak_reserved = max(row["peak_reserved_gib"] for row in step_rows)
    result = {
        "name": name,
        "short_batch_size": short_batch_size,
        "checkpoint_minimum": checkpoint_minimum,
        "deterministic_algorithms": bool(deterministic_algorithms),
        "steps": step_rows,
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in step_rows)
        / len(step_rows),
        "peak_allocated_gib": short_peak_allocated,
        "peak_reserved_gib": short_peak_reserved,
        "sample_order_sha256": sample_digest.hexdigest().upper(),
        "source_states": {name: pool.state() for name, pool in pools.items()},
        "long_context_smoke": long_rows,
    }
    del optimizer, model, pools, base_trainable, exact_trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    for path in (config_path, checkpoint_path, output_path):
        if path.drive.upper() != "C:":
            raise ValueError("benchmark is C-only")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    variants = (
        ("original_batch1_checkpointed", 1, 1, False),
        ("short_batch1_uncheckpointed", 1, 8192, False),
        ("short_batch2_uncheckpointed", 2, 8192, True),
    )
    rows = []
    samples = {}
    for name, batch_size, checkpoint_minimum, long_smoke in variants:
        result, current_samples = run_variant(
            name,
            config_path,
            checkpoint_path,
            device,
            short_batch_size=batch_size,
            checkpoint_minimum=checkpoint_minimum,
            measured_steps=args.steps,
            long_smoke=long_smoke,
        )
        rows.append(result)
        samples[name] = current_samples
    reference = samples[variants[0][0]]
    comparisons = {
        name: compare_samples(reference, samples[name]) for name, *_ in variants[1:]
    }
    reference_steps = rows[0]["steps"]
    finite = all(
        math.isfinite(step["base_loss"] + step["exact_loss"])
        for row in rows
        for step in row["steps"]
    )
    order_equal = len({row["sample_order_sha256"] for row in rows}) == 1
    states_equal = len(
        {json.dumps(row["source_states"], sort_keys=True) for row in rows}
    ) == 1
    loss_deltas = {
        row["name"]: max(
            abs(current["base_loss"] - reference["base_loss"])
            for current, reference in zip(row["steps"], reference_steps)
        )
        for row in rows[1:]
    }
    long_safe = all(
        item["checkpointing_enabled"] and item["finite_loss"]
        for item in rows[-1]["long_context_smoke"]
    ) and {item["sequence_length"] for item in rows[-1]["long_context_smoke"]} == {
        8192,
        16384,
    }
    candidate_equivalence = {
        "short_batch1_uncheckpointed": (
            comparisons["short_batch1_uncheckpointed"]["max_abs"] == 0.0
            and loss_deltas["short_batch1_uncheckpointed"] <= 2e-4
        ),
        "short_batch2_uncheckpointed": (
            comparisons["short_batch2_uncheckpointed"]["relative_l2"] <= 1e-6
            and loss_deltas["short_batch2_uncheckpointed"] <= 2e-4
        ),
    }
    eligible = [
        row
        for row in rows[1:]
        if candidate_equivalence[row["name"]]
        and float(row["mean_tokens_per_second"]) >= 1000.0
    ]
    selected_variant = (
        max(eligible, key=lambda row: float(row["mean_tokens_per_second"]))["name"]
        if eligible
        else None
    )
    equivalence_pass = selected_variant is not None
    output = {
        "status": "PASS" if finite and order_equal and states_equal and long_safe and equivalence_pass else "FAIL",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "trainer": str(SPEED_PATH),
        "trainer_sha256": sha256(SPEED_PATH),
        "measured_steps": args.steps,
        "variants": rows,
        "sampled_update_comparisons_vs_original": comparisons,
        "base_loss_max_abs_deltas_vs_original": loss_deltas,
        "candidate_equivalence": candidate_equivalence,
        "selected_variant": selected_variant,
        "equivalence_tolerances": {
            "short_batch1_sampled_update_max_abs": 0.0,
            "short_batch2_sampled_update_relative_l2": 1e-6,
            "base_loss_max_abs": 2e-4,
        },
        "all_sample_orders_identical": order_equal,
        "all_source_states_identical": states_equal,
        "long_context_checkpoint_safety_pass": long_safe,
        "equivalence_pass": equivalence_pass,
    }
    atomic_json(output_path, output)
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
