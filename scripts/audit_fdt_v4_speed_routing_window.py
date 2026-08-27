from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_speed_route_audit", TRAINER)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def update_rows(digest, payload):
    for row_index in range(int(payload["input_ids"].size(0))):
        for key in sorted(payload):
            tensor = payload[key][row_index].contiguous()
            digest.update(key.encode("ascii"))
            digest.update(tensor.numpy().tobytes())


def pools_from(config_data, train, source_states):
    directories = {
        "natural": ROOT / config_data["natural_lm_dir"],
        "factual": ROOT / config_data["factual_dir"],
        "exact_copy": ROOT / config_data["exact_copy_dir"],
        "long_context": ROOT / config_data["long_context_dir"],
        "generated_prefix": ROOT / config_data["generated_prefix_dir"],
    }
    seed = int(train["seed"])
    pools = {
        name: trainer.RowPool(
            path,
            config_data.get(f"{name}_split", config_data.get("split", "train")),
            seed + index * 1009,
            exact=name == "exact_copy",
            generated_prefix=name == "generated_prefix",
        )
        for index, (name, path) in enumerate(directories.items())
    }
    for name, state in source_states.items():
        if name in pools:
            pools[name].restore(state)
    return pools


def audit_variant(model, config_data, train, source_states, start_step, batch_size, device):
    pools = pools_from(config_data, train, source_states)
    cycle = trainer.deterministic_source_cycle(train["source_batch_fractions"], slots=100)
    effective_samples = int(train["batch_size"]) * int(train["grad_accum_steps"])
    window_masks = None
    digest = hashlib.sha256()
    per_step_dead = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for step in range(int(start_step), int(start_step) + 25):
            groups = trainer.planned_base_forward_groups(
                pools, cycle, step, effective_samples, batch_size, 512
            )
            step_masks = None
            for host_batch in groups:
                update_rows(digest, host_batch)
                batch = trainer.move_batch(host_batch, device)
                trainer.set_sequence_gradient_checkpointing(
                    model, int(batch["input_ids"].size(1)), train
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_logits=False,
                    )
                metrics = trainer.routing_diagnostics(output)
                if metrics["dead_anchor_fraction"] is None or not metrics["active_anchor_masks"]:
                    raise RuntimeError("route diagnostics missing during 25-step audit")
                step_masks = trainer.merge_active_anchor_masks(
                    step_masks, metrics["active_anchor_masks"]
                )
                del output, batch
            window_masks = trainer.merge_active_anchor_masks(window_masks, step_masks or [])
            per_step_dead.append(trainer.dead_anchor_fraction_from_masks(step_masks))
    return {
        "short_batch_size": batch_size,
        "sample_order_sha256": digest.hexdigest().upper(),
        "source_states": {name: pool.state() for name, pool in pools.items()},
        "window_dead_anchor_fraction": trainer.dead_anchor_fraction_from_masks(window_masks),
        "maximum_single_step_dead_anchor_fraction": max(per_step_dead),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if output_path.exists() or any(
        path.drive.upper() != "C:" for path in (config_path, checkpoint_path, output_path)
    ):
        raise ValueError("route audit requires a fresh C-only output")
    config, train, data, _ = trainer.model_config_and_settings(
        config_path, ROOT / "runs" / "fdt_v4_speed_route_audit_view"
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = trainer.build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    source_states = payload["source_states"]
    start_step = int(payload["optimizer_step"])
    model.to("cuda").eval()
    del payload
    variants = [
        audit_variant(model, data, train, source_states, start_step, batch_size, torch.device("cuda"))
        for batch_size in (1, 2)
    ]
    order_equal = variants[0]["sample_order_sha256"] == variants[1]["sample_order_sha256"]
    states_equal = variants[0]["source_states"] == variants[1]["source_states"]
    safety_limit = float(train["dead_anchor_fraction_limit"])
    passed = order_equal and states_equal and all(
        float(row["window_dead_anchor_fraction"]) <= safety_limit for row in variants
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "trainer_sha256": sha256(TRAINER),
        "start_step": start_step,
        "window_steps": 25,
        "dead_anchor_fraction_limit": safety_limit,
        "all_sample_orders_identical": order_equal,
        "all_source_states_identical": states_equal,
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(report))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
