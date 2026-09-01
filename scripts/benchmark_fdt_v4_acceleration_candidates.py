from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py"
CANDIDATE_PATH = ROOT / "scripts" / "train_fdt_v4_curriculum_speed_observable.py"
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_fdt_v4_training_speed.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


original = load_module("fdt_v4_accel_original", ORIGINAL_PATH)
candidate = load_module("fdt_v4_accel_candidate", CANDIDATE_PATH)
benchmark = load_module("fdt_v4_accel_benchmark", BENCHMARK_PATH)


class TrainerView:
    def __init__(
        self,
        module,
        lm_chunk_size: int,
        max_merged_short_groups: int | None = None,
    ):
        self.module = module
        self.lm_chunk_size = int(lm_chunk_size)
        self.max_merged_short_groups = max_merged_short_groups

    def __getattr__(self, name: str):
        return getattr(self.module, name)

    def model_config_and_settings(self, config_path: Path, output_dir: Path):
        config, train_cfg, data_cfg, raw = self.module.model_config_and_settings(
            config_path, output_dir
        )
        train_cfg = dict(train_cfg)
        train_cfg["lm_loss_sequence_chunk_size"] = self.lm_chunk_size
        return config, train_cfg, data_cfg, raw

    def planned_base_forward_groups(
        self,
        pools,
        source_cycle,
        optimizer_step,
        effective_samples,
        short_batch_size,
        short_sequence_max_length,
    ):
        groups = self.module.planned_base_forward_groups(
            pools,
            source_cycle,
            optimizer_step,
            effective_samples,
            short_batch_size,
            short_sequence_max_length,
        )
        if self.max_merged_short_groups is None:
            return groups
        result = []
        retained = 0
        for group in groups:
            batch_size = int(group["input_ids"].size(0))
            sequence_length = int(group["input_ids"].size(1))
            is_merged_short = batch_size > 1 and sequence_length <= int(
                short_sequence_max_length
            )
            if is_merged_short and retained >= int(self.max_merged_short_groups):
                result.extend(
                    [
                        {key: value[index : index + 1] for key, value in group.items()}
                        for index in range(batch_size)
                    ]
                )
            else:
                result.append(group)
                if is_merged_short:
                    retained += 1
        return result


def max_step_delta(rows: list[dict[str, Any]], reference: list[dict[str, Any]], key: str) -> float:
    return max(abs(float(left[key]) - float(right[key])) for left, right in zip(rows, reference))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    for path in (config_path, checkpoint_path, output_path):
        if path.drive.upper() != "C:":
            raise ValueError("acceleration benchmark is C-only")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    variants = (
        ("live_batch1_chunk256", TrainerView(original, 256), 1, False),
        ("no_grad_batch1_chunk256", TrainerView(candidate, 256), 1, False),
        ("no_grad_batch2_one_pair_chunk256", TrainerView(candidate, 256, 1), 2, True),
        ("no_grad_batch2_two_pairs_chunk256", TrainerView(candidate, 256, 2), 2, True),
        ("no_grad_batch2_three_pairs_chunk256", TrainerView(candidate, 256, 3), 2, True),
        ("no_grad_batch2_chunk256", TrainerView(candidate, 256), 2, True),
        ("no_grad_batch2_chunk512", TrainerView(candidate, 512), 2, True),
        ("no_grad_batch3_chunk256", TrainerView(candidate, 256), 3, True),
        ("no_grad_batch4_one_group_chunk256", TrainerView(candidate, 256, 1), 4, True),
        ("no_grad_batch4_chunk256", TrainerView(candidate, 256), 4, True),
    )
    rows: list[dict[str, Any]] = []
    samples: dict[str, dict[str, list[float]]] = {}
    for name, trainer, short_batch_size, long_smoke in variants:
        benchmark.speed = trainer
        result, parameter_samples = benchmark.run_variant(
            name,
            config_path,
            checkpoint_path,
            torch.device("cuda"),
            short_batch_size=short_batch_size,
            checkpoint_minimum=8192,
            measured_steps=args.steps,
            long_smoke=long_smoke,
            deterministic_algorithms=False,
        )
        result["lm_loss_sequence_chunk_size"] = trainer.lm_chunk_size
        rows.append(result)
        samples[name] = parameter_samples

    reference_name = variants[0][0]
    reference_samples = samples[reference_name]
    reference_steps = rows[0]["steps"]
    comparisons = {
        name: benchmark.compare_samples(reference_samples, samples[name])
        for name, *_ in variants[1:]
    }
    base_loss_deltas = {
        row["name"]: max_step_delta(row["steps"], reference_steps, "base_loss")
        for row in rows[1:]
    }
    exact_loss_deltas = {
        row["name"]: max_step_delta(row["steps"], reference_steps, "exact_loss")
        for row in rows[1:]
    }
    pointer_accuracy_deltas = {
        row["name"]: max_step_delta(
            row["steps"], reference_steps, "exact_pointer_accuracy"
        )
        for row in rows[1:]
    }
    order_equal = len({row["sample_order_sha256"] for row in rows}) == 1
    states_equal = len(
        {json.dumps(row["source_states"], sort_keys=True) for row in rows}
    ) == 1
    finite = all(
        math.isfinite(float(step["base_loss"]) + float(step["exact_loss"]))
        for row in rows
        for step in row["steps"]
    )
    long_safe = {
        row["name"]: (
            {item["sequence_length"] for item in row["long_context_smoke"]}
            == {8192, 16384}
            and all(
                item["checkpointing_enabled"] and item["finite_loss"]
                for item in row["long_context_smoke"]
            )
        )
        for row in rows
        if row["long_context_smoke"]
    }

    equivalence = {
        "no_grad_batch1_chunk256": (
            comparisons["no_grad_batch1_chunk256"]["max_abs"] == 0.0
            and base_loss_deltas["no_grad_batch1_chunk256"] == 0.0
            and exact_loss_deltas["no_grad_batch1_chunk256"] == 0.0
            and pointer_accuracy_deltas["no_grad_batch1_chunk256"] == 0.0
        ),
    }
    for name in comparisons:
        if name == "no_grad_batch1_chunk256":
            continue
        equivalence[name] = (
            comparisons[name]["relative_l2"] <= 1e-6
            and comparisons[name]["max_abs"] <= 1e-5
            and base_loss_deltas[name] <= 1e-3
            and exact_loss_deltas[name] <= 1e-3
            and pointer_accuracy_deltas[name] <= 0.25
            and long_safe.get(name, False)
        )
    eligible = [
        row
        for row in rows[1:]
        if equivalence[row["name"]]
        and float(row["mean_tokens_per_second"])
        > float(rows[0]["mean_tokens_per_second"]) * 1.05
        and float(row["peak_reserved_gib"]) < 15.5
    ]
    selected = (
        max(eligible, key=lambda row: float(row["mean_tokens_per_second"]))["name"]
        if eligible
        else None
    )
    output = {
        "status": "PASS" if finite and order_equal and states_equal and selected else "NO_PROMOTION",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": benchmark.sha256(checkpoint_path),
        "config": str(config_path),
        "config_sha256": benchmark.sha256(config_path),
        "original_trainer_sha256": benchmark.sha256(ORIGINAL_PATH),
        "candidate_trainer_sha256": benchmark.sha256(CANDIDATE_PATH),
        "measured_steps": int(args.steps),
        "variants": rows,
        "sampled_update_comparisons_vs_live": comparisons,
        "base_loss_max_abs_deltas_vs_live": base_loss_deltas,
        "exact_loss_max_abs_deltas_vs_live": exact_loss_deltas,
        "pointer_accuracy_max_abs_deltas_vs_live": pointer_accuracy_deltas,
        "all_sample_orders_identical": order_equal,
        "all_source_states_identical": states_equal,
        "all_finite": finite,
        "long_context_safety": long_safe,
        "equivalence": equivalence,
        "selected_variant": selected,
        "selection_contract": {
            "minimum_speedup": 1.05,
            "sampled_update_relative_l2_max": 1e-6,
            "sampled_update_max_abs": 1e-5,
            "base_loss_max_abs": 1e-3,
            "exact_loss_max_abs": 1e-3,
            "pointer_accuracy_max_abs": 0.25,
            "peak_reserved_gib_max": 15.5,
        },
    }
    benchmark.atomic_json(output_path, output)
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
