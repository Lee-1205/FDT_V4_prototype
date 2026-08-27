from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_fdt_v4_training_speed.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_runtime_benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-smoke", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in (config, checkpoint, output, BENCHMARK_PATH):
        if path.drive.upper() != "C:":
            raise ValueError("determinism diagnosis is C-only")
    if output.exists():
        raise FileExistsError(output)

    device = benchmark.torch.device("cuda")
    rows = []
    samples = {}
    for name, deterministic in (
        ("live_deterministic", True),
        ("candidate_fast_backend", False),
    ):
        result, current = benchmark.run_variant(
            name,
            config,
            checkpoint,
            device,
            short_batch_size=1,
            checkpoint_minimum=8192,
            measured_steps=1,
            long_smoke=bool(args.long_smoke),
            deterministic_algorithms=deterministic,
        )
        rows.append(result)
        samples[name] = current

    comparison = benchmark.compare_samples(
        samples["live_deterministic"], samples["candidate_fast_backend"]
    )
    left = rows[0]["steps"][0]
    right = rows[1]["steps"][0]
    loss_delta = abs(float(left["base_loss"]) - float(right["base_loss"]))
    exact_delta = abs(float(left["exact_loss"]) - float(right["exact_loss"]))
    sample_order_equal = rows[0]["sample_order_sha256"] == rows[1]["sample_order_sha256"]
    source_states_equal = rows[0]["source_states"] == rows[1]["source_states"]
    long_loss_deltas = []
    long_contract_pass = True
    if args.long_smoke:
        left_long = rows[0]["long_context_smoke"]
        right_long = rows[1]["long_context_smoke"]
        long_contract_pass = (
            [item["sequence_length"] for item in left_long]
            == [item["sequence_length"] for item in right_long]
            and all(item["checkpointing_enabled"] and item["finite_loss"] for item in left_long + right_long)
        )
        long_loss_deltas = [
            abs(float(left_item["loss"]) - float(right_item["loss"]))
            for left_item, right_item in zip(left_long, right_long)
        ]
        long_contract_pass = long_contract_pass and all(
            delta <= 2e-4 for delta in long_loss_deltas
        )
    equivalent = (
        comparison["max_abs"] == 0.0
        and loss_delta <= 2e-4
        and exact_delta <= 2e-4
        and sample_order_equal
        and source_states_equal
        and long_contract_pass
    )
    report = {
        "status": "PASS" if equivalent else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": benchmark.sha256(checkpoint),
        "config": str(config),
        "config_sha256": benchmark.sha256(config),
        "benchmark_sha256": benchmark.sha256(BENCHMARK_PATH),
        "trainer_sha256": benchmark.sha256(benchmark.SPEED_PATH),
        "variants": rows,
        "sampled_update_comparison": comparison,
        "base_loss_abs_delta": loss_delta,
        "exact_loss_abs_delta": exact_delta,
        "sample_order_equal": sample_order_equal,
        "source_states_equal": source_states_equal,
        "long_context_loss_abs_deltas": long_loss_deltas,
        "long_context_contract_pass": long_contract_pass,
        "equivalence_pass": equivalent,
    }
    benchmark.atomic_json(output, report)
    print(json.dumps(report))
    return 0 if equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
