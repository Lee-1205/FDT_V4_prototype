from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdt_rlm.config import ModelConfig, load_yaml_like  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from train_fdt_v4_curriculum_bridge import (  # noqa: E402
    RowPool,
    atomic_json,
    convert_v20_state_dict,
    load_payload,
    require_c_path,
    sha256_file,
    validation_snapshot,
)


def release(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_controls(
    model: torch.nn.Module,
    pool: RowPool,
    config: ModelConfig,
    device: torch.device,
    controls: list[dict[str, float]],
    batches: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    fixed_state = pool.state()
    for item in controls:
        model.set_transition_alpha(item["alpha"])
        if "legacy_scale" in item:
            model.set_legacy_position_scale(item["legacy_scale"])
        model.set_anchor_recency_bias(item["recency"])
        model.config.routing_logit_quantization = item["quantum"]
        loss = validation_snapshot(
            model,
            pool,
            config,
            device,
            2.0,
            batches=batches,
            fixed_state=fixed_state,
        )
        results.append({**item, "validation_loss": loss, "finite": math.isfinite(loss)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--path-probe", action="store_true")
    parser.add_argument(
        "--transition-mode", choices=("lerp", "phase", "output_blend")
    )
    parser.add_argument("--legacy-position-scale", type=float)
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.config is None):
        raise ValueError("pass exactly one of --checkpoint or --config")
    checkpoint = (
        require_c_path(args.checkpoint, "checkpoint")
        if args.checkpoint is not None
        else None
    )
    config_path = (
        require_c_path(args.config, "config") if args.config is not None else None
    )
    parent = require_c_path(args.parent, "parent")
    validation_dir = require_c_path(args.validation_dir, "validation directory")
    output = require_c_path(args.output, "output")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(f"diagnostic output is not fresh: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    payload = load_payload(checkpoint) if checkpoint is not None else None
    config = (
        ModelConfig(**payload["model_config"])
        if payload is not None
        else ModelConfig(**load_yaml_like(config_path)["model"])
    )
    if config.model_type != "fdt_v4":
        raise ValueError("checkpoint is not FDT v4")
    if args.transition_mode is not None:
        config.rope_transition_mode = args.transition_mode
    if args.legacy_position_scale is not None:
        config.legacy_position_scale = float(args.legacy_position_scale)
    current = {
        "alpha": float(config.rope_transition_alpha),
        "recency": float(config.anchor_recency_bias),
        "quantum": float(config.routing_logit_quantization),
    }
    model = build_model(config)
    if payload is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
    else:
        parent_payload_for_conversion = load_payload(parent)
        convert_v20_state_dict(model, parent_payload_for_conversion)
        del parent_payload_for_conversion
    model = model.to(device=device, dtype=torch.float32).eval()
    pool = RowPool(validation_dir, "validation", 20260829 + 7001)
    controls = [
        {"name": "legacy_controls", "alpha": 0.0, "recency": 4.0, "quantum": 0.0},
        {"name": "alpha_only", "alpha": current["alpha"], "recency": 4.0, "quantum": 0.0},
        {"name": "recency_only", "alpha": 0.0, "recency": current["recency"], "quantum": 0.0},
        {"name": "quantum_only", "alpha": 0.0, "recency": 4.0, "quantum": current["quantum"]},
        {"name": "current_controls", **current},
    ]
    if args.path_probe:
        controls = [
            {"name": "start", "alpha": 0.0, "recency": 4.0, "quantum": 0.0, "legacy_scale": 1.0},
            {"name": "rope_025", "alpha": 0.25, "recency": 4.0, "quantum": 0.0, "legacy_scale": 1.0},
            {"name": "rope_050", "alpha": 0.50, "recency": 4.0, "quantum": 0.0, "legacy_scale": 1.0},
            {"name": "rope_075", "alpha": 0.75, "recency": 4.0, "quantum": 0.0, "legacy_scale": 1.0},
            {"name": "rope_100", "alpha": 1.00, "recency": 4.0, "quantum": 0.0, "legacy_scale": 1.0},
            {"name": "legacy_075", "alpha": 1.00, "recency": 4.0, "quantum": 0.0, "legacy_scale": 0.75},
            {"name": "legacy_050", "alpha": 1.00, "recency": 4.0, "quantum": 0.0, "legacy_scale": 0.50},
            {"name": "legacy_025", "alpha": 1.00, "recency": 4.0, "quantum": 0.0, "legacy_scale": 0.25},
            {"name": "finish", "alpha": 1.00, "recency": 0.12476396288738318, "quantum": 0.0001, "legacy_scale": 0.0},
        ]
    candidate = evaluate_controls(model, pool, config, device, controls, args.batches)
    release(model)
    model = None

    parent_payload = load_payload(parent)
    parent_config = ModelConfig(**parent_payload["model_config"])
    parent_model = build_model(parent_config)
    parent_model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    del parent_payload
    parent_model = parent_model.to(device=device, dtype=torch.float32).eval()
    parent_pool = RowPool(validation_dir, "validation", 20260829 + 7001)
    parent_loss = validation_snapshot(
        parent_model,
        parent_pool,
        parent_config,
        device,
        2.0,
        batches=args.batches,
        fixed_state=parent_pool.state(),
    )
    release(parent_model)

    by_name = {row["name"]: row for row in candidate}
    report = {
        "schema": "fdt_v4_1_transition_control_ablation_v1",
        "status": "PASS" if all(row["finite"] for row in candidate) and math.isfinite(parent_loss) else "FAIL",
        "official_operations": "unquantized_fp32",
        "candidate_source": (
            {"kind": "checkpoint", "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            if checkpoint is not None
            else {"kind": "parent_conversion", "config": str(config_path), "config_sha256": sha256_file(config_path)}
        ),
        "parent": {"path": str(parent), "sha256": sha256_file(parent)},
        "validation": {"directory": str(validation_dir), "batches": args.batches},
        "parent_validation_loss": parent_loss,
        "candidate": candidate,
        "decomposition": ({
            "weight_update_delta_at_legacy_controls": by_name["legacy_controls"]["validation_loss"] - parent_loss,
            "alpha_increment": by_name["alpha_only"]["validation_loss"] - by_name["legacy_controls"]["validation_loss"],
            "recency_increment": by_name["recency_only"]["validation_loss"] - by_name["legacy_controls"]["validation_loss"],
            "quantum_increment": by_name["quantum_only"]["validation_loss"] - by_name["legacy_controls"]["validation_loss"],
            "combined_transition_increment": by_name["current_controls"]["validation_loss"] - by_name["legacy_controls"]["validation_loss"],
        } if not args.path_probe else {}),
    }
    atomic_json(output, report)
    atomic_json(output.with_name(output.name + ".sha256.json"), {"path": str(output), "sha256": sha256_file(output)})
    print(json.dumps(report["decomposition"], sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
