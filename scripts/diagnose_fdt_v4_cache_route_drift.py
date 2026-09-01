from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def snapshot_route(x: torch.Tensor, result: tuple, top_k: int) -> dict[str, torch.Tensor | float]:
    full_distance, indices, membership = result[2], result[3], result[4]
    ordered = full_distance[:, -1].float().sort(dim=-1).values
    boundary_margin = ordered[:, top_k] - ordered[:, top_k - 1]
    return {
        "input": x[:, -1].detach().float().cpu(),
        "all_indices": indices.detach().cpu(),
        "all_membership": membership.detach().float().cpu(),
        "indices": indices[:, -1].detach().cpu(),
        "membership": membership[:, -1].detach().float().cpu(),
        "distance": full_distance[:, -1].detach().float().cpu(),
        "boundary_margin": float(boundary_margin.min().detach().cpu()),
    }


def compare_routes(incremental: dict, recomputed: dict) -> dict:
    left_indices = incremental["indices"]
    right_indices = recomputed["indices"]
    left_set = set(int(item) for item in left_indices.flatten().tolist())
    right_set = set(int(item) for item in right_indices.flatten().tolist())
    return {
        "top1_equal": bool(torch.equal(left_indices[..., :1], right_indices[..., :1])),
        "topk_order_equal": bool(torch.equal(left_indices, right_indices)),
        "topk_set_equal": left_set == right_set,
        "incremental_indices": left_indices.flatten().tolist(),
        "recomputed_indices": right_indices.flatten().tolist(),
        "route_input_max_abs_error": float(
            (incremental["input"] - recomputed["input"]).abs().max()
        ),
        "membership_max_abs_error": float(
            (incremental["membership"] - recomputed["membership"]).abs().max()
        ),
        "distance_max_abs_error": float(
            (incremental["distance"] - recomputed["distance"]).abs().max()
        ),
        "incremental_boundary_margin": incremental["boundary_margin"],
        "recomputed_boundary_margin": recomputed["boundary_margin"],
    }


def compare_prefix_routes(shorter: dict, longer: dict) -> dict:
    shorter_indices = shorter["all_indices"]
    prefix_indices = longer["all_indices"][:, : shorter_indices.size(1)]
    mismatch = shorter_indices.ne(prefix_indices).any(dim=-1)
    mismatch_positions = mismatch.nonzero(as_tuple=False)
    shorter_membership = shorter["all_membership"]
    prefix_membership = longer["all_membership"][:, : shorter_membership.size(1)]
    return {
        "prefix_route_mismatch_tokens": int(mismatch.sum()),
        "first_prefix_route_mismatch": (
            int(mismatch_positions[0, 1]) if mismatch_positions.numel() else None
        ),
        "prefix_membership_max_abs_error": float(
            (shorter_membership - prefix_membership).abs().max()
        ),
    }


def diagnose_context(model, config: ModelConfig, context: int) -> dict:
    snapshots: dict[int, dict] = {}
    originals = []
    for layer_index in model.anchor_layer_indices:
        layer = model.blocks[layer_index].anchor
        original = layer.route
        originals.append((layer, original))

        def wrapped(x, *args, _index=layer_index, _original=original, **kwargs):
            result = _original(x, *args, **kwargs)
            diagnostic_result = result
            if result[2] is None:
                diagnostic_result = _original(x, return_distance=True)
            snapshots[_index] = snapshot_route(x, diagnostic_result, config.top_k)
            return result

        layer.route = wrapped

    try:
        ids = (
            (torch.arange(context, device="cuda", dtype=torch.long) * 65537 + 29)
            % config.vocab_size
        ).unsqueeze(0)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            output, cache = model.prefill(ids, mask)
            prefix_routes = dict(snapshots)
            next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            snapshots.clear()
            incremental, incremental_cache = model.decode_step(next_id, cache)
            incremental_routes = dict(snapshots)
            snapshots.clear()
            extended = torch.cat((ids, next_id), dim=1)
            recomputed, recomputed_cache = model.prefill(
                extended, torch.ones_like(extended)
            )
            recomputed_routes = dict(snapshots)
        rows = []
        for layer_index in model.anchor_layer_indices:
            row = {"layer_index": layer_index}
            row.update(
                compare_routes(
                    incremental_routes[layer_index], recomputed_routes[layer_index]
                )
            )
            row.update(
                compare_prefix_routes(
                    prefix_routes[layer_index], recomputed_routes[layer_index]
                )
            )
            incremental_state = incremental_cache["layers"][layer_index][1]
            recomputed_state = recomputed_cache["layers"][layer_index][1]
            row["anchor_summary_max_abs_error"] = float(
                (
                    incremental_state.numerator.float()
                    - recomputed_state.numerator.float()
                )
                .abs()
                .max()
                .cpu()
            )
            row["anchor_mass_max_abs_error"] = float(
                (incremental_state.mass.float() - recomputed_state.mass.float())
                .abs()
                .max()
                .cpu()
            )
            rows.append(row)
        return {
            "context": context,
            "extended_context": context + 1,
            "decode_logit_max_abs_error": float(
                (incremental["logits"].float() - recomputed["logits"].float())
                .abs()
                .max()
            ),
            "first_topk_mismatch_layer": next(
                (row["layer_index"] for row in rows if not row["topk_order_equal"]),
                None,
            ),
            "layers": rows,
        }
    finally:
        for layer, original in originals:
            layer.route = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--fp32-anchor-state", action="store_true")
    parser.add_argument("--routing-logit-quantum", type=float, default=0.0)
    parser.add_argument("--routing-boundary-smoothing-epsilon", type=float, default=0.0)
    parser.add_argument("--routing-boundary-extra-candidates", type=int, default=0)
    parser.add_argument("--routing-membership-quantum", type=float, default=0.0)
    parser.add_argument("--inference-prefix-stable-group-size", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output must use a fresh path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    if args.fp32_anchor_state:
        config.anchor_decode_state_fp32 = True
    if args.routing_logit_quantum > 0.0:
        config.routing_logit_quantization = float(args.routing_logit_quantum)
    if args.routing_boundary_smoothing_epsilon < 0.0:
        raise ValueError("routing boundary smoothing epsilon cannot be negative")
    if args.routing_boundary_extra_candidates < 0:
        raise ValueError("routing boundary extra candidates cannot be negative")
    if args.routing_membership_quantum < 0.0:
        raise ValueError("routing membership quantum cannot be negative")
    if args.inference_prefix_stable_group_size < 0:
        raise ValueError("inference prefix stable group size cannot be negative")
    config.routing_boundary_smoothing_epsilon = float(
        args.routing_boundary_smoothing_epsilon
    )
    config.routing_boundary_extra_candidates = int(
        args.routing_boundary_extra_candidates
    )
    config.routing_membership_quantization = float(args.routing_membership_quantum)
    config.inference_prefix_stable_group_size = int(
        args.inference_prefix_stable_group_size
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32).eval()
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dtype": "float32",
        "quantization": "none",
        "anchor_decode_state_fp32": bool(config.anchor_decode_state_fp32),
        "routing_logit_quantization": float(config.routing_logit_quantization),
        "routing_boundary_smoothing_epsilon": float(
            config.routing_boundary_smoothing_epsilon
        ),
        "routing_boundary_extra_candidates": int(
            config.routing_boundary_extra_candidates
        ),
        "routing_membership_quantization": float(
            config.routing_membership_quantization
        ),
        "inference_prefix_stable_group_size": int(
            config.inference_prefix_stable_group_size
        ),
        "contexts": [diagnose_context(model, config, value) for value in args.contexts],
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
