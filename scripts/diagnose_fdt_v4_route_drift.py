from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT.parent / "source" / "src"
SCRIPTS = ROOT / "scripts" if (ROOT / "scripts").is_dir() else ROOT.parent / "training" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_fdt_v4_cache_integrity import anchor_state_diagnostics  # noqa: E402
from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def capture_anchor_inputs(model, call: Callable[[], Any]) -> tuple[Any, dict[int, torch.Tensor]]:
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in model.anchor_layer_indices:
        norm = model.blocks[layer_index].anchor_norm

        def hook(_module, _inputs, output, index=layer_index):
            captures[index] = output[:, -1:].detach().clone()

        handles.append(norm.register_forward_hook(hook))
    try:
        result = call()
    finally:
        for handle in handles:
            handle.remove()
    return result, captures


def _route_logits(layer, anchor_input: torch.Tensor) -> torch.Tensor:
    q = F.normalize(layer.q_proj(anchor_input), dim=-1)
    anchor_keys, _ = layer.normalized_anchors()
    cosine = torch.matmul(q, anchor_keys.t())
    if layer.config.routing_type == "cosine":
        logits = cosine / max(layer.config.cosine_temperature, 1e-5)
    elif layer.config.routing_type == "gaussian":
        distance = (2.0 - 2.0 * cosine).clamp_min(0.0)
        sigma = layer.log_sigma.exp().clamp(layer.config.sigma_min, layer.config.sigma_max)
        logits = -distance / (2.0 * sigma.view(1, 1, -1).pow(2))
    else:
        raise ValueError(f"unsupported routing type: {layer.config.routing_type}")
    return torch.nan_to_num(logits, nan=0.0).clamp(
        -layer.config.membership_logit_clip,
        layer.config.membership_logit_clip,
    )


def route_trace(layer, anchor_input: torch.Tensor, position: int) -> dict[str, Any]:
    v, _, _, indices, membership = layer.route(anchor_input, return_distance=False)
    logits = _route_logits(layer, anchor_input)
    boundary_values = torch.topk(logits, k=min(layer.top_k + 1, layer.num_anchors), dim=-1).values
    boundary_margin = None
    if boundary_values.size(-1) > layer.top_k:
        boundary_margin = float((boundary_values[..., layer.top_k - 1] - boundary_values[..., layer.top_k]).item())

    index = indices[:, 0]
    weights = membership[:, 0]
    scales = layer._decode_recency_scales[position].to(device=weights.device, dtype=weights.dtype)
    stacked_weights = weights.unsqueeze(-1) * scales
    dim = int(v.size(-1))
    numerator = torch.zeros(1, layer.num_anchors, 2, dim, device=v.device, dtype=v.dtype)
    expanded = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, dim)
    numerator.scatter_add_(1, expanded, stacked_weights.unsqueeze(-1) * v[:, 0, None, None, :])
    mass = torch.zeros(1, layer.num_anchors, 2, device=v.device, dtype=v.dtype)
    mass.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, 2), stacked_weights)
    return {
        "anchor_input": anchor_input,
        "indices": indices,
        "membership": membership,
        "boundary_margin": boundary_margin,
        "numerator_contribution": numerator,
        "mass_contribution": mass,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def compare_traces(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_indices = left["indices"].reshape(-1)
    right_indices = right["indices"].reshape(-1)
    left_set = set(int(value) for value in left_indices.tolist())
    right_set = set(int(value) for value in right_indices.tolist())
    flat_left = left["anchor_input"].float().reshape(1, -1)
    flat_right = right["anchor_input"].float().reshape(1, -1)
    return {
        "anchor_input_max_abs_error": _max_abs(left["anchor_input"], right["anchor_input"]),
        "anchor_input_cosine": float(F.cosine_similarity(flat_left, flat_right, dim=-1).item()),
        "indices_equal": bool(torch.equal(left["indices"], right["indices"])),
        "top1_equal": bool(left_indices[0].eq(right_indices[0]).item()),
        "topk_overlap": len(left_set & right_set),
        "topk_size": int(left_indices.numel()),
        "left_indices": [int(value) for value in left_indices.tolist()],
        "right_indices": [int(value) for value in right_indices.tolist()],
        "membership_max_abs_error": _max_abs(left["membership"], right["membership"]),
        "left_membership": [float(value) for value in left["membership"].reshape(-1).float().tolist()],
        "right_membership": [float(value) for value in right["membership"].reshape(-1).float().tolist()],
        "left_boundary_margin": left["boundary_margin"],
        "right_boundary_margin": right["boundary_margin"],
        "numerator_contribution_max_abs_error": _max_abs(left["numerator_contribution"], right["numerator_contribution"]),
        "mass_contribution_max_abs_error": _max_abs(left["mass_contribution"], right["mass_contribution"]),
    }


def evaluate_context(model, config: ModelConfig, context: int, device: str = "cuda") -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ids = ((torch.arange(context, device=device, dtype=torch.long) * 65537 + 29) % config.vocab_size).unsqueeze(0)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        (prefill, cache), _ = capture_anchor_inputs(model, lambda: model.prefill(ids, mask))
        next_id = prefill["logits"][:, -1].argmax(dim=-1, keepdim=True)
        (incremental, incremental_cache), decode_inputs = capture_anchor_inputs(
            model, lambda: model.decode_step(next_id, cache)
        )
        extended = torch.cat((ids, next_id), dim=1)
        extended_mask = torch.ones_like(extended)
        (full, full_cache), full_inputs = capture_anchor_inputs(
            model, lambda: model.prefill(extended, extended_mask)
        )
        (_, repeat_cache), repeat_inputs = capture_anchor_inputs(
            model, lambda: model.prefill(extended, extended_mask)
        )

        layers = []
        for layer_index in model.anchor_layer_indices:
            layer = model.blocks[layer_index].anchor
            decode_trace = route_trace(layer, decode_inputs[layer_index], context)
            full_trace = route_trace(layer, full_inputs[layer_index], context)
            repeat_trace = route_trace(layer, repeat_inputs[layer_index], context)
            layers.append(
                {
                    "layer_index": int(layer_index),
                    "decode_vs_full": compare_traces(decode_trace, full_trace),
                    "full_vs_repeat_full": compare_traces(full_trace, repeat_trace),
                }
            )
        state = anchor_state_diagnostics(incremental_cache, full_cache)
        repeat_state = anchor_state_diagnostics(full_cache, repeat_cache)
        logit_error = _max_abs(incremental["logits"][:, -1], full["logits"][:, -1])

    route_failures = [row["layer_index"] for row in layers if not row["decode_vs_full"]["indices_equal"]]
    repeat_route_failures = [row["layer_index"] for row in layers if not row["full_vs_repeat_full"]["indices_equal"]]
    record = {
        "context": int(context),
        "prompt_context": int(context),
        "extended_context": int(context + 1),
        "appended_token_id": int(next_id.item()),
        "decode_full_max_abs_logit_error": logit_error,
        "decode_full_token_agreement": bool(incremental["logits"][:, -1].argmax(-1).eq(full["logits"][:, -1].argmax(-1)).all()),
        "first_decode_full_route_divergence_layer": route_failures[0] if route_failures else None,
        "decode_full_route_divergence_layers": route_failures,
        "full_repeat_route_divergence_layers": repeat_route_failures,
        "anchor_state": state,
        "repeat_full_anchor_state": repeat_state,
        "layers": layers,
    }
    if device.startswith("cuda"):
        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace FDT v4 appended-token route drift")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16383])
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the main-checkpoint route audit")
    if args.output.exists():
        raise FileExistsError("route-drift output is immutable and must use a fresh path")
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32).eval()
    contexts = [evaluate_context(model, config, int(context)) for context in args.contexts]
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_stage_status": payload.get("stage_status"),
        "dtype": "float32",
        "quantization": "none",
        "gpu": torch.cuda.get_device_name(0),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": args.git_commit or "UNKNOWN",
        },
        "contexts": contexts,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "contexts": args.contexts, "gpu": report["gpu"]}))


if __name__ == "__main__":
    main()
