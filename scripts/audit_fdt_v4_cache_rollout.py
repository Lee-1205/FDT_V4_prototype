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


def audit_rollout(model, config: ModelConfig, context: int, steps: int, tolerance: float) -> dict:
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    originals = []
    for layer_index in model.anchor_layer_indices:
        layer = model.blocks[layer_index].anchor
        original = layer.route
        originals.append((layer, original))

        def wrapped(x, *args, _index=layer_index, _original=original, **kwargs):
            result = _original(x, *args, **kwargs)
            captures[_index] = (
                result[3][:, -1].detach().cpu(),
                result[4][:, -1].detach().float().cpu(),
            )
            return result

        layer.route = wrapped

    try:
        ids = (
            (torch.arange(context, device="cuda", dtype=torch.long) * 65537 + 29)
            % config.vocab_size
        ).unsqueeze(0)
        with torch.inference_mode():
            output, cache = model.prefill(ids, torch.ones_like(ids))
            next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            maximum_logit_error = 0.0
            maximum_membership_error = 0.0
            first_token_mismatch = None
            first_route_mismatch = None
            generated = []
            rows = []
            for step in range(1, steps + 1):
                captures.clear()
                cached, cache = model.decode_step(next_id, cache)
                cached_routes = dict(captures)
                ids = torch.cat((ids, next_id), dim=1)
                generated.append(int(next_id.item()))

                captures.clear()
                recomputed = model(ids, attention_mask=torch.ones_like(ids))
                recomputed_routes = dict(captures)
                cached_logits = cached["logits"][:, -1].float()
                full_logits = recomputed["logits"][:, -1].float()
                logit_error = float((cached_logits - full_logits).abs().max())
                token_equal = bool(
                    cached_logits.argmax(-1).eq(full_logits.argmax(-1)).all()
                )
                route_equal = True
                membership_error = 0.0
                for layer_index in model.anchor_layer_indices:
                    cached_indices, cached_membership = cached_routes[layer_index]
                    full_indices, full_membership = recomputed_routes[layer_index]
                    route_equal &= bool(torch.equal(cached_indices, full_indices))
                    membership_error = max(
                        membership_error,
                        float((cached_membership - full_membership).abs().max()),
                    )
                maximum_logit_error = max(maximum_logit_error, logit_error)
                maximum_membership_error = max(
                    maximum_membership_error, membership_error
                )
                if not token_equal and first_token_mismatch is None:
                    first_token_mismatch = step
                if not route_equal and first_route_mismatch is None:
                    first_route_mismatch = step
                rows.append(
                    {
                        "step": step,
                        "context": int(ids.size(1)),
                        "logit_max_abs_error": logit_error,
                        "token_equal": token_equal,
                        "all_anchor_topk_equal": route_equal,
                        "membership_max_abs_error": membership_error,
                    }
                )
                next_id = cached_logits.argmax(dim=-1, keepdim=True)
        status = (
            "PASS"
            if maximum_logit_error <= tolerance
            and first_token_mismatch is None
            and first_route_mismatch is None
            else "FAIL"
        )
        return {
            "status": status,
            "initial_context": context,
            "steps": steps,
            "final_context": int(ids.size(1)),
            "tolerance": tolerance,
            "max_logit_abs_error": maximum_logit_error,
            "max_membership_abs_error": maximum_membership_error,
            "first_token_mismatch": first_token_mismatch,
            "first_route_mismatch": first_route_mismatch,
            "generated_token_ids": generated,
            "rows": rows,
        }
    finally:
        for layer, original in originals:
            layer.route = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--tolerance", type=float, default=3e-4)
    parser.add_argument("--inference-prefix-stable-group-size", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("rollout audit output must use a fresh path")
    if args.context + args.steps > 16_384:
        raise ValueError("rollout exceeds the configured 16K context")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    if args.inference_prefix_stable_group_size < 0:
        raise ValueError("inference prefix stable group size cannot be negative")
    config.inference_prefix_stable_group_size = int(
        args.inference_prefix_stable_group_size
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32).eval()
    torch.cuda.reset_peak_memory_stats()
    rollout = audit_rollout(
        model, config, int(args.context), int(args.steps), float(args.tolerance)
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_stage_status": payload.get("stage_status"),
        "dtype": "float32",
        "quantization": "none",
        "inference_prefix_stable_group_size": int(
            config.inference_prefix_stable_group_size
        ),
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "rollout": rollout,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"status": rollout["status"], **{key: rollout[key] for key in ("initial_context", "steps", "final_context", "max_logit_abs_error", "max_membership_abs_error", "first_token_mismatch", "first_route_mismatch")}}))


if __name__ == "__main__":
    main()
