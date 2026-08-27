from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState
from fdt_rlm.models import build_model
from fdt_rlm.models.fdt_v3 import (
    sparse_chunked_prefix_summaries,
    sparse_segmented_prefix_summaries,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=127,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=64,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=16_384,
        dropout=0.0,
        use_rope=True,
        num_anchors=16,
        top_k=4,
        router_dim=32,
        anchor_layer_indices=[0, 2],
        local_attention_window=16,
        anchor_scan_chunk_size=4,
        exact_memory_enabled=True,
        exact_memory_mode="copy",
        exact_pointer_chunk_size=16,
        exact_pointer_chunk_anchors=2,
        exact_pointer_candidate_chunks=2,
    )


def check_sparse_scan(device: torch.device) -> dict[str, float]:
    torch.manual_seed(901)
    bsz, seq_len, top_k, anchors, dim = 1, 137, 2, 7, 3
    routing_logits = torch.randn(
        bsz, seq_len, top_k, device=device, requires_grad=True
    )
    membership = routing_logits.softmax(-1)
    positions = torch.arange(seq_len, device=device).view(1, -1)
    indices = torch.stack(
        (positions.remainder(anchors), (positions + 3).remainder(anchors)), dim=-1
    )
    values = torch.randn(bsz, seq_len, dim, device=device, requires_grad=True)
    mask = torch.ones(bsz, seq_len, device=device)

    chunked = sparse_chunked_prefix_summaries(
        membership, indices, values, mask, anchors, 16_384, 1.5, 4
    )
    reference = sparse_segmented_prefix_summaries(
        membership, indices, values, mask, anchors, 16_384, 1.5
    )
    chunked_output = sum(
        torch.einsum("bnk,bnkd->bnd", membership, item) for item in chunked
    )
    reference_output = sum(
        torch.einsum("bnk,bnkd->bnd", membership, item) for item in reference
    )
    chunked_grads = torch.autograd.grad(
        chunked_output.square().mean(), (routing_logits, values), retain_graph=True
    )
    reference_grads = torch.autograd.grad(
        reference_output.square().mean(), (routing_logits, values)
    )
    output_error = float((chunked_output - reference_output).abs().max().detach())
    route_grad_error = float(
        (chunked_grads[0] - reference_grads[0]).abs().max().detach()
    )
    value_grad_error = float(
        (chunked_grads[1] - reference_grads[1]).abs().max().detach()
    )
    if output_error > 2e-5 or route_grad_error > 2e-5 or value_grad_error > 2e-5:
        raise RuntimeError("CUDA sparse scan differs from segmented reference")
    return {
        "output_max_abs_error": output_error,
        "route_grad_max_abs_error": route_grad_error,
        "value_grad_max_abs_error": value_grad_error,
    }


def check_model(device: torch.device) -> dict[str, object]:
    torch.manual_seed(902)
    model = build_model(tiny_config()).to(device=device, dtype=torch.bfloat16).train()
    model.set_gradient_checkpointing(True)
    ids = ((torch.arange(257, device=device) * 17 + 3) % 126 + 1).unsqueeze(0)
    mask = torch.ones_like(ids)
    output = model(ids, attention_mask=mask)
    loss = output["logits"].float().square().mean()
    loss.backward()
    gradient = model.blocks[0].local_attention.qkv.weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("checkpointed CUDA backward produced an invalid gradient")

    model.eval()
    with torch.no_grad():
        output = model(ids, attention_mask=mask)
        routes = model.exact_route_indices(output["hidden"])
        memory = model.build_exact_memory(output["hidden"], ids, mask)
        state = LexicalPointerDecodeState(source_length=ids.size(1))
        _, diagnostics = state.prepare_logits(
            model.exact_pointer,
            output["logits"][:, -1].float(),
            output["hidden"][:, -1:],
            ids,
            mask,
            min_gate=0.0,
            anchor_memory=memory,
            query_anchor_ids=routes[:, -1],
            max_candidate_chunks=2,
        )
    if not diagnostics.get("candidate_ids"):
        raise RuntimeError("exact memory returned no CUDA candidate")
    return {
        "loss": float(loss.detach()),
        "gradient_finite": True,
        "exact_mode": diagnostics.get("mode"),
        "exact_candidates": len(diagnostics["candidate_ids"][0]),
        "exact_endpoint_source": diagnostics.get("span_end_source"),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "sparse_scan": check_sparse_scan(device),
        "model": check_model(device),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
