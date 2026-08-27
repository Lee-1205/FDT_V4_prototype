from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "diagnose_fdt_v4_route_drift.py"
    spec = importlib.util.spec_from_file_location("diagnose_fdt_v4_route_drift", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=96,
        dropout=0.0,
        use_rope=True,
        num_anchors=16,
        top_k=4,
        router_dim=16,
        anchor_layer_indices=[0],
        local_attention_window=8,
        anchor_scan_chunk_size=16,
        aggregation_impl="sparse_chunked_scan",
        exact_memory_enabled=False,
        exact_memory_mode="off",
    )


@torch.no_grad()
def test_cpu_route_trace_records_appended_token_path_and_contribution():
    module = load_module()
    torch.manual_seed(73)
    config = tiny_config()
    model = build_model(config).to(dtype=torch.float32).eval()

    record = module.evaluate_context(model, config, context=64, device="cpu")

    assert record["decode_full_token_agreement"] is True
    assert record["extended_context"] == 65
    assert len(record["layers"]) == 1
    layer = record["layers"][0]
    assert layer["layer_index"] == 0
    assert layer["decode_vs_full"]["topk_size"] == config.top_k
    assert layer["decode_vs_full"]["left_boundary_margin"] is not None
    assert layer["full_vs_repeat_full"]["indices_equal"] is True
    assert record["anchor_state"]["structure_match"] is True


def test_route_drift_atomic_json(tmp_path):
    module = load_module()
    output = tmp_path / "route.json"
    module.atomic_json(output, {"status": "diagnostic"})
    assert output.is_file()
    assert not list(tmp_path.glob("*.tmp"))
