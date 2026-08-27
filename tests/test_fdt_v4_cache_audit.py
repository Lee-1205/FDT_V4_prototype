from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "audit_fdt_v4_cache_integrity.py"
    spec = importlib.util.spec_from_file_location("audit_fdt_v4_cache_integrity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cache_audit_atomic_json(tmp_path):
    module = load_module()
    output = tmp_path / "cache.json"
    module.atomic_json(output, {"dtype": "float32", "quantization": "none"})
    assert json.loads(output.read_text(encoding="utf-8"))["quantization"] == "none"
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_audit_uses_fixed_tolerance_and_main_contexts():
    source = (ROOT / "scripts" / "audit_fdt_v4_cache_integrity.py").read_text(encoding="utf-8")
    assert "default=3e-4" in source
    assert "8192" in source
    assert "16383" in source
    assert "extended_context" in source
    assert "anchor_state_max_abs_error" in source
    assert "local_kv_state_max_abs_error" in source


def test_tensor_error_preserves_fp64_cache_precision():
    module = load_module()
    left = torch.tensor([44559.921001], dtype=torch.float64)
    right = torch.tensor([44559.921002], dtype=torch.float64)
    error = module._tensor_error(left, right)
    assert 9e-7 < error["max_abs_error"] < 1.1e-6


def _cache(numerator, mass):
    anchor = SimpleNamespace(numerator=numerator, mass=mass)
    return {"layers": [(None, anchor)]}


def test_raw_accumulator_fail_is_not_relaxed_by_normalized_summary_match():
    module = load_module()
    numerator = torch.tensor([[[[1000.0], [2000.0]]]])
    mass = torch.tensor([[[10.0, 20.0]]])
    left = _cache(numerator, mass)
    right = _cache(numerator * 1.001, mass * 1.001)
    diagnostics = module.anchor_state_diagnostics(left, right)
    assert diagnostics["raw_max_abs_error"] > 3e-4
    assert diagnostics["normalized_max_abs_error"] < 1e-5
    classification = module.classify_anchor_mismatch(
        diagnostics,
        module.anchor_state_diagnostics(right, right),
        prefill_error=1e-5,
        decode_error=1e-5,
        tolerance=3e-4,
    )
    assert classification["classification"] == "FLOAT_ACCUMULATION_ORDER_CANDIDATE"
    assert classification["requires_sol"] is False


def test_normalized_anchor_drift_is_major_and_routes_to_sol():
    module = load_module()
    left = _cache(torch.tensor([[[[1.0], [2.0]]]]), torch.tensor([[[1.0, 1.0]]]))
    right = _cache(torch.tensor([[[[1.0], [2.1]]]]), torch.tensor([[[1.0, 1.0]]]))
    diagnostics = module.anchor_state_diagnostics(left, right)
    classification = module.classify_anchor_mismatch(
        diagnostics,
        module.anchor_state_diagnostics(right, right),
        prefill_error=1e-5,
        decode_error=1e-5,
        tolerance=3e-4,
    )
    assert classification["classification"] == "POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT"
    assert classification["requires_sol"] is True


@pytest.mark.parametrize(
    ("normalized", "repeat_normalized"),
    [
        (0.014455318450927734, 0.014453411102294922),
        (0.01383829116821289, 0.013837814331054688),
    ],
)
def test_4096_8192_repeat_baseline_prevents_false_major_but_raw_stays_fail(normalized, repeat_normalized):
    module = load_module()
    state = {
        "structure_match": True,
        "raw_max_abs_error": 0.5,
        "normalized_max_abs_error": normalized,
    }
    repeat = {
        "structure_match": True,
        "raw_max_abs_error": 0.9,
        "normalized_max_abs_error": repeat_normalized,
    }
    classification = module.classify_anchor_mismatch(
        state,
        repeat,
        prefill_error=1.2e-5,
        decode_error=7.2e-5,
        tolerance=3e-4,
    )
    assert classification["classification"] == "INDETERMINATE_CUDA_REDUCTION_BASELINE"
    assert classification["requires_sol"] is False
    assert classification["normalized_excess_over_repeat_baseline"] <= 3e-4
    assert classification["raw_status_relaxed"] is False
    assert module.strict_integrity_status(True, True, 1.2e-5, 7.2e-5, 0.5, 3e-4) == "FAIL"


def test_normalized_error_materially_above_repeat_baseline_remains_major():
    module = load_module()
    state = {"structure_match": True, "raw_max_abs_error": 0.5, "normalized_max_abs_error": 0.015}
    repeat = {"structure_match": True, "raw_max_abs_error": 0.5, "normalized_max_abs_error": 0.010}
    classification = module.classify_anchor_mismatch(state, repeat, 1e-5, 1e-5, 3e-4)
    assert classification["normalized_excess_over_repeat_baseline"] > 3e-4
    assert classification["classification"] == "POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT"
    assert classification["requires_sol"] is True


@torch.no_grad()
def test_small_cpu_fp32_incremental_anchor_state_matches_full_recompute():
    module = load_module()
    torch.manual_seed(41)
    config = ModelConfig(
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=160,
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
    model = build_model(config).to(dtype=torch.float32).eval()
    ids = ((torch.arange(128) * 17 + 3) % config.vocab_size).unsqueeze(0)
    output, cache = model.prefill(ids, torch.ones_like(ids))
    next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
    incremental, incremental_cache = model.decode_step(next_id, cache)
    extended = torch.cat((ids, next_id), dim=1)
    recomputed = model(extended, attention_mask=torch.ones_like(extended))["logits"][:, -1:]
    _, recomputed_cache = model.prefill(extended, torch.ones_like(extended))
    diagnostics = module.anchor_state_diagnostics(incremental_cache, recomputed_cache)
    local = module.local_state_diagnostics(incremental_cache, recomputed_cache)
    assert torch.allclose(incremental["logits"], recomputed, atol=3e-4, rtol=0.0)
    assert diagnostics["raw_max_abs_error"] <= 3e-4
    assert diagnostics["normalized_max_abs_error"] <= 3e-4
    assert local["structure_match"] is True
    assert local["length_match"] is True
    assert local["max_abs_error"] <= 3e-4


def test_local_cache_length_mismatch_is_a_structural_failure():
    module = load_module()
    left = {"length": 8, "layers": []}
    right = {"length": 9, "layers": []}
    diagnostics = module.local_state_diagnostics(left, right)
    assert diagnostics["structure_match"] is False
    assert diagnostics["length_match"] is False
    assert diagnostics["max_abs_error"] == float("inf")
