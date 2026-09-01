from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "generate_fdt_v4.py"
    spec = importlib.util.spec_from_file_location("generate_fdt_v4_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_explicit_span_map_is_loaded_without_heuristic_resegmentation(tmp_path):
    module = load_module()
    contract = tmp_path / "spans.json"
    contract.write_text('{"span_end_positions": [2, 2, 2, 4, 4]}', encoding="utf-8")
    ends = module.load_exact_span_end_positions(contract, 5)
    assert ends.tolist() == [[2, 2, 2, 4, 4]]


def test_product_loop_penalty_uses_calibrated_floor_unless_explicitly_disabled():
    module = load_module()
    assert module.resolve_ngram_loop_penalty(8.0, None) == 13.0
    assert module.resolve_ngram_loop_penalty(20.0, None) == 20.0
    assert module.resolve_ngram_loop_penalty(8.0, 0.0) == 0.0


def test_exact_copy_requires_an_explicit_span_contract():
    module = load_module()
    assert not module.exact_copy_contract_enabled(True, "copy", None)
    assert not module.exact_copy_contract_enabled(False, "copy", torch.ones(1, 1))
    assert module.exact_copy_contract_enabled(True, "copy", torch.ones(1, 1))


def test_free_generation_receives_token_and_ngram_loop_controls():
    module = load_module()
    logits = torch.zeros(1, 20)
    logits[0, 3] = 2.0
    logits[0, 5] = 4.0
    generated = torch.tensor([[3, 4, 5, 3, 4]])

    controlled = module.apply_non_copy_generation_controls(
        logits,
        generated,
        copy_active=False,
        repetition_penalty=1.1,
        ngram_order=3,
        ngram_penalty=8.0,
        ngram_window=96,
        ngram_hard_block_after=2,
    )

    assert controlled[0, 3].item() < logits[0, 3].item()
    assert controlled[0, 5].item() < 0.0
    assert torch.equal(logits, torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 4.0] + [0.0] * 14]))


def test_exact_copy_mode_is_exempt_from_all_repetition_controls():
    module = load_module()
    logits = torch.randn(1, 20)
    original = logits.clone()
    generated = torch.tensor([[3, 4, 5, 3, 4]])

    controlled = module.apply_non_copy_generation_controls(
        logits,
        generated,
        copy_active=True,
        repetition_penalty=1.1,
        ngram_order=3,
        ngram_penalty=8.0,
        ngram_window=96,
        ngram_hard_block_after=2,
    )

    assert controlled is logits
    assert torch.equal(controlled, original)


def test_mixed_copy_exempts_only_the_pointer_candidate():
    module = load_module()
    logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
    generated = torch.tensor([[1, 2, 1, 2]])

    controlled = module.apply_non_copy_generation_controls(
        logits,
        generated,
        copy_active=False,
        exempt_token_ids=torch.tensor([[1]]),
        repetition_penalty=1.2,
        ngram_order=3,
        ngram_penalty=8.0,
        ngram_window=96,
        ngram_hard_block_after=2,
    )

    assert controlled[0, 1] == logits[0, 1]
    assert controlled[0, 2] < logits[0, 2]


def test_persistent_loop_is_hard_blocked_within_recent_window():
    module = load_module()
    logits = torch.zeros(1, 20)
    generated = torch.tensor([[3, 4, 5, 3, 4, 5, 3, 4]])

    controlled = module.apply_non_copy_generation_controls(
        logits,
        generated,
        copy_active=False,
        repetition_penalty=1.0,
        ngram_order=3,
        ngram_penalty=8.0,
        ngram_window=96,
        ngram_hard_block_after=2,
    )

    assert torch.isneginf(controlled[0, 5])


def test_loop_outside_recent_window_is_not_penalized():
    module = load_module()
    logits = torch.zeros(1, 20)
    generated = torch.tensor([[3, 4, 5, 8, 8, 8, 8, 3, 4]])

    controlled = module.apply_non_copy_generation_controls(
        logits,
        generated,
        copy_active=False,
        repetition_penalty=1.0,
        ngram_order=3,
        ngram_penalty=8.0,
        ngram_window=5,
        ngram_hard_block_after=2,
    )

    assert controlled[0, 5].item() == 0.0
