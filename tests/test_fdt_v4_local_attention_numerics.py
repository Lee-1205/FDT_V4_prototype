import pytest
import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.fdt_v4 import CausalFDTv4LM, RotaryCausalWindowAttention


def attention_config(**overrides):
    values = dict(
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=32,
        n_layers=1,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=32,
        dropout=0.0,
        use_rope=True,
        local_attention_window=4,
        num_anchors=8,
        top_k=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_cached_step_matches_full_last_token_after_ring_rollover(monkeypatch):
    torch.manual_seed(20260823)
    attention = RotaryCausalWindowAttention(attention_config()).eval()
    prefix = torch.randn(1, 7, 32)
    next_token = torch.randn(1, 1, 32)
    prefix_mask = torch.ones(1, 7, dtype=torch.bool)

    _, state = attention(prefix, prefix_mask, return_state=True)

    def reject_sdpa(*args, **kwargs):
        raise AssertionError(
            "cached v4 attention must use the explicit full-attention path"
        )

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", reject_sdpa)
    cached, state = attention.step(
        next_token,
        state,
        torch.ones(1, 1, dtype=torch.bool),
        position=7,
    )
    full = attention(
        torch.cat((prefix, next_token), dim=1),
        torch.ones(1, 8, dtype=torch.bool),
    )[:, -1:]

    torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-6)

    _, projected_keys, _ = attention._project(
        torch.cat((prefix, next_token), dim=1),
        position_offset=0,
    )
    order = (torch.arange(attention.window) + state.cursor) % attention.window
    chronological_keys = state.key.index_select(2, order)
    torch.testing.assert_close(
        chronological_keys,
        projected_keys[:, :, -attention.window :],
        atol=2e-6,
        rtol=2e-6,
    )


def test_masked_cached_step_matches_full_attention():
    torch.manual_seed(7)
    attention = RotaryCausalWindowAttention(attention_config()).eval()
    prefix = torch.randn(1, 5, 32)
    next_token = torch.randn(1, 1, 32)
    prefix_mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
    _, state = attention(prefix, prefix_mask, return_state=True)

    cached, _ = attention.step(
        next_token,
        state,
        torch.ones(1, 1, dtype=torch.bool),
        position=5,
    )
    full = attention(
        torch.cat((prefix, next_token), dim=1),
        torch.cat((prefix_mask, torch.ones(1, 1, dtype=torch.bool)), dim=1),
    )[:, -1:]
    torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-6)


def test_rope_accepts_final_16k_position_and_rejects_overflow():
    attention = RotaryCausalWindowAttention(
        attention_config(max_seq_len=16_384)
    ).eval()
    token = torch.randn(1, 1, 32)

    q, k, v = attention._project(token, position_offset=16_383)
    assert torch.isfinite(q).all()
    assert torch.isfinite(k).all()
    assert torch.isfinite(v).all()

    with pytest.raises(ValueError, match="max_seq_len"):
        attention._project(token, position_offset=16_384)


def test_model_enforces_16k_prefill_and_decode_boundaries_before_compute():
    config = attention_config(
        max_seq_len=16_384,
        exact_memory_enabled=False,
        exact_memory_mode="off",
    )
    model = CausalFDTv4LM(config).eval()

    with pytest.raises(ValueError, match="prefill input exceeds max_seq_len"):
        model.prefill(torch.zeros(1, 16_385, dtype=torch.long))

    full_cache = {
        "backend": "fdt_v4_incremental",
        "length": 16_384,
        "layers": [],
    }
    with pytest.raises(ValueError, match="decode cache reached max_seq_len"):
        model.decode_step(torch.zeros(1, 1, dtype=torch.long), full_cache)
