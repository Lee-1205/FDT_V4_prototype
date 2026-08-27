import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.lexical_pointer import (
    AnchorIndexedExactMemory,
    LexicalPointerDecodeState,
    SparseLexicalPointer,
)
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import apply_ngram_loop_penalty_


def tiny_config(**overrides):
    values = dict(
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=32,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=32,
        dropout=0.0,
        use_rope=True,
        num_anchors=16,
        top_k=4,
        router_dim=16,
        anchor_layer_indices=[0, 2],
        local_attention_window=8,
        anchor_scan_chunk_size=4,
        exact_memory_enabled=True,
        exact_memory_mode="copy",
        exact_pointer_chunk_size=4,
        exact_pointer_chunk_anchors=2,
        exact_pointer_candidate_chunks=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_v4_uses_rope_without_learned_position_table():
    model = build_model(tiny_config())
    assert model.position_embedding is None
    assert model.blocks[0].local_attention.rope is not None
    assert model.blocks[1].anchor is None
    assert model.exact_pointer is not None


def test_v4_is_causal():
    torch.manual_seed(1)
    model = build_model(tiny_config()).eval()
    left = torch.tensor([[2, 3, 4, 5, 6, 7]])
    right = left.clone()
    right[:, 4:] = torch.tensor([[44, 45]])
    a = model(left, attention_mask=torch.ones_like(left))["logits"][:, :4]
    b = model(right, attention_mask=torch.ones_like(right))["logits"][:, :4]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_v4_incremental_cache_matches_full_recompute():
    torch.manual_seed(2)
    model = build_model(tiny_config(exact_memory_enabled=False, exact_memory_mode="off")).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    output, cache = model.prefill(ids, torch.ones_like(ids))
    next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
    incremental, _ = model.decode_step(next_id, cache)
    full_ids = torch.cat((ids, next_id), dim=1)
    full = model(full_ids, attention_mask=torch.ones_like(full_ids))["logits"][:, -1:]
    torch.testing.assert_close(incremental["logits"], full, atol=3e-4, rtol=3e-4)


def test_v4_gradient_checkpointing_preserves_forward_and_exact_routes():
    torch.manual_seed(23)
    model = build_model(tiny_config()).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 1]])
    mask = torch.ones_like(ids)
    reference = model(ids, attention_mask=mask)
    reference_routes = model.exact_route_indices(reference["hidden"]).clone()

    model.set_gradient_checkpointing(True)
    checkpointed = model(ids, attention_mask=mask)
    checkpointed_routes = model.exact_route_indices(checkpointed["hidden"]).clone()

    torch.testing.assert_close(checkpointed["logits"], reference["logits"], atol=1e-6, rtol=1e-6)
    assert torch.equal(checkpointed_routes, reference_routes)
    assert len(checkpointed["anchor_stats"]) == len(model.anchor_layer_indices)
    assert all(torch.isfinite(item["entropy_normalized"]) for item in checkpointed["anchor_stats"])
    checkpointed["logits"].float().mean().backward()
    assert model.blocks[0].local_attention.qkv.weight.grad is not None


def test_v4_exact_memory_stores_lossless_tokens_and_compact_keys():
    torch.manual_seed(3)
    model = build_model(tiny_config()).eval()
    ids = torch.tensor([[2, 11, 12, 13, 11, 12, 13, 1]])
    output = model(ids, attention_mask=torch.ones_like(ids))
    memory = model.build_exact_memory(output["hidden"], ids, torch.ones_like(ids))
    assert torch.equal(memory.token_ids.long(), ids)
    assert memory.key_vectors.shape == (1, ids.size(1) - 1, model.config.exact_pointer_dim)
    assert memory.storage_bytes() > ids.numel() * 4


def test_ngram_loop_penalty_blocks_only_repeated_completion():
    logits = torch.zeros(1, 20)
    generated = torch.tensor([[3, 4, 5, 3, 4]])
    apply_ngram_loop_penalty_(logits, generated, ngram_order=3, penalty=8.0)
    assert logits[0, 5].item() == -8.0
    assert logits[0, 6].item() == 0.0


def test_exact_memory_full_scan_is_correctness_fallback():
    pointer = SparseLexicalPointer(hidden_dim=2, pointer_dim=2, window=2)
    with torch.no_grad():
        pointer.q_proj.weight.copy_(torch.eye(2))
        pointer.gate_proj.weight.zero_()
        pointer.gate_proj.bias.fill_(10.0)
        pointer.logit_scale.fill_(torch.tensor(3.0))
    memory = AnchorIndexedExactMemory(
        token_ids=torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32),
        valid_mask=torch.ones(1, 5, dtype=torch.bool),
        anchor_ids=torch.ones(1, 5, 1, dtype=torch.int16),
        chunk_anchor_ids=torch.tensor([[[1], [2], [3]]], dtype=torch.int16),
        chunk_valid=torch.ones(1, 3, dtype=torch.bool),
        commit_scores=torch.zeros(1, 5, dtype=torch.float16),
        chunk_commit_score=torch.zeros(1, 3),
        chunk_size=2,
        key_vectors=torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]),
    )
    positions, valid = memory.candidate_key_positions(torch.tensor([[1]]), max_chunks=1)
    base_logits = torch.zeros(1, 32)
    mixed, diagnostics = pointer.mix_next_logits(
        base_logits,
        hidden=torch.tensor([[[1.0, 0.0]]]),
        input_ids=memory.token_ids.long(),
        attention_mask=memory.valid_mask.long(),
        source_length=5,
        min_gate=0.0,
        anchor_memory=memory,
        query_anchor_ids=torch.tensor([[1]]),
        indexed_candidates=(positions, valid),
        full_scan_fallback=True,
        candidate_cap=4,
    )
    assert diagnostics["used_full_scan_fallback"] is True
    assert int(mixed.argmax(dim=-1)) == 14


def test_v4_global_exact_loss_crosses_local_window_and_source_chunk_boundary():
    torch.manual_seed(11)
    model = build_model(tiny_config(max_seq_len=96)).train()
    ids = torch.full((1, 90), 5, dtype=torch.long)
    ids[:, 16:19] = 41
    ids[:, 82:85] = 41
    ids[:, 85:] = torch.tensor([[51, 52, 53, 54, 1]])
    labels = torch.full_like(ids, -100)
    labels[:, 82:] = ids[:, 82:]
    mask = torch.ones_like(ids)

    output = model(ids, attention_mask=mask)
    result = model.exact_memory_loss(
        output["hidden"],
        ids,
        labels,
        mask,
        source_chunk_size=16,
        query_chunk_size=2,
    )

    assert result.contract_valid.item() == 1.0
    assert result.copyable_rate.item() > 0.0
    assert result.cursor_continuation_rate.item() > 0.0
    assert result.max_copy_distance.item() > model.config.exact_pointer_window
    assert result.scanned_source_tokens.item() > model.config.exact_pointer_window
    assert torch.isfinite(result.loss)
    assert result.pointer_loss.item() > 0.0
    result.loss.backward()
    assert model.exact_pointer.q_proj.weight.grad is not None
    assert model.exact_pointer.k_proj.weight.grad is not None
    assert model.exact_pointer.q_proj.weight.grad.abs().sum().item() > 0.0
    assert model.exact_pointer.k_proj.weight.grad.abs().sum().item() > 0.0


def test_v4_exact_loss_rejects_all_token_labels_without_negative_gate_training():
    torch.manual_seed(12)
    model = build_model(tiny_config()).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1]])
    mask = torch.ones_like(ids)
    output = model(ids, attention_mask=mask)

    result = model.exact_memory_loss(output["hidden"], ids, ids, mask)

    assert result.contract_valid.item() == 0.0
    assert result.loss.item() == 0.0
    assert result.gate_loss.item() == 0.0
    assert result.commit_loss.item() == 0.0
    result.loss.backward()
    for parameter in model.exact_pointer.parameters():
        assert parameter.grad is None or parameter.grad.abs().sum().item() == 0.0


def test_repeated_token_cursor_uses_only_the_first_full_scan():
    pointer = SparseLexicalPointer(hidden_dim=2, pointer_dim=2, window=2)
    with torch.no_grad():
        pointer.q_proj.weight.copy_(torch.eye(2))
        pointer.gate_proj.weight.zero_()
        pointer.gate_proj.bias.fill_(10.0)
        pointer.logit_scale.fill_(torch.tensor(3.0))
    memory = AnchorIndexedExactMemory(
        token_ids=torch.tensor([[9, 7, 7, 7, 7]], dtype=torch.int32),
        valid_mask=torch.ones(1, 5, dtype=torch.bool),
        anchor_ids=torch.ones(1, 5, 1, dtype=torch.int16),
        chunk_anchor_ids=torch.tensor([[[2], [1], [3]]], dtype=torch.int16),
        chunk_valid=torch.ones(1, 3, dtype=torch.bool),
        commit_scores=torch.ones(1, 5, dtype=torch.float16),
        chunk_commit_score=torch.zeros(1, 3),
        chunk_size=2,
        key_vectors=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
    )
    calls = []
    original_mix = pointer.mix_next_logits

    def counted_mix(*args, **kwargs):
        calls.append(bool(kwargs.get("full_scan_fallback")))
        return original_mix(*args, **kwargs)

    pointer.mix_next_logits = counted_mix
    state = LexicalPointerDecodeState(source_length=5)
    base = torch.zeros(1, 32)
    hidden = torch.tensor([[[1.0, 0.0]]])
    first_logits, first = state.prepare_logits(
        pointer,
        base,
        hidden,
        memory.token_ids.long(),
        memory.valid_mask.long(),
        min_gate=0.0,
        anchor_memory=memory,
        query_anchor_ids=torch.tensor([[1]]),
        max_candidate_chunks=1,
    )
    selected = int(first_logits.argmax(dim=-1))
    assert selected == 7
    assert first["used_full_scan_fallback"] is True
    assert first["cursor_continuation_supported"][0][0] is True
    state.commit(selected, first)

    second_logits, second = state.prepare_logits(
        pointer,
        base,
        hidden,
        memory.token_ids.long(),
        memory.valid_mask.long(),
        min_gate=0.0,
        anchor_memory=memory,
        query_anchor_ids=torch.tensor([[1]]),
        max_candidate_chunks=1,
    )
    assert int(second_logits.argmax(dim=-1)) == 7
    assert second["mode"] == "cursor"
    assert second["full_scan_count"] == 1
    assert calls == [True]


def test_rope_cache_keys_are_distinct_at_audit_context_boundaries():
    torch.manual_seed(13)
    model = build_model(
        tiny_config(
            max_seq_len=16384,
            exact_memory_enabled=False,
            exact_memory_mode="off",
        )
    ).eval()
    attention = model.blocks[0].local_attention
    token_state = torch.randn(1, 1, model.config.dim)
    token_mask = torch.ones(1, 1, dtype=torch.bool)
    cached_keys = []
    for context_length in (512, 1024, 2048, 4096, 8192, 16384):
        _, state = attention.step(
            token_state,
            None,
            token_mask,
            position=context_length - 1,
        )
        cached_keys.append(state.key[:, :, :1].clone())

    for left, right in zip(cached_keys, cached_keys[1:]):
        assert torch.isfinite(left).all()
        assert not torch.allclose(left, right, atol=1e-7, rtol=1e-7)
