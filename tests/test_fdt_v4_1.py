from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import torch
import pytest

from fdt_rlm.config import ModelConfig
from fdt_rlm.lexical_pointer import (
    AnchorIndexedExactMemory,
    LexicalPointerDecodeState,
    SparseLexicalPointer,
)
from fdt_rlm.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def _load_bridge_trainer():
    path = ROOT / "scripts" / "train_fdt_v4_curriculum_bridge.py"
    spec = importlib.util.spec_from_file_location("fdt_v4_1_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_generated_prefix_builder():
    path = ROOT / "scripts" / "prepare_fdt_v4_1_generated_prefix_recovery.py"
    spec = importlib.util.spec_from_file_location("fdt_v4_1_prefix_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_evaluator():
    path = ROOT / "scripts" / "evaluate_fdt_v4.py"
    spec = importlib.util.spec_from_file_location("fdt_v4_1_evaluator_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_config(**overrides):
    values = dict(
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        num_anchors=16,
        top_k=4,
        router_dim=16,
        anchor_layer_indices=[0],
        local_attention_window=8,
        anchor_scan_chunk_size=16,
        aggregation_impl="sparse_chunked_scan",
        anchor_recency_bias=4.0,
        anchor_recency_reference_len=64,
        exact_memory_enabled=False,
        exact_memory_mode="off",
    )
    values.update(overrides)
    return ModelConfig(**values)


@torch.no_grad()
def test_loop_controller_starts_as_exact_noop_and_adds_only_small_parameters():
    torch.manual_seed(70)
    baseline = build_model(
        _base_config(model_type="fdt_v4", max_seq_len=64, use_rope=True)
    ).eval()
    controlled = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=64,
            use_rope=True,
            loop_controller_rank=4,
        )
    ).eval()
    incompatible = controlled.load_state_dict(baseline.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "loop_controller.down.weight",
        "loop_controller.up.weight",
    }
    assert incompatible.unexpected_keys == []
    ids = torch.randint(2, 97, (2, 16))
    mask = torch.ones_like(ids)
    torch.testing.assert_close(
        baseline(ids, attention_mask=mask)["logits"],
        controlled(ids, attention_mask=mask)["logits"],
        rtol=0.0,
        atol=0.0,
    )
    added = sum(parameter.numel() for parameter in controlled.loop_controller.parameters())
    assert added == 4 * (32 + 97)


@torch.no_grad()
def test_adapter_overlay_loader_reconstructs_parent_plus_controller(tmp_path):
    evaluator = _load_evaluator()
    base_config = _base_config(model_type="fdt_v4", max_seq_len=64, use_rope=True)
    target_config = _base_config(
        model_type="fdt_v4",
        max_seq_len=64,
        use_rope=True,
        loop_controller_rank=4,
    )
    parent_model = build_model(base_config).eval()
    target_model = build_model(target_config).eval()
    target_model.load_state_dict(parent_model.state_dict(), strict=False)
    target_model.loop_controller.up.weight.normal_(mean=0.0, std=0.01)
    parent = tmp_path / "parent.pt"
    overlay = tmp_path / "overlay.pt"
    torch.save(
        {
            "model_config": asdict(base_config),
            "model_state_dict": parent_model.state_dict(),
        },
        parent,
    )
    torch.save(
        {
            "model_config": asdict(target_config),
            "checkpoint_format": "fdt_v4_adapter_overlay_v1",
            "parent_checkpoint": str(parent.resolve()),
            "parent_checkpoint_sha256": evaluator.sha256_file(parent),
            "adapter_state_dict": {
                name: tensor
                for name, tensor in target_model.state_dict().items()
                if name.startswith("loop_controller.")
            },
        },
        overlay,
    )
    loaded, _, metadata = evaluator.load_checkpoint(overlay)
    ids = torch.randint(2, 97, (1, 12))
    mask = torch.ones_like(ids)
    torch.testing.assert_close(
        loaded(ids, attention_mask=mask)["logits"],
        target_model(ids, attention_mask=mask)["logits"],
    )
    assert metadata["checkpoint_format"] == "fdt_v4_adapter_overlay_v1"


def test_chunked_lm_retention_trains_only_enabled_loop_controller():
    trainer = _load_bridge_trainer()
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=64,
            use_rope=True,
            loop_controller_rank=4,
        )
    ).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.loop_controller.parameters():
        parameter.requires_grad_(True)
    ids = torch.randint(2, 97, (2, 16))
    mask = torch.ones_like(ids)
    hidden = model(ids, attention_mask=mask, return_logits=False)["hidden"]
    loss = trainer.chunked_weighted_lm_loss(
        model,
        hidden,
        ids,
        mask,
        0,
        1,
        2.0,
        8,
        False,
    )
    loss.backward()
    assert model.loop_controller.up.weight.grad is not None
    assert model.loop_controller.up.weight.grad.abs().sum().item() > 0.0
    assert model.lm_head.weight.grad is None


@torch.no_grad()
def test_alpha_zero_preserves_v3_logits_with_legacy_positions():
    torch.manual_seed(71)
    source = build_model(
        _base_config(model_type="fdt_v3", max_seq_len=64, use_rope=False)
    ).eval()
    target = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_alpha=0.0,
            legacy_position_transition_max_len=64,
        )
    ).eval()
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, tensor in target_state.items():
        source_name = (
            "position_embedding.weight"
            if name == "legacy_position_embedding.weight"
            else name
        )
        if source_name in source_state and source_state[source_name].shape == tensor.shape:
            tensor.copy_(source_state[source_name])
    target.load_state_dict(target_state, strict=True)

    ids = ((torch.arange(48) * 13 + 5) % source.config.vocab_size).unsqueeze(0)
    mask = torch.ones_like(ids)
    source_logits = source(ids, attention_mask=mask)["logits"]
    target_logits = target(ids, attention_mask=mask)["logits"]
    assert torch.equal(source_logits, target_logits)


@torch.no_grad()
def test_alpha_one_fully_disables_legacy_position_table():
    torch.manual_seed(73)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_alpha=1.0,
            legacy_position_transition_max_len=64,
        )
    ).eval()
    ids = torch.arange(32).unsqueeze(0) % model.config.vocab_size
    before = model(ids)["logits"]
    model.legacy_position_embedding.weight.normal_(mean=0.0, std=100.0)
    after = model(ids)["logits"]
    assert torch.equal(before, after)


@torch.no_grad()
def test_phase_rope_transition_preserves_qk_norms_and_endpoints():
    torch.manual_seed(74)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_alpha=0.0,
            rope_transition_mode="phase",
            legacy_position_transition_max_len=64,
            legacy_position_scale=1.0,
        )
    ).eval()
    attention = model.blocks[0].local_attention
    x = torch.randn(2, 41, model.config.dim)
    q_zero, k_zero, _ = attention._project(x)
    model.set_transition_alpha(0.37)
    q_mid, k_mid, _ = attention._project(x)
    assert torch.allclose(q_mid.norm(dim=-1), q_zero.norm(dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(k_mid.norm(dim=-1), k_zero.norm(dim=-1), atol=1e-6, rtol=1e-6)
    model.set_transition_alpha(1.0)
    q_one, k_one, _ = attention._project(x)
    attention.transition_mode = "lerp"
    q_reference, k_reference, _ = attention._project(x)
    assert torch.equal(q_one, q_reference)
    assert torch.equal(k_one, k_reference)


def test_explicit_legacy_position_scale_is_independent_of_rope_alpha():
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_alpha=0.0,
            legacy_position_transition_max_len=64,
            legacy_position_scale=1.0,
        )
    )
    model.set_transition_alpha(0.5)
    assert model.legacy_position_scale == 1.0
    model.set_legacy_position_scale(0.75)
    assert model.legacy_position_scale == 0.75


@torch.no_grad()
def test_output_blend_transition_is_linear_between_exact_attention_endpoints():
    torch.manual_seed(75)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_mode="output_blend",
            rope_transition_alpha=0.0,
            legacy_position_transition_max_len=64,
            legacy_position_scale=1.0,
        )
    ).eval()
    attention = model.blocks[0].local_attention
    x = torch.randn(2, 41, model.config.dim)
    mask = torch.ones(2, 41, dtype=torch.long)
    model.set_transition_alpha(0.0)
    legacy = attention(x, mask)
    model.set_transition_alpha(1.0)
    rope = attention(x, mask)
    model.set_transition_alpha(0.25)
    middle = attention(x, mask)
    assert torch.allclose(middle, torch.lerp(legacy, rope, 0.25), atol=1e-6, rtol=1e-6)


def test_quantized_route_selection_has_deterministic_anchor_id_ties():
    torch.manual_seed(79)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=64,
            use_rope=True,
            routing_logit_quantization=1e-4,
        )
    )
    anchor = model.blocks[0].anchor
    with torch.no_grad():
        anchor.anchor_keys.fill_(1.0)
    x = torch.randn(1, 3, model.config.dim, requires_grad=True)
    route = anchor.route(x)
    assert route[3][0, 0].tolist() == [0, 1, 2, 3]
    route[4][..., 0].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_boundary_smoothed_route_is_normalized_canonical_and_differentiable():
    torch.manual_seed(20260830)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=64,
            use_rope=True,
            routing_boundary_smoothing_epsilon=1e-3,
            routing_boundary_extra_candidates=2,
        )
    )
    anchor = model.blocks[0].anchor
    with torch.no_grad():
        anchor.anchor_keys.fill_(1.0)
    x = torch.randn(1, 3, model.config.dim, requires_grad=True)
    route = anchor.route(x)
    indices, membership = route[3], route[4]
    assert indices.shape[-1] == model.config.top_k + 2
    assert torch.equal(indices, indices.sort(dim=-1).values)
    torch.testing.assert_close(
        membership.sum(dim=-1),
        torch.ones_like(membership[..., 0]),
    )
    membership[..., 0].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_membership_quantization_uses_straight_through_gradients():
    torch.manual_seed(20260831)
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=64,
            use_rope=True,
            routing_membership_quantization=1e-4,
        )
    )
    anchor = model.blocks[0].anchor
    x = torch.randn(1, 3, model.config.dim, requires_grad=True)
    membership = anchor.route(x)[4]
    scaled = membership.detach() / 1e-4
    torch.testing.assert_close(scaled, scaled.round(), atol=2e-3, rtol=0.0)
    membership[..., 0].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("copy_length", [4, 8, 16, 32, 64])
@torch.no_grad()
def test_high_confidence_hard_copy_is_lossless_end_to_end(copy_length):
    pointer = SparseLexicalPointer(hidden_dim=2, pointer_dim=2, window=8)
    pointer.q_proj.weight.copy_(torch.eye(2))
    pointer.k_proj.weight.copy_(torch.eye(2))
    pointer.gate_proj.weight.zero_()
    pointer.gate_proj.bias.fill_(10.0)
    pointer.logit_scale.fill_(3.0)

    code = [10 + index for index in range(copy_length)]
    input_ids = torch.tensor([[3, *code]], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    routes = torch.ones(1, input_ids.size(1), 1, dtype=torch.long)
    commits = torch.ones_like(input_ids, dtype=torch.float32)
    keys = torch.full((1, input_ids.size(1), 2), -1.0)
    keys[..., 1] = 0.0
    keys[:, 0] = torch.tensor([1.0, 0.0])
    span_ends = torch.full_like(input_ids, copy_length)
    memory = AnchorIndexedExactMemory.from_prompt(
        input_ids,
        routes,
        mask,
        chunk_size=8,
        chunk_anchor_count=1,
        commit_scores=commits,
        key_vectors=keys,
        span_end_positions=span_ends,
    )
    state = LexicalPointerDecodeState(
        source_length=input_ids.size(1),
        max_activation_steps=copy_length + 2,
        max_copy_tokens=copy_length + 2,
    )
    hidden = torch.tensor([[[1.0, 0.0]]])
    base = torch.full((1, 256), -5.0)
    base[0, 250] = 20.0
    copied = []
    for step in range(copy_length):
        logits, diagnostics = state.prepare_logits(
            pointer,
            base,
            hidden,
            input_ids,
            mask,
            min_gate=0.0,
            anchor_memory=memory,
            query_anchor_ids=torch.tensor([[1]]),
            max_candidate_chunks=1,
            full_scan_fallback=True,
            commit_threshold=0.5,
            hard_copy=True,
            hard_copy_gate_threshold=0.9,
            hard_copy_pointer_threshold=0.9,
            hard_copy_margin_threshold=1.0,
        )
        selected = int(logits.argmax(dim=-1))
        if step == 0:
            assert diagnostics["mode"] == "hard_copy"
            assert diagnostics["source_positions"][0][0] == 0
            assert diagnostics["hard_copy_eligible"] is True
        else:
            assert diagnostics["mode"] == "cursor"
        copied.append(selected)
        state.commit(selected, diagnostics)
    assert copied == code


@torch.no_grad()
def test_registered_payload_is_lossless_across_tokenization_boundaries():
    pointer = SparseLexicalPointer(hidden_dim=2, pointer_dim=2, window=8)
    pointer.q_proj.weight.copy_(torch.eye(2))
    pointer.k_proj.weight.copy_(torch.eye(2))
    pointer.gate_proj.weight.zero_()
    pointer.gate_proj.bias.fill_(10.0)
    pointer.logit_scale.fill_(3.0)

    source = torch.tensor([[3, 99]])
    mask = torch.ones_like(source)
    routes = torch.ones(1, 2, 1, dtype=torch.long)
    keys = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    registered = torch.tensor([[True, False]])
    payload = torch.tensor([[[27, 38, 41]]])
    memory = AnchorIndexedExactMemory.from_prompt(
        source,
        routes,
        mask,
        chunk_size=8,
        chunk_anchor_count=1,
        commit_scores=torch.zeros_like(source, dtype=torch.float32),
        key_vectors=keys,
        registered_key_mask=registered,
        registered_key_positions=torch.tensor([[0]]),
        registered_payload_ids=payload,
        registered_payload_lengths=torch.tensor([[3]]),
    )
    state = LexicalPointerDecodeState(source_length=2, max_copy_tokens=8)
    hidden = torch.tensor([[[1.0, 0.0]]])
    base = torch.full((1, 128), -5.0)
    copied = []
    for step in range(3):
        logits, diagnostics = state.prepare_logits(
            pointer,
            base,
            hidden,
            source,
            mask,
            min_gate=0.0,
            anchor_memory=memory,
            query_anchor_ids=torch.tensor([[1]]),
            max_candidate_chunks=1,
            full_scan_fallback=True,
            hard_copy=True,
            hard_copy_gate_threshold=0.9,
            hard_copy_pointer_threshold=0.9,
            hard_copy_margin_threshold=1.0,
        )
        assert diagnostics["mode"] == ("hard_copy" if step == 0 else "cursor")
        selected = int(logits.argmax(dim=-1))
        copied.append(selected)
        state.commit(selected, diagnostics)
    assert copied == [27, 38, 41]
    assert state.full_scan_count == 1


def test_token_budget_planner_controls_actual_tokens_and_exactly_resumes():
    trainer = _load_bridge_trainer()
    fractions = {
        "natural": 0.56,
        "factual": 0.40,
        "bridge_context": 0.02,
        "long_context": 0.02,
    }
    lengths = {
        "natural": 512,
        "factual": 512,
        "bridge_context": 4096,
        "long_context": 16384,
    }
    planner = trainer.TokenBudgetSourcePlanner(fractions)
    for _ in range(50000):
        source = planner.choose_source()
        planner.record(source, lengths[source])
    total = sum(planner.active_tokens.values())
    for source, target in fractions.items():
        observed = planner.active_tokens[source] / total
        assert abs(observed - target) < 0.001

    restored = trainer.TokenBudgetSourcePlanner(fractions)
    restored.restore(planner.state())
    assert restored.state() == planner.state()
    assert restored.choose_source() == planner.choose_source()


def test_architecture_transition_starts_exact_and_ramps_routing_stability():
    trainer = _load_bridge_trainer()
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_alpha=0.0,
            routing_logit_quantization=0.0,
        )
    )
    settings = {
        "architecture_transition_tokens": 100,
        "rope_transition_alpha_start": 0.0,
        "rope_transition_alpha_end": 1.0,
        "anchor_recency_bias_start": 4.0,
        "anchor_recency_bias_end": 0.125,
        "routing_logit_quantization_start": 0.0,
        "routing_logit_quantization_end": 1e-4,
    }
    initial = trainer.apply_architecture_transition(model, settings, 0)
    assert initial["rope_transition_alpha"] == 0.0
    assert initial["anchor_recency_bias"] == 4.0
    assert initial["routing_logit_quantization"] == 0.0

    middle = trainer.apply_architecture_transition(model, settings, 50)
    assert middle["rope_transition_alpha"] == 0.5
    assert middle["anchor_recency_bias"] == pytest.approx(2.0625)
    assert middle["routing_logit_quantization"] == pytest.approx(5e-5)

    final = trainer.apply_architecture_transition(model, settings, 100)
    assert final["rope_transition_alpha"] == 1.0
    assert final["anchor_recency_bias"] == 0.125
    assert final["routing_logit_quantization"] == pytest.approx(1e-4)


def test_architecture_transition_can_stage_rope_before_legacy_position_removal():
    trainer = _load_bridge_trainer()
    model = build_model(
        _base_config(
            model_type="fdt_v4",
            max_seq_len=128,
            use_rope=True,
            rope_transition_mode="phase",
            rope_transition_alpha=0.0,
            legacy_position_transition_max_len=128,
            legacy_position_scale=1.0,
        )
    )
    settings = {
        "architecture_transition_tokens": 100,
        "rope_transition_alpha_start": 0.0,
        "rope_transition_alpha_end": 1.0,
        "rope_transition_start_fraction": 0.0,
        "rope_transition_end_fraction": 0.5,
        "legacy_position_scale_start": 1.0,
        "legacy_position_scale_end": 0.0,
        "legacy_position_transition_start_fraction": 0.5,
        "legacy_position_transition_end_fraction": 1.0,
    }
    first = trainer.apply_architecture_transition(model, settings, 25)
    assert first["rope_transition_alpha"] == 0.5
    assert first["legacy_position_scale"] == 1.0
    middle = trainer.apply_architecture_transition(model, settings, 50)
    assert middle["rope_transition_alpha"] == 1.0
    assert middle["legacy_position_scale"] == 1.0
    final = trainer.apply_architecture_transition(model, settings, 100)
    assert final["rope_transition_alpha"] == 1.0
    assert final["legacy_position_scale"] == 0.0


def test_resume_model_config_allows_only_scheduled_transition_controls_to_move():
    trainer = _load_bridge_trainer()
    configured = _base_config(
        model_type="fdt_v4",
        use_rope=True,
        rope_transition_mode="output_blend",
        rope_transition_alpha=0.0,
        legacy_position_scale=1.0,
    ).__dict__
    checkpoint = dict(configured)
    checkpoint.update(
        rope_transition_alpha=0.01245,
        legacy_position_scale=1.0,
        anchor_recency_bias=3.5,
        routing_logit_quantization=1e-5,
    )
    assert trainer.resume_model_configs_compatible(checkpoint, configured)
    checkpoint["rope_transition_mode"] = "lerp"
    assert not trainer.resume_model_configs_compatible(checkpoint, configured)


def test_loop_detector_selects_only_third_or_later_ngram_closure():
    builder = _load_generated_prefix_builder()
    closes, occurrences = builder.closes_degenerate_ngram(
        [1, 2, 3, 1, 2, 3, 1, 2],
        3,
        order=3,
        prior_occurrences=2,
        window=96,
    )
    assert closes is True
    assert occurrences == 2

    closes, occurrences = builder.closes_degenerate_ngram(
        [1, 2, 3, 1, 2],
        3,
        order=3,
        prior_occurrences=2,
        window=96,
    )
    assert closes is False
    assert occurrences == 1


def test_recovery_row_penalizes_real_closure_at_clean_boundary_only():
    builder = _load_generated_prefix_builder()
    source = list(range(80))
    generated = [7, 8, 9, 7, 8, 9, 7, 8]
    row = builder.construct_recovery_row(
        source,
        16,
        generated,
        9,
        sequence_length=64,
        recovery_tokens=16,
        ngram_order=3,
        prior_occurrences=2,
    )
    assert row is not None
    boundary = int(row["recovery_boundary"])
    assert boundary == 24
    assert row["labels"][:boundary].eq(-100).all()
    assert int(row["labels"][boundary]) == source[boundary]
    assert int(row["loop_negative_ids"][boundary]) == 9
    assert int(row["loop_negative_mask"].sum()) == 1
    assert int(row["loop_negative_prior_occurrences"][boundary]) == 2


def test_trajectory_unlikelihood_row_marks_only_generated_loop_tokens():
    builder = _load_generated_prefix_builder()
    prompt = [3, 4, 5]
    generated = [7, 8, 9, 7, 8, 9, 7, 8, 9]
    events = [
        {
            "generated_index": 8,
            "negative_token": 9,
            "prior_occurrences": 2,
            "closing_ngram": [7, 8, 9],
        }
    ]
    row = builder.construct_trajectory_unlikelihood_row(
        prompt,
        generated,
        events,
        sequence_length=32,
        ngram_order=3,
    )
    assert row is not None
    position = len(prompt) + 8
    assert int(row["loop_unlikelihood_only"]) == 1
    assert row["labels"].eq(-100).all()
    assert int(row["loop_negative_mask"].sum()) == 1
    assert int(row["loop_negative_ids"][position]) == 9
    assert int(row["input_ids"][position]) == 9


def test_batched_generated_prefix_builder_finds_real_loop_per_row():
    builder = _load_generated_prefix_builder()

    class FakeConfig:
        rope_transition_mode = "output_blend"
        rope_transition_alpha = 0.25

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.config = FakeConfig()

        def forward(self, input_ids, attention_mask):
            logits = torch.zeros(
                input_ids.size(0), input_ids.size(1), 10, device=input_ids.device
            )
            next_token = (4, 5, 6)[(input_ids.size(1) - 2) % 3]
            logits[:, -1, next_token] = 10.0 + self.marker
            return {"logits": logits}

    results = builder.generate_batch_to_loop_closure(
        FakeModel(),
        [[1, 2], [2, 3], [3, 4]],
        max_new_tokens=12,
        eos_token_id=9,
        ngram_order=3,
        prior_occurrences=2,
        ngram_window=12,
        device=torch.device("cpu"),
    )
    assert all(result is not None for result in results)
    assert all(result["negative_token"] == 6 for result in results)
    assert all(
        result["generated_prefix"] == [4, 5, 6, 4, 5, 6, 4, 5]
        for result in results
    )


def test_gradient_norm_finiteness_does_not_stack_devices():
    trainer = _load_bridge_trainer()
    assert trainer.gradient_norms_are_finite(torch.tensor(0.7), torch.tensor(0.0))
    assert not trainer.gradient_norms_are_finite(
        torch.tensor(float("inf")), torch.tensor(0.0)
    )
