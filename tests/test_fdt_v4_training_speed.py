from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_fdt_v4_curriculum_speed_test",
    ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py",
)
speed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(speed)


class _HeadModel:
    def __init__(self, hidden_dim: int, vocab_size: int):
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = type("Config", (), {"lm_logit_clip": 30.0})()


class _CheckpointModel:
    def __init__(self):
        self.enabled = None

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.enabled = enabled


class _Pool:
    def __init__(self, marker: int, sequence_length: int):
        self.marker = marker
        self.sequence_length = sequence_length
        self.cursor = 0

    def next_batch(self, batch_size: int):
        assert batch_size == 1
        value = self.marker + self.cursor
        self.cursor += 1
        ids = torch.full((1, self.sequence_length), value, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def test_short_checkpoint_policy_keeps_8k_and_16k_checkpointed():
    model = _CheckpointModel()
    config = {
        "activation_checkpointing": True,
        "activation_checkpointing_min_sequence_length": 8192,
        "lm_loss_checkpointing_min_sequence_length": 8192,
    }
    assert not speed.set_sequence_gradient_checkpointing(model, 512, config)
    assert model.enabled is False
    assert speed.set_sequence_gradient_checkpointing(model, 8192, config)
    assert model.enabled is True
    assert speed.set_sequence_gradient_checkpointing(model, 16384, config)
    assert speed.lm_loss_checkpointing_enabled(8192, config)
    assert not speed.lm_loss_checkpointing_enabled(512, config)


def test_grouped_loss_matches_sum_of_single_row_losses_and_gradients():
    torch.manual_seed(7)
    model = _HeadModel(6, 13)
    hidden = torch.randn(2, 8, 6, requires_grad=True)
    labels = torch.randint(0, 13, (2, 8))
    attention = torch.ones(2, 8, dtype=torch.long)
    attention[1, -2:] = 0

    grouped = speed.chunked_weighted_lm_loss(
        model, hidden, labels, attention, 0, 2, 2.0, 4, False
    )
    grouped.backward()
    grouped_hidden_grad = hidden.grad.detach().clone()
    grouped_weight_grad = model.lm_head.weight.grad.detach().clone()

    model.lm_head.weight.grad = None
    hidden_single = hidden.detach().clone().requires_grad_(True)
    singles = sum(
        speed.chunked_weighted_lm_loss(
            model,
            hidden_single[index : index + 1],
            labels[index : index + 1],
            attention[index : index + 1],
            0,
            2,
            2.0,
            4,
            False,
        )
        for index in range(2)
    )
    singles.backward()

    torch.testing.assert_close(grouped, singles, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grouped_hidden_grad, hidden_single.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grouped_weight_grad, model.lm_head.weight.grad, rtol=1e-6, atol=1e-6)


def test_short_batch2_preserves_eight_row_source_order_and_long_singletons():
    pools = {"a": _Pool(10, 512), "b": _Pool(20, 512), "long": _Pool(30, 8192)}
    cycle = ["a", "b", "a", "long", "b", "a", "b", "a"]
    groups = speed.planned_base_forward_groups(pools, cycle, 0, 8, 2, 512)

    assert [batch["input_ids"].shape[0] for batch in groups] == [2, 1, 1, 2, 2]
    assert [batch["input_ids"].shape[1] for batch in groups] == [512, 512, 8192, 512, 512]
    flattened = []
    for batch in groups:
        flattened.extend(batch["input_ids"][:, 0].tolist())
    assert flattened == [10, 20, 11, 30, 21, 12, 22, 13]


def test_speed_config_selects_exact_short_batch_runtime():
    from fdt_rlm.config import load_yaml_like

    raw = load_yaml_like(ROOT / "configs" / "fdt_v4_main_426m_speed_r1.yaml")
    train = raw["train"]
    assert train["batch_size"] == 1
    assert train["grad_accum_steps"] == 8
    assert train["short_sequence_batch_size"] == 1
    assert train["short_sequence_max_length"] == 512
    assert train["activation_checkpointing_min_sequence_length"] == 8192
    assert train["lm_loss_checkpointing_min_sequence_length"] == 8192
    assert train["validation_batches"] == 8
    assert train["overfit_gate_version"] == "fixed_validation_v2"


def test_exact_copy_curriculum_advances_and_caps_each_row_independently():
    assert [speed.exact_curriculum_cap(value) for value in (0.0, 0.1, 0.25, 0.5, 0.75)] == [4, 8, 16, 32, 64]
    batch = {
        "copy_target_mask": torch.tensor(
            [[0, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0]], dtype=torch.long
        )
    }
    result = speed.cap_copy_targets(batch, 2)["copy_target_mask"]
    assert result.tolist() == [
        [False, True, True, False, False, False],
        [True, False, True, False, False, False],
    ]


def test_generated_prefix_curriculum_ramps_without_overshoot():
    assert speed.linear_ramp(0.0, 0.02, 0.20) == 0.0
    assert speed.linear_ramp(0.11, 0.02, 0.20) == pytest.approx(0.5)
    assert speed.linear_ramp(0.25, 0.02, 0.20) == 1.0


def test_resume_payload_is_released_before_training_loop():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    release = source.index("resume_payload = None")
    collect = source.index("gc.collect()", release)
    loop = source.index("while tokens_seen - start_tokens < target_tokens:")
    assert release < collect < loop


def test_uncheckpointed_anchor_stats_keep_active_masks_for_safety_gate():
    output = {
        "anchor_stats": [
            SimpleNamespace(
                entropy=torch.tensor(0.5),
                load_prob=torch.tensor([0.25, 0.0, 0.75]),
                top1_membership=torch.tensor(0.8),
                membership=torch.ones(1, 3),
            )
        ]
    }
    metrics = speed.routing_diagnostics(output)
    assert metrics["dead_anchor_fraction"] == pytest.approx(1.0 / 3.0)
    assert len(metrics["active_anchor_masks"]) == 1
    torch.testing.assert_close(
        metrics["active_anchor_masks"][0], torch.tensor([True, False, True])
    )


def test_resume_routing_window_waits_for_a_full_fresh_window():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    increment = source.index("routing_window_observed_steps += 1")
    gate = source.index("window_due = routing_window_observed_steps >= routing_window_steps")
    reset = source.index("routing_window_observed_steps = 0", gate)
    assert increment < gate < reset
    assert "if (step + 1) % routing_window_steps == 0:" not in source


def test_device_resident_routing_diagnostics_defer_scalar_conversion():
    output = {
        "anchor_stats": [
            SimpleNamespace(
                entropy=torch.tensor(0.4),
                load_prob=torch.tensor([0.2, 0.0, 0.8]),
                top1_membership=torch.tensor(0.7),
                membership=torch.ones(1, 3),
            )
        ]
    }
    metrics = speed.routing_diagnostics(output, device_resident=True)
    assert isinstance(metrics["entropy_normalized"], torch.Tensor)
    assert isinstance(metrics["dead_anchor_fraction"], torch.Tensor)
    assert isinstance(metrics["top1_membership"], torch.Tensor)
    dead = speed.dead_anchor_fraction_from_masks(
        metrics["active_anchor_masks"], device_resident=True
    )
    assert isinstance(dead, torch.Tensor)
    assert float(dead) == pytest.approx(1.0 / 3.0)


def test_resume_throughput_uses_only_session_tokens():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    assert "session_start_tokens = tokens_seen" in source
    assert "session_tokens = tokens_seen - session_start_tokens" in source
    assert "tps = session_tokens / session_elapsed" in source
    assert '"throughput_contract": "session_local_v2"' in source


def test_zero_weight_prefix_preserves_cursor_without_building_a_graph():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    advance = source.index('host_prefix_batch = pools["generated_prefix"].next_batch')
    positive_gate = source.index("if prefix_scale > 0.0:", advance)
    move = source.index("prefix_batch = move_batch(host_prefix_batch, device)", positive_gate)
    assert advance < positive_gate < move
    assert 'diagnostic_values["generated_prefix_zero_weight_skipped"] = 1.0' in source


def test_loss_finiteness_is_synchronized_once_before_clipping():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    assert "finite_loss_flags.append(torch.isfinite(micro_total.detach()))" in source
    assert "if finite_loss_flags and not bool(torch.stack(finite_loss_flags).all()):" in source
    assert 'if not torch.isfinite(micro_total):' not in source


def test_live_trainer_selects_the_verified_fast_backend_by_default():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum_speed.py").read_text(
        encoding="utf-8"
    )
    assert 'train_cfg.get("deterministic_algorithms", False)' in source
    assert '"cuda_backend_contract": "verified_fast_equivalent_v1"' in source
