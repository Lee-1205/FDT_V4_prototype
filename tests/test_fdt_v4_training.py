import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = load_script("fdt_v4_curriculum_test_module", "train_fdt_v4_curriculum.py")
supervisor = load_script("luna_fdt_v4_supervisor_test_module", "luna_fdt_v4_supervisor.py")


def tiny_v4_config():
    from fdt_rlm.config import ModelConfig

    return ModelConfig(
        vocab_size=31,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v4",
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=16,
        dropout=0.0,
        use_rope=True,
        anchor_layer_indices=[0],
        num_anchors=8,
        top_k=2,
        router_dim=16,
        local_attention_window=4,
        exact_memory_enabled=True,
        exact_memory_mode="copy",
    )


def test_weighted_eos_loss_is_finite_and_uses_eos_weight():
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    ids = torch.tensor([[2, 3, 1, 4]])
    mask = torch.ones_like(ids)
    loss = trainer.weighted_lm_loss(logits, ids, mask, 0, 1, 2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_chunked_lm_loss_matches_full_logits_loss():
    from fdt_rlm.models import build_model

    torch.manual_seed(20260823)
    model = build_model(tiny_v4_config())
    hidden = torch.randn(1, 9, model.config.dim, requires_grad=True)
    labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1, 0]])
    mask = torch.ones_like(labels)
    full = trainer.weighted_lm_loss(
        model.lm_head(hidden), labels, mask, 0, 1, 2.0
    )
    chunked = trainer.chunked_weighted_lm_loss(
        model, hidden, labels, mask, 0, 1, 2.0, 3
    )
    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-6)


def test_exact_objective_is_never_called_on_generic_lm_batch(monkeypatch):
    from fdt_rlm.models import build_model

    model = build_model(tiny_v4_config())
    calls = []
    original = model.exact_memory_loss

    def wrapped(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "exact_memory_loss", wrapped)
    ids = torch.tensor([[2, 3, 4, 5, 6, 1]])
    mask = torch.ones_like(ids)
    output = model(ids, attention_mask=mask)
    lm_loss = trainer.weighted_lm_loss(output["logits"], ids, mask, 0, 1, 2.0)
    lm_loss.backward()
    assert calls == []

    model.zero_grad(set_to_none=True)
    output = model(ids, attention_mask=mask)
    exact = model.exact_memory_loss(output["hidden"], ids, ids, mask)
    exact.loss.backward()
    assert calls == [True]


def test_exact_pointer_uses_a_dedicated_learning_rate_and_clip_domain():
    from fdt_rlm.models import build_model

    model = build_model(tiny_v4_config())
    groups, base_parameters, exact_parameters = trainer.optimizer_parameter_groups(
        model,
        {
            "learning_rate": 2.5e-9,
            "weight_decay": 0.01,
            "exact_pointer_learning_rate": 1e-4,
            "exact_pointer_weight_decay": 0.0,
        },
    )

    assert base_parameters
    assert exact_parameters
    assert {group["group_name"] for group in groups} == {"base_model", "exact_pointer"}
    by_name = {group["group_name"]: group for group in groups}
    assert by_name["base_model"]["lr"] == 2.5e-9
    assert by_name["exact_pointer"]["lr"] == 1e-4
    assert by_name["exact_pointer"]["weight_decay"] == 0.0
    assert not ({id(parameter) for parameter in base_parameters} & {id(parameter) for parameter in exact_parameters})


def test_detached_exact_objective_updates_pointer_but_not_base_model():
    from fdt_rlm.models import build_model

    model = build_model(tiny_v4_config())
    ids = torch.tensor([[1, 4, 7, 9, 4, 7, 9, 1]])
    labels = ids.clone()
    labels[:, :4] = -100
    copy_target_mask = torch.zeros_like(ids)
    copy_target_mask[:, 4:7] = 1
    copy_source_positions = torch.full_like(ids, -1)
    copy_source_positions[:, 4:7] = torch.tensor([[1, 2, 3]])
    batch = {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.ones_like(ids),
        "prompt_mask": labels.eq(-100),
        "source_boundary": torch.tensor([4]),
        "copy_source_positions": copy_source_positions,
        "copy_target_mask": copy_target_mask,
    }
    loss, _ = trainer.exact_copy_objective(model, batch, 1.0, detach_hidden=True)
    loss.backward()
    pointer_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("exact_pointer.")
    ]
    base_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("exact_pointer.")
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in pointer_grads
    )
    assert all(grad is None or grad.abs().sum() == 0 for grad in base_grads)


def test_trainer_runs_exact_objective_after_base_microbatch_loop():
    source = (ROOT / "scripts" / "train_fdt_v4_curriculum.py").read_text(encoding="utf-8")
    loop_start = source.index("for micro_index in range(accumulation):")
    exact_start = source.index("if step % int(train_cfg.get(\"exact_batch_every_steps\", 1)) == 0:")
    clip_start = source.index("base_grad_norm = torch.nn.utils.clip_grad_norm_")

    assert loop_start < exact_start < clip_start
    assert "prevents a second full-model graph" in source[loop_start:clip_start]
    assert '"diagnostic_metrics": diagnostic_values' in source
    assert "total_loss = sum(objective_values.values())" in source


def test_row_pool_resume_is_deterministic(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "shards" / "train").mkdir(parents=True)
    payload = {
        "input_ids": torch.arange(48).view(6, 8),
        "attention_mask": torch.ones(6, 8, dtype=torch.bool),
        "labels": torch.arange(48).view(6, 8),
    }
    torch.save(payload, dataset / "shards" / "train" / "shard.pt")
    first = trainer.RowPool(dataset, "train", 17)
    first.next_batch(3)
    state = first.state()
    expected = first.next_batch(2)
    second = trainer.RowPool(dataset, "train", 17)
    second.restore(state)
    actual = second.next_batch(2)
    torch.testing.assert_close(expected["input_ids"], actual["input_ids"])
    torch.testing.assert_close(expected["labels"], actual["labels"])


def test_source_cycle_is_deterministic_and_keeps_rare_8k_rows():
    fractions = {"natural": 0.58, "factual": 0.40, "long_context": 0.02}
    first = trainer.deterministic_source_cycle(fractions, slots=100)
    second = trainer.deterministic_source_cycle(fractions, slots=100)
    assert first == second
    assert first.count("natural") == 58
    assert first.count("factual") == 40
    assert first.count("long_context") == 2


def test_metadata_preflight_is_426m_class_and_24k():
    from fdt_rlm.config import load_yaml_like

    raw = load_yaml_like(ROOT / "configs" / "fdt_v4_main_426m.yaml")
    config = trainer.model_config_from_yaml(raw)
    report = trainer.metadata_model_preflight(config)
    assert report["meta_device"] is True
    assert report["parameter_class"] == "426M"
    assert report["vocab_size"] == 24576


def test_exact_contract_requires_explicit_labels_and_prompt_fields(tmp_path):
    dataset = tmp_path / "exact"
    (dataset / "shards" / "train").mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    ids = torch.tensor([[4, 5, 6, 2]])
    torch.save({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, dataset / "shards" / "train" / "shard.pt")
    with pytest.raises(ValueError, match="labels"):
        trainer.preflight_dataset_contract({"exact_copy": dataset}, {"split": "train"})


def test_exact_contract_rejects_supervised_target_inside_source_boundary(tmp_path):
    dataset = tmp_path / "exact_inside_prompt"
    (dataset / "shards" / "train").mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    ids = torch.tensor([[1, 10, 10, 2]])
    payload = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([[-100, -100, 10, -100]]),
        "prompt_mask": torch.tensor([[1, 1, 0, 0]]),
        "source_boundary": torch.tensor([3]),
        "copy_source_positions": torch.tensor([[-1, -1, 1, -1]]),
        "copy_target_mask": torch.tensor([[0, 0, 1, 0]]),
    }
    torch.save(payload, dataset / "shards" / "train" / "shard.pt")

    with pytest.raises(ValueError, match="outside the prompt boundary"):
        trainer.preflight_dataset_contract(
            {"exact_copy": dataset}, {"split": "train"}
        )


def test_routing_diagnostics_exposes_dead_anchor_safety_signal():
    output = {
        "anchor_stats": [
            {
                "entropy_normalized": torch.tensor(0.6),
                "dead_anchor_fraction": torch.tensor(0.02),
                "top1_membership": torch.tensor(0.7),
            },
            {
                "entropy_normalized": torch.tensor(0.4),
                "dead_anchor_fraction": torch.tensor(0.0),
                "top1_membership": torch.tensor(0.5),
            },
        ]
    }
    diagnostics = trainer.routing_diagnostics(output)
    assert diagnostics["entropy_normalized"] == pytest.approx(0.5)
    assert diagnostics["dead_anchor_fraction"] == pytest.approx(0.02)
    assert diagnostics["top1_membership"] == pytest.approx(0.6)


def test_routing_dead_anchor_gate_uses_accumulated_mask_union():
    first = [torch.tensor([True, False, False, True])]
    second = [torch.tensor([False, True, False, True])]
    third = [torch.tensor([False, False, True, False])]
    merged = trainer.merge_active_anchor_masks(None, first)
    merged = trainer.merge_active_anchor_masks(merged, second)
    assert trainer.dead_anchor_fraction_from_masks(merged) == 0.25
    merged = trainer.merge_active_anchor_masks(merged, third)
    assert trainer.dead_anchor_fraction_from_masks(merged) == 0.0


def test_v20_conversion_allowlist_leaves_rope_and_exact_new():
    from fdt_rlm.models import build_model

    model = build_model(tiny_v4_config())
    source = {name: value.detach().clone() for name, value in model.state_dict().items() if trainer._key_allowed(name)}
    payload = {"model_config": {"model_type": "fdt_v3", "vocab_size": 24576, "pad_token_id": 0, "eos_token_id": 2}, "model_state_dict": source}
    manifest = trainer.convert_v20_state_dict(model, payload)
    assert manifest["anchor_transfer_verified"] is True
    assert manifest["rope_and_exact_memory_initialized_new"] is True
    assert "blocks.0.anchor.q_proj.weight" in manifest["converted_keys"]
    assert all("exact_pointer" not in name for name in manifest["converted_keys"])


def test_handoff_schema_separates_evaluation_and_incident(tmp_path):
    run = ROOT / "runs" / "test_fdt_v4_handoff_schema"
    run.mkdir(parents=True, exist_ok=True)
    payload = {"stage_status": "PAUSED", "optimizer_state_included": True, "model_config": {"model_type": "fdt_v4"}, "train_config": {"output_dir": str(run)}, "source_states": {}, "model_state_dict": {}, "optimizer_state_dict": {}}
    torch.save(payload, run / "latest_recovery.pt")
    torch.save(payload, run / "latest.pt")
    raw = {"terra": {"handoff_path": "terra_handoff.json"}}
    destination = supervisor.write_handoff(raw, run, run / "latest_recovery.pt", "PAUSED", classification="incident", severity="critical", trigger="test")
    handoff = json.loads(destination.read_text(encoding="utf-8"))
    assert handoff["status"] == "ABNORMAL"
    assert handoff["handoff_type"] == "INCIDENT"
    for path in run.iterdir():
        path.unlink()
    run.rmdir()


def test_main_yaml_has_only_426m_architecture():
    from fdt_rlm.config import load_yaml_like

    raw = load_yaml_like(ROOT / "configs" / "fdt_v4_main_426m.yaml")
    model = raw["model"]
    assert model["dim"] == 1216
    assert model["n_layers"] == 20
    assert model["n_heads"] == 19
    assert model["local_attention_window"] == 64
    assert model["num_anchors"] == 256
    assert model["top_k"] == 8
    assert model["router_dim"] == 256
    assert model["anchor_layer_indices"] == list(range(0, 20, 2))
    assert model["max_seq_len"] == 16384
    assert raw["train"]["exact_pointer_learning_rate"] > raw["train"]["learning_rate"]
    assert raw["train"]["exact_batch_every_steps"] == 1
    assert raw["train"]["detach_exact_hidden"] is True
    assert model["exact_memory_commit_threshold"] == 0.5
    assert raw["train"]["parent_checkpoint"].endswith(
        "fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2/latest.pt"
    )
    assert raw["train"]["eos_loss_weight"] == 2.0
    assert raw["train"]["generated_prefix_min_step"] >= 1
    assert raw["train"]["require_generated_prefix_recovery"] is True
    assert raw["data"]["generated_prefix_dir"]
    assert raw["train"]["activation_checkpointing"] is True
    assert raw["train"]["dead_anchor_fraction_limit"] == 0.01
    assert raw["train"]["require_long_context_curriculum"] is True
    assert raw["train"]["long_context_min_sequence_length"] == 8192
    assert raw["train"]["long_context_require_8k_bucket"] is True
    assert raw["train"]["long_context_require_16k_shard"] is True
    assert raw["train"]["require_routing_diagnostics"] is True
    assert raw["train"]["source_batch_fractions"]["long_context"] == 0.02


def test_generated_prefix_contract_and_unlikelihood_loss(tmp_path):
    dataset = tmp_path / "generated_prefix"
    (dataset / "shards" / "train").mkdir(parents=True)
    (dataset / "manifest.json").write_text('{"rows": 1}', encoding="utf-8")
    ids = torch.tensor([[4, 5, 4, 5, 7, 1]])
    labels = ids.clone()
    labels[:, :3] = -100
    negative_ids = torch.zeros_like(ids)
    negative_ids[:, 3:] = torch.tensor([[6, 6, 6]])
    negative_mask = torch.zeros_like(ids)
    negative_mask[:, 3:] = 1
    payload = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
        "loop_negative_ids": negative_ids,
        "loop_negative_mask": negative_mask,
    }
    torch.save(payload, dataset / "shards" / "train" / "shard.pt")
    report = trainer.preflight_dataset_contract(
        {"generated_prefix": dataset},
        {"split": "train"},
    )
    assert report["sources"]["generated_prefix"]["shards"][0]["generated_prefix_contract"] is True

    class FakeModel:
        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 11, requires_grad=True)
            return {"logits": logits}

    total, metrics = trainer.generated_prefix_recovery_objective(
        FakeModel(),
        payload,
        pad_token_id=0,
        eos_token_id=1,
        eos_weight=2.0,
        recovery_weight=0.01,
        unlikelihood_weight=0.01,
    )
    assert torch.isfinite(total)
    assert torch.isfinite(metrics["recovery_lm"])
    assert torch.isfinite(metrics["loop_unlikelihood"])
    total.backward()


def test_main_architecture_validator_rejects_probe_shape():
    config = tiny_v4_config()
    with pytest.raises(ValueError, match="architecture mismatch"):
        trainer.validate_main_architecture(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_seq_len", 8192),
        ("use_rope", False),
        ("exact_memory_mode", "store"),
        ("exact_memory_copy_cursor", False),
        ("exact_memory_commit_threshold", 0.0),
        ("generation_ngram_hard_block_after", 0),
    ],
)
def test_main_architecture_validator_rejects_missing_core_contract(field, value):
    from fdt_rlm.config import load_yaml_like

    raw = load_yaml_like(ROOT / "configs" / "fdt_v4_main_426m.yaml")
    config = trainer.model_config_from_yaml(raw)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="architecture mismatch"):
        trainer.validate_main_architecture(config)


def test_atomic_json_has_no_temporary_residue(tmp_path):
    path = tmp_path / "result.json"
    trainer.atomic_json(path, {"status": "READY"})
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_luna_escalation_policy_is_machine_readable_and_single_model():
    policy = supervisor.ESCALATION_POLICY
    assert policy["routine_owner"] == "luna"
    assert policy["abnormality_owner"] == "terra"
    assert policy["major_remedy_owner"] == "sol"
    assert policy["luna_may_redesign"] is False
    assert policy["single_trainable_model"] == "fdt_v4_main_426m"
    assert "116m" in policy["forbidden_model_variants"]


def test_luna_supervisor_state_is_separate_from_trainer_output(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = runs / "main"
    state = runs / "main_supervisor"
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    assert supervisor.supervisor_state_path(run, None) == state
    assert supervisor.supervisor_state_path(run, state) == state
    with pytest.raises(ValueError, match="must be separate"):
        supervisor.supervisor_state_path(run, run)


def test_dataset_preflight_rejects_validation_only_contract(tmp_path):
    dataset = tmp_path / "validation_only"
    (dataset / "shards" / "validation").mkdir(parents=True)
    (dataset / "manifest.json").write_text('{"rows": 1}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no train shards"):
        trainer.preflight_dataset_contract(
            {"natural": dataset},
            {"split": "train"},
        )


def test_trainer_passes_explicit_labels_to_exact_loss_and_rejects_all_token_labels():
    captured = []

    class FakeModel:
        def __call__(self, input_ids, attention_mask, return_logits=True):
            return {"hidden": torch.ones(input_ids.size(0), input_ids.size(1), 4, requires_grad=True)}

        def exact_memory_loss(self, hidden, input_ids, labels, attention_mask, **kwargs):
            captured.append((labels.detach().clone(), kwargs))
            return SimpleNamespace(loss=hidden.sum() * 0.0 + 1.0)

    ids = torch.tensor([[1, 4, 5, 4, 5]])
    labels = ids.clone()
    labels[:, :3] = -100
    batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids), "prompt_mask": torch.tensor([[1, 1, 1, 0, 0]]), "source_boundary": torch.tensor([3]), "copy_source_positions": torch.tensor([[-1, -1, -1, 1, 2]]), "copy_target_mask": torch.tensor([[0, 0, 0, 1, 1]])}
    trainer.exact_copy_objective(FakeModel(), batch, 1.0)
    torch.testing.assert_close(captured[0][0], labels)
    torch.testing.assert_close(captured[0][1]["copy_source_positions"], batch["copy_source_positions"])
    torch.testing.assert_close(captured[0][1]["copy_target_mask"], batch["copy_target_mask"])
    assert not torch.equal(captured[0][0], ids)
    all_token = dict(batch, labels=ids.clone(), prompt_mask=torch.zeros_like(ids), copy_target_mask=torch.zeros_like(ids))
    with pytest.raises(ValueError, match="all-token labels"):
        trainer.exact_copy_objective(FakeModel(), all_token, 1.0)
