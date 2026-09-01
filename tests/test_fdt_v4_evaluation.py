import importlib.util
import json
from pathlib import Path

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_config():
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
        num_anchors=8,
        top_k=2,
        router_dim=16,
        anchor_layer_indices=[0],
        local_attention_window=8,
        anchor_scan_chunk_size=4,
        exact_memory_enabled=True,
        exact_memory_mode="copy",
        exact_pointer_chunk_size=4,
        exact_pointer_chunk_anchors=2,
        exact_pointer_candidate_chunks=2,
    )


def make_checkpoint(path: Path):
    model = build_model(tiny_config()).eval()
    torch.save({"model_config": vars(model.config), "model_state_dict": model.state_dict()}, path)


def test_exact_case_marks_context_limits_and_all_ablations():
    module = load_module("evaluate_fdt_v4_test", "scripts/evaluate_fdt_v4.py")
    model = build_model(tiny_config()).eval()
    report = module.evaluate_exact_memory(model, model.config, None)
    assert report["status"] == "ok"
    cell = report["matrix"]["4"]["512"]
    assert cell["status"] == "unsupported"
    modes = report["matrix"]["4"]["512"]
    assert "reason" in modes
    available = report["matrix"]["4"]["64"]
    assert set(available["modes"]) == {"off", "store", "retrieve", "copy"}


def test_exact_ablation_semantics_do_not_mix_retrieve_logits():
    module = load_module("evaluate_fdt_v4_ablations_test", "scripts/evaluate_fdt_v4.py")
    model = build_model(tiny_config()).eval()
    prompt, _, _ = module.exact_case_prompt(model.config.vocab_size, 4, 8)
    off, off_trace = module.generate(model, model.config, prompt, 1, "off")
    store, store_trace = module.generate(model, model.config, prompt, 1, "store")
    retrieve, retrieve_trace = module.generate(model, model.config, prompt, 1, "retrieve")
    copy, copy_trace = module.generate(model, model.config, prompt, 1, "copy")
    assert off_trace[0]["exact_memory_built"] is False
    assert store_trace[0]["exact_memory_built"] is True
    assert store_trace[0]["logits_mixed"] is False
    assert retrieve_trace[0]["logits_mixed"] is False
    assert copy_trace[0]["logits_mixed"] is True
    assert off == store


def test_intermediate_output_blend_uses_full_recompute_generation():
    module = load_module("evaluate_fdt_v4_output_blend_test", "scripts/evaluate_fdt_v4.py")
    config = tiny_config()
    config.rope_transition_mode = "output_blend"
    config.rope_transition_alpha = 0.25
    config.legacy_position_scale = 1.0
    model = build_model(config).eval()
    generated, trace = module.generate(model, config, [2, 3, 4], 3, "off")
    assert generated
    assert all(item["decode_backend"] == "full_recompute" for item in trace)
    cache_result = module.cache_integrity(
        model,
        config,
        [{"input_ids": [2, 3, 4, 5]}],
    )
    assert cache_result["status"] == "unsupported"
    assert "alpha endpoints" in cache_result["reason"]


def test_cpu_fp32_evaluation_writes_atomic_result(tmp_path):
    module = load_module("evaluate_fdt_v4_cpu_test", "scripts/evaluate_fdt_v4.py")
    checkpoint = tmp_path / "tiny.pt"
    dataset = tmp_path / "rows.pt"
    output = tmp_path / "result.json"
    make_checkpoint(checkpoint)
    torch.save({"input_ids": torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])}, dataset)
    result = module.evaluate(checkpoint, output, dataset_path=dataset, dataset_limit=2)
    assert output.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert result["official_evaluation"] == {"quantization": "none", "dtype": "float32", "device": "cpu", "gpu_launched": False}
    assert result["teacher_forced"]["status"] == "ok"
    assert result["cache_full_recompute"]["status"] == "ok"
    assert result["integrity_audit"]["status"] == "PASS"
    assert output.with_name(output.name + ".sha256").read_text(encoding="ascii").strip() == result["output"]["sha256"]
    assert set(result["audit_axes"]) == {
        "ARCHITECTURE",
        "EXACT_MEMORY",
        "GENERATION_STABILITY",
        "LONG_CONTEXT",
        "INFERENCE_INTEGRITY",
        "PERFORMANCE",
        "QUALITY",
        "REPRODUCIBILITY",
    }
    assert result["audit_axes"]["PERFORMANCE"]["status"] == "NOT_TESTED"


def test_json_dataset_without_tokenizer_cannot_silently_evaluate_zero_rows(tmp_path):
    module = load_module("evaluate_fdt_v4_empty_rows_test", "scripts/evaluate_fdt_v4.py")
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({"prompt": "Question:", "target": " answer"}) + "\n", encoding="utf-8")
    rows = module.load_dataset(dataset, tokenizer=None, limit=1)
    assert rows == []
    try:
        module.require_nonempty_rows(rows, dataset, "evaluation")
    except ValueError as exc:
        assert "require --tokenizer" in str(exc)
    else:
        raise AssertionError("zero-row evaluation dataset unexpectedly accepted")


def test_exact_memory_audit_can_be_explicitly_deferred(tmp_path):
    module = load_module("evaluate_fdt_v4_deferred_exact_test", "scripts/evaluate_fdt_v4.py")
    checkpoint = tmp_path / "tiny.pt"
    dataset = tmp_path / "rows.pt"
    output = tmp_path / "result.json"
    make_checkpoint(checkpoint)
    torch.save({"input_ids": torch.tensor([[2, 3, 4, 5]])}, dataset)
    result = module.evaluate(
        checkpoint,
        output,
        dataset_path=dataset,
        dataset_limit=1,
        run_exact_memory=False,
    )
    assert result["exact_memory"]["status"] == "unsupported"
    assert "separately preserved" in result["exact_memory"]["reason"]


def test_terra_handoff_checks_digest_and_emits_integrity(tmp_path):
    evaluator = load_module("evaluate_fdt_v4_terra_test", "scripts/evaluate_fdt_v4.py")
    terra = load_module("terra_fdt_v4_test", "scripts/terra_fdt_v4_evaluator.py")
    checkpoint = tmp_path / "tiny.pt"
    dataset = tmp_path / "rows.pt"
    handoff = tmp_path / "handoff.json"
    result = tmp_path / "terra-result.json"
    make_checkpoint(checkpoint)
    torch.save({"input_ids": torch.tensor([[2, 3, 4, 5]])}, dataset)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text('{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"a":2,"b":3,"c":4,"d":5},"unk_token":"<pad>"}}', encoding="utf-8")
    handoff.write_text(json.dumps({
        "status": "READY",
        "handoff_type": "EVALUATION",
        "handoff_id": "cpu-test",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": evaluator.sha256_file(checkpoint),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_json_sha256": evaluator.sha256_file(tokenizer_json),
        "tensor_dataset": str(dataset),
        "tensor_dataset_sha256": evaluator.sha256_file(dataset),
        "dataset_limit": 1,
        "comparator_checkpoint": str(checkpoint),
        "comparator_checkpoint_sha256": evaluator.sha256_file(checkpoint),
        "repetition_tensor_dataset": str(dataset),
        "repetition_tensor_dataset_sha256": evaluator.sha256_file(dataset),
        "bootstrap_samples": 8,
    }), encoding="utf-8")
    payload = terra.run_handoff(handoff, result)
    assert payload["status"] == "RESULT"
    assert payload["integrity_digest"] == terra.sha256_file(result)
    assert result.is_file()
    assert result.with_name(result.name + ".sha256").read_text(encoding="ascii").strip() == payload["integrity_digest"]
    evaluation = json.loads(Path(payload["evaluation"]["path"]).read_text(encoding="utf-8"))
    assert evaluation["paired_bootstrap"]["status"] == "ok"
    assert evaluation["independent_repetition"]["status"] == "ok"


def test_luna_major_change_routes_to_sol_and_116m_is_blocked(tmp_path):
    terra = load_module("terra_fdt_v4_classification_test", "scripts/terra_fdt_v4_evaluator.py")
    incident = load_module("terra_fdt_v4_incident_test", "scripts/terra_fdt_v4_incident.py")
    decision = terra.classify_luna_abnormality({
        "status": "ABNORMAL",
        "event": "loss regression",
        "requested_change": "reduce to 116M parameters",
        "proposed_parameter_count": 116_000_000,
    })
    assert decision["route"] == "Sol"
    assert decision["change_class"] == "model_scale"
    assert not decision["auto_apply"]
    handoff = tmp_path / "luna.json"
    result = tmp_path / "incident.json"
    handoff.write_text(json.dumps({"status": "ABNORMAL", "handoff_type": "INCIDENT", "event": "CUDA OOM"}), encoding="utf-8")
    payload = incident.consume_luna_handoff(handoff, result)
    assert payload["decision"]["change_class"] == "runtime_config"
    assert payload["model_policy"]["forbid_116m"]
    assert payload["integrity_digest"] == incident.sha256_file(result)
    assert result.with_name(result.name + ".sha256").read_text(encoding="ascii").strip() == payload["integrity_digest"]


def test_unified_handoff_rejects_legacy_status_or_json_dataset(tmp_path):
    terra = load_module("terra_fdt_v4_manifest_test", "scripts/terra_fdt_v4_evaluator.py")
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"status": "READY", "checkpoint": str(tmp_path / "none.pt")}), encoding="utf-8")
    try:
        terra.load_ready_handoff(handoff)
    except ValueError as exc:
        assert "handoff_type EVALUATION" in str(exc)
    else:
        raise AssertionError("legacy manifest unexpectedly accepted")
