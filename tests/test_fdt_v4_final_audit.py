import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("final_audit_fdt_v4_test", ROOT / "scripts" / "final_audit_fdt_v4.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_checkpoint(path: Path):
    torch.save({"model_config": {"model_type": "fdt_v4"}, "model_state_dict": {"embedding.weight": torch.ones(3, 4)}}, path)


def write_config(path: Path):
    path.write_text("""model:\n  model_type: fdt_v4\n  vocab_size: 24576\n  pad_token_id: 0\n  eos_token_id: 2\n  dim: 1216\n  n_layers: 20\n  n_heads: 19\n  local_attention_window: 64\n  anchor_layer_indices: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]\n  num_anchors: 256\n  top_k: 8\n  router_dim: 256\n  routing_type: cosine\n  cosine_temperature: 0.25\n  tie_embeddings: true\n  use_rope: true\n  exact_memory_enabled: true\n  exact_memory_copy_cursor: true\n  exact_memory_full_scan_fallback: true\n""", encoding="utf-8")


def fake_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def test_final_audit_creates_required_immutable_bundle(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "git_state", lambda repo, commit: {"status": "ok", "head": "a" * 40, "branch": "main", "requested_commit": commit, "requested_commit_exists": True})
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    run = tmp_path / "run"
    run.mkdir()
    tiny_checkpoint(checkpoint)
    write_config(config)
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text("context,decode_ms_per_token,status\n512,1.2,PASS\n", encoding="utf-8")
    exact = tmp_path / "exact.csv"
    exact.write_text("length,position,distractors,whole_sequence_exact,status\n4,front,0,1,PASS\n", encoding="utf-8")
    evidence = tmp_path / "evaluation.json"
    evidence.write_text(json.dumps({"audit_axes": {"INFERENCE_INTEGRITY": {"status": "PASS"}, "QUALITY": {"status": "NOT TESTED"}}}), encoding="utf-8")
    output = tmp_path / "final"
    result = module.create_audit(fake_repo(tmp_path), "a" * 40, run, checkpoint, config, None, output, None, [benchmark], [exact], [], [evidence], [])
    assert result["gpu_launched"] is False
    assert result["final_verdict"] == "PARTIAL"
    for name in ("FINAL_AUDIT.md", "final_audit_results.json", "benchmark.csv", "exact_copy_matrix.csv", "profiler_summary.json", "failed_tests.txt", "git_commit_hash_manifest.json"):
        assert (output / name).is_file()
    parsed = json.loads((output / "final_audit_results.json").read_text(encoding="utf-8"))
    assert parsed["axes"]["ARCHITECTURE"]["status"] == "PASS"
    assert parsed["axes"]["PERFORMANCE"]["status"] == "NOT TESTED"
    assert len(parsed["audit_tool"]["sha256"]) == 64
    assert not list(tmp_path.glob(".final.*"))


def test_missing_evidence_never_becomes_global_pass(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "git_state", lambda repo, commit: {"status": "ok", "head": "b" * 40, "branch": "main", "requested_commit": commit, "requested_commit_exists": True})
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    run = tmp_path / "run"
    run.mkdir()
    tiny_checkpoint(checkpoint)
    write_config(config)
    result = module.create_audit(fake_repo(tmp_path), None, run, checkpoint, config, None, tmp_path / "final", None, [], [], [], [], [])
    assert result["final_verdict"] == "PARTIAL"
    assert result["axes"]["GENERATION_STABILITY"]["status"] == "NOT TESTED"
    assert result["axes"]["LONG_CONTEXT"]["status"] == "NOT TESTED"


def test_existing_audit_directory_is_never_overwritten(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "git_state", lambda repo, commit: {"status": "ok", "head": "c" * 40})
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    run = tmp_path / "run"
    run.mkdir()
    tiny_checkpoint(checkpoint)
    write_config(config)
    output = tmp_path / "final"
    output.mkdir()
    try:
        module.create_audit(fake_repo(tmp_path), None, run, checkpoint, config, None, output, None, [], [], [], [], [])
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable output directory was overwritten")


def test_checkpoint_metadata_counts_tied_storage_once(tmp_path):
    module = load_module()
    checkpoint = tmp_path / "tied.pt"
    tied = torch.ones(3, 4)
    torch.save({"model_state_dict": {"token_embedding.weight": tied, "lm_head.weight": tied}}, checkpoint)

    metadata = module.checkpoint_metadata(checkpoint)

    assert metadata["parameter_count"] == 12
    assert metadata["state_dict_tensor_elements"] == 24
    assert metadata["tied_state_aliases"] == [["lm_head.weight", "token_embedding.weight"]]
