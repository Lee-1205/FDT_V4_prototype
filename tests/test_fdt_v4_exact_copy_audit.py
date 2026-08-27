import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("audit_fdt_v4_exact_copy_test", ROOT / "scripts" / "audit_fdt_v4_exact_copy.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_is_exactly_60_deterministic_cells_and_covers_all_kinds():
    module = load_module()
    first = module.matrix_specs(17)
    second = module.matrix_specs(17)
    assert first == second
    assert len(first) == 60
    assert {spec.length for spec in first} == {4, 8, 16, 32, 64}
    assert {spec.position for spec in first} == {"front", "middle", "end"}
    assert {spec.distractors for spec in first} == {0, 1, 4, 16}
    assert {spec.string_kind for spec in first} == set(module.STRING_KINDS)
    repeated = next(spec for spec in first if spec.string_kind == "repeated_character")
    case = module.build_case(repeated)
    assert len(case["target"]) == repeated.length
    assert len(set(case["target"])) == 1
    repeated_many = next(spec for spec in first if spec.string_kind == "repeated_character" and spec.distractors == 16)
    crowded = module.build_case(repeated_many)
    assert crowded["target"] not in crowded["decoys"]
    assert len(set(crowded["decoys"])) == 16


def test_metrics_and_copy_active_repetition_exemption_are_explicit():
    module = load_module()
    metrics = module.text_metrics("ABXD", "ABCD", [1, 2, 9, 4], [1, 2, 3, 4])
    assert metrics["whole_string_exact"] is False
    assert metrics["character_accuracy"] == 0.75
    assert metrics["token_accuracy"] == 0.75
    assert metrics["character_edit_distance"] == 1
    assert metrics["first_divergence"] == 2

    base = torch.tensor([[0.0, 4.0, 2.0]])
    proposed = torch.tensor([[-100.0, -100.0, 0.0]])
    history = torch.tensor([[2]])
    selected, copy_active, penalty_applied = module.copy_safe_logits(base, proposed, history, {"mode": "cursor", "mix_gate": 1.0})
    torch.testing.assert_close(selected, proposed)
    assert copy_active is True
    assert penalty_applied is False

    selected, copy_active, penalty_applied = module.copy_safe_logits(base, proposed, history, {"mode": "base", "mix_gate": 0.0})
    assert copy_active is False
    assert penalty_applied is True
    assert selected[0, 2] == base[0, 2] / 1.10


def test_prompt_target_alignment_exposes_cross_boundary_tokenization():
    module = load_module()
    exact = module.prompt_target_token_alignment([10, 11, 12, 13], [12, 13], 1)
    assert exact == {
        "prompt_target_token_alignment_exact": True,
        "prompt_target_token_ids": [12, 13],
    }
    mismatch = module.prompt_target_token_alignment([10, 11, 99, 13], [12, 13], 1)
    assert mismatch["prompt_target_token_alignment_exact"] is False
    assert mismatch["prompt_target_token_ids"] == [99, 13]


def test_atomic_bundle_contains_json_csv_and_no_temp_residue(tmp_path):
    module = load_module()
    cells = []
    for spec in module.matrix_specs(23):
        cells.append({"cell_id": spec.cell_id, "length_chars": spec.length, "target_position": spec.position, "distractor_count": spec.distractors, "string_kind": spec.string_kind, "status": "NOT TESTED", "cursor_trace": []})
    report = {"schema": "fixture", "cells": cells}
    output = tmp_path / "audit"
    module.write_atomic_bundle(output, report)
    assert (output / "fdt_v4_exact_copy_audit.json").is_file()
    assert (output / "fdt_v4_exact_copy_matrix.csv").is_file()
    assert (output / "sha256.json").is_file()
    parsed = json.loads((output / "fdt_v4_exact_copy_audit.json").read_text(encoding="utf-8"))
    assert len(parsed["cells"]) == 60
    assert not list(tmp_path.glob(".audit.*"))
    try:
        module.write_atomic_bundle(output, report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable audit output was overwritten")


def test_strict_exact_axis_requires_every_cell(monkeypatch, tmp_path):
    module = load_module()
    cells = []
    for spec in module.matrix_specs(29):
        cells.append(
            {
                "cell_id": spec.cell_id,
                "status": "ok",
                "whole_string_exact": True,
                "exact_retrieval_success": True,
                "copy_gate_activated": True,
                "full_scan_count": 1,
            }
        )

    class Tokenizer:
        def __len__(self):
            return 8

    class Config:
        vocab_size = 8

    parameter = torch.nn.Parameter(torch.zeros(()))
    model = torch.nn.Module()
    model.register_parameter("sentinel", parameter)
    monkeypatch.setattr(module, "tokenizer_paths", lambda path: (tmp_path, tmp_path / "tokenizer.json"))
    monkeypatch.setattr(module, "load_checkpoint", lambda checkpoint, device: (model, Config(), {"sha256": "x"}))
    monkeypatch.setattr(module, "load_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(module, "evaluate_cell", lambda model, config, tokenizer, spec: cells[spec.cell_id])
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")

    report = module.run_audit(checkpoint, tmp_path, tmp_path / "pass", device_name="cpu")
    assert report["audit_axes"]["EXACT_MEMORY"]["status"] == "PASS"
    assert report["evaluator"]["sha256"] == module.sha256_file(Path(module.__file__))

    cells[-1]["whole_string_exact"] = False
    report = module.run_audit(checkpoint, tmp_path, tmp_path / "fail", device_name="cpu")
    assert report["audit_axes"]["EXACT_MEMORY"]["status"] == "FAIL"
