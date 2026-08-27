import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_fdt_v4_exact_copy_failure_test",
        ROOT / "scripts" / "diagnose_fdt_v4_exact_copy_failure.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_untrained_pointer_failure_is_separated_from_cursor_and_alignment():
    module = load_module()
    audit = {
        "summary": {
            "tested_cells": 60,
            "whole_string_exact_rate": 0.0,
            "retrieval_success_rate": 0.0,
            "copy_gate_activation_rate": 1.0,
            "max_full_scan_count": 1,
        },
        "checkpoint": {"exact_weight_provenance": "UNTRAINED_WARM_START"},
        "cells": [
            {
                "cursor_trace": [{"mode": "mixed", "source_positions": [[7]]}],
                "prompt_target_token_alignment_exact": None,
            }
        ],
    }
    conversion = {"new_random_components": ["exact_pointer.q_proj.weight"], "skipped_new_or_denied_keys": ["exact_pointer.k_proj.weight"]}
    report = module.classify(audit, conversion)
    classification = report["classification"]
    assert classification["immediate_cause"] == "UNTRAINED_EXACT_POINTER_SELECTION_PARAMETERS"
    assert classification["evaluator_prompt_alignment"]["status"] == "NOT_MEASURED_IN_THIS_LEGACY_AUDIT"
    assert classification["retrieval_candidate_selection"]["status"] == "FAILED_BEFORE_CURSOR"
    assert classification["decode_cursor"]["status"] == "NOT_REACHED"

    audit["cells"][0]["cursor_trace"].append({"mode": "cursor"})
    report = module.classify(audit, conversion)
    assert report["classification"]["decode_cursor"]["status"] == "NOT_CAUSAL_FOR_INITIAL_RETRIEVAL_FAILURE"


def test_diagnosis_output_is_atomic_and_immutable(tmp_path):
    module = load_module()
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    audit.write_text(json.dumps({"summary": {}, "checkpoint": {}, "cells": [{"cursor_trace": []}]}), encoding="utf-8")
    manifest.write_text(json.dumps({}), encoding="utf-8")
    output = tmp_path / "diagnosis"
    report = module.classify(module.read_json(audit), module.read_json(manifest))
    module.write_atomic(output, report)
    assert (output / "fdt_v4_exact_copy_failure_diagnosis.json").is_file()
    assert (output / "sha256.json").is_file()
    try:
        module.write_atomic(output, report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("diagnosis output was overwritten")
