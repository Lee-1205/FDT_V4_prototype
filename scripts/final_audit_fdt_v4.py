from __future__ import annotations

"""Read-only, evidence-conservative final audit assembler for FDT v4.

This program never instantiates a model, launches CUDA, edits a repository, or
creates a checkpoint.  It only hashes the supplied immutable inputs, extracts
architecture facts from source/configuration, and assembles supplied evidence
into an immutable audit bundle.  Missing evidence stays ``NOT TESTED``.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT.parent / "source" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AXES = (
    "ARCHITECTURE",
    "EXACT_MEMORY",
    "GENERATION_STABILITY",
    "LONG_CONTEXT",
    "INFERENCE_INTEGRITY",
    "PERFORMANCE",
    "QUALITY",
    "REPRODUCIBILITY",
)
VALID_STATUSES = {"PASS", "PARTIAL", "FAIL", "NOT TESTED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def git_value(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state(repo: Path, requested_commit: str | None) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"status": "NOT_TESTED", "reason": "repository is not a git checkout", "path": str(repo)}
    head = git_value(repo, "rev-parse", "HEAD")
    branch = git_value(repo, "branch", "--show-current")
    status = git_value(repo, "status", "--short")
    commit_exists = True
    if requested_commit:
        try:
            subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{requested_commit}^{{commit}}"], capture_output=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            commit_exists = False
    return {
        "status": "ok" if head else "NOT_TESTED",
        "path": str(repo),
        "head": head,
        "branch": branch,
        "working_tree_dirty": bool(status),
        "working_tree_status": status.splitlines() if status else [],
        "requested_commit": requested_commit,
        "requested_commit_exists": commit_exists,
        "requested_matches_head": bool(requested_commit and head and requested_commit.lower() == head.lower()),
    }


def load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"configuration must contain an object: {path}")
        return value
    from fdt_rlm.config import load_yaml_like

    value = load_yaml_like(path)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return value


def flattened_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", config)
    if not isinstance(model, dict):
        return {}
    return model


def source_architecture(config: dict[str, Any], source_root: Path) -> dict[str, Any]:
    model = flattened_config(config)
    files = {
        "model": source_root / "fdt_rlm" / "models" / "fdt_v4.py",
        "fuzzy_anchor": source_root / "fdt_rlm" / "models" / "fdt_v3.py",
        "anchor_runtime": source_root / "fdt_rlm" / "models" / "next_causal_lm.py",
        "causal_base": source_root / "fdt_rlm" / "models" / "causal_lm.py",
        "pointer": source_root / "fdt_rlm" / "lexical_pointer.py",
        "config": source_root / "fdt_rlm" / "config.py",
    }
    texts = {name: path.read_text(encoding="utf-8") if path.is_file() else "" for name, path in files.items()}
    required = {
        "causal_local_attention": "RotaryCausalWindowAttention",
        "rope_qk": "RotaryEmbedding",
        "rmsnorm": "RMSNorm",
        "swiglu": "SwiGLU",
        "fuzzy_anchor": "V3FuzzyAnchorLayer",
        "cosine_routing": "cosine / max",
        "top_k_sparsity": "torch.topk",
        "tied_embedding": "self.lm_head.weight = self.token_embedding.weight",
        "learned_absolute_disabled_under_rope": "self.position_embedding = None if config.use_rope",
        "raw_token_identity": "token_ids",
        "anchor_indexed_exact_memory": "AnchorIndexedExactMemory",
        "copy_cursor": "copy_cursor",
        "bounded_candidate_control": "candidate",
    }
    combined = "\n".join(texts.values())
    found = {name: token in combined for name, token in required.items()}
    expected = {
        "model_type": "fdt_v4",
        "local_attention_window": 64,
        "num_anchors": 256,
        "top_k": 8,
        "router_dim": 256,
        "routing_type": "cosine",
        "cosine_temperature": 0.25,
        "tie_embeddings": True,
        "use_rope": True,
        "exact_memory_enabled": True,
        "exact_memory_copy_cursor": True,
    }
    observed = {key: model.get(key) for key in expected}
    mismatches = {key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value}
    anchor_indices = model.get("anchor_layer_indices")
    alternating = anchor_indices == list(range(0, 20, 2))
    status = "PASS" if all(found.values()) and not mismatches and alternating else "FAIL"
    warnings: list[str] = []
    if model.get("exact_memory_full_scan_fallback"):
        warnings.append("exact-memory full-source fallback is enabled; performance evidence must demonstrate it is not used on every decode.")
    if not alternating:
        warnings.append("anchor layer indices do not match the expected alternating 20-layer main configuration.")
    return {
        "status": status,
        "config": observed | {
            "hidden_dim": model.get("dim"),
            "n_layers": model.get("n_layers"),
            "n_heads": model.get("n_heads"),
            "dropout": model.get("dropout"),
            "anchor_layer_indices": anchor_indices,
            "exact_pointer_chunk_size": model.get("exact_pointer_chunk_size"),
            "exact_pointer_candidate_chunks": model.get("exact_pointer_candidate_chunks"),
            "exact_memory_candidate_cap": model.get("exact_memory_candidate_cap"),
            "repetition_scope": model.get("generation_repetition_scope"),
            "repetition_ngram_order": model.get("generation_ngram_order"),
        },
        "source_files": {name: str(path) for name, path in files.items()},
        "source_file_hashes": {name: sha256_file(path) if path.is_file() else None for name, path in files.items()},
        "source_invariants": found,
        "config_mismatches": mismatches,
        "alternating_anchor_layers": alternating,
        "warnings": warnings,
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    state = payload.get("model_state_dict", payload.get("state_dict"))
    parameter_count = None
    state_dict_tensor_elements = None
    tied_state_aliases: list[list[str]] = []
    if isinstance(state, dict):
        tensors = [(str(name), value) for name, value in state.items() if isinstance(value, torch.Tensor)]
        state_dict_tensor_elements = sum(int(value.numel()) for _, value in tensors)
        storage_groups: dict[tuple[int, int, int, str], list[str]] = {}
        for name, value in tensors:
            identity = (value.untyped_storage().data_ptr(), int(value.storage_offset()), int(value.numel()), str(value.dtype))
            storage_groups.setdefault(identity, []).append(name)
        parameter_count = sum(identity[2] for identity in storage_groups)
        tied_state_aliases = sorted(sorted(names) for names in storage_groups.values() if len(names) > 1)
    config = payload.get("model_config")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "keys": sorted(str(key) for key in payload),
        "parameter_count": parameter_count,
        "state_dict_tensor_elements": state_dict_tensor_elements,
        "tied_state_aliases": tied_state_aliases,
        "model_config": config if isinstance(config, dict) else None,
    }


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "os": platform.platform(),
        "system_ram_bytes": _system_ram_bytes(),
        "gpu_launched": False,
        "device": "cpu",
    }


def _system_ram_bytes() -> int | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        return None
    return None


def read_evidence(path: Path, kind: str) -> dict[str, Any]:
    record: dict[str, Any] = {"kind": kind, "path": str(path), "sha256": sha256_file(path), "format": path.suffix.lower()}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        record["rows"] = rows
    else:
        record["payload"] = json.loads(path.read_text(encoding="utf-8"))
    return record


def walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_mappings(child)


def explicit_axis_statuses(evidence: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    values = {axis: [] for axis in AXES}
    failures: list[str] = []
    for item in evidence:
        body = item.get("payload", {"rows": item.get("rows", [])})
        for mapping in walk_mappings(body):
            axes = mapping.get("audit_axes") or mapping.get("axis_verdicts")
            if isinstance(axes, dict):
                for axis, result in axes.items():
                    if axis not in values:
                        continue
                    status = result.get("status") if isinstance(result, dict) else result
                    if isinstance(status, str) and status.upper() in VALID_STATUSES:
                        values[axis].append({"status": status.upper(), "source": item["path"]})
            axis = mapping.get("axis")
            status = mapping.get("status")
            if isinstance(axis, str) and axis.upper() in values and isinstance(status, str) and status.upper() in VALID_STATUSES:
                values[axis.upper()].append({"status": status.upper(), "source": item["path"]})
            test_name = mapping.get("test") or mapping.get("name")
            if isinstance(status, str) and status.upper() == "FAIL" and isinstance(test_name, str):
                failures.append(f"{item['path']}: {test_name}")
    return values, failures


def conservative_axis(statuses: list[dict[str, str]], default: str = "NOT TESTED") -> dict[str, Any]:
    if not statuses:
        return {"status": default, "evidence": []}
    severity = {"FAIL": 3, "PARTIAL": 2, "NOT TESTED": 1, "PASS": 0}
    selected = max(statuses, key=lambda item: severity[item["status"]])
    return {"status": selected["status"], "evidence": statuses}


def derive_axes(architecture: dict[str, Any], git: dict[str, Any], evidence: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    explicit, failures = explicit_axis_statuses(evidence)
    axes = {axis: conservative_axis(explicit[axis]) for axis in AXES}
    axes["ARCHITECTURE"] = {
        "status": architecture["status"],
        "evidence": [{"source": "source+config", "status": architecture["status"]}],
        "reason": "Source-derived invariant check; this is not a quality claim.",
    }
    if not explicit["EXACT_MEMORY"]:
        axes["EXACT_MEMORY"] = {"status": "PARTIAL" if architecture["source_invariants"].get("raw_token_identity") and architecture["source_invariants"].get("copy_cursor") else "FAIL", "evidence": [{"source": "source+config", "status": "PARTIAL"}], "reason": "Implementation is present, but no supplied deterministic copy matrix establishes functional success."}
    if not explicit["REPRODUCIBILITY"]:
        reproducible = git.get("status") == "ok" and bool(git.get("head"))
        axes["REPRODUCIBILITY"] = {"status": "PARTIAL" if reproducible else "NOT TESTED", "evidence": [{"source": "repository-state", "status": "PARTIAL"}] if reproducible else [], "reason": "Hashes and commit are recorded; a rerun has not been demonstrated by this read-only audit."}
    return axes, failures


def collect_rows(evidence: list[dict[str, Any]], kind: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in evidence:
        if item["kind"] != kind:
            continue
        for row in item.get("rows", []):
            rows.append({"source": item["path"], **{str(key): str(value) for key, value in row.items()}})
        payload = item.get("payload")
        if isinstance(payload, dict):
            candidate = payload.get("rows") or payload.get("matrix") or payload.get("benchmarks") or payload.get("cells")
            if isinstance(candidate, list):
                for row in candidate:
                    if isinstance(row, dict):
                        rows.append({"source": item["path"], **{str(key): json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value) for key, value in row.items()}})
    return rows


def csv_text(rows: list[dict[str, str]], fallback_columns: list[str]) -> str:
    columns = list(fallback_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def python_cap_evidence(config: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    train_cap = None
    train_config = config.get("train") if isinstance(config.get("train"), dict) else {}
    for key in ("python_completion_cap", "code_completion_cap", "completion_cap"):
        if key in train_config:
            train_cap = train_config[key]
            break
    text_hits: list[dict[str, Any]] = []
    for item in evidence:
        serialized = json.dumps(item.get("payload", item.get("rows", [])), ensure_ascii=False)
        for match in re.finditer(r"(?:python|code)[^\n]{0,80}(?:96|192)|(?:96|192)[^\n]{0,80}(?:python|code)", serialized, flags=re.I):
            text_hits.append({"source": item["path"], "excerpt": match.group(0)[:180]})
    return {"training_completion_cap": train_cap if train_cap is not None else "UNKNOWN", "evaluation_or_runtime_96_192_evidence": text_hits, "conclusion": "UNKNOWN unless an explicit supplied config/result identifies the evaluation or runtime cap."}


def markdown_report(results: dict[str, Any]) -> str:
    axes = results["axes"]
    lines = ["# FDT Final Audit", "", "## 1. Executive Verdict", "", f"FINAL VERDICT: {results['final_verdict']}", "", "## 2. Exact Repository State", "", "```json", json.dumps(results["repository"], indent=2, ensure_ascii=False), "```", "", "## 3. Final Architecture", "", "```json", json.dumps(results["architecture"], indent=2, ensure_ascii=False), "```", "", "## 4. Changes From Previous Pure-FDT", "", "NOT TESTED: no supplied source-diff evidence was interpreted as a benchmark or capability result.", "", "## 5. RoPE Verification", "", f"{axes['LONG_CONTEXT']['status']}: functional RoPE/cache evidence must come from supplied tests or evaluation.", "", "## 6. Fuzzy Semantic Memory Verification", "", f"{axes['ARCHITECTURE']['status']}: source-derived configuration only.", "", "## 7. Exact Episodic Memory Verification", "", f"{axes['EXACT_MEMORY']['status']}", "", "## 8. Exact-Copy Matrix", "", f"Merged rows: {results['artifacts']['exact_copy_rows']}", "", "## 9. Generation / Loop Analysis", "", f"{axes['GENERATION_STABILITY']['status']}", "", "## 10. EOS / Exposure Analysis", "", "NOT TESTED unless explicitly present in supplied evidence.", "", "## 11. Long-Context Results", "", f"{axes['LONG_CONTEXT']['status']}", "", "## 12. Incremental Cache Integrity", "", f"{axes['INFERENCE_INTEGRITY']['status']}", "", "## 13. GPU Profile on Current Hardware", "", f"{axes['PERFORMANCE']['status']}. This read-only CPU audit did not launch GPU work.", "", "## 14. Optimization A/B Results", "", "NOT TESTED unless supplied as current-hardware evidence.", "", "## 15. Dense Control Comparison", "", "NOT TESTED unless supplied as a fair control.", "", "## 16. Quality / BPB Results", "", f"{axes['QUALITY']['status']}", "", "## 17. Regression Tests", "", "See `failed_tests.txt`; this tool does not execute GPU or main-checkpoint tests.", "", "## 18. Reproducibility", "", f"{axes['REPRODUCIBILITY']['status']}", "", "## 19. Remaining Failures", ""]
    lines += [f"- {item}" for item in results["critical_failures"]] or ["- No explicit failed test was supplied; absence of evidence is not success."]
    lines += ["", "## 20. Final Decision", "", "Global PASS is intentionally never inferred by this assembler. Functional axes require supplied, hashed evidence.", "", "## 21. Recommended Next Experiment", "", "Run the missing bounded CPU/GPU evidence through the designated evaluators, then reassemble a new immutable audit directory.", "", "## Python 96 / 192 Evidence", "", "```json", json.dumps(results["python_96_192"], indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def create_audit(
    repo: Path,
    commit: str | None,
    run_dir: Path,
    checkpoint: Path,
    config_path: Path,
    tokenizer: Path | None,
    output_dir: Path,
    visibility_check: Path | None,
    benchmark_paths: list[Path],
    exact_paths: list[Path],
    profile_paths: list[Path],
    evaluation_paths: list[Path],
    test_paths: list[Path],
) -> dict[str, Any]:
    inputs = [repo, run_dir, checkpoint, config_path, *benchmark_paths, *exact_paths, *profile_paths, *evaluation_paths, *test_paths]
    if tokenizer:
        inputs.append(tokenizer)
    if visibility_check:
        inputs.append(visibility_check)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"immutable audit output already exists: {output_dir}")
    config = load_mapping(config_path)
    architecture = source_architecture(config, SRC)
    repository = git_state(repo, commit)
    checkpoint_info = checkpoint_metadata(checkpoint)
    tokenizer_info = {"status": "NOT TESTED"}
    if tokenizer:
        tokenizer_info = {"status": "ok", "path": str(tokenizer), "sha256": sha256_file(tokenizer)}
    evidence: list[dict[str, Any]] = []
    for kind, paths in (("benchmark", benchmark_paths), ("exact_copy", exact_paths), ("profile", profile_paths), ("evaluation", evaluation_paths), ("test", test_paths)):
        evidence.extend(read_evidence(path, kind) for path in paths)
    axes, failed = derive_axes(architecture, repository, evidence)
    if architecture["status"] == "FAIL":
        failed.append("source/config architecture invariant mismatch")
    if repository.get("requested_commit") and not repository.get("requested_commit_exists"):
        failed.append("requested git commit does not exist in supplied repository")
    final_verdict = "FAIL" if any(axis["status"] == "FAIL" for axis in axes.values()) else "PARTIAL"
    benchmark_rows = collect_rows(evidence, "benchmark")
    exact_rows = collect_rows(evidence, "exact_copy")
    profiles = [item for item in evidence if item["kind"] == "profile"]
    visibility = {"status": "NOT TESTED"}
    if visibility_check:
        visibility = {"status": "recorded_only", "path": str(visibility_check), "sha256": sha256_file(visibility_check), "visibility_changed": False}
    results = {
        "schema": "fdt_v4_final_audit_v1",
        "created_at_unix": time.time(),
        "read_only": True,
        "gpu_launched": False,
        "audit_tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "final_verdict": final_verdict,
        "axes": axes,
        "repository": repository,
        "run": {"path": str(run_dir), "sha256": None},
        "checkpoint": checkpoint_info,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path), "model": flattened_config(config)},
        "tokenizer": tokenizer_info,
        "environment": runtime_environment(),
        "private_visibility_check": visibility,
        "architecture": architecture,
        "python_96_192": python_cap_evidence(config, evidence),
        "supplied_evidence": [{key: value for key, value in item.items() if key not in {"payload", "rows"}} for item in evidence],
        "artifacts": {"benchmark_rows": len(benchmark_rows), "exact_copy_rows": len(exact_rows), "profile_sources": len(profiles)},
        "critical_failures": sorted(set(failed)),
        "remaining_research_questions": ["All NOT TESTED axes require actual hashed evaluator or profiler evidence.", "A source presence check cannot establish free-generation quality, 8K correctness, or current-GPU performance."],
    }
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        atomic_json(temporary / "final_audit_results.json", results)
        atomic_text(temporary / "FINAL_AUDIT.md", markdown_report(results))
        atomic_text(temporary / "benchmark.csv", csv_text(benchmark_rows, ["source", "context", "prefill_ms", "decode_ms_per_token", "status"]))
        atomic_text(temporary / "exact_copy_matrix.csv", csv_text(exact_rows, ["source", "length_chars", "target_position", "distractor_count", "whole_string_exact", "status"]))
        atomic_json(temporary / "profiler_summary.json", {"status": "NOT TESTED" if not profiles else "SUPPLIED", "sources": profiles})
        atomic_text(temporary / "failed_tests.txt", "\n".join(results["critical_failures"]) + ("\n" if results["critical_failures"] else ""))
        atomic_json(temporary / "git_commit_hash_manifest.json", {"repository": repository, "audit_tool": results["audit_tool"], "checkpoint": checkpoint_info, "config_sha256": results["config"]["sha256"], "tokenizer": tokenizer_info, "source_file_hashes": architecture["source_file_hashes"], "visibility_check": visibility})
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    results["output_dir"] = str(output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only conservative final audit assembler for FDT v4")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-visibility-check", type=Path)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--exact-copy", type=Path, action="append", default=[])
    parser.add_argument("--profile", type=Path, action="append", default=[])
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--test-result", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = create_audit(args.repo.resolve(), args.commit, args.run_dir.resolve(), args.checkpoint.resolve(), args.config.resolve(), args.tokenizer.resolve() if args.tokenizer else None, args.output_dir.resolve(), args.private_visibility_check.resolve() if args.private_visibility_check else None, [path.resolve() for path in args.benchmark], [path.resolve() for path in args.exact_copy], [path.resolve() for path in args.profile], [path.resolve() for path in args.evaluation], [path.resolve() for path in args.test_result])
    print(json.dumps({"final_verdict": result["final_verdict"], "output_dir": result["output_dir"], "gpu_launched": False}))


if __name__ == "__main__":
    main()
