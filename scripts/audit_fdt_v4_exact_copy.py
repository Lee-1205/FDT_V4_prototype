from __future__ import annotations

"""Deterministic, CPU-FP32 exact-copy audit for an FDT v4 checkpoint.

The audit is intentionally read-only: it never trains, mutates a checkpoint,
uses quantization, or launches CUDA.  It evaluates the checkpoint exactly as
stored and records unknown exact-weight provenance as UNKNOWN.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.next_tools import apply_repetition_penalty_  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


LENGTHS = (4, 8, 16, 32, 64)
POSITIONS = ("front", "middle", "end")
DISTRACTORS = (0, 1, 4, 16)
STRING_KINDS = ("digits", "alphanumeric", "hex", "uuid_like", "mixed_identifier", "repeated_character")
REPETITION_PENALTY = 1.10


@dataclass(frozen=True)
class MatrixSpec:
    cell_id: int
    length: int
    position: str
    distractors: int
    string_kind: str
    seed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def matrix_specs(seed: int = 20260823) -> list[MatrixSpec]:
    specs: list[MatrixSpec] = []
    cell = 0
    for length in LENGTHS:
        for position in POSITIONS:
            for distractors in DISTRACTORS:
                specs.append(MatrixSpec(cell, length, position, distractors, STRING_KINDS[cell % len(STRING_KINDS)], seed + cell * 7919))
                cell += 1
    if len(specs) != 60:
        raise AssertionError("exact-copy matrix contract must contain exactly 60 cells")
    return specs


def _cycle(alphabet: str, length: int, offset: int) -> str:
    return "".join(alphabet[(offset + index * 7) % len(alphabet)] for index in range(length))


def deterministic_string(kind: str, length: int, seed: int, variant: int = 0) -> str:
    offset = (seed + variant * 37) % 997
    if kind == "digits":
        return _cycle("0123456789", length, offset)
    if kind == "alphanumeric":
        return _cycle("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", length, offset)
    if kind == "hex":
        return _cycle("0123456789abcdef", length, offset)
    if kind == "uuid_like":
        raw = _cycle("0123456789abcdef", length, offset)
        value = list(raw)
        for location in (8, 13, 18, 23):
            if location < length:
                value[location] = "-"
        return "".join(value)
    if kind == "mixed_identifier":
        return _cycle("Az_9-bY7.Cx5", length, offset)
    if kind == "repeated_character":
        character = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"[(offset + variant) % 34]
        return character * length
    raise ValueError(f"unknown string kind: {kind}")


def build_case(spec: MatrixSpec) -> dict[str, Any]:
    target = deterministic_string(spec.string_kind, spec.length, spec.seed)
    decoys = [deterministic_string(spec.string_kind, spec.length, spec.seed, index + 1) for index in range(spec.distractors)]
    left = "Context section alpha contains ordinary prose about calibration and identity.\n"
    right = "Context section omega contains unrelated prose and no requested answer.\n"
    target_record = f"PRIMARY RECORD | <<VALUE>>{target}<<END>>\n"
    decoy_lines = [f"DISTRACTOR {index:02d} | <<DECOY>>{value}<<END>>\n" for index, value in enumerate(decoys)]
    decoy_records = "".join(decoy_lines)
    if spec.position == "front":
        body = target_record + left + decoy_records + right
    elif spec.position == "middle":
        midpoint = len(decoy_lines) // 2
        body = left + "".join(decoy_lines[:midpoint]) + target_record + "".join(decoy_lines[midpoint:]) + right
    else:
        body = left + decoy_records + right + target_record
    prompt = body + "Output the content between PRIMARY <<VALUE>> and <<END>> verbatim. Answer:<<VALUE>>"
    target_char_start = prompt.index(target)
    return {"prompt": prompt, "target": target, "target_char_start": target_char_start, "decoys": decoys}


def tokenizer_paths(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_file() and path.name == "tokenizer.json":
        return path.parent, path
    tokenizer_json = path / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer directory must contain tokenizer.json: {path}")
    return path, tokenizer_json


def checkpoint_metadata(payload: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    state = payload.get("model_state_dict")
    exact_tensors = {name: tensor for name, tensor in state.items() if "exact_pointer" in name} if isinstance(state, dict) else {}
    exact_parameters = sum(int(tensor.numel()) for tensor in exact_tensors.values() if isinstance(tensor, torch.Tensor))
    exact_nonzero = sum(int(torch.count_nonzero(tensor).item()) for tensor in exact_tensors.values() if isinstance(tensor, torch.Tensor))
    stage_fields = {
        key: payload.get(key)
        for key in ("stage_status", "optimizer_step", "step", "sample_cursor", "token_counter", "tokens_seen", "audit_status")
        if key in payload
    }
    train_config = payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}
    conversion = payload.get("conversion_manifest") or payload.get("warm_start_manifest")
    audit_metadata = payload.get("audit_metadata")
    explicit_provenance = payload.get("exact_memory_training") or payload.get("exact_memory_provenance")
    if explicit_provenance is None and isinstance(audit_metadata, dict):
        if audit_metadata.get("new_v4_parameters_trained") is False:
            explicit_provenance = "UNTRAINED_WARM_START"
        elif audit_metadata.get("new_v4_parameters_trained") is True:
            explicit_provenance = "CHECKPOINT_DECLARED_V4_PARAMETERS_TRAINED"
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "checkpoint_keys": sorted(str(key) for key in payload),
        "stage_metadata": stage_fields,
        "train_config_stage": train_config.get("stage") or train_config.get("run_name"),
        "audit_metadata": audit_metadata if audit_metadata is not None else "UNKNOWN",
        "conversion_manifest": conversion if conversion is not None else "UNKNOWN",
        "exact_weight_provenance": explicit_provenance if explicit_provenance is not None else "UNKNOWN",
        "exact_weights": {
            "tensor_count": len(exact_tensors),
            "parameter_count": exact_parameters,
            "nonzero_parameters": exact_nonzero,
            "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in exact_tensors.values()),
            "warning": "Weight presence/nonzero values do not prove that exact memory was trained. Provenance remains UNKNOWN unless checkpoint metadata says otherwise.",
        },
    }


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError("checkpoint must contain model_config and model_state_dict")
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v4":
        raise ValueError(f"checkpoint must be fdt_v4, got {config.model_type!r}")
    if not config.exact_memory_enabled or config.exact_memory_mode != "copy":
        raise ValueError("checkpoint must enable exact memory in copy mode")
    model = build_model(config).to(device="cpu", dtype=torch.float32).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    return model, config, checkpoint_metadata(payload, checkpoint)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_value in enumerate(left, 1):
        current = [index]
        for other_index, right_value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (left_value != right_value)))
        previous = current
    return previous[-1]


def first_divergence(left: str, right: str) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def text_metrics(generated: str, target: str, generated_ids: list[int], target_ids: list[int]) -> dict[str, Any]:
    char_matches = sum(left == right for left, right in zip(generated, target))
    token_matches = sum(left == right for left, right in zip(generated_ids, target_ids))
    return {
        "whole_string_exact": generated == target,
        "character_accuracy": char_matches / max(len(target), 1),
        "token_accuracy": token_matches / max(len(target_ids), 1),
        "character_edit_distance": edit_distance(generated, target),
        "first_divergence": first_divergence(generated, target),
    }


def _candidate_count(diagnostics: dict[str, Any]) -> int:
    candidates = diagnostics.get("candidate_ids") or []
    return sum(len(row) for row in candidates if isinstance(row, list))


def compact_trace(diagnostics: dict[str, Any], selected_id: int, copy_active: bool, penalty_applied: bool) -> dict[str, Any]:
    positions = diagnostics.get("source_positions") or []
    return {
        "mode": diagnostics.get("mode", "base"),
        "selected_id": selected_id,
        "gate": diagnostics.get("gate"),
        "mix_gate": diagnostics.get("mix_gate"),
        "pointer_confidence": diagnostics.get("pointer_confidence"),
        "score_margin": diagnostics.get("score_margin"),
        "commit_confidence": diagnostics.get("commit_confidence"),
        "span_end_source": diagnostics.get("span_end_source"),
        "hard_copy_eligible": bool(diagnostics.get("hard_copy_eligible", False)),
        "source_positions": positions,
        "candidate_count": _candidate_count(diagnostics),
        "used_full_scan_fallback": bool(diagnostics.get("used_full_scan_fallback", False)),
        "full_scan_attempted": bool(diagnostics.get("full_scan_attempted", False)),
        "full_scan_count": int(diagnostics.get("full_scan_count", 0)),
        "copy_active": copy_active,
        "repetition_penalty_applied": penalty_applied,
        "copy_repetition_exempt": copy_active and not penalty_applied,
    }


def copy_safe_logits(base_logits: torch.Tensor, proposed_logits: torch.Tensor, generated_only: torch.Tensor, diagnostics: dict[str, Any]) -> tuple[torch.Tensor, bool, bool]:
    copy_active = diagnostics.get("mode") == "cursor" or float(diagnostics.get("mix_gate", 0.0) or 0.0) > 0.0
    if copy_active:
        return proposed_logits, True, False
    penalized = base_logits.clone()
    apply_repetition_penalty_(penalized, generated_only, REPETITION_PENALTY)
    return penalized, False, generated_only.numel() > 0


def _prepare_step(model: torch.nn.Module, config: ModelConfig, output: dict[str, Any], generated: torch.Tensor, generated_only: torch.Tensor, memory: Any, state: LexicalPointerDecodeState) -> tuple[torch.Tensor, dict[str, Any], bool, bool]:
    base = output["logits"][:, -1].float()
    query = model.exact_route_indices(output["hidden"])[:, -1]
    proposed, diagnostics = state.prepare_logits(
        model.exact_pointer,
        base,
        output["hidden"],
        generated,
        torch.ones_like(generated),
        min_gate=0.0,
        anchor_memory=memory,
        query_anchor_ids=query,
        max_candidate_chunks=config.exact_pointer_candidate_chunks,
        full_scan_fallback=config.exact_memory_full_scan_fallback,
        fallback_margin=config.exact_memory_fallback_margin,
        candidate_cap=config.exact_memory_candidate_cap,
        commit_threshold=config.exact_memory_commit_threshold,
        hard_copy=config.exact_memory_hard_copy,
        hard_copy_gate_threshold=config.exact_memory_hard_copy_gate_threshold,
        hard_copy_pointer_threshold=config.exact_memory_hard_copy_pointer_threshold,
        hard_copy_margin_threshold=config.exact_memory_hard_copy_margin_threshold,
    )
    selected_logits, copy_active, penalty_applied = copy_safe_logits(base, proposed, generated_only, diagnostics)
    return selected_logits, diagnostics, copy_active, penalty_applied


def supports_incremental_cache(config: ModelConfig) -> bool:
    return not (
        getattr(config, "rope_transition_mode", "lerp") == "output_blend"
        and 0.0 < float(getattr(config, "rope_transition_alpha", 1.0)) < 1.0
    )


@torch.inference_mode()
def exact_span_metadata(
    prompt_length: int,
    source_key_position: int,
    target_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Declare the immutable source span registered with Exact Memory."""
    ends = torch.arange(prompt_length, dtype=torch.long, device=device).view(1, -1)
    source_start = source_key_position + 1
    source_end = source_start + len(target_ids) - 1
    if source_key_position < 0 or source_end >= prompt_length:
        raise ValueError("declared Exact Memory span is outside the prompt")
    ends[:, source_start : source_end + 1] = source_end
    registered_keys = torch.zeros(
        (1, prompt_length), dtype=torch.bool, device=device
    )
    registered_keys[:, source_key_position] = True
    key_positions = torch.tensor(
        [[source_key_position]], dtype=torch.long, device=device
    )
    payload_ids = torch.tensor([[target_ids]], dtype=torch.long, device=device)
    payload_lengths = torch.tensor(
        [[len(target_ids)]], dtype=torch.long, device=device
    )
    return ends, registered_keys, key_positions, payload_ids, payload_lengths


def free_generate(
    model: torch.nn.Module,
    config: ModelConfig,
    prompt_ids: list[int],
    target_ids: list[int],
    source_key_position: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = prompt.clone()
    generated_only = generated[:, :0]
    cached = supports_incremental_cache(config)
    if cached:
        output, cache = model.prefill(prompt, torch.ones_like(prompt))
    else:
        output = model(prompt, attention_mask=torch.ones_like(prompt))
        cache = None
    span_ends, registered_keys, key_positions, payload_ids, payload_lengths = exact_span_metadata(
        len(prompt_ids), source_key_position, target_ids, device
    )
    memory = model.build_exact_memory(
        output["hidden"],
        prompt,
        torch.ones_like(prompt),
        source_length=len(prompt_ids),
        span_end_positions=span_ends,
        registered_key_mask=registered_keys,
        registered_key_positions=key_positions,
        registered_payload_ids=payload_ids,
        registered_payload_lengths=payload_lengths,
    )
    max_new_tokens = len(target_ids)
    state = LexicalPointerDecodeState(source_length=len(prompt_ids), max_activation_steps=max_new_tokens, max_copy_tokens=max_new_tokens)
    selected_ids: list[int] = []
    trace: list[dict[str, Any]] = []
    for _ in range(min(max_new_tokens, config.max_seq_len - len(prompt_ids))):
        logits, diagnostics, copy_active, penalty_applied = _prepare_step(model, config, output, generated, generated_only, memory, state)
        selected = int(logits.argmax(dim=-1).item())
        state.commit(selected, diagnostics)
        selected_ids.append(selected)
        trace.append(compact_trace(diagnostics, selected, copy_active, penalty_applied))
        if selected == config.eos_token_id:
            break
        token = torch.tensor([[selected]], dtype=torch.long, device=device)
        generated = torch.cat((generated, token), dim=1)
        generated_only = torch.cat((generated_only, token), dim=1)
        if cached:
            output, cache = model.decode_step(token, cache)
        else:
            output = model(generated, attention_mask=torch.ones_like(generated))
    return selected_ids, trace


@torch.inference_mode()
def teacher_forced(
    model: torch.nn.Module,
    config: ModelConfig,
    prompt_ids: list[int],
    target_ids: list[int],
    source_key_position: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = prompt.clone()
    generated_only = generated[:, :0]
    cached = supports_incremental_cache(config)
    if cached:
        output, cache = model.prefill(prompt, torch.ones_like(prompt))
    else:
        output = model(prompt, attention_mask=torch.ones_like(prompt))
        cache = None
    span_ends, registered_keys, key_positions, payload_ids, payload_lengths = exact_span_metadata(
        len(prompt_ids), source_key_position, target_ids, device
    )
    memory = model.build_exact_memory(
        output["hidden"],
        prompt,
        torch.ones_like(prompt),
        source_length=len(prompt_ids),
        span_end_positions=span_ends,
        registered_key_mask=registered_keys,
        registered_key_positions=key_positions,
        registered_payload_ids=payload_ids,
        registered_payload_lengths=payload_lengths,
    )
    state = LexicalPointerDecodeState(source_length=len(prompt_ids), max_activation_steps=len(target_ids), max_copy_tokens=len(target_ids))
    ranks: list[int] = []
    probabilities: list[float] = []
    trace: list[dict[str, Any]] = []
    for gold in target_ids:
        logits, diagnostics, copy_active, penalty_applied = _prepare_step(model, config, output, generated, generated_only, memory, state)
        gold_logit = logits[0, gold]
        ranks.append(int((logits[0] > gold_logit).sum().item()) + 1)
        probabilities.append(float(F.softmax(logits, dim=-1)[0, gold].item()))
        state.commit(int(gold), diagnostics)
        trace.append(compact_trace(diagnostics, int(gold), copy_active, penalty_applied))
        token = torch.tensor([[gold]], dtype=torch.long, device=device)
        generated = torch.cat((generated, token), dim=1)
        generated_only = torch.cat((generated_only, token), dim=1)
        if cached:
            output, cache = model.decode_step(token, cache)
        else:
            output = model(generated, attention_mask=torch.ones_like(generated))
    return {
        "gold_ranks": ranks,
        "gold_probabilities": probabilities,
        "mean_gold_rank": sum(ranks) / max(len(ranks), 1),
        "mean_gold_probability": sum(probabilities) / max(len(probabilities), 1),
        "top1_rate": sum(rank == 1 for rank in ranks) / max(len(ranks), 1),
        "trace": trace,
    }


def _source_key_position(tokenizer: Any, prompt: str, target: str) -> int:
    char_start = prompt.index(target)
    prefix_ids = tokenizer.encode(prompt[:char_start], add_special_tokens=False)
    return len(prefix_ids) - 1


def prompt_target_token_alignment(
    prompt_ids: list[int],
    target_ids: list[int],
    source_key_position: int,
) -> dict[str, Any]:
    """Record whether independent target tokenization matches its prompt span.

    This is diagnostic evidence only. Exact-copy scoring remains unchanged and
    continues to use the key position immediately before the target span.
    """
    start = int(source_key_position) + 1
    observed = prompt_ids[start : start + len(target_ids)]
    return {
        "prompt_target_token_alignment_exact": observed == target_ids,
        "prompt_target_token_ids": observed,
    }


def evaluate_cell(model: torch.nn.Module, config: ModelConfig, tokenizer: Any, spec: MatrixSpec) -> dict[str, Any]:
    case = build_case(spec)
    prompt_ids = list(tokenizer.encode(case["prompt"], add_special_tokens=False))
    target_ids = list(tokenizer.encode(case["target"], add_special_tokens=False))
    base = {
        "cell_id": spec.cell_id,
        "length_chars": spec.length,
        "target_position": spec.position,
        "distractor_count": spec.distractors,
        "string_kind": spec.string_kind,
        "target": case["target"],
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "target_roundtrip_exact": tokenizer.decode(target_ids, skip_special_tokens=True) == case["target"],
    }
    if not target_ids:
        return base | {"status": "NOT TESTED", "reason": "target tokenized to an empty sequence"}
    if len(prompt_ids) + len(target_ids) > config.max_seq_len:
        return base | {"status": "NOT TESTED", "reason": f"requires {len(prompt_ids) + len(target_ids)} tokens but max_seq_len is {config.max_seq_len}"}
    source_key_position = _source_key_position(tokenizer, case["prompt"], case["target"])
    alignment = prompt_target_token_alignment(prompt_ids, target_ids, source_key_position)
    free_ids, free_trace = free_generate(
        model, config, prompt_ids, target_ids, source_key_position
    )
    free_text = tokenizer.decode(free_ids, skip_special_tokens=True)
    forced = teacher_forced(
        model, config, prompt_ids, target_ids, source_key_position
    )
    all_traces = free_trace + forced["trace"]
    first_positions = free_trace[0].get("source_positions", []) if free_trace else []
    top1_source_position = (
        int(first_positions[0][0])
        if first_positions and first_positions[0]
        else None
    )
    return base | text_metrics(free_text, case["target"], free_ids, target_ids) | {
        "status": "ok",
        "source_key_position": source_key_position,
        **alignment,
        "free_output": free_text,
        "free_output_token_ids": free_ids,
        "teacher_forced_gold_ranks": forced["gold_ranks"],
        "teacher_forced_gold_probabilities": forced["gold_probabilities"],
        "teacher_forced_mean_gold_rank": forced["mean_gold_rank"],
        "teacher_forced_mean_gold_probability": forced["mean_gold_probability"],
        "teacher_forced_top1_rate": forced["top1_rate"],
        "exact_retrieval_success": top1_source_position == source_key_position,
        "source_top1_position": top1_source_position,
        "copy_gate_activated": any(trace["copy_active"] for trace in all_traces),
        "max_copy_gate": max((float(trace.get("mix_gate") or 0.0) for trace in all_traces), default=0.0),
        "max_candidate_count": max((trace["candidate_count"] for trace in all_traces), default=0),
        "full_scan_used": any(trace["used_full_scan_fallback"] for trace in all_traces),
        "full_scan_count": max((trace["full_scan_count"] for trace in all_traces), default=0),
        "cursor_trace": free_trace,
        "teacher_forced_cursor_trace": forced["trace"],
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def csv_text(cells: list[dict[str, Any]]) -> str:
    preferred = ["cell_id", "length_chars", "target_position", "distractor_count", "string_kind", "status", "whole_string_exact", "character_accuracy", "token_accuracy", "character_edit_distance", "first_divergence", "teacher_forced_mean_gold_rank", "teacher_forced_mean_gold_probability", "free_output", "exact_retrieval_success", "copy_gate_activated", "max_candidate_count", "full_scan_used", "full_scan_count", "cursor_trace"]
    fields = list(preferred)
    for cell in cells:
        for key in cell:
            if key not in fields:
                fields.append(key)
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for cell in cells:
        writer.writerow({key: csv_value(value) for key, value in cell.items()})
    return handle.getvalue()


def write_atomic_bundle(output_dir: Path, report: dict[str, Any]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        json_path = temporary / "fdt_v4_exact_copy_audit.json"
        csv_path = temporary / "fdt_v4_exact_copy_matrix.csv"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        csv_path.write_text(csv_text(report["cells"]), encoding="utf-8", newline="")
        (temporary / "sha256.json").write_text(json.dumps({"fdt_v4_exact_copy_audit.json": sha256_file(json_path), "fdt_v4_exact_copy_matrix.csv": sha256_file(csv_path)}, indent=2) + "\n", encoding="ascii")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_audit(
    checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int = 20260823,
    device_name: str = "cpu",
    git_commit: str | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    tokenizer_dir, tokenizer_json = tokenizer_paths(tokenizer_path)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model, config, checkpoint_info = load_checkpoint(checkpoint, device)
    tokenizer = load_tokenizer(str(tokenizer_dir))
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"tokenizer vocabulary {len(tokenizer)} != checkpoint vocabulary {config.vocab_size}")
    cells = [evaluate_cell(model, config, tokenizer, spec) for spec in matrix_specs(seed)]
    tested = [cell for cell in cells if cell["status"] == "ok"]
    exact_pass = (
        len(tested) == 60
        and all(bool(cell.get("whole_string_exact")) for cell in tested)
        and all(bool(cell.get("exact_retrieval_success")) for cell in tested)
        and all(bool(cell.get("copy_gate_activated")) for cell in tested)
        and all(int(cell.get("full_scan_count", 0)) <= 1 for cell in tested)
    )
    exact_status = "PASS" if exact_pass else "FAIL" if tested else "NOT TESTED"
    report = {
        "schema": "fdt_v4_exact_copy_audit_v1",
        "created_at_unix": time.time(),
        "official_evaluation": {
            "device": str(device),
            "dtype": "float32",
            "quantization": "none",
            "gpu_launched": device.type == "cuda",
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda, "os": platform.platform()},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "git_commit": git_commit or "UNKNOWN"},
        "checkpoint": checkpoint_info,
        "tokenizer": {"path": str(tokenizer_dir), "tokenizer_json_sha256": sha256_file(tokenizer_json), "vocab_size": len(tokenizer)},
        "protocol": {"seed": seed, "matrix_cells": 60, "lengths_chars": list(LENGTHS), "positions": list(POSITIONS), "distractors": list(DISTRACTORS), "string_kinds": list(STRING_KINDS), "exact_mode": "copy", "memory_contract": "explicit_registered_span_with_standalone_payload", "retrieval_criterion": "registered_source_top1", "repetition_penalty": REPETITION_PENALTY, "repetition_scope": "generated", "copy_repetition_exemption": True, "temperature": 0.0, "decoding": "greedy", "model_runtime": "cached" if supports_incremental_cache(config) else "full_recompute_transition"},
        "summary": {"tested_cells": len(tested), "not_tested_cells": 60 - len(tested), "whole_string_exact_rate": sum(bool(cell.get("whole_string_exact")) for cell in tested) / max(len(tested), 1), "retrieval_success_rate": sum(bool(cell.get("exact_retrieval_success")) for cell in tested) / max(len(tested), 1), "copy_gate_activation_rate": sum(bool(cell.get("copy_gate_activated")) for cell in tested) / max(len(tested), 1), "max_full_scan_count": max((int(cell.get("full_scan_count", 0)) for cell in tested), default=0)},
        "audit_axes": {
            "EXACT_MEMORY": {
                "status": exact_status,
                "criterion": "all 60 cells must be exact, retrieve the target, activate copy, and scan the full source at most once",
            }
        },
        "cells": cells,
    }
    write_atomic_bundle(output_dir.resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-FP32 deterministic FDT v4 exact-copy matrix")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    report = run_audit(args.checkpoint, args.tokenizer, args.output_dir, args.seed, args.device, args.git_commit)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "tested_cells": report["summary"]["tested_cells"], "whole_string_exact_rate": report["summary"]["whole_string_exact_rate"], "device": args.device, "gpu_launched": args.device == "cuda"}))


if __name__ == "__main__":
    main()
