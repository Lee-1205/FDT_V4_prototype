from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT.parent / "source" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def anchor_state_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(anchor_state_diagnostics(left, right)["raw_max_abs_error"])


def _chronological_local_state(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(state.count)
    capacity = int(state.key.size(2))
    if count < capacity:
        order = torch.arange(count, device=state.key.device)
    else:
        order = (torch.arange(capacity, device=state.key.device) + int(state.cursor)) % capacity
    return (
        state.key.index_select(2, order),
        state.value.index_select(2, order),
        state.mask.index_select(1, order),
    )


def local_state_diagnostics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_length = int(left.get("length", -1))
    right_length = int(right.get("length", -1))
    if left_length != right_length:
        return {
            "structure_match": False,
            "length_match": False,
            "left_length": left_length,
            "right_length": right_length,
            "max_abs_error": float("inf"),
            "layers": [],
        }
    if len(left.get("layers", [])) != len(right.get("layers", [])):
        return {"structure_match": False, "length_match": True, "max_abs_error": float("inf"), "layers": []}
    rows = []
    maximum = 0.0
    for layer_index, ((left_local, _), (right_local, _)) in enumerate(
        zip(left["layers"], right["layers"])
    ):
        if left_local is None or right_local is None:
            if left_local is not right_local:
                return {"structure_match": False, "max_abs_error": float("inf"), "layers": rows}
            continue
        if int(left_local.count) != int(right_local.count):
            return {"structure_match": False, "max_abs_error": float("inf"), "layers": rows}
        left_key, left_value, left_mask = _chronological_local_state(left_local)
        right_key, right_value, right_mask = _chronological_local_state(right_local)
        if not torch.equal(left_mask, right_mask):
            return {"structure_match": False, "max_abs_error": float("inf"), "layers": rows}
        key_error = _tensor_error(left_key, right_key)["max_abs_error"]
        value_error = _tensor_error(left_value, right_value)["max_abs_error"]
        layer_error = max(key_error, value_error)
        maximum = max(maximum, layer_error)
        rows.append(
            {
                "layer_index": layer_index,
                "count": int(left_local.count),
                "key_max_abs_error": key_error,
                "value_max_abs_error": value_error,
                "max_abs_error": layer_error,
            }
        )
    return {
        "structure_match": True,
        "length_match": True,
        "left_length": left_length,
        "right_length": right_length,
        "max_abs_error": maximum,
        "layers": rows,
    }


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    comparison_dtype = torch.promote_types(left.dtype, right.dtype)
    if comparison_dtype in {torch.float16, torch.bfloat16}:
        comparison_dtype = torch.float32
    left_values = left.to(dtype=comparison_dtype)
    right_values = right.to(dtype=comparison_dtype)
    difference = (left_values - right_values).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    scale = max(
        float(left_values.abs().max().item()) if left.numel() else 0.0,
        float(right_values.abs().max().item()) if right.numel() else 0.0,
        1e-12,
    )
    return {"max_abs_error": maximum, "scale": scale, "scale_relative_error": maximum / scale}


def anchor_state_diagnostics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Separate raw accumulator error from the summary consumed by decode.

    Numerator and mass grow with context length.  Their raw absolute error is
    retained as the strict integrity criterion, while normalized summaries and
    repeat-recompute measurements distinguish a state semantic mismatch from
    floating-point reduction-order sensitivity.
    """
    layers: list[dict[str, Any]] = []
    if len(left.get("layers", [])) != len(right.get("layers", [])):
        return {
            "structure_match": False,
            "raw_max_abs_error": float("inf"),
            "normalized_max_abs_error": float("inf"),
            "layers": [],
        }
    raw_errors: list[float] = []
    normalized_errors: list[float] = []
    for layer_index, ((_, left_anchor), (_, right_anchor)) in enumerate(zip(left["layers"], right["layers"])):
        if left_anchor is None or right_anchor is None:
            if left_anchor is not right_anchor:
                return {
                    "structure_match": False,
                    "raw_max_abs_error": float("inf"),
                    "normalized_max_abs_error": float("inf"),
                    "layers": layers,
                }
            continue
        left_values = vars(left_anchor)
        right_values = vars(right_anchor)
        if left_values.keys() != right_values.keys():
            return {
                "structure_match": False,
                "raw_max_abs_error": float("inf"),
                "normalized_max_abs_error": float("inf"),
                "layers": layers,
            }
        if not isinstance(left_values.get("numerator"), torch.Tensor) or not isinstance(left_values.get("mass"), torch.Tensor):
            return {
                "structure_match": False,
                "raw_max_abs_error": float("inf"),
                "normalized_max_abs_error": float("inf"),
                "layers": layers,
            }
        numerator = _tensor_error(left_values["numerator"], right_values["numerator"])
        mass = _tensor_error(left_values["mass"], right_values["mass"])
        left_normalized = bool(left_values.get("normalized", False))
        right_normalized = bool(right_values.get("normalized", False))
        if left_normalized != right_normalized:
            return {
                "structure_match": False,
                "raw_max_abs_error": float("inf"),
                "normalized_max_abs_error": float("inf"),
                "layers": layers,
            }
        if left_normalized:
            left_summary = left_values["numerator"].float()
            right_summary = right_values["numerator"].float()
        else:
            left_summary = left_values["numerator"].float() / left_values["mass"].float().clamp_min(1e-6).unsqueeze(-1)
            right_summary = right_values["numerator"].float() / right_values["mass"].float().clamp_min(1e-6).unsqueeze(-1)
        active = (left_values["mass"].float() > 0) | (right_values["mass"].float() > 0)
        summary_difference = (left_summary - right_summary).abs()
        normalized_error = float(summary_difference[active.unsqueeze(-1).expand_as(summary_difference)].max().item()) if active.any() else 0.0
        raw_error = max(numerator["max_abs_error"], mass["max_abs_error"])
        raw_errors.append(raw_error)
        normalized_errors.append(normalized_error)
        layers.append(
            {
                "layer_index": layer_index,
                "numerator": numerator,
                "mass": mass,
                "raw_max_abs_error": raw_error,
                "normalized_summary_max_abs_error": normalized_error,
                "state_representation": "weighted_mean" if left_normalized else "numerator",
            }
        )
    return {
        "structure_match": True,
        "raw_max_abs_error": max(raw_errors, default=0.0),
        "normalized_max_abs_error": max(normalized_errors, default=0.0),
        "layers": layers,
    }


def classify_anchor_mismatch(
    state: dict[str, Any],
    repeat_recompute: dict[str, Any],
    prefill_error: float,
    decode_error: float,
    tolerance: float,
) -> dict[str, Any]:
    raw_error = float(state["raw_max_abs_error"])
    normalized_error = float(state["normalized_max_abs_error"])
    repeat_raw_error = float(repeat_recompute["raw_max_abs_error"])
    repeat_normalized_error = float(repeat_recompute["normalized_max_abs_error"])
    normalized_excess = max(normalized_error - repeat_normalized_error, 0.0)
    common = {
        "incremental_vs_full_raw_error": raw_error,
        "incremental_vs_full_normalized_error": normalized_error,
        "full_vs_full_repeat_raw_error": repeat_raw_error,
        "full_vs_full_repeat_normalized_error": repeat_normalized_error,
        "normalized_excess_over_repeat_baseline": normalized_excess,
        "tolerance": float(tolerance),
        "raw_status_relaxed": False,
    }
    if not state.get("structure_match", False) or prefill_error > tolerance or decode_error > tolerance:
        return {
            "classification": "POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT",
            "severity": "major",
            "requires_sol": True,
            "reason": "Cache structure or observable logits exceed the unchanged tolerance.",
            **common,
        }
    if normalized_error > tolerance:
        if repeat_normalized_error > tolerance and normalized_excess <= tolerance:
            return {
                "classification": "INDETERMINATE_CUDA_REDUCTION_BASELINE",
                "severity": "measurement",
                "requires_sol": False,
                "reason": "The full-vs-full normalized repeat baseline already fails, and incremental-vs-full adds no error above that baseline beyond the unchanged tolerance. Raw integrity remains FAIL; absence of causal drift is not proven.",
                **common,
            }
        return {
            "classification": "POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT",
            "severity": "major",
            "requires_sol": True,
            "reason": "Normalized incremental-vs-full error exceeds both the unchanged tolerance and the measured full-vs-full repeat baseline by more than tolerance.",
            **common,
        }
    if raw_error <= tolerance:
        return {"classification": "MATCH", "severity": "none", "requires_sol": False, **common}
    if repeat_raw_error > tolerance or repeat_normalized_error > tolerance:
        return {
            "classification": "COMPARATOR_EXPOSED_NONDETERMINISTIC_REDUCTION",
            "severity": "measurement",
            "requires_sol": False,
            "reason": "Two full recomputes already disagree above tolerance; raw scatter reduction is not a stable equality oracle.",
            **common,
        }
    return {
        "classification": "FLOAT_ACCUMULATION_ORDER_CANDIDATE",
        "severity": "diagnostic",
        "requires_sol": False,
        "reason": "Raw accumulators exceed tolerance while consumed normalized summaries and logits do not. Raw FAIL is retained pending device evidence.",
        **common,
    }


def strict_integrity_status(
    prefill_agreement: bool,
    decode_agreement: bool,
    prefill_error: float,
    decode_error: float,
    raw_anchor_state_error: float,
    tolerance: float,
) -> str:
    """Keep the original raw absolute criterion independent of diagnosis."""
    return (
        "PASS"
        if prefill_agreement
        and decode_agreement
        and prefill_error <= tolerance
        and decode_error <= tolerance
        and raw_anchor_state_error <= tolerance
        else "FAIL"
    )


def evaluate_context(model, config: ModelConfig, context: int, tolerance: float) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = ((torch.arange(context, device="cuda", dtype=torch.long) * 65537 + 29) % config.vocab_size).unsqueeze(0)
    mask = torch.ones_like(ids)
    record: dict[str, Any] = {
        "context": context,
        "prompt_context": context,
        "extended_context": context + 1,
        "dtype": "float32",
        "quantization": "none",
    }
    try:
        with torch.inference_mode():
            full = model(ids, attention_mask=mask)
            full_last = full["logits"][:, -1].clone()
            del full
            prefill, cache = model.prefill(ids, mask)
            prefill_error = float((prefill["logits"][:, -1].float() - full_last.float()).abs().max().item())
            prefill_agreement = bool(prefill["logits"][:, -1].argmax(-1).eq(full_last.argmax(-1)).all())
            next_id = full_last.argmax(dim=-1, keepdim=True)
            incremental, incremental_cache = model.decode_step(next_id, cache)
            extended = torch.cat((ids, next_id), dim=1)
            extended_mask = torch.ones_like(extended)
            recomputed = model(extended, attention_mask=extended_mask)
            recomputed_last = recomputed["logits"][:, -1].clone()
            del recomputed
            decode_error = float((incremental["logits"][:, -1].float() - recomputed_last.float()).abs().max().item())
            decode_agreement = bool(incremental["logits"][:, -1].argmax(-1).eq(recomputed_last.argmax(-1)).all())
            recomputed_output, recomputed_cache = model.prefill(extended, extended_mask)
            del recomputed_output
            repeat_output, repeat_recomputed_cache = model.prefill(extended, extended_mask)
            del repeat_output
            state = anchor_state_diagnostics(incremental_cache, recomputed_cache)
            repeat_state = anchor_state_diagnostics(recomputed_cache, repeat_recomputed_cache)
            local_state = local_state_diagnostics(incremental_cache, recomputed_cache)
            repeat_local_state = local_state_diagnostics(
                recomputed_cache, repeat_recomputed_cache
            )
            state_error = max(
                float(state["raw_max_abs_error"]),
                float(local_state["max_abs_error"]),
            )
            diagnosis = classify_anchor_mismatch(state, repeat_state, prefill_error, decode_error, tolerance)
            if (
                not local_state.get("structure_match", False)
                or float(local_state["max_abs_error"]) > tolerance
            ):
                diagnosis = {
                    "classification": "LOCAL_KV_CACHE_DRIFT",
                    "severity": "major",
                    "requires_sol": True,
                    "reason": "Chronological local KV state exceeds the unchanged tolerance.",
                    "local_kv_state_max_abs_error": local_state["max_abs_error"],
                    "tolerance": tolerance,
                    "raw_status_relaxed": False,
                }
        record.update(
            {
                "prefill_full_max_abs_logit_error": prefill_error,
                "prefill_full_token_agreement": prefill_agreement,
                "decode_full_max_abs_logit_error": decode_error,
                "decode_full_token_agreement": decode_agreement,
                "anchor_state_max_abs_error": state_error,
                "anchor_state_raw_max_abs_error": state_error,
                "anchor_state_normalized_max_abs_error": state["normalized_max_abs_error"],
                "anchor_state_component_diagnostics": state,
                    "local_kv_state_max_abs_error": local_state["max_abs_error"],
                    "local_kv_state_diagnostics": local_state,
                    "repeat_recompute_local_kv_state_max_abs_error": repeat_local_state[
                        "max_abs_error"
                    ],
                    "repeat_recompute_local_kv_state_diagnostics": repeat_local_state,
                "repeat_recompute_anchor_state_raw_max_abs_error": repeat_state["raw_max_abs_error"],
                "repeat_recompute_anchor_state_normalized_max_abs_error": repeat_state["normalized_max_abs_error"],
                "anchor_state_mismatch_diagnosis": diagnosis,
                "tolerance": tolerance,
                "status": strict_integrity_status(
                    prefill_agreement,
                    decode_agreement,
                    prefill_error,
                    decode_error,
                    state_error,
                    tolerance,
                ),
            }
        )
        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    except torch.cuda.OutOfMemoryError as error:
        record.update({"status": "FAIL", "failure": "CUDA_OUT_OF_MEMORY", "error": str(error)})
        torch.cuda.empty_cache()
    except Exception as error:
        record.update({"status": "FAIL", "failure": type(error).__name__, "error": str(error)})
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit main FDT v4 cached/full FP32 integrity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16383])
    parser.add_argument("--tolerance", type=float, default=3e-4)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for main-checkpoint integrity audit")
    if args.output.exists():
        raise FileExistsError("cache-integrity output is immutable and must use a fresh path")
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32).eval()
    rows = [evaluate_context(model, config, int(context), float(args.tolerance)) for context in args.contexts]
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_stage_status": payload.get("stage_status"),
        "dtype": "float32",
        "quantization": "none",
        "gpu": torch.cuda.get_device_name(0),
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "git_commit": args.git_commit or "UNKNOWN"},
        "contexts": rows,
        "audit_axes": {
            "INFERENCE_INTEGRITY": {
                "status": "PASS" if rows and all(row.get("status") == "PASS" for row in rows) else "FAIL"
            },
            "LONG_CONTEXT": {
                "status": "PASS"
                if rows
                and all(row.get("status") == "PASS" for row in rows)
                and any(row.get("context") == 8192 for row in rows)
                and any(row.get("extended_context") == 16384 for row in rows)
                else "FAIL"
            },
        },
        "anchor_state_policy": {
            "raw_absolute_tolerance_unchanged": float(args.tolerance),
            "raw_state_failure_is_not_relaxed": True,
            "normalized_summary_is_diagnostic_only": True,
            "repeat_recompute_is_measurement_nondeterminism_control": True,
            "normalized_drift_requires_excess_over_repeat_baseline": True,
            "repeat_baseline_does_not_change_raw_status": True,
        },
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
