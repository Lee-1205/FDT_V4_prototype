from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdt_rlm.config import ModelConfig, load_yaml_like  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
from train_fdt_v4_curriculum_bridge import (  # noqa: E402
    atomic_json,
    convert_v20_state_dict,
    load_payload,
    require_c_path,
    sha256_file,
)


def load_validation_rows(directory: Path, limit: int) -> tuple[list[list[int]], dict[str, Any]]:
    directory = require_c_path(directory, "validation directory")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[list[int]] = []
    shards: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        shard_path = require_c_path(directory / str(shard["file"]), "validation shard")
        actual_sha = sha256_file(shard_path)
        expected_sha = str(shard["sha256"]).upper()
        if actual_sha != expected_sha:
            raise ValueError(f"validation shard hash mismatch: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        input_ids = payload["input_ids"]
        attention_mask = payload.get("attention_mask")
        for index in range(int(input_ids.size(0))):
            active = input_ids[index]
            if attention_mask is not None:
                active = active[attention_mask[index].bool()]
            ids = [int(value) for value in active.tolist()]
            if len(ids) >= 2:
                rows.append(ids)
            if len(rows) >= limit:
                break
        shards.append(
            {
                "path": str(shard_path),
                "sha256": actual_sha,
                "rows_declared": int(shard["rows"]),
            }
        )
        if len(rows) >= limit:
            break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} validation rows, found {len(rows)}")
    return rows, {
        "directory": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "shards": shards,
        "rows": len(rows),
        "active_tokens": sum(len(row) for row in rows),
    }


def load_generation_prompts(
    path: Path,
    tokenizer: Any,
    limit: int,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = require_c_path(path, "generation inputs")
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("category", "")) not in {"natural", "factual", "retrieval"}:
                continue
            ids = list(tokenizer.encode(str(row["prompt"]), add_special_tokens=False))
            if not ids:
                continue
            prompts.append(
                {
                    "category": str(row.get("category", "")),
                    "prompt": str(row["prompt"]),
                    "prompt_ids": [int(value) for value in ids[-max_prompt_tokens:]],
                }
            )
            if len(prompts) >= limit:
                break
    if len(prompts) != limit:
        raise ValueError(f"requested {limit} generation prompts, found {len(prompts)}")
    return prompts, {"path": str(path), "sha256": sha256_file(path), "rows": len(prompts)}


def release_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_parent_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    payload = load_payload(checkpoint)
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v3" or config.use_rope:
        raise ValueError("behavior comparator must be the learned-position V20 FDT v3")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, config, {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_candidate_model(
    raw_config: dict[str, Any],
    parent_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    config = ModelConfig(**raw_config["model"])
    if config.model_type != "fdt_v4":
        raise ValueError("candidate must be FDT v4")
    if config.rope_transition_alpha != 0.0:
        raise ValueError("behavior-preservation audit requires alpha=0")
    model = build_model(config)
    parent_payload = load_payload(parent_checkpoint)
    conversion = convert_v20_state_dict(model, parent_payload)
    del parent_payload
    model = model.to(device=device, dtype=torch.float32).eval()
    conversion["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    conversion["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, config, conversion


@torch.inference_mode()
def collect_reference(
    model: torch.nn.Module,
    rows: list[list[int]],
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    logits_by_row: list[torch.Tensor] = []
    loss_sum = 0.0
    correct = 0
    tokens = 0
    for ids in rows:
        tensor = torch.tensor([ids], device=device, dtype=torch.long)
        logits = model(tensor, attention_mask=torch.ones_like(tensor))["logits"][0, :-1].float()
        labels = tensor[0, 1:]
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int(logits.argmax(dim=-1).eq(labels).sum())
        tokens += int(labels.numel())
        logits_by_row.append(logits.cpu())
    return logits_by_row, {
        "nll": loss_sum / tokens,
        "top1": correct / tokens,
        "tokens": tokens,
        "finite": math.isfinite(loss_sum),
    }


@torch.inference_mode()
def compare_candidate(
    model: torch.nn.Module,
    rows: list[list[int]],
    references: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    loss_sum = 0.0
    correct = 0
    top1_agreement = 0
    tokens = 0
    max_abs = 0.0
    absolute_sum = 0.0
    compared_logits = 0
    exact_rows = 0
    row_metrics: list[dict[str, Any]] = []
    for row_index, (ids, reference_cpu) in enumerate(zip(rows, references)):
        tensor = torch.tensor([ids], device=device, dtype=torch.long)
        logits = model(tensor, attention_mask=torch.ones_like(tensor))["logits"][0, :-1].float()
        labels = tensor[0, 1:]
        reference = reference_cpu.to(device=device)
        difference = (logits - reference).abs()
        row_max = float(difference.max())
        row_mean = float(difference.mean())
        candidate_top1 = logits.argmax(dim=-1)
        reference_top1 = reference.argmax(dim=-1)
        row_exact = bool(torch.equal(logits, reference))
        exact_rows += int(row_exact)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int(candidate_top1.eq(labels).sum())
        top1_agreement += int(candidate_top1.eq(reference_top1).sum())
        tokens += int(labels.numel())
        max_abs = max(max_abs, row_max)
        absolute_sum += float(difference.sum())
        compared_logits += int(difference.numel())
        row_metrics.append(
            {
                "row_index": row_index,
                "tokens": int(labels.numel()),
                "max_abs_logit_delta": row_max,
                "mean_abs_logit_delta": row_mean,
                "bit_exact": row_exact,
            }
        )
        del reference, difference, logits
    return {
        "nll": loss_sum / tokens,
        "top1": correct / tokens,
        "top1_agreement": top1_agreement / tokens,
        "tokens": tokens,
        "max_abs_logit_delta": max_abs,
        "mean_abs_logit_delta": absolute_sum / compared_logits,
        "bit_exact_rows": exact_rows,
        "rows": row_metrics,
        "finite": math.isfinite(loss_sum) and math.isfinite(max_abs),
    }


@torch.inference_mode()
def greedy_full_recompute(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    max_seq_len: int,
    device: torch.device,
) -> list[int]:
    generated = list(prompt_ids)
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        context = generated[-max_seq_len:]
        tensor = torch.tensor([context], device=device, dtype=torch.long)
        logits = model(tensor, attention_mask=torch.ones_like(tensor))["logits"][:, -1]
        next_token = int(logits.argmax(dim=-1).item())
        generated.append(next_token)
        new_tokens.append(next_token)
    return new_tokens


def collect_generations(
    model: torch.nn.Module,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    max_seq_len: int,
    device: torch.device,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    records = []
    for index, prompt in enumerate(prompts):
        generated = greedy_full_recompute(
            model,
            prompt["prompt_ids"],
            max_new_tokens,
            max_seq_len,
            device,
        )
        records.append(
            {
                "row_index": index,
                "category": prompt["category"],
                "generated_ids": generated,
                "generated_text": tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ),
            }
        )
    return records


def compare_generations(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    exact = 0
    for new, old in zip(candidate, reference):
        is_exact = new["generated_ids"] == old["generated_ids"]
        exact += int(is_exact)
        first_divergence = next(
            (
                index
                for index, (new_id, old_id) in enumerate(
                    zip(new["generated_ids"], old["generated_ids"])
                )
                if new_id != old_id
            ),
            None,
        )
        rows.append(
            {
                **new,
                "reference_generated_ids": old["generated_ids"],
                "reference_generated_text": old["generated_text"],
                "exact": is_exact,
                "first_divergence": first_divergence,
            }
        )
    return {"exact_rows": exact, "total_rows": len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit behavior preservation before the FDT v4.1 RoPE transition"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--generation-inputs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--generation-rows", type=int, default=4)
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--routing-logit-quantum", type=float, default=None)
    parser.add_argument("--routing-boundary-smoothing-epsilon", type=float, default=None)
    parser.add_argument("--routing-boundary-extra-candidates", type=int, default=None)
    parser.add_argument("--routing-membership-quantum", type=float, default=None)
    parser.add_argument("--inference-prefix-stable-group-size", type=int, default=None)
    args = parser.parse_args()

    config_path = require_c_path(args.config, "config")
    parent_path = require_c_path(args.parent, "parent checkpoint")
    output_path = require_c_path(args.output, "audit output")
    if output_path.exists() or output_path.with_name(output_path.name + ".tmp").exists():
        raise FileExistsError(f"audit output is not fresh: {output_path}")
    raw_config = load_yaml_like(config_path)
    configured_quantum = float(
        args.routing_logit_quantum
        if args.routing_logit_quantum is not None
        else raw_config["model"].get("routing_logit_quantization", 0.0)
    )
    if configured_quantum < 0.0:
        raise ValueError("routing logit quantization cannot be negative")
    configured_smoothing = float(
        args.routing_boundary_smoothing_epsilon
        if args.routing_boundary_smoothing_epsilon is not None
        else raw_config["model"].get("routing_boundary_smoothing_epsilon", 0.0)
    )
    configured_extra_candidates = int(
        args.routing_boundary_extra_candidates
        if args.routing_boundary_extra_candidates is not None
        else raw_config["model"].get("routing_boundary_extra_candidates", 0)
    )
    if configured_smoothing < 0.0 or configured_extra_candidates < 0:
        raise ValueError("routing boundary smoothing settings cannot be negative")
    configured_membership_quantum = float(
        args.routing_membership_quantum
        if args.routing_membership_quantum is not None
        else raw_config["model"].get("routing_membership_quantization", 0.0)
    )
    if configured_membership_quantum < 0.0:
        raise ValueError("routing membership quantum cannot be negative")
    configured_inference_group = int(
        args.inference_prefix_stable_group_size
        if args.inference_prefix_stable_group_size is not None
        else raw_config["model"].get("inference_prefix_stable_group_size", 0)
    )
    if configured_inference_group < 0:
        raise ValueError("inference prefix stable group size cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer_dir = require_c_path(args.tokenizer, "tokenizer")
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer = load_tokenizer(str(tokenizer_dir))
    rows, validation_metadata = load_validation_rows(args.validation_dir, args.rows)
    prompts, generation_metadata = load_generation_prompts(
        args.generation_inputs,
        tokenizer,
        args.generation_rows,
        384,
    )

    parent_model: torch.nn.Module | None = None
    candidate_model: torch.nn.Module | None = None
    try:
        parent_model, parent_config, parent_metadata = load_parent_model(parent_path, device)
        reference_logits, reference_metrics = collect_reference(parent_model, rows, device)
        reference_generations = collect_generations(
            parent_model,
            prompts,
            args.generation_tokens,
            parent_config.max_seq_len,
            device,
            tokenizer,
        )
        release_model(parent_model)
        parent_model = None

        candidate_model, candidate_config, conversion = load_candidate_model(
            raw_config,
            parent_path,
            device,
        )
        candidate_config.routing_logit_quantization = 0.0
        candidate_config.routing_boundary_smoothing_epsilon = 0.0
        candidate_config.routing_boundary_extra_candidates = 0
        candidate_config.routing_membership_quantization = 0.0
        candidate_config.inference_prefix_stable_group_size = 0
        exact_metrics = compare_candidate(candidate_model, rows, reference_logits, device)
        exact_generations = collect_generations(
            candidate_model,
            prompts,
            args.generation_tokens,
            candidate_config.max_seq_len,
            device,
            tokenizer,
        )

        candidate_config.routing_logit_quantization = configured_quantum
        candidate_config.routing_boundary_smoothing_epsilon = configured_smoothing
        candidate_config.routing_boundary_extra_candidates = configured_extra_candidates
        candidate_config.routing_membership_quantization = configured_membership_quantum
        candidate_config.inference_prefix_stable_group_size = configured_inference_group
        for module in candidate_model.modules():
            if hasattr(module, "inference_group_size"):
                module.inference_group_size = configured_inference_group
        configured_metrics = compare_candidate(candidate_model, rows, reference_logits, device)
        configured_generations = collect_generations(
            candidate_model,
            prompts,
            args.generation_tokens,
            candidate_config.max_seq_len,
            device,
            tokenizer,
        )
    finally:
        release_model(parent_model)
        release_model(candidate_model)

    exact_generation_comparison = compare_generations(
        exact_generations,
        reference_generations,
    )
    configured_generation_comparison = compare_generations(
        configured_generations,
        reference_generations,
    )
    exact_nll_delta = exact_metrics["nll"] - reference_metrics["nll"]
    configured_nll_delta = configured_metrics["nll"] - reference_metrics["nll"]
    configured_nll_relative = configured_nll_delta / reference_metrics["nll"]
    checks = {
        "parent_sha_matches_config": parent_metadata["sha256"]
        == "7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179",
        "alpha_zero_quantum_zero_nll_exact": abs(exact_nll_delta) <= 1e-8,
        "alpha_zero_quantum_zero_logits_exact": exact_metrics["max_abs_logit_delta"] == 0.0,
        "alpha_zero_quantum_zero_top1_exact": exact_metrics["top1_agreement"] == 1.0,
        "alpha_zero_quantum_zero_generation_exact": exact_generation_comparison["exact_rows"]
        == exact_generation_comparison["total_rows"],
        "configured_nll_relative_within_0_1pct": abs(configured_nll_relative) <= 0.001,
        "configured_top1_agreement_at_least_99_9pct": configured_metrics["top1_agreement"]
        >= 0.999,
        "configured_max_abs_logit_delta_within_0_01": configured_metrics[
            "max_abs_logit_delta"
        ]
        <= 0.01,
        "configured_generation_exact": configured_generation_comparison["exact_rows"]
        == configured_generation_comparison["total_rows"],
        "all_metrics_finite": bool(
            reference_metrics["finite"]
            and exact_metrics["finite"]
            and configured_metrics["finite"]
        ),
    }
    report = {
        "schema": "fdt_v4_1_behavior_preservation_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "official_operations": "unquantized_fp32",
        "device": str(device),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "tokenizer": {
            "directory": str(tokenizer_dir),
            "tokenizer_json_sha256": sha256_file(tokenizer_json),
        },
        "parent": parent_metadata,
        "validation": validation_metadata,
        "generation_inputs": generation_metadata,
        "conversion": conversion,
        "reference": {
            "metrics": reference_metrics,
            "generations": reference_generations,
        },
        "alpha_zero_quantum_zero": {
            "metrics": exact_metrics,
            "nll_delta": exact_nll_delta,
            "generation_comparison": exact_generation_comparison,
        },
        "alpha_zero_configured_quantum": {
            "routing_logit_quantization": configured_quantum,
            "routing_boundary_smoothing_epsilon": configured_smoothing,
            "routing_boundary_extra_candidates": configured_extra_candidates,
            "routing_membership_quantization": configured_membership_quantum,
            "inference_prefix_stable_group_size": configured_inference_group,
            "metrics": configured_metrics,
            "nll_delta": configured_nll_delta,
            "nll_relative_delta": configured_nll_relative,
            "generation_comparison": configured_generation_comparison,
        },
        "checks": checks,
        "decision": (
            "ALLOW_BEHAVIOR_PRESERVING_TRANSITION_PILOT"
            if all(checks.values())
            else "BLOCK_TRANSITION_AND_REPAIR"
        ),
        "limitations": [
            "This gate verifies the alpha-zero conversion and configured stable routing on fixed 512-token validation rows.",
            "It does not claim that the untrained Exact Memory pointer retrieves a correct source span.",
            "It does not authorize the later main curriculum until the transition pilot and objective pilots pass.",
        ],
    }
    atomic_json(output_path, report)
    atomic_json(
        output_path.with_name(output_path.name + ".sha256.json"),
        {"path": str(output_path), "sha256": sha256_file(output_path)},
    )
    print(json.dumps({"status": report["status"], "decision": report["decision"], "output": str(output_path)}))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
