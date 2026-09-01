from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def shard_paths(directory: Path, split: str) -> list[Path]:
    paths = sorted((directory / "shards" / split).glob("*.pt"))
    if not paths:
        paths = sorted((directory / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards under {directory}")
    return paths


def closes_degenerate_ngram(
    generated: list[int],
    candidate: int,
    *,
    order: int = 3,
    prior_occurrences: int = 2,
    window: int = 96,
) -> tuple[bool, int]:
    trial = [*generated[-max(int(window), order):], int(candidate)]
    if len(trial) < order:
        return False, 0
    closing = tuple(trial[-order:])
    earlier = trial[:-1]
    occurrences = sum(
        tuple(earlier[index : index + order]) == closing
        for index in range(max(len(earlier) - order + 1, 0))
    )
    return occurrences >= int(prior_occurrences), occurrences


def construct_recovery_row(
    source: list[int],
    prompt_length: int,
    generated_prefix: list[int],
    negative_token: int,
    *,
    sequence_length: int,
    recovery_tokens: int,
    ngram_order: int,
    prior_occurrences: int,
) -> dict[str, torch.Tensor] | None:
    recovery_start = int(prompt_length) + len(generated_prefix)
    clean = source[recovery_start : recovery_start + int(recovery_tokens)]
    prefix = source[:prompt_length] + list(generated_prefix)
    if not clean or int(negative_token) == int(clean[0]):
        return None
    active = prefix + clean
    if len(active) > int(sequence_length):
        return None
    boundary = len(prefix)
    input_ids = torch.zeros(sequence_length, dtype=torch.int32)
    labels = torch.full((sequence_length,), -100, dtype=torch.int32)
    attention_mask = torch.zeros(sequence_length, dtype=torch.uint8)
    negative_ids = torch.full((sequence_length,), -1, dtype=torch.int32)
    negative_mask = torch.zeros(sequence_length, dtype=torch.uint8)
    occurrence_counts = torch.zeros(sequence_length, dtype=torch.uint8)
    input_ids[: len(active)] = torch.tensor(active, dtype=torch.int32)
    labels[boundary : len(active)] = torch.tensor(clean, dtype=torch.int32)
    attention_mask[: len(active)] = 1
    negative_ids[boundary] = int(negative_token)
    negative_mask[boundary] = 1
    occurrence_counts[boundary] = min(int(prior_occurrences), 255)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loop_negative_ids": negative_ids,
        "loop_negative_mask": negative_mask,
        "loop_negative_prior_occurrences": occurrence_counts,
        "recovery_boundary": torch.tensor(boundary, dtype=torch.int32),
        "loop_ngram_order": torch.tensor(ngram_order, dtype=torch.uint8),
    }


def construct_trajectory_unlikelihood_row(
    prompt: list[int],
    generated: list[int],
    loop_events: list[dict[str, Any]],
    *,
    sequence_length: int,
    ngram_order: int,
) -> dict[str, torch.Tensor] | None:
    """Preserve a real failed trajectory and mark only its loop-closing tokens."""
    active = list(prompt) + list(generated)
    if not generated or not loop_events or len(active) > int(sequence_length):
        return None
    input_ids = torch.zeros(sequence_length, dtype=torch.int32)
    labels = torch.full((sequence_length,), -100, dtype=torch.int32)
    attention_mask = torch.zeros(sequence_length, dtype=torch.uint8)
    negative_ids = torch.full((sequence_length,), -1, dtype=torch.int32)
    negative_mask = torch.zeros(sequence_length, dtype=torch.uint8)
    occurrence_counts = torch.zeros(sequence_length, dtype=torch.uint8)
    input_ids[: len(active)] = torch.tensor(active, dtype=torch.int32)
    attention_mask[: len(active)] = 1
    for event in loop_events:
        position = len(prompt) + int(event["generated_index"])
        token = int(event["negative_token"])
        if position <= 0 or position >= len(active) or int(active[position]) != token:
            raise ValueError("loop event is not aligned to its generated trajectory")
        negative_ids[position] = token
        negative_mask[position] = 1
        occurrence_counts[position] = min(int(event["prior_occurrences"]), 255)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loop_negative_ids": negative_ids,
        "loop_negative_mask": negative_mask,
        "loop_negative_prior_occurrences": occurrence_counts,
        "recovery_boundary": torch.tensor(len(prompt), dtype=torch.int32),
        "loop_ngram_order": torch.tensor(ngram_order, dtype=torch.uint8),
        "loop_unlikelihood_only": torch.tensor(1, dtype=torch.uint8),
    }


def active_row(payload: dict[str, torch.Tensor], index: int) -> list[int]:
    ids = payload["input_ids"][index]
    mask = payload["attention_mask"][index].bool()
    return [int(value) for value in ids[mask].tolist()]


def active_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    active = ids[mask.bool()].to(torch.int32).contiguous().numpy().tobytes()
    return hashlib.sha256(active).hexdigest().upper()


def dataset_active_hashes(directory: Path, split: str) -> set[str]:
    hashes: set[str] = set()
    for path in shard_paths(directory, split):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for ids, mask in zip(payload["input_ids"], payload["attention_mask"]):
            hashes.add(active_hash(ids, mask))
    return hashes


def load_sources(
    sources: dict[str, Path],
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for category, directory in sources.items():
        manifest = directory / "manifest.json"
        category_shards = []
        rows = 0
        for path in shard_paths(directory, split):
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            category_shards.append({"path": path, "payload": payload})
            rows += int(payload["input_ids"].size(0))
        loaded[category] = category_shards
        metadata[category] = {
            "directory": str(directory),
            "manifest_sha256": sha256_file(manifest),
            "rows": rows,
            "shards": [
                {"path": str(item["path"]), "sha256": sha256_file(item["path"])}
                for item in category_shards
            ],
        }
    return loaded, metadata


def source_candidates(
    loaded: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, int, int]]:
    candidates = []
    for category, shards in loaded.items():
        for shard_index, item in enumerate(shards):
            rows = int(item["payload"]["input_ids"].size(0))
            candidates.extend((category, shard_index, row) for row in range(rows))
    return candidates


def load_frozen_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    config = ModelConfig(**payload["model_config"])
    if config.model_type not in {"fdt_v3", "fdt_v4"}:
        raise ValueError("generated-prefix builder requires an FDT v3/v4 checkpoint")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, config, {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "model_type": config.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "float32",
        "quantization": "none",
    }


@torch.inference_mode()
def generate_to_loop_closure(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    ngram_order: int,
    prior_occurrences: int,
    ngram_window: int,
    device: torch.device,
) -> dict[str, Any] | None:
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    config = model.config
    cached = not (
        getattr(config, "rope_transition_mode", "lerp") == "output_blend"
        and 0.0 < float(getattr(config, "rope_transition_alpha", 1.0)) < 1.0
    )
    if cached:
        output, cache = model.prefill(prompt, torch.ones_like(prompt))
    else:
        output = model(prompt, attention_mask=torch.ones_like(prompt))
        cache = None
    generated_tensor = prompt
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        candidate = int(output["logits"][:, -1].argmax(dim=-1).item())
        closes, occurrences = closes_degenerate_ngram(
            generated,
            candidate,
            order=ngram_order,
            prior_occurrences=prior_occurrences,
            window=ngram_window,
        )
        if closes:
            return {
                "generated_prefix": generated,
                "negative_token": candidate,
                "prior_occurrences": occurrences,
                "closing_ngram": [*generated, candidate][-ngram_order:],
            }
        if candidate == int(eos_token_id):
            return None
        generated.append(candidate)
        token = torch.tensor([[candidate]], device=device, dtype=torch.long)
        if cached:
            output, cache = model.decode_step(token, cache)
        else:
            generated_tensor = torch.cat((generated_tensor, token), dim=1)
            output = model(
                generated_tensor,
                attention_mask=torch.ones_like(generated_tensor),
            )
    return None


@torch.inference_mode()
def generate_batch_to_loop_closure(
    model: torch.nn.Module,
    prompt_rows: list[list[int]],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    ngram_order: int,
    prior_occurrences: int,
    ngram_window: int,
    device: torch.device,
) -> list[dict[str, Any] | None]:
    """Generate equal-length intermediate-transition prompts in one FP32 batch."""
    if not prompt_rows:
        return []
    if len({len(row) for row in prompt_rows}) != 1:
        raise ValueError("batched generated-prefix prompts must have equal lengths")
    config = model.config
    full_recompute = bool(
        getattr(config, "rope_transition_mode", "lerp") == "output_blend"
        and 0.0 < float(getattr(config, "rope_transition_alpha", 1.0)) < 1.0
    )
    if not full_recompute:
        return [
            generate_to_loop_closure(
                model,
                row,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                ngram_order=ngram_order,
                prior_occurrences=prior_occurrences,
                ngram_window=ngram_window,
                device=device,
            )
            for row in prompt_rows
        ]

    generated_tensor = torch.tensor(prompt_rows, device=device, dtype=torch.long)
    generated_rows: list[list[int]] = [[] for _ in prompt_rows]
    active_original = list(range(len(prompt_rows)))
    results: list[dict[str, Any] | None] = [None for _ in prompt_rows]
    output = model(generated_tensor, attention_mask=torch.ones_like(generated_tensor))
    for _ in range(int(max_new_tokens)):
        candidates = output["logits"][:, -1].argmax(dim=-1).tolist()
        continuing: list[int] = []
        next_tokens: list[int] = []
        for local_index, candidate_value in enumerate(candidates):
            candidate = int(candidate_value)
            generated = generated_rows[local_index]
            closes, occurrences = closes_degenerate_ngram(
                generated,
                candidate,
                order=ngram_order,
                prior_occurrences=prior_occurrences,
                window=ngram_window,
            )
            if closes:
                results[active_original[local_index]] = {
                    "generated_prefix": list(generated),
                    "negative_token": candidate,
                    "prior_occurrences": occurrences,
                    "closing_ngram": [*generated, candidate][-ngram_order:],
                }
            elif candidate != int(eos_token_id):
                generated.append(candidate)
                continuing.append(local_index)
                next_tokens.append(candidate)
        if not continuing:
            break
        keep = torch.tensor(continuing, device=device, dtype=torch.long)
        generated_tensor = generated_tensor.index_select(0, keep)
        generated_tensor = torch.cat(
            (
                generated_tensor,
                torch.tensor(next_tokens, device=device, dtype=torch.long).unsqueeze(1),
            ),
            dim=1,
        )
        active_original = [active_original[index] for index in continuing]
        generated_rows = [generated_rows[index] for index in continuing]
        output = model(generated_tensor, attention_mask=torch.ones_like(generated_tensor))
    return results


@torch.inference_mode()
def generate_batch_loop_trajectories(
    model: torch.nn.Module,
    prompt_rows: list[list[int]],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    ngram_order: int,
    prior_occurrences: int,
    ngram_window: int,
    device: torch.device,
    backend: str = "full_recompute",
) -> list[dict[str, Any] | None]:
    """Collect every third-or-later n-gram closure on each greedy trajectory."""
    if not prompt_rows:
        return []
    if len({len(row) for row in prompt_rows}) != 1:
        raise ValueError("batched trajectory prompts must have equal lengths")
    if backend not in {"full_recompute", "incremental_cache"}:
        raise ValueError(f"unsupported trajectory generation backend: {backend}")
    generated_tensor = torch.tensor(prompt_rows, device=device, dtype=torch.long)
    generated_rows: list[list[int]] = [[] for _ in prompt_rows]
    event_rows: list[list[dict[str, Any]]] = [[] for _ in prompt_rows]
    if backend == "incremental_cache":
        output, cache = model.prefill(
            generated_tensor,
            attention_mask=torch.ones_like(generated_tensor),
        )
        finished = [False for _ in prompt_rows]
        for _ in range(int(max_new_tokens)):
            candidates = output["logits"][:, -1].argmax(dim=-1).tolist()
            next_tokens: list[int] = []
            for row_index, candidate_value in enumerate(candidates):
                candidate = int(candidate_value)
                if finished[row_index]:
                    next_tokens.append(int(eos_token_id))
                    continue
                if candidate == int(eos_token_id):
                    finished[row_index] = True
                    next_tokens.append(candidate)
                    continue
                generated = generated_rows[row_index]
                closes, occurrences = closes_degenerate_ngram(
                    generated,
                    candidate,
                    order=ngram_order,
                    prior_occurrences=prior_occurrences,
                    window=ngram_window,
                )
                generated_index = len(generated)
                generated.append(candidate)
                if closes:
                    event_rows[row_index].append(
                        {
                            "generated_index": generated_index,
                            "negative_token": candidate,
                            "prior_occurrences": occurrences,
                            "closing_ngram": generated[-ngram_order:],
                        }
                    )
                next_tokens.append(candidate)
            if all(finished):
                break
            token = torch.tensor(next_tokens, device=device, dtype=torch.long).unsqueeze(1)
            output, cache = model.decode_step(token, cache)
        return [
            {
                "generated_prefix": list(generated),
                "loop_events": list(events),
            }
            if events
            else None
            for generated, events in zip(generated_rows, event_rows)
        ]

    active_original = list(range(len(prompt_rows)))
    results: list[dict[str, Any] | None] = [None for _ in prompt_rows]
    output = model(generated_tensor, attention_mask=torch.ones_like(generated_tensor))
    for _ in range(int(max_new_tokens)):
        candidates = output["logits"][:, -1].argmax(dim=-1).tolist()
        continuing: list[int] = []
        next_tokens: list[int] = []
        for local_index, candidate_value in enumerate(candidates):
            candidate = int(candidate_value)
            generated = generated_rows[local_index]
            events = event_rows[local_index]
            if candidate == int(eos_token_id):
                if events:
                    results[active_original[local_index]] = {
                        "generated_prefix": list(generated),
                        "loop_events": list(events),
                    }
                continue
            closes, occurrences = closes_degenerate_ngram(
                generated,
                candidate,
                order=ngram_order,
                prior_occurrences=prior_occurrences,
                window=ngram_window,
            )
            generated_index = len(generated)
            generated.append(candidate)
            if closes:
                events.append(
                    {
                        "generated_index": generated_index,
                        "negative_token": candidate,
                        "prior_occurrences": occurrences,
                        "closing_ngram": generated[-ngram_order:],
                    }
                )
            continuing.append(local_index)
            next_tokens.append(candidate)
        if not continuing:
            break
        keep = torch.tensor(continuing, device=device, dtype=torch.long)
        generated_tensor = generated_tensor.index_select(0, keep)
        generated_tensor = torch.cat(
            (
                generated_tensor,
                torch.tensor(next_tokens, device=device, dtype=torch.long).unsqueeze(1),
            ),
            dim=1,
        )
        active_original = [active_original[index] for index in continuing]
        generated_rows = [generated_rows[index] for index in continuing]
        event_rows = [event_rows[index] for index in continuing]
        output = model(generated_tensor, attention_mask=torch.ones_like(generated_tensor))
    for local_index, original_index in enumerate(active_original):
        if event_rows[local_index]:
            results[original_index] = {
                "generated_prefix": list(generated_rows[local_index]),
                "loop_events": list(event_rows[local_index]),
            }
    return results


def stack_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("generated recovery row fields changed")
    return {field: torch.stack([row[field] for row in rows]) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build recovery data from real penalty-off frozen-model loop closures"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--factual-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prompt-min", type=int, default=64)
    parser.add_argument("--prompt-max", type=int, default=192)
    parser.add_argument("--recovery-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--ngram-order", type=int, default=3)
    parser.add_argument("--prior-occurrences", type=int, default=2)
    parser.add_argument("--ngram-window", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument(
        "--objective-contract",
        choices=("recovery", "trajectory_unlikelihood"),
        default="recovery",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("full_recompute", "incremental_cache"),
        default="full_recompute",
    )
    args = parser.parse_args()

    checkpoint = require_c_path(args.checkpoint, "checkpoint")
    tokenizer_dir = require_c_path(args.tokenizer, "tokenizer")
    sources = {
        "natural": require_c_path(args.natural_dir, "natural dataset"),
        "factual": require_c_path(args.factual_dir, "factual dataset"),
    }
    validation = require_c_path(args.validation_dir, "validation dataset")
    output = require_c_path(args.output_dir, "output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards" / "train"
    shard_dir.mkdir(parents=True, exist_ok=False)

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer = load_tokenizer(str(tokenizer_dir))
    loaded, source_metadata = load_sources(sources, args.split)
    candidates = source_candidates(loaded)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if int(args.generation_batch_size) < 1:
        raise ValueError("generation batch size must be positive")
    if int(args.generation_batch_size) > 1 and int(args.prompt_min) != int(args.prompt_max):
        raise ValueError("batched generation requires one fixed prompt length")
    device = torch.device(args.device)
    model, config, model_metadata = load_frozen_model(checkpoint, device)

    rows: list[dict[str, torch.Tensor]] = []
    raw_rows: list[dict[str, Any]] = []
    captured_hashes: set[str] = set()
    pending: list[tuple[int, str, int, int, list[int], int]] = []

    def consume_pending() -> bool:
        prompt_rows = [
            source[:prompt_length] for _, _, _, _, source, prompt_length in pending
        ]
        if args.objective_contract == "trajectory_unlikelihood":
            failures = generate_batch_loop_trajectories(
                model,
                prompt_rows,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=config.eos_token_id,
                ngram_order=args.ngram_order,
                prior_occurrences=args.prior_occurrences,
                ngram_window=args.ngram_window,
                device=device,
                backend=args.generation_backend,
            )
        else:
            failures = generate_batch_to_loop_closure(
                model,
                prompt_rows,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=config.eos_token_id,
                ngram_order=args.ngram_order,
                prior_occurrences=args.prior_occurrences,
                ngram_window=args.ngram_window,
                device=device,
            )
        for entry, failure in zip(pending, failures):
            candidate_index, category, shard_index, row_index, source, prompt_length = entry
            if failure is None:
                continue
            if args.objective_contract == "trajectory_unlikelihood":
                row = construct_trajectory_unlikelihood_row(
                    source[:prompt_length],
                    failure["generated_prefix"],
                    failure["loop_events"],
                    sequence_length=args.sequence_length,
                    ngram_order=args.ngram_order,
                )
            else:
                row = construct_recovery_row(
                    source,
                    prompt_length,
                    failure["generated_prefix"],
                    failure["negative_token"],
                    sequence_length=args.sequence_length,
                    recovery_tokens=args.recovery_tokens,
                    ngram_order=args.ngram_order,
                    prior_occurrences=failure["prior_occurrences"],
                )
            if row is None:
                continue
            active = row["input_ids"][row["attention_mask"].bool()]
            row_sha = hashlib.sha256(
                active.to(torch.int32).contiguous().numpy().tobytes()
            ).hexdigest().upper()
            if row_sha in captured_hashes:
                continue
            captured_hashes.add(row_sha)
            rows.append(row)
            boundary = int(row["recovery_boundary"])
            raw_row = {
                    "row_sha256": row_sha,
                    "source_category": category,
                    "source_shard": str(loaded[category][shard_index]["path"]),
                    "source_row": row_index,
                    "candidate_order": candidate_index,
                    "prompt_length": prompt_length,
                    "generated_prefix_tokens": len(failure["generated_prefix"]),
                    "recovery_boundary": boundary,
                    "objective_contract": args.objective_contract,
                    "loop_event_count": int(row["loop_negative_mask"].sum()),
                    "prompt_text": tokenizer.decode(
                        source[:prompt_length], skip_special_tokens=True
                    ),
                    "generated_failure_prefix_text": tokenizer.decode(
                        failure["generated_prefix"], skip_special_tokens=True
                    ),
                }
            if args.objective_contract == "trajectory_unlikelihood":
                raw_row["loop_events"] = failure["loop_events"]
            else:
                raw_row.update(
                    {
                        "negative_token": int(failure["negative_token"]),
                        "closing_ngram": failure["closing_ngram"],
                        "prior_occurrences": int(failure["prior_occurrences"]),
                        "negative_token_text": tokenizer.decode(
                            [failure["negative_token"]], skip_special_tokens=True
                        ),
                        "clean_recovery_text": tokenizer.decode(
                            row["input_ids"][boundary:][
                                row["attention_mask"][boundary:].bool()
                            ].tolist(),
                            skip_special_tokens=True,
                        ),
                    }
                )
            raw_rows.append(raw_row)
            if len(rows) % 128 == 0:
                print(
                    json.dumps(
                        {
                            "event": "generation_progress",
                            "captured_rows": len(rows),
                            "requested_rows": int(args.rows),
                            "candidate_rows_scanned": candidate_index + 1,
                            "generation_batch_size": int(args.generation_batch_size),
                        }
                    ),
                    flush=True,
                )
            if len(rows) >= int(args.rows):
                return True
        return False

    complete = False
    for candidate_index, (category, shard_index, row_index) in enumerate(candidates):
        payload = loaded[category][shard_index]["payload"]
        source = active_row(payload, row_index)
        reserved_tokens = int(args.max_new_tokens)
        if args.objective_contract == "recovery":
            reserved_tokens += int(args.recovery_tokens)
        maximum_prompt = min(int(args.prompt_max), len(source) - reserved_tokens)
        if maximum_prompt < int(args.prompt_min):
            continue
        prompt_length = rng.randint(int(args.prompt_min), maximum_prompt)
        pending.append(
            (candidate_index, category, shard_index, row_index, source, prompt_length)
        )
        if len(pending) < int(args.generation_batch_size):
            continue
        complete = consume_pending()
        pending.clear()
        if complete:
            break
    if not complete and pending:
        consume_pending()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(rows) < int(args.rows):
        incident = {
            "status": "INSUFFICIENT_REAL_LOOP_FAILURES",
            "requested_rows": int(args.rows),
            "captured_rows": len(rows),
            "candidate_rows_scanned": len(candidates),
            "model": model_metadata,
        }
        atomic_json(output / "incident.json", incident)
        raise RuntimeError(json.dumps(incident))

    payload = stack_rows(rows)
    shard = shard_dir / "shard_00000.pt"
    atomic_torch_save(shard, payload)
    raw_path = output / "raw_failure_rows.jsonl"
    raw_temporary = raw_path.with_name(raw_path.name + ".tmp")
    with raw_temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(raw_temporary, raw_path)

    overlap_hashes = set()
    for directory in (*sources.values(), validation):
        split = "validation" if directory == validation else args.split
        overlap_hashes.update(dataset_active_hashes(directory, split))
    output_hashes = {
        active_hash(ids, mask)
        for ids, mask in zip(payload["input_ids"], payload["attention_mask"])
    }
    overlap = sorted(output_hashes & overlap_hashes)
    if overlap:
        raise ValueError(f"generated-prefix output has {len(overlap)} exact row overlaps")

    trajectory_only = args.objective_contract == "trajectory_unlikelihood"
    manifest = {
        "schema_version": (
            "fdt_v4_1_model_failure_unlikelihood_v1"
            if trajectory_only
            else "fdt_v4_1_model_failure_recovery_v1"
        ),
        "name": "generated_prefix",
        "split": "train",
        "construction": (
            "frozen_model_penalty_off_full_trajectory_trigram_unlikelihood"
            if trajectory_only
            else "frozen_model_penalty_off_real_trigram_loop_closure"
        ),
        "objective_contract": args.objective_contract,
        "penalty_scope": "natural_and_factual_only_excludes_exact_copy_code_json",
        "model": model_metadata,
        "tokenizer": {
            "directory": str(tokenizer_dir),
            "tokenizer_json_sha256": sha256_file(tokenizer_json),
        },
        "sources": source_metadata,
        "validation": {
            "directory": str(validation),
            "manifest_sha256": sha256_file(validation / "manifest.json"),
        },
        "seed": int(args.seed),
        "candidate_rows_scanned": raw_rows[-1]["candidate_order"] + 1,
        "unique_rows": len(output_hashes),
        "exact_row_overlap": len(overlap),
        "sequence_length": int(args.sequence_length),
        "ngram_order": int(args.ngram_order),
        "prior_occurrences_required": int(args.prior_occurrences),
        "generation_batch_size": int(args.generation_batch_size),
        "generation_backend": args.generation_backend,
        "shards": [
            {
                "file": str(shard.relative_to(output)),
                "rows": int(payload["input_ids"].size(0)),
                "active_tokens": int(payload["attention_mask"].sum()),
                "sha256": sha256_file(shard),
                "fields": sorted(payload),
            }
        ],
        "raw_failure_rows": {
            "path": str(raw_path.relative_to(output)),
            "sha256": sha256_file(raw_path),
        },
    }
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output / "sha256.json",
        {
            "manifest_sha256": sha256_file(manifest_path),
            "shard_sha256": sha256_file(shard),
            "raw_failure_rows_sha256": sha256_file(raw_path),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "rows": len(rows),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    )


if __name__ == "__main__":
    main()
