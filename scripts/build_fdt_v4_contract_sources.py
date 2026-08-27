from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def shard_paths(dataset: Path, split: str) -> list[Path]:
    paths = sorted((dataset / "shards" / split).glob("*.pt"))
    if not paths:
        paths = sorted((dataset / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {dataset}")
    return paths


def load_category_rows(
    dataset: Path,
    category_id: int,
    count: int,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
    }
    remaining = int(count)
    for path in shard_paths(dataset, "train"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        category_ids = payload.get("category_ids")
        if not isinstance(category_ids, torch.Tensor):
            raise ValueError(f"Source lacks category_ids: {path}")
        selected = category_ids.eq(int(category_id)).nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        selected = selected[:remaining]
        for name in chunks:
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Source lacks {name}: {path}")
            chunks[name].append(value.index_select(0, selected).cpu())
        remaining -= int(selected.numel())
        if remaining <= 0:
            break
    if remaining > 0:
        raise ValueError(
            f"Category {category_id} has fewer than {count} rows; missing {remaining}"
        )
    return {name: torch.cat(values, dim=0) for name, values in chunks.items()}


def compact_lm_payload(
    payload: dict[str, torch.Tensor],
    *,
    full_lm_labels: bool,
) -> dict[str, torch.Tensor]:
    ids = payload["input_ids"].to(torch.int32).contiguous()
    mask = payload["attention_mask"].to(torch.uint8).contiguous()
    if full_lm_labels:
        labels = ids.clone()
        labels.masked_fill_(~mask.bool(), -100)
    else:
        labels = payload["labels"].to(torch.int32).contiguous()
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def active_tokens(payload: dict[str, torch.Tensor], row: int) -> torch.Tensor:
    mask = payload["attention_mask"][row].bool()
    return payload["input_ids"][row][mask].to(torch.int32)


def build_exact_copy(
    natural: dict[str, torch.Tensor],
    rows: int,
    seed: int,
    sequence_length: int = 512,
) -> dict[str, torch.Tensor]:
    rng = random.Random(seed)
    output = {
        "input_ids": torch.zeros(rows, sequence_length, dtype=torch.int32),
        "labels": torch.full((rows, sequence_length), -100, dtype=torch.int32),
        "attention_mask": torch.zeros(rows, sequence_length, dtype=torch.uint8),
        "prompt_mask": torch.zeros(rows, sequence_length, dtype=torch.uint8),
        "source_boundary": torch.zeros(rows, dtype=torch.int32),
        "copy_source_positions": torch.full(
            (rows, sequence_length), -1, dtype=torch.int32
        ),
        "copy_target_mask": torch.zeros(rows, sequence_length, dtype=torch.uint8),
    }
    target_lengths = (8, 16, 32, 64)
    distractor_lengths = (16, 64, 128, 256, 384)
    natural_rows = int(natural["input_ids"].size(0))
    for row in range(rows):
        source = active_tokens(natural, row % natural_rows)
        distractor = active_tokens(natural, (row * 7919 + 17) % natural_rows)
        target_length = target_lengths[row % len(target_lengths)]
        source_start = 1 + rng.randrange(max(int(source.numel()) - target_length - 1, 1))
        copied = source[source_start : source_start + target_length]
        if copied.numel() < target_length:
            copied = source[1 : 1 + target_length]
        if copied.numel() < target_length:
            raise ValueError("Natural source is too short for exact-copy construction")
        maximum_distractor = sequence_length - target_length - target_length - 2
        distractor_length = min(
            distractor_lengths[(row // len(target_lengths)) % len(distractor_lengths)],
            maximum_distractor,
            int(distractor.numel()),
        )
        header = source[:1]
        prompt = torch.cat((header, copied, distractor[:distractor_length]))
        boundary = int(prompt.numel())
        stop = boundary + target_length
        output["input_ids"][row, :boundary] = prompt
        output["input_ids"][row, boundary:stop] = copied
        output["labels"][row, boundary:stop] = copied
        output["attention_mask"][row, :stop] = 1
        output["prompt_mask"][row, :boundary] = 1
        output["source_boundary"][row] = boundary
        output["copy_target_mask"][row, boundary:stop] = 1
        output["copy_source_positions"][row, boundary:stop] = torch.arange(
            1, 1 + target_length, dtype=torch.int32
        )
    return output


def build_generated_prefix(
    natural: dict[str, torch.Tensor],
    rows: int,
    seed: int,
    sequence_length: int = 512,
) -> dict[str, torch.Tensor]:
    rng = random.Random(seed)
    ids = torch.zeros(rows, sequence_length, dtype=torch.int32)
    mask = torch.zeros(rows, sequence_length, dtype=torch.uint8)
    negative_ids = torch.full((rows, sequence_length), -1, dtype=torch.int32)
    negative_mask = torch.zeros(rows, sequence_length, dtype=torch.uint8)
    natural_rows = int(natural["input_ids"].size(0))
    for row in range(rows):
        left = active_tokens(natural, row % natural_rows)
        right = active_tokens(natural, (row * 6151 + 31) % natural_rows)
        left_count = min(sequence_length // 2, int(left.numel()))
        combined = torch.cat((left[-left_count:], right))[:sequence_length]
        active = int(combined.numel())
        ids[row, :active] = combined
        mask[row, :active] = 1
        offset = 16 + rng.randrange(8)
        for position in range(offset, active, 32):
            candidate = int(combined[position - 3])
            target = int(combined[position])
            if candidate == target:
                candidate = int(combined[position - 5])
            if candidate == target:
                candidate = (target + 1) % 24576
                if candidate == 0:
                    candidate = 1
            negative_ids[row, position] = candidate
            negative_mask[row, position] = 1
    labels = ids.clone()
    labels.masked_fill_(~mask.bool(), -100)
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "loop_negative_ids": negative_ids,
        "loop_negative_mask": negative_mask,
    }


def token_stream(natural: dict[str, torch.Tensor]) -> Iterable[int]:
    for row in range(int(natural["input_ids"].size(0))):
        for token in active_tokens(natural, row).tolist():
            yield int(token)


def build_long_context(
    natural: dict[str, torch.Tensor],
    rows: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    needed = rows * sequence_length
    values: list[int] = []
    while len(values) < needed:
        before = len(values)
        values.extend(token_stream(natural))
        if len(values) == before:
            raise ValueError("Natural source has no active tokens")
    ids = torch.tensor(values[:needed], dtype=torch.int32).reshape(
        rows, sequence_length
    )
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones(rows, sequence_length, dtype=torch.uint8),
    }


def row_hashes(payload: dict[str, torch.Tensor]) -> list[str]:
    hashes: list[str] = []
    for ids, mask in zip(payload["input_ids"], payload["attention_mask"]):
        active = ids[mask.bool()].to(torch.int32).contiguous().numpy().tobytes()
        hashes.append(hashlib.sha256(active).hexdigest().upper())
    return hashes


def write_source(
    root: Path,
    name: str,
    split: str,
    shards: list[tuple[str, dict[str, torch.Tensor]]],
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    destination = root / name / "shards" / split
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hashes: list[str] = []
    for filename, payload in shards:
        path = destination / filename
        atomic_torch_save(path, payload)
        current_hashes = row_hashes(payload)
        if len(current_hashes) != len(set(current_hashes)):
            raise ValueError(f"Duplicate active rows inside {name}/{filename}")
        hashes.extend(current_hashes)
        records.append(
            {
                "file": str(path),
                "sha256": sha256_file(path),
                "rows": int(payload["input_ids"].size(0)),
                "sequence_length": int(payload["input_ids"].size(1)),
                "fields": sorted(payload),
                "active_tokens": int(payload["attention_mask"].sum().item()),
            }
        )
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"Duplicate active rows across {name} shards")
    manifest = {
        "schema_version": "fdt_v4_contract_source_v1",
        "name": name,
        "split": split,
        "provenance": provenance,
        "shards": records,
        "unique_active_rows": len(set(hashes)),
    }
    atomic_json(root / name / "manifest.json", manifest)
    return records, hashes


def write_copied_source(
    root: Path,
    name: str,
    split: str,
    source: Path,
    payload: dict[str, torch.Tensor],
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    destination = root / name / "shards" / split
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "shard_00000.pt"
    shutil.copy2(source, target)
    hashes = row_hashes(payload)
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"Duplicate active rows inside {name}")
    record = {
        "file": str(target),
        "sha256": sha256_file(target),
        "rows": int(payload["input_ids"].size(0)),
        "sequence_length": int(payload["input_ids"].size(1)),
        "fields": sorted(payload),
        "active_tokens": int(payload["attention_mask"].sum().item()),
    }
    atomic_json(
        root / name / "manifest.json",
        {
            "schema_version": "fdt_v4_contract_source_v1",
            "name": name,
            "split": split,
            "provenance": provenance,
            "shards": [record],
            "unique_active_rows": len(hashes),
        },
    )
    return [record], hashes


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = require_c_path(args.base_source, "base source")
    validation = require_c_path(args.validation_source, "validation source")
    output = require_c_path(args.output_dir, "output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_manifest_path = base / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if int(base_manifest.get("post_build_exact_overlap", -1)) != 0:
        raise ValueError("Base source did not pass its exact-overlap audit")

    natural_raw = load_category_rows(
        base, args.natural_category_id, args.natural_rows
    )
    factual_raw = load_category_rows(
        base, args.factual_category_id, args.factual_rows
    )
    natural = compact_lm_payload(natural_raw, full_lm_labels=True)
    factual = compact_lm_payload(factual_raw, full_lm_labels=False)
    exact = build_exact_copy(natural, args.exact_rows, args.seed + 11)
    generated_prefix = build_generated_prefix(
        natural, args.generated_prefix_rows, args.seed + 29
    )
    long_8k = build_long_context(natural, args.long_8k_rows, 8192)
    long_16k = build_long_context(natural, args.long_16k_rows, 16384)

    validation_shard = shard_paths(validation, args.validation_split)[0]
    validation_payload = torch.load(
        validation_shard, map_location="cpu", weights_only=False
    )
    validation_compact = {
        "input_ids": validation_payload["input_ids"],
        "labels": validation_payload["labels"],
        "attention_mask": validation_payload["attention_mask"],
    }

    provenance = {
        "base_source": str(base),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "base_post_build_exact_overlap": 0,
        "seed": int(args.seed),
        "historical_payload_limitation": base_manifest.get(
            "historical_payload_limitation", "inherited from audited base source"
        ),
    }
    all_hashes: dict[str, list[str]] = {}
    source_records: dict[str, Any] = {}
    for name, split, shards, details in (
        (
            "natural",
            "train",
            [("shard_00000.pt", natural)],
            {**provenance, "construction": "fresh natural rows with full LM labels"},
        ),
        (
            "factual",
            "train",
            [("shard_00000.pt", factual)],
            {**provenance, "construction": "fresh factual QA completion rows"},
        ),
        (
            "exact_copy",
            "train",
            [("shard_00000.pt", exact)],
            {
                **provenance,
                "construction": "explicit prompt-to-target source mappings with distractors",
            },
        ),
        (
            "generated_prefix",
            "train",
            [("shard_00000.pt", generated_prefix)],
            {
                **provenance,
                "construction": "cross-document clean continuations with explicit loop-negative ids",
            },
        ),
        (
            "long_context",
            "train",
            [("shard_00000_8k.pt", long_8k), ("shard_00001_16k.pt", long_16k)],
            {
                **provenance,
                "construction": "packed fresh natural rows at exact 8K and 16K lengths",
            },
        ),
    ):
        records, hashes = write_source(output, name, split, shards, details)
        source_records[name] = records
        all_hashes[name] = hashes

    validation_records, validation_hashes = write_copied_source(
        output,
        "validation",
        args.validation_split,
        validation_shard,
        validation_compact,
        {
            "source": str(validation),
            "source_shard": str(validation_shard),
            "source_shard_sha256": sha256_file(validation_shard),
            "construction": "byte-identical fixed validation tensor",
        },
    )
    source_records["validation"] = validation_records
    all_hashes["validation"] = validation_hashes

    train_names = ("natural", "factual", "exact_copy", "generated_prefix", "long_context")
    seen: dict[str, str] = {}
    cross_collisions: list[dict[str, str]] = []
    for name in train_names:
        for value in all_hashes[name]:
            previous = seen.get(value)
            if previous is not None:
                cross_collisions.append({"sha256": value, "left": previous, "right": name})
            else:
                seen[value] = name
    validation_overlap = sorted(set(seen) & set(all_hashes["validation"]))
    if cross_collisions or validation_overlap:
        raise ValueError(
            f"Cross-source overlap detected: train={len(cross_collisions)}, validation={len(validation_overlap)}"
        )

    manifest = {
        "schema_version": "fdt_v4_contract_sources_v1",
        "base_source": str(base),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "source_records": source_records,
        "cross_source_exact_overlap": 0,
        "train_validation_exact_overlap": 0,
        "total_unique_train_rows": len(seen),
        "requirements": {
            "natural_language_primary": True,
            "factual_knowledge_primary": True,
            "explicit_exact_memory_mapping": True,
            "generated_prefix_loop_negatives": True,
            "distinct_8k_shard": True,
            "exact_16k_shard": True,
            "conversational_sft": False,
        },
    }
    atomic_json(output / "manifest.json", manifest)
    result = {
        "output_dir": str(output),
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "total_unique_train_rows": len(seen),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited FDT v4 source contracts from a fresh knowledge tranche"
    )
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--natural-category-id", type=int, default=4)
    parser.add_argument("--factual-category-id", type=int, default=0)
    parser.add_argument("--natural-rows", type=int, default=20_000)
    parser.add_argument("--factual-rows", type=int, default=20_000)
    parser.add_argument("--exact-rows", type=int, default=4_096)
    parser.add_argument("--generated-prefix-rows", type=int, default=2_048)
    parser.add_argument("--long-8k-rows", type=int, default=64)
    parser.add_argument("--long-16k-rows", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
