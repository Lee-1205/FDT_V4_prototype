"""Prepare explicit-contract FDT v4 shards from existing tensor shards.

This utility never invents exact-copy or loop-negative supervision.
Natural/factual sources may inherit input_ids as labels, but exact-copy and
generated-prefix inputs must already contain their full explicit contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import shutil
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def c_path(value: Path, label: str) -> Path:
    path = value.expanduser().resolve()
    if path.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def source_shards(source: Path, split: str) -> list[Path]:
    paths = sorted((source / "shards" / split).glob("*.pt"))
    if not paths:
        paths = sorted((source / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {source}")
    return paths


def prepare_source(
    source: Path,
    destination: Path,
    split: str,
    contract: str,
) -> list[dict[str, Any]]:
    output = destination / "shards" / split
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(source_shards(source, split)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = payload.get("input_ids")
        mask = payload.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or ids.shape != mask.shape:
            raise ValueError(f"Source shard lacks input_ids/attention_mask contract: {path}")
        labels = payload.get("labels", ids.clone())
        if not isinstance(labels, torch.Tensor) or labels.shape != ids.shape:
            raise ValueError(f"Source shard labels are missing or mismatched: {path}")
        result: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
        }
        if contract == "exact_copy":
            required = ("prompt_mask", "source_boundary", "copy_source_positions", "copy_target_mask")
            missing = [name for name in required if name not in payload]
            if missing:
                raise ValueError(f"Exact source shard lacks explicit fields {missing}: {path}")
            for name in required:
                value = payload[name]
                valid_shape = isinstance(value, torch.Tensor) and (value.shape == ids.shape or (name == "source_boundary" and value.shape == ids.shape[:1]))
                if not valid_shape:
                    raise ValueError(f"Exact field {name} is not shape-matched: {path}")
                result[name] = value
            if bool(result["labels"].masked_select(result["prompt_mask"].bool()).ne(-100).any()):
                raise ValueError(f"Exact prompt labels must be -100: {path}")
            prompt_mask = result["prompt_mask"].bool()
            target_mask = result["copy_target_mask"].bool()
            if bool((target_mask & ~mask.bool()).any()):
                raise ValueError(f"Exact copy targets must be active tokens: {path}")
            source_positions = result["copy_source_positions"].long()
            boundary = result["source_boundary"]
            boundaries = (
                boundary.long().view(-1, 1).expand_as(ids)
                if boundary.ndim == 1
                else boundary.long()
            )
            rows, targets = target_mask.nonzero(as_tuple=True)
            mapped = source_positions[rows, targets]
            if bool(
                (
                    mapped.le(0)
                    | mapped.ge(boundaries[rows, targets])
                    | mapped.ge(targets)
                    | targets.lt(boundaries[rows, targets])
                    | boundaries[rows, targets].gt(ids.size(1))
                ).any()
            ):
                raise ValueError(f"Exact source mapping is outside the prompt boundary: {path}")
            if bool((~prompt_mask[rows, mapped]).any()) or bool(
                ids[rows, mapped].ne(result["labels"][rows, targets]).any()
            ):
                raise ValueError(f"Exact source mapping does not match prompt tokens: {path}")
        elif contract == "generated_prefix":
            required = ("loop_negative_ids", "loop_negative_mask")
            missing = [name for name in required if name not in payload]
            if missing:
                raise ValueError(
                    f"Generated-prefix source lacks explicit fields {missing}: {path}"
                )
            for name in required:
                value = payload[name]
                if not isinstance(value, torch.Tensor) or value.shape != ids.shape:
                    raise ValueError(f"Generated-prefix field {name} is not shape-matched: {path}")
                result[name] = value
            negative_mask = result["loop_negative_mask"].bool()
            if not bool(negative_mask.any()):
                raise ValueError(f"Generated-prefix source has no loop-negative tokens: {path}")
            if bool(result["labels"].masked_select(negative_mask).eq(-100).any()):
                raise ValueError(
                    f"Generated-prefix negatives require clean supervised labels: {path}"
                )
            if bool(
                result["loop_negative_ids"].masked_select(negative_mask).eq(
                    result["labels"].masked_select(negative_mask)
                ).any()
            ):
                raise ValueError(
                    f"Generated-prefix negatives must differ from clean labels: {path}"
                )
        target = output / f"shard_{index:05d}.pt"
        if "labels" in payload:
            shutil.copy2(path, target)
        else:
            torch.save(result, target)
        records.append({"file": str(target.relative_to(destination)), "sha256": sha256(target), "rows": int(ids.size(0)), "tokens": int(mask.sum().item())})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build explicit FDT v4 train/validation tensor contracts")
    parser.add_argument("--natural-source", type=Path, required=True)
    parser.add_argument("--factual-source", type=Path, required=True)
    parser.add_argument("--exact-source", type=Path, required=True)
    parser.add_argument("--generated-prefix-source", type=Path, required=True)
    parser.add_argument("--long-context-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    output = c_path(args.output_dir, "output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output}")
    for path in (
        args.natural_source,
        args.factual_source,
        args.exact_source,
        args.generated_prefix_source,
        args.long_context_source,
    ):
        c_path(path, "source")
    tokenizer = c_path(args.tokenizer_dir, "tokenizer directory") / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    sources = {
        "natural": prepare_source(c_path(args.natural_source, "natural source"), output / "natural", args.split, "lm"),
        "factual": prepare_source(c_path(args.factual_source, "factual source"), output / "factual", args.split, "lm"),
        "exact_copy": prepare_source(c_path(args.exact_source, "exact source"), output / "exact_copy", args.split, "exact_copy"),
        "generated_prefix": prepare_source(
            c_path(args.generated_prefix_source, "generated-prefix source"),
            output / "generated_prefix",
            args.split,
            "generated_prefix",
        ),
        "long_context": prepare_source(
            c_path(args.long_context_source, "long-context source"),
            output / "long_context",
            args.split,
            "lm",
        ),
    }
    validation_source = c_path(args.validation_source or args.natural_source, "validation source")
    sources["validation"] = prepare_source(
        validation_source,
        output / "validation",
        args.validation_split,
        "lm",
    )
    source_inputs = {
        "natural": c_path(args.natural_source, "natural source"),
        "factual": c_path(args.factual_source, "factual source"),
        "exact_copy": c_path(args.exact_source, "exact source"),
        "generated_prefix": c_path(
            args.generated_prefix_source, "generated-prefix source"
        ),
        "long_context": c_path(args.long_context_source, "long-context source"),
        "validation": validation_source,
    }
    source_splits = {
        "natural": args.split,
        "factual": args.split,
        "exact_copy": args.split,
        "generated_prefix": args.split,
        "long_context": args.split,
        "validation": args.validation_split,
    }
    for name, records in sources.items():
        input_manifest = source_inputs[name] / "manifest.json"
        atomic_json(
            output / name / "manifest.json",
            {
                "schema_version": "fdt_v4_curriculum_source_manifest_v1",
                "name": name,
                "split": source_splits[name],
                "source_directory": str(source_inputs[name]),
                "source_manifest_sha256": (
                    sha256(input_manifest) if input_manifest.exists() else "MISSING"
                ),
                "shards": records,
            },
        )
    manifest = {"schema_version": "fdt_v4_curriculum_manifest_v4", "tokenizer_dir": str(c_path(args.tokenizer_dir, "tokenizer directory")), "tokenizer_json": str(tokenizer), "tokenizer_json_sha256": sha256(tokenizer), "sources": sources, "evaluation_tensor_dataset": sources["validation"][0]["file"], "evaluation_tensor_dataset_sha256": sources["validation"][0]["sha256"], "explicit_labels": True, "exact_prompt_source_contract": True, "generated_prefix_loop_negative_contract": True, "long_context_training_contract": "contains_at_least_one_8192_token_shard_and_never_exceeds_16384", "historical_overlap_limitations": "Only available payloads were scanned; pruned historical payloads require nonce construction evidence."}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "manifest": str(output / "manifest.json"), "manifest_sha256": sha256(output / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
