from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fdt_v4 as evaluator
import prepare_fdt_v4_1_generated_prefix_recovery as builder


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_raw_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def atomic_torch_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict:
    checkpoint = require_c_path(args.checkpoint, "checkpoint")
    source = require_c_path(args.source_dataset, "source dataset")
    output = require_c_path(args.output_dir, "output dataset")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    shard_path = source / "shards" / "train" / "shard_00000.pt"
    raw_path = source / "raw_failure_rows.jsonl"
    payload = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    limit = min(int(args.limit), int(payload["input_ids"].size(0)))
    raw_rows = load_raw_rows(raw_path, limit)
    if len(raw_rows) != limit:
        raise ValueError("raw failure rows do not align with the source shard")

    model, config, metadata = evaluator.load_checkpoint(checkpoint)
    if int(config.loop_controller_rank) != 0:
        raise ValueError("escape targets must come from the immutable controller-free parent")
    model = model.to(device=args.device, dtype=torch.float32).eval()
    sequence_length = int(payload["input_ids"].size(1))
    max_candidates = int(args.max_candidates)
    tensors = {
        "input_ids": torch.zeros((limit, sequence_length), dtype=torch.int32),
        "labels": torch.full((limit, sequence_length), -100, dtype=torch.int32),
        "attention_mask": torch.zeros((limit, sequence_length), dtype=torch.uint8),
        "loop_negative_ids": torch.full(
            (limit, sequence_length), -1, dtype=torch.int32
        ),
        "loop_negative_mask": torch.zeros(
            (limit, sequence_length), dtype=torch.uint8
        ),
        "loop_negative_prior_occurrences": torch.zeros(
            (limit, sequence_length), dtype=torch.uint8
        ),
        "recovery_boundary": torch.zeros(limit, dtype=torch.int32),
        "loop_ngram_order": torch.full((limit,), 3, dtype=torch.uint8),
        "loop_unlikelihood_only": torch.ones(limit, dtype=torch.uint8),
        "loop_candidate_ids": torch.zeros(
            (limit, max_candidates), dtype=torch.int32
        ),
        "loop_candidate_mask": torch.zeros(
            (limit, max_candidates), dtype=torch.uint8
        ),
        "loop_escape_ids": torch.zeros(limit, dtype=torch.int32),
        "loop_contrast_position": torch.zeros(limit, dtype=torch.int32),
    }
    evidence_rows = []
    candidate_counts = []
    escape_ranks = []
    escape_probabilities = []
    negative_probabilities = []
    excluded_special = {int(config.pad_token_id), int(config.eos_token_id), 1}

    for start in range(0, limit, int(args.batch_size)):
        stop = min(start + int(args.batch_size), limit)
        boundaries = payload["recovery_boundary"][start:stop].long()
        active_end = int(boundaries.max().item())
        input_ids = payload["input_ids"][start:stop, :active_end].long().to(args.device)
        attention_mask = payload["attention_mask"][start:stop, :active_end].long().to(
            args.device
        )
        logits = model(input_ids, attention_mask=attention_mask)["logits"].float()
        for local_index, boundary_value in enumerate(boundaries.tolist()):
            row_index = start + local_index
            boundary = int(boundary_value)
            prediction = logits[local_index, boundary - 1]
            probabilities = torch.softmax(prediction, dim=-1)
            ranked = torch.topk(prediction, k=int(args.top_k)).indices.tolist()
            prompt_length = int(raw_rows[row_index]["prompt_length"])
            generated = [
                int(value)
                for value in payload["input_ids"][
                    row_index, prompt_length:boundary
                ].tolist()
            ]
            loop_candidates = []
            escape = None
            escape_rank = None
            for rank, token_value in enumerate(ranked, start=1):
                token = int(token_value)
                closes, _ = builder.closes_degenerate_ngram(
                    generated,
                    token,
                    order=3,
                    prior_occurrences=2,
                    window=96,
                )
                if closes and len(loop_candidates) < max_candidates:
                    loop_candidates.append(token)
                elif not closes and token not in excluded_special and escape is None:
                    escape = token
                    escape_rank = rank
            negative = int(raw_rows[row_index]["negative_token"])
            if negative not in loop_candidates:
                raise ValueError("recorded loop token was not recovered in parent top-k")
            if escape is None:
                raise ValueError("no non-loop escape candidate was found")
            escape_closes, _ = builder.closes_degenerate_ngram(
                generated,
                escape,
                order=3,
                prior_occurrences=2,
                window=96,
            )
            if escape_closes:
                raise AssertionError("selected escape still closes a loop")

            prefix = payload["input_ids"][row_index, :boundary].to(torch.int32)
            tensors["input_ids"][row_index, :boundary] = prefix
            tensors["input_ids"][row_index, boundary] = negative
            tensors["attention_mask"][row_index, : boundary + 1] = 1
            tensors["loop_negative_ids"][row_index, boundary] = negative
            tensors["loop_negative_mask"][row_index, boundary] = 1
            tensors["loop_negative_prior_occurrences"][row_index, boundary] = min(
                int(raw_rows[row_index]["prior_occurrences"]), 255
            )
            tensors["recovery_boundary"][row_index] = boundary
            tensors["loop_contrast_position"][row_index] = boundary
            tensors["loop_candidate_ids"][row_index, : len(loop_candidates)] = torch.tensor(
                loop_candidates, dtype=torch.int32
            )
            tensors["loop_candidate_mask"][row_index, : len(loop_candidates)] = 1
            tensors["loop_escape_ids"][row_index] = escape
            candidate_counts.append(len(loop_candidates))
            escape_ranks.append(int(escape_rank))
            escape_probabilities.append(float(probabilities[escape].item()))
            negative_probabilities.append(float(probabilities[negative].item()))
            evidence_rows.append(
                {
                    **raw_rows[row_index],
                    "loop_candidate_ids": loop_candidates,
                    "loop_candidate_count": len(loop_candidates),
                    "escape_token": escape,
                    "escape_rank": int(escape_rank),
                    "escape_probability": float(probabilities[escape].item()),
                    "negative_probability": float(probabilities[negative].item()),
                }
            )
        del logits, input_ids, attention_mask

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        target_shard = temporary / "shards" / "train" / "shard_00000.pt"
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(target_shard, tensors)
        raw_output = temporary / "raw_escape_rows.jsonl"
        with raw_output.open("w", encoding="utf-8") as handle:
            for row in evidence_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": "fdt_v4_1_loop_escape_contrast_v1",
            "name": "generated_prefix",
            "split": "train",
            "construction": "parent_topk_loop_set_and_highest_nonloop_self_escape",
            "objective_contract": "model_native_loop_controller_contrast",
            "parent": metadata,
            "source_dataset": {
                "path": str(source),
                "manifest_sha256": sha256_file(source / "manifest.json"),
                "shard_sha256": sha256_file(shard_path),
                "raw_rows_sha256": sha256_file(raw_path),
            },
            "rows": limit,
            "sequence_length": sequence_length,
            "active_tokens": int(tensors["attention_mask"].sum().item()),
            "top_k_inspected": int(args.top_k),
            "max_loop_candidates": max_candidates,
            "mean_loop_candidates": sum(candidate_counts) / limit,
            "mean_escape_rank": sum(escape_ranks) / limit,
            "mean_escape_probability": sum(escape_probabilities) / limit,
            "mean_negative_probability": sum(negative_probabilities) / limit,
            "shards": [
                {
                    "file": "shards\\train\\shard_00000.pt",
                    "rows": limit,
                    "sha256": sha256_file(target_shard),
                    "fields": sorted(tensors),
                }
            ],
            "raw_escape_rows": {
                "file": "raw_escape_rows.jsonl",
                "sha256": sha256_file(raw_output),
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (temporary / "sha256.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": sha256_file(temporary / "manifest.json"),
                    "shard_sha256": sha256_file(target_shard),
                    "raw_escape_rows_sha256": sha256_file(raw_output),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-native loop escape contrast rows from real failures"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--limit", type=int, default=8192)
    args = parser.parse_args()
    report = build(args)
    manifest = args.output_dir.resolve() / "manifest.json"
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "rows": report["rows"],
                "mean_loop_candidates": report["mean_loop_candidates"],
                "mean_escape_rank": report["mean_escape_rank"],
            }
        )
    )


if __name__ == "__main__":
    main()
