from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]


def c_path(path: Path, label: str) -> Path:
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


def row_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    active = ids[mask.bool()].to(torch.int32).contiguous().numpy().tobytes()
    return hashlib.sha256(active).hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def shard_paths(source: Path, split: str) -> list[Path]:
    paths = sorted((source / "shards" / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {source}")
    return paths


def token_stream(paths: Iterable[Path]) -> Iterable[int]:
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = payload.get("input_ids")
        mask = payload.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError(f"Natural shard lacks input_ids/attention_mask: {path}")
        for row_ids, row_mask in zip(ids, mask):
            for token in row_ids[row_mask.bool()].tolist():
                yield int(token)
        del payload, ids, mask
        gc.collect()


def pack_after_offset(
    paths: list[Path],
    *,
    skip_tokens: int,
    rows_by_length: list[tuple[int, int]],
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    required = sum(rows * length for length, rows in rows_by_length)
    stream = token_stream(paths)
    skipped = 0
    while skipped < int(skip_tokens):
        try:
            next(stream)
        except StopIteration as exc:
            raise ValueError("Natural stream ended before the bridge offset") from exc
        skipped += 1
    values: list[int] = []
    for _ in range(required):
        try:
            values.append(next(stream))
        except StopIteration as exc:
            raise ValueError("Natural stream is too short for the bridge payload") from exc
    packed: list[tuple[int, dict[str, torch.Tensor]]] = []
    cursor = 0
    for length, rows in rows_by_length:
        count = int(length) * int(rows)
        ids = torch.tensor(values[cursor : cursor + count], dtype=torch.int32).reshape(rows, length)
        packed.append(
            (
                length,
                {
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones(rows, length, dtype=torch.uint8),
                },
            )
        )
        cursor += count
    return packed


def audit_available_payloads(
    audit_root: Path,
    output: Path,
    target_hashes: dict[int, set[str]],
) -> dict[str, Any]:
    collisions: list[dict[str, Any]] = []
    scanned_files = 0
    candidate_rows = 0
    non_row_payloads = 0
    for path in sorted(audit_root.rglob("*.pt")):
        if output == path or output in path.parents:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scanned_files += 1
        if not isinstance(payload, dict):
            non_row_payloads += 1
            del payload
            gc.collect()
            continue
        ids = payload.get("input_ids")
        mask = payload.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or ids.ndim != 2 or ids.shape != mask.shape:
            non_row_payloads += 1
            del payload, ids, mask
            gc.collect()
            continue
        lengths = mask.long().sum(dim=1)
        target_lengths = torch.tensor(sorted(target_hashes), dtype=lengths.dtype)
        for row in torch.isin(lengths, target_lengths).nonzero(as_tuple=False).flatten().tolist():
            length = int(lengths[row])
            candidate_rows += 1
            digest = row_hash(ids[row], mask[row])
            if digest in target_hashes[length]:
                collisions.append({"path": str(path), "row": int(row), "length": length, "sha256": digest})
        del payload, ids, mask, lengths
        gc.collect()
    return {
        "available_payload_files_scanned": scanned_files,
        "same_length_rows_hashed": candidate_rows,
        "non_row_payload_files": non_row_payloads,
        "exact_row_collisions": collisions,
        "post_build_exact_overlap": len(collisions),
        "historical_limitation": "Hash-journaled payloads pruned before this audit cannot be rescanned.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build disjoint 2K/4K natural-language bridge shards")
    parser.add_argument("--natural-source", type=Path, required=True)
    parser.add_argument("--existing-long-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows-2k", type=int, default=256)
    parser.add_argument("--rows-4k", type=int, default=128)
    args = parser.parse_args()

    natural = c_path(args.natural_source, "natural source")
    existing_long = c_path(args.existing_long_source, "existing long source")
    output = c_path(args.output_dir, "output")
    audit_root = c_path(args.audit_root, "audit root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "shards" / args.split
    destination.mkdir(parents=True, exist_ok=True)

    old_manifest_path = existing_long / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    consumed_prefix = sum(int(item["active_tokens"]) for item in old_manifest["shards"])
    packed = pack_after_offset(
        shard_paths(natural, args.split),
        skip_tokens=consumed_prefix,
        rows_by_length=[(2048, args.rows_2k), (4096, args.rows_4k)],
    )

    records: list[dict[str, Any]] = []
    target_hashes: dict[int, set[str]] = {}
    for index, (length, payload) in enumerate(packed):
        path = destination / f"shard_{index:05d}_{length // 1024}k.pt"
        atomic_torch_save(path, payload)
        hashes = {row_hash(ids, mask) for ids, mask in zip(payload["input_ids"], payload["attention_mask"])}
        if len(hashes) != int(payload["input_ids"].size(0)):
            raise ValueError(f"Duplicate rows inside {length}-token bridge shard")
        target_hashes[length] = hashes
        records.append(
            {
                "file": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "rows": int(payload["input_ids"].size(0)),
                "sequence_length": length,
                "active_tokens": int(payload["attention_mask"].sum()),
                "fields": sorted(payload),
                "row_hash_set_sha256": hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest().upper(),
            }
        )

    if target_hashes[2048] & target_hashes[4096]:
        raise ValueError("2K and 4K bridge rows overlap")
    audit = audit_available_payloads(audit_root, output, target_hashes)
    if audit["post_build_exact_overlap"] != 0:
        raise ValueError(f"Available-payload overlap detected: {audit['exact_row_collisions'][:3]}")
    natural_manifest = natural / "manifest.json"
    manifest = {
        "schema_version": "fdt_v4_context_bridge_v1",
        "name": "bridge_context",
        "split": args.split,
        "construction": "disjoint contiguous natural-language stream after the tokens consumed by the accepted 8K/16K source",
        "natural_source": str(natural),
        "natural_manifest_sha256": sha256_file(natural_manifest),
        "existing_long_source": str(existing_long),
        "existing_long_manifest_sha256": sha256_file(old_manifest_path),
        "natural_stream_skip_tokens": consumed_prefix,
        "length_order": [2048, 4096],
        "shards": records,
        "audit": audit,
        "post_build_exact_overlap": 0,
        "conversational_sft": False,
    }
    atomic_json(output / "manifest.json", manifest)
    result = {
        "status": "PASS",
        "output": str(output),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "shards": records,
        "audit": audit,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
