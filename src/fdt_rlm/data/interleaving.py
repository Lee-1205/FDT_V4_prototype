from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


DEFAULT_BUCKET_ORDER = (
    "fineweb_edu",
    "fineweb",
    "code_python",
    "code_javascript_typescript",
    "code_c_cpp",
    "code_java",
    "code_rust_go",
    "code_structured_docs",
)


def bucket_sequence_counts(
    manifest: Mapping[str, Any],
    split: str = "train",
    bucket_order: Sequence[str] = DEFAULT_BUCKET_ORDER,
) -> dict[str, int]:
    seq_len = int(manifest["seq_len"])
    token_counts = {bucket: 0 for bucket in bucket_order}
    split_sequences = 0
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        split_sequences += int(shard["sequences"])
        for bucket, tokens in shard.get("buckets", {}).items():
            if bucket not in token_counts:
                raise ValueError(f"Unknown source bucket in manifest: {bucket}")
            token_counts[bucket] += int(tokens)

    for bucket, tokens in token_counts.items():
        if tokens % seq_len:
            raise ValueError(f"Bucket {bucket} has {tokens} tokens, not divisible by seq_len={seq_len}")
    counts = {bucket: token_counts[bucket] // seq_len for bucket in bucket_order}
    if sum(counts.values()) != split_sequences:
        raise ValueError("Manifest bucket counts do not cover the complete split")
    return counts


def interleave_remaining_indices(
    bucket_counts: Mapping[str, int],
    consumed_prefix: int = 0,
) -> torch.Tensor:
    """Interleave the unconsumed suffix of contiguous source ranges.

    The weighted-deficit schedule keeps every prefix close to the remaining
    source mixture while preserving order inside each source and replaying no
    index below ``consumed_prefix``.
    """
    if consumed_prefix < 0:
        raise ValueError("consumed_prefix must be non-negative")

    names = list(bucket_counts)
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    cursor = 0
    for name in names:
        count = int(bucket_counts[name])
        if count < 0:
            raise ValueError(f"Negative sequence count for {name}")
        starts[name] = cursor
        cursor += count
        ends[name] = cursor
    if consumed_prefix > cursor:
        raise ValueError("consumed_prefix exceeds the dataset length")

    next_index = {name: max(starts[name], consumed_prefix) for name in names}
    remaining = {name: max(0, ends[name] - next_index[name]) for name in names}
    total = sum(remaining.values())
    emitted = {name: 0 for name in names}
    order = torch.empty(total, dtype=torch.long)

    for position in range(total):
        target_step = position + 1
        chosen = max(
            (name for name in names if emitted[name] < remaining[name]),
            key=lambda name: target_step * remaining[name] - emitted[name] * total,
        )
        order[position] = next_index[chosen]
        next_index[chosen] += 1
        emitted[chosen] += 1
    return order


def source_counts_for_prefix(
    order: torch.Tensor,
    bucket_counts: Mapping[str, int],
    prefix_size: int,
) -> dict[str, int]:
    prefix = order[:prefix_size]
    result: dict[str, int] = {}
    start = 0
    for bucket, count_value in bucket_counts.items():
        end = start + int(count_value)
        result[bucket] = int(((prefix >= start) & (prefix < end)).sum().item())
        start = end
    return result
