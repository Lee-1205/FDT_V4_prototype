from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class ShardInfo:
    path: str
    split: str
    tokens: int
    sequences: int
    buckets: Dict[str, int] = field(default_factory=dict)


class TokenShardWriter:
    def __init__(self, output_dir: str | Path, seq_len: int, shard_tokens: int):
        self.output_dir = Path(output_dir)
        self.seq_len = seq_len
        self.shard_tokens = shard_tokens
        self.buffers: Dict[str, List[List[int]]] = {"train": [], "validation": [], "test": []}
        self.bucket_counts: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
        self.indices = {"train": 0, "validation": 0, "test": 0}
        self.shards: List[ShardInfo] = []

    def add(self, split: str, tokens: List[int], bucket: str) -> None:
        if len(tokens) != self.seq_len:
            raise ValueError("Packed sequence length mismatch.")
        self.buffers[split].append([int(x) for x in tokens])
        self.bucket_counts[split][bucket] = self.bucket_counts[split].get(bucket, 0) + len(tokens)
        if len(self.buffers[split]) * self.seq_len >= self.shard_tokens:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers.get(split, [])
        if not rows:
            return
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"shard_{self.indices[split]:06d}.pt"
        tensor = torch.tensor(rows, dtype=torch.long)
        torch.save({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}, path)
        tokens = int(tensor.numel())
        self.shards.append(ShardInfo(
            path=str(path),
            split=split,
            tokens=tokens,
            sequences=int(tensor.size(0)),
            buckets=dict(self.bucket_counts[split]),
        ))
        self.buffers[split] = []
        self.bucket_counts[split] = {}
        self.indices[split] += 1

    def close(self) -> List[ShardInfo]:
        for split in list(self.buffers):
            self.flush(split)
        return self.shards

