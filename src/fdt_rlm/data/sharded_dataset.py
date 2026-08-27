from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import IterableDataset


class ShardedTokenDataset(IterableDataset):
    def __init__(self, dataset_dir: str | Path, split: str = "train"):
        self.dataset_dir = Path(dataset_dir)
        self.split = split

    def __iter__(self):
        shard_dir = self.dataset_dir / "shards" / self.split
        for path in sorted(shard_dir.glob("*.pt")):
            data = torch.load(path, map_location="cpu")
            ids = data["input_ids"]
            mask = data.get("attention_mask", torch.ones_like(ids))
            for row, row_mask in zip(ids, mask):
                yield {"input_ids": row.long(), "attention_mask": row_mask.long()}

