from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset


def iter_local_texts(path: str | Path) -> Iterator[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def iter_hf_texts(
    dataset_name: str,
    split: str,
    text_field: str = "text",
    dataset_config: str = "",
    streaming: bool = True,
) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    for row in dataset:
        text = row.get(text_field, "")
        if isinstance(text, str) and text.strip():
            yield text


class PackedTextDataset(IterableDataset):
    """Pack documents into fixed token blocks with EOS separators."""

    def __init__(
        self,
        texts: Iterable[str] | Callable[[], Iterable[str]],
        tokenizer,
        seq_len: int,
        max_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_blocks = max_blocks

    def __iter__(self):
        buffer = []
        eos_id = int(self.tokenizer.eos_token_id)
        produced = 0
        texts = self.texts() if callable(self.texts) else iter(self.texts)
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buffer.extend(int(x) for x in ids)
            buffer.append(eos_id)
            while len(buffer) >= self.seq_len:
                block = buffer[: self.seq_len]
                del buffer[: self.seq_len]
                tensor = torch.tensor(block, dtype=torch.long)
                yield {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                }
                produced += 1
                if self.max_blocks is not None and produced >= self.max_blocks:
                    return


def pack_token_stream(tokens: Iterable[int], seq_len: int):
    buffer = []
    for token in tokens:
        buffer.append(int(token))
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            del buffer[:seq_len]
