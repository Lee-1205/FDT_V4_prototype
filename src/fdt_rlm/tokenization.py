from __future__ import annotations

from pathlib import Path
from typing import Any


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<rlm>", "<read>", "<search>", "<call>", "<compare>", "<merge>", "<stop>", "<result>", "<state>", "<confidence>"]


def load_tokenizer(name_or_path: str = "gpt2") -> Any:
    """Load a byte-level BPE style tokenizer for English/code experiments.

    `gpt2` is the default bootstrap tokenizer because it is byte-level BPE and
    widely available. A project-trained tokenizer can later be passed by path.
    """

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    path = Path(name_or_path)
    if path.exists() and (path / "tokenizer.json").exists():
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(path / "tokenizer.json"),
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            sep_token="<sep>",
            additional_special_tokens=[tok for tok in SPECIAL_TOKENS if tok not in {"<pad>", "<bos>", "<eos>", "<sep>"}],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
