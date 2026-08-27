from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from .dedup import normalized_text_hash


class ContaminationFilter:
    def __init__(self):
        self.hashes: Dict[str, str] = {}

    def add_benchmark_texts(self, name: str, texts: Iterable[str]) -> None:
        for text in texts:
            self.hashes[normalized_text_hash(text)] = name

    def add_benchmark_file(self, name: str, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            import json

            texts = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    texts.append(row.get("prompt") or row.get("text") or row.get("canonical_solution") or "")
            self.add_benchmark_texts(name, texts)
        else:
            self.add_benchmark_texts(name, [path.read_text(encoding="utf-8")])

    def check_document(self, text: str):
        return self.hashes.get(normalized_text_hash(text))

