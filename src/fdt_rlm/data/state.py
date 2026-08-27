from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .manifest import read_json, write_json


@dataclass
class DatasetState:
    documents_seen: int = 0
    documents_kept: int = 0
    tokens_by_bucket: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetState":
        path = Path(path)
        if not path.exists():
            return cls()
        data = read_json(path)
        return cls(
            documents_seen=int(data.get("documents_seen", 0)),
            documents_kept=int(data.get("documents_kept", 0)),
            tokens_by_bucket={str(k): int(v) for k, v in data.get("tokens_by_bucket", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        write_json(path, {
            "documents_seen": self.documents_seen,
            "documents_kept": self.documents_kept,
            "tokens_by_bucket": self.tokens_by_bucket,
        })

