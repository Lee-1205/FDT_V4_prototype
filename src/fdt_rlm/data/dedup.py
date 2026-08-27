from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from .filters import normalize_web_text


def normalized_text_hash(text: str) -> str:
    normalized = normalize_web_text(text)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


class SQLiteDeduper:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, source TEXT, split TEXT)")
        self.conn.commit()

    def check_and_add(self, digest: str, source: str, split: str = "") -> bool:
        try:
            self.conn.execute("INSERT INTO seen(hash, source, split) VALUES (?, ?, ?)", (digest, source, split))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self) -> None:
        self.conn.close()


class MemoryDeduper:
    def __init__(self):
        self.seen = set()

    def check_and_add(self, digest: str, source: str = "", split: str = "") -> bool:
        if digest in self.seen:
            return False
        self.seen.add(digest)
        return True


def build_deduper(path: Optional[str | Path]):
    if path:
        return SQLiteDeduper(path)
    return MemoryDeduper()

