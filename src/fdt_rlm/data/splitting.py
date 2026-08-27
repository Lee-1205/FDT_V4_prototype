from __future__ import annotations

import hashlib
from typing import Mapping


def assign_split(key: str, ratios: Mapping[str, float], seed: int = 42) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(16**16)
    train = float(ratios.get("train", 0.99))
    validation = float(ratios.get("validation", 0.005))
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"

