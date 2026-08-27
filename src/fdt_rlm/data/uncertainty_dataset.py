from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class UncertaintyRecord:
    task_id: str
    step: int
    correct_without_more_context: bool
    additional_read_helpful: bool
    recursive_call_helpful: bool
    signals: Dict[str, float]

    def to_dict(self):
        return asdict(self)
