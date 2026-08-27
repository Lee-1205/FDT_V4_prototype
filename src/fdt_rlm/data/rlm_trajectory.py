from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RLMTask:
    task_id: str
    family: str
    difficulty: str
    question: str
    context: str
    gold_answer: str
    gold_evidence: List[str]
    gold_trajectory: List[Dict[str, Any]]
    files: Dict[str, str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
