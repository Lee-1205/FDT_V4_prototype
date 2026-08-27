from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    step: int
    depth: int
    raw_output: str
    action: Dict[str, Any] | None
    observation: str
    ok: bool
    signals: Dict[str, float] = field(default_factory=dict)
    budget_remaining: Dict[str, int] = field(default_factory=dict)


@dataclass
class RLMTrace:
    events: List[TraceEvent] = field(default_factory=list)

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [asdict(event) for event in self.events]}
