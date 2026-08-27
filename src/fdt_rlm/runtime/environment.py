from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence

from .actions import Action, ActionName


@dataclass
class ActionResult:
    ok: bool
    observation: str
    result_ref: str | None = None
    tokens_read: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


class Environment(Protocol):
    allowed_actions: Sequence[ActionName]

    def execute(self, action: Action) -> ActionResult: ...

    def resolve_ref(self, result_ref: str) -> str: ...
