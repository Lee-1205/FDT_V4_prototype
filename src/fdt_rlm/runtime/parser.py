from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .actions import Action, ActionValidationError


@dataclass(frozen=True)
class ParseResult:
    action: Optional[Action]
    error: Optional[str]
    repaired: bool = False


def _candidate(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
        return stripped, True
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start and (start != 0 or end != len(stripped) - 1):
        return stripped[start : end + 1], True
    return stripped, False


def parse_action(text: str) -> ParseResult:
    candidate, repaired = _candidate(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseResult(None, f"invalid JSON: {exc.msg}", repaired)
    if not isinstance(value, dict):
        return ParseResult(None, "action payload must be an object", repaired)
    try:
        return ParseResult(Action.from_mapping(value), None, repaired)
    except ActionValidationError as exc:
        return ParseResult(None, str(exc), repaired)
