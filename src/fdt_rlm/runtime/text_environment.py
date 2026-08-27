from __future__ import annotations

import re
from typing import Dict

from .actions import Action, ActionName
from .environment import ActionResult


class TextEnvironment:
    allowed_actions = (
        ActionName.READ,
        ActionName.SEARCH,
        ActionName.COMPARE,
        ActionName.MERGE,
        ActionName.CALL,
        ActionName.STOP,
    )

    def __init__(self, context: str, search_window: int = 48):
        self.context = context
        self.tokens = context.split()
        self.search_window = search_window
        self.results: Dict[str, str] = {}
        self._next_ref = 0

    def _store(self, text: str) -> str:
        result_ref = f"result_{self._next_ref}"
        self._next_ref += 1
        self.results[result_ref] = text
        return result_ref

    def resolve_ref(self, result_ref: str) -> str:
        if result_ref not in self.results:
            raise KeyError(f"unknown context reference: {result_ref}")
        return self.results[result_ref]

    def execute(self, action: Action) -> ActionResult:
        if action.name == ActionName.READ:
            start = action.arguments["start"]
            end = action.arguments["end"]
            if start >= len(self.tokens):
                return ActionResult(False, "READ start is outside context")
            end = min(end, len(self.tokens))
            text = " ".join(self.tokens[start:end])
            ref = self._store(text)
            return ActionResult(True, text, ref, end - start, {"start": start, "end": end})
        if action.name == ActionName.SEARCH:
            query = action.arguments["query"].strip()
            if not query:
                return ActionResult(False, "SEARCH query is empty")
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            matches = []
            for match in pattern.finditer(self.context):
                left = max(0, match.start() - self.search_window)
                right = min(len(self.context), match.end() + self.search_window)
                matches.append(self.context[left:right])
                if len(matches) >= 5:
                    break
            if not matches:
                return ActionResult(True, "No matches", self._store(""), 0, {"matches": 0})
            text = "\n---\n".join(matches)
            ref = self._store(text)
            return ActionResult(True, text, ref, len(text.split()), {"matches": len(matches)})
        if action.name in {ActionName.COMPARE, ActionName.MERGE}:
            try:
                parts = [self.resolve_ref(ref) for ref in action.arguments["result_refs"]]
            except KeyError as exc:
                return ActionResult(False, str(exc))
            separator = "\n---COMPARE---\n" if action.name == ActionName.COMPARE else "\n"
            text = separator.join(parts)
            return ActionResult(True, text, self._store(text), 0)
        return ActionResult(False, f"{action.name.value} is not handled by TextEnvironment")
