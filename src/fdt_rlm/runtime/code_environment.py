from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Dict, Mapping

from .actions import Action, ActionName
from .environment import ActionResult


class CodeEnvironment:
    """Read-only in-memory repository environment for deterministic evaluation."""

    allowed_actions = (
        ActionName.LIST_FILES,
        ActionName.READ_FILE,
        ActionName.GREP,
        ActionName.FIND_SYMBOL,
        ActionName.FIND_IMPORTS,
        ActionName.CALL,
        ActionName.STOP,
    )

    def __init__(self, files: Mapping[str, str]):
        self.files = {str(PurePosixPath(path)): text for path, text in files.items()}
        self.results: Dict[str, str] = {}
        self._next_ref = 0

    def _store(self, text: str) -> str:
        ref = f"result_{self._next_ref}"
        self._next_ref += 1
        self.results[ref] = text
        return ref

    def resolve_ref(self, result_ref: str) -> str:
        if result_ref not in self.results:
            raise KeyError(f"unknown context reference: {result_ref}")
        return self.results[result_ref]

    def execute(self, action: Action) -> ActionResult:
        if action.name == ActionName.LIST_FILES:
            root = str(PurePosixPath(action.arguments["path"]))
            paths = sorted(path for path in self.files if path.startswith(root))
            text = "\n".join(paths)
            return ActionResult(True, text, self._store(text), 0, {"files": len(paths)})
        if action.name == ActionName.READ_FILE:
            path = str(PurePosixPath(action.arguments["path"]))
            if path not in self.files:
                return ActionResult(False, f"unknown file: {path}")
            lines = self.files[path].splitlines()
            start = action.arguments["start"]
            end = min(action.arguments["end"], len(lines))
            if start >= len(lines):
                return ActionResult(False, "READ_FILE start is outside file")
            text = "\n".join(lines[start:end])
            return ActionResult(True, text, self._store(text), len(text.split()), {"path": path})
        if action.name == ActionName.GREP:
            pattern = action.arguments["pattern"]
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                return ActionResult(False, f"invalid pattern: {exc}")
            matches = []
            for path, content in sorted(self.files.items()):
                for line_no, line in enumerate(content.splitlines(), start=1):
                    if regex.search(line):
                        matches.append(f"{path}:{line_no}:{line}")
            text = "\n".join(matches[:50])
            return ActionResult(True, text, self._store(text), len(text.split()), {"matches": len(matches)})
        if action.name == ActionName.FIND_SYMBOL:
            name = re.escape(action.arguments["name"])
            regex = re.compile(rf"\b(?:def|class|fn|func|function)\s+{name}\b")
            return self.execute(Action(ActionName.GREP, {"pattern": regex.pattern}))
        if action.name == ActionName.FIND_IMPORTS:
            path = str(PurePosixPath(action.arguments["path"]))
            if path not in self.files:
                return ActionResult(False, f"unknown file: {path}")
            imports = [
                line
                for line in self.files[path].splitlines()
                if re.match(r"\s*(?:from\s+\S+\s+import|import\s+|use\s+|require\()", line)
            ]
            text = "\n".join(imports)
            return ActionResult(True, text, self._store(text), len(text.split()))
        return ActionResult(False, f"{action.name.value} is not handled by CodeEnvironment")
