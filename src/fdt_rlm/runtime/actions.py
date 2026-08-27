from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping


class ActionName(str, Enum):
    READ = "READ"
    SEARCH = "SEARCH"
    CALL = "CALL"
    COMPARE = "COMPARE"
    MERGE = "MERGE"
    STOP = "STOP"
    LIST_FILES = "LIST_FILES"
    READ_FILE = "READ_FILE"
    GREP = "GREP"
    FIND_SYMBOL = "FIND_SYMBOL"
    FIND_IMPORTS = "FIND_IMPORTS"


class ActionValidationError(ValueError):
    pass


REQUIRED: Dict[ActionName, Dict[str, type | tuple[type, ...]]] = {
    ActionName.READ: {"start": int, "end": int},
    ActionName.SEARCH: {"query": str},
    ActionName.CALL: {"question": str, "context_ref": str},
    ActionName.COMPARE: {"result_refs": list},
    ActionName.MERGE: {"result_refs": list},
    ActionName.STOP: {"answer": str},
    ActionName.LIST_FILES: {"path": str},
    ActionName.READ_FILE: {"path": str, "start": int, "end": int},
    ActionName.GREP: {"pattern": str},
    ActionName.FIND_SYMBOL: {"name": str},
    ActionName.FIND_IMPORTS: {"path": str},
}


@dataclass(frozen=True)
class Action:
    name: ActionName
    arguments: Dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Action":
        raw_name = value.get("action")
        if not isinstance(raw_name, str):
            raise ActionValidationError("action must be a string")
        try:
            name = ActionName(raw_name.upper())
        except ValueError as exc:
            raise ActionValidationError(f"unknown action: {raw_name}") from exc
        arguments = {key: item for key, item in value.items() if key != "action"}
        for field, expected in REQUIRED[name].items():
            if field not in arguments:
                raise ActionValidationError(f"{name.value} requires {field}")
            if not isinstance(arguments[field], expected) or isinstance(arguments[field], bool):
                raise ActionValidationError(
                    f"{name.value}.{field} must be {getattr(expected, '__name__', expected)}"
                )
        if name in {ActionName.COMPARE, ActionName.MERGE}:
            refs = arguments["result_refs"]
            if len(refs) < 1 or not all(isinstance(item, str) for item in refs):
                raise ActionValidationError("result_refs must be a non-empty list of strings")
        if name in {ActionName.READ, ActionName.READ_FILE}:
            if arguments["start"] < 0 or arguments["end"] <= arguments["start"]:
                raise ActionValidationError("READ range must satisfy 0 <= start < end")
        return cls(name=name, arguments=arguments)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.name.value, **self.arguments}

    def fingerprint(self) -> str:
        items = sorted((key, repr(value)) for key, value in self.arguments.items())
        return f"{self.name.value}:{items}"
