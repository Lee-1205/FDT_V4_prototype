from __future__ import annotations

"""Classify atomic Luna abnormality handoffs without changing the model."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from terra_fdt_v4_evaluator import (
    INCIDENT_HANDOFF_STATUS,
    INCIDENT_HANDOFF_TYPE,
    classify_luna_abnormality,
    sha256_file,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def consume_luna_handoff(handoff: Path, result: Path) -> dict[str, Any]:
    handoff = handoff.resolve()
    if handoff.suffix == ".tmp" or not handoff.is_file():
        raise ValueError("Luna handoff must be a completed non-temporary file")
    source = json.loads(handoff.read_text(encoding="utf-8"))
    if source.get("status") != INCIDENT_HANDOFF_STATUS or source.get("handoff_type") != INCIDENT_HANDOFF_TYPE:
        raise ValueError("incident handoff must declare status ABNORMAL and handoff_type INCIDENT")
    decision = classify_luna_abnormality(source)
    payload = {
        "schema": "terra_fdt_v4_incident_v2",
        "status": "CLASSIFIED",
        "source": "Luna",
        "handoff": str(handoff),
        "handoff_sha256": sha256_file(handoff),
        "incident_id": source.get("incident_id") or source.get("handoff_id"),
        "model_policy": {
            "lineages_allowed": 1,
            "target_parameter_count": 426_000_000,
            "forbid_116m": True,
        },
        "decision": decision,
        "completed_at_unix": time.time(),
    }
    atomic_json(result.resolve(), payload)
    payload["integrity_digest"] = sha256_file(result.resolve())
    atomic_text(result.resolve().with_name(result.name + ".sha256"), payload["integrity_digest"] + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Terra FDT v4 Luna-abnormality incident classifier")
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = consume_luna_handoff(args.handoff, args.result)
    print(json.dumps({"result": str(args.result.resolve()), "integrity_digest": payload["integrity_digest"]}))


if __name__ == "__main__":
    main()
