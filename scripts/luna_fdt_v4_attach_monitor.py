from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def c_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_status(pid: int) -> tuple[bool, int | None]:
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
    )
    if not handle:
        return False, None
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False, None
        return exit_code.value == STILL_ACTIVE, int(exit_code.value)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_after_exit(run: Path, trainer_pid: int, exit_code: int | None) -> dict[str, Any]:
    import torch

    recovery = run / "latest_recovery.pt"
    model = run / "latest.pt"
    temporary = [path.name for path in run.glob("*.tmp")]
    if not recovery.is_file() or temporary:
        raise RuntimeError("trainer exited without a clean atomic recovery checkpoint")
    payload = torch.load(recovery, map_location="cpu", weights_only=False)
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "train_config",
        "source_states",
        "torch_rng_state",
        "python_random_state",
    )
    valid = payload.get("optimizer_state_included") and all(
        key in payload for key in required
    )
    if not valid:
        raise RuntimeError("post-exit recovery lacks exact-resume state")
    result = {
        "status": "VERIFIED",
        "trainer_pid": trainer_pid,
        "process_exit_code": exit_code,
        "stage_status": payload.get("stage_status"),
        "optimizer_step": int(payload.get("optimizer_step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "optimizer_state_included": True,
        "recovery_checkpoint": str(recovery),
        "recovery_sha256": sha256(recovery),
        "model_checkpoint": str(model) if model.is_file() else None,
        "model_sha256": sha256(model) if model.is_file() else None,
        "temp_residue": temporary,
        "next_route": "terra_evaluation" if payload.get("stage_status") == "COMPLETE" else "terra_incident_review",
    }
    del payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    run = c_path(args.run_dir, "run")
    state = c_path(args.state_dir, "state")
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in run.parents or runs_root not in state.parents or run == state:
        raise ValueError("run and state must be separate workspace run directories")
    if state.exists() and any(state.iterdir()):
        raise FileExistsError(f"attach-monitor state is not fresh: {state}")
    state.mkdir(parents=True, exist_ok=True)
    (state / "monitor.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    atomic_json(
        state / "attached.json",
        {
            "status": "ATTACHED",
            "trainer_pid": args.trainer_pid,
            "run_dir": str(run),
            "monitor_pid": os.getpid(),
            "checkpoint_loaded_while_training": False,
        },
    )
    while True:
        active, exit_code = process_status(args.trainer_pid)
        if not active:
            break
        time.sleep(max(args.poll_seconds, 1.0))
    try:
        result = verify_after_exit(run, args.trainer_pid, exit_code)
    except Exception as exc:
        result = {
            "status": "VERIFICATION_ERROR",
            "trainer_pid": args.trainer_pid,
            "process_exit_code": exit_code,
            "error": str(exc),
            "next_route": "terra_incident_review",
        }
    atomic_json(state / "post_exit_verification.json", result)
    atomic_json(run / "terra_handoff.json", result)
    return 0 if result["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
