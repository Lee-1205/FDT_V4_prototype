from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().with_name("train_fdt_v4_curriculum_speed.py")

ESCALATION_POLICY = {
    "routine_owner": "luna",
    "abnormality_owner": "terra",
    "major_remedy_owner": "sol",
    "lightweight_remedy_owner": "terra_or_luna",
    "single_trainable_model": "fdt_v4_main_426m",
    "forbidden_model_variants": ["fdt_v4_probe_116m", "116m", "scratch_probe"],
    "luna_abnormality_action": "checkpoint_pause_and_atomic_handoff",
    "luna_may_redesign": False,
    "luna_may_continue_after_gate_failure": False,
}


def c_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def run_path(path: Path) -> Path:
    resolved = c_path(path, "run")
    runs_root = (ROOT / "runs").resolve()
    if runs_root not in resolved.parents:
        raise ValueError("run must be below the workspace runs directory")
    if resolved.is_symlink():
        raise ValueError("run may not be a symlink")
    return resolved


def supervisor_state_path(run: Path, configured: Path | None) -> Path:
    state = run_path(configured) if configured is not None else run.parent / f"{run.name}_supervisor"
    if state == run:
        raise ValueError("Supervisor state must be separate from the trainer output directory")
    return state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = c_path(path, "handoff")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_marker(path: Path, text: str) -> None:
    path = c_path(path, "stop marker")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def release_checkpoint_memory(*payloads: object) -> None:
    """Release resume verification payloads before the trainer owns host memory."""
    del payloads
    gc.collect()
    if os.name == "nt":
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())


def verified_recovery(path: Path, run: Path) -> dict[str, Any]:
    path = c_path(path, "recovery checkpoint")
    if path.parent != run or path.name.endswith(".tmp") or not path.exists():
        raise ValueError("Recovery checkpoint is not an owned, complete file")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model_state_dict", "optimizer_state_dict", "model_config", "train_config", "source_states")
    if not payload.get("optimizer_state_included") or any(key not in payload for key in required):
        raise ValueError("Recovery checkpoint is missing resumability state")
    if payload.get("model_config", {}).get("model_type") != "fdt_v4":
        raise ValueError("Recovery checkpoint is not fdt_v4")
    if Path(payload["train_config"].get("output_dir", "")).resolve() != run:
        raise ValueError("Recovery checkpoint output ownership mismatch")
    if not all((run / name).exists() for name in ("latest_recovery.pt",)):
        raise ValueError("Recovery checkpoint disappeared during verification")
    return payload


def handoff_path(config_raw: dict[str, Any], run: Path) -> Path:
    configured = str(config_raw.get("terra", {}).get("handoff_path", "artifacts/fdt_v4_terra_handoff.json"))
    path = Path(configured)
    resolved = c_path(path if path.is_absolute() else run / path, "Terra handoff")
    if resolved.parent != run:
        raise ValueError("Terra handoff must be run-owned; global handoff paths are forbidden")
    return resolved


def write_handoff(
    config_raw: dict[str, Any],
    run: Path,
    checkpoint: Path,
    status: str,
    *,
    severity: str = "info",
    classification: str = "completed",
    trigger: str = "normal_completion",
    terra_instruction: str = "evaluate_verified_checkpoint",
) -> Path:
    payload = verified_recovery(checkpoint, run)
    model_path = run / "latest.pt"
    if not model_path.exists() or model_path.name.endswith(".tmp"):
        raise ValueError("Model checkpoint is not ready for Terra handoff")
    destination = handoff_path(config_raw, run)
    abnormal = classification != "completed"
    terra = config_raw.get("terra", {})
    evaluation_fields: dict[str, Any] = {}
    if not abnormal:
        dataset = c_path(ROOT / str(terra.get("tensor_dataset", "")), "evaluation tensor dataset")
        repetition = c_path(ROOT / str(terra.get("repetition_tensor_dataset", terra.get("tensor_dataset", ""))), "repetition tensor dataset")
        tokenizer_dir = c_path(ROOT / str(config_raw.get("data", {}).get("tokenizer_dir", "")), "tokenizer directory")
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        if not dataset.is_file() or not repetition.is_file() or not tokenizer_json.is_file():
            raise ValueError("Normal evaluation handoff requires fixed tensor datasets and tokenizer.json")
        expected_dataset_sha = str(terra.get("tensor_dataset_sha256", "")).upper()
        expected_repetition_sha = str(terra.get("repetition_tensor_dataset_sha256", "")).upper()
        if not expected_dataset_sha or expected_dataset_sha.startswith("REQUIRED") or sha256(dataset) != expected_dataset_sha:
            raise ValueError("Normal evaluation handoff requires a pinned tensor dataset SHA-256")
        if not expected_repetition_sha or expected_repetition_sha.startswith("REQUIRED") or sha256(repetition) != expected_repetition_sha:
            raise ValueError("Normal evaluation handoff requires a pinned repetition dataset SHA-256")
        evaluation_fields = {
            "tensor_dataset": str(dataset),
            "tensor_dataset_sha256": sha256(dataset),
            "repetition_tensor_dataset": str(repetition),
            "repetition_tensor_dataset_sha256": sha256(repetition),
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_json_sha256": sha256(tokenizer_json),
        }
    config_file = ROOT / "configs" / "fdt_v4_main_426m.yaml"
    atomic_json(destination, {
        "schema_version": "fdt_terra_handoff_v1",
        "artifact_role": "verified_checkpoint_evaluation_input",
        "status": "ABNORMAL" if abnormal else "READY",
        "handoff_status": "ABNORMAL" if abnormal else "READY",
        "handoff_type": "INCIDENT" if abnormal else "EVALUATION",
        "created_at": time.time(),
        "source_run": str(run),
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "recovery_checkpoint": str(checkpoint),
        "recovery_sha256": sha256(checkpoint),
        "stage_status": status,
        "severity": severity,
        "classification": classification,
        "trigger": trigger,
        "luna_action": "PAUSED_AND_HANDOFF" if classification != "completed" else "HANDOFF_AFTER_COMPLETION",
        "terra_instruction": terra_instruction,
        "required_remedy_classification": "pending_terra" if abnormal else "evaluation_only",
        "next_route": "terra",
        "sol_route_condition": "Terra must classify major_architecture_objective_data_contract before Sol is routed",
        "escalation_policy": ESCALATION_POLICY,
        "optimizer_step": int(payload.get("optimizer_step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "model_type": payload["model_config"]["model_type"],
        "evaluator": terra.get("evaluator", ""),
        "config_sha256": sha256(config_file) if config_file.is_file() else "UNPINNED",
        "precision_contract": payload.get("train_config", {}).get("precision", "UNPINNED"),
        "environment_contract": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda},
        "evaluation_executed": False,
        "handoff_contract": "Terra owns evaluation; supervisor only writes verified immutable input metadata",
        **evaluation_fields,
    })
    return destination


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(c_path(path, "config").read_text(encoding="utf-8")) or {}


def main(args: argparse.Namespace) -> int:
    config_path = c_path(args.config, "config")
    run = run_path(args.run_dir)
    state = supervisor_state_path(run, args.state_dir)
    raw = load_config(config_path)
    run.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    atomic_marker(state / "supervisor.pid", f"{os.getpid()}\n")
    log = state / "supervisor_log.jsonl"
    recovery = run / "latest_recovery.pt"
    restarts = 0
    while True:
        stop_marker = run / "STOP_REQUESTED"
        gate_files = [run / "GATE_FAILURE.json", run / "gate_failure.json"]
        if stop_marker.exists() and args.resume_paused:
            stop_marker.unlink()
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "resume_paused_marker_consumed"}) + "\n")
        if (stop_marker.exists() and not args.resume_paused) or any(path.exists() for path in gate_files):
            if not recovery.exists():
                raise RuntimeError("Escalation requires an atomic recovery checkpoint")
            payload = verified_recovery(recovery, run)
            classification = "gate_failure" if any(path.exists() for path in gate_files) else "user_or_safety_stop"
            destination = write_handoff(
                raw,
                run,
                recovery,
                str(payload.get("stage_status", "PAUSED")),
                severity="critical" if classification == "gate_failure" else "warning",
                classification=classification,
                trigger="gate_or_stop_marker",
                terra_instruction="diagnose_only_no_luna_redesign",
            )
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "escalated_before_launch", "path": str(destination), "classification": classification}) + "\n")
            return 0
        resume_candidate: Path | None = None
        if recovery.exists():
            existing = verified_recovery(recovery, run)
            existing_status = str(existing.get("stage_status", ""))
            if existing_status in {"COMPLETE", "SAFETY_STOP"} or (existing_status == "PAUSED" and not args.resume_paused):
                destination = write_handoff(
                    raw,
                    run,
                    recovery,
                    existing_status,
                    severity="info" if existing_status == "COMPLETE" else "warning",
                    classification="completed" if existing_status == "COMPLETE" else "paused_recovery",
                    trigger="existing_verified_recovery",
                    terra_instruction="evaluate_verified_checkpoint" if existing_status == "COMPLETE" else "diagnose_only_no_luna_redesign",
                )
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "existing_recovery_handoff", "path": str(destination), "status": existing_status}) + "\n")
                return 0
            if existing_status == "PAUSED" and args.resume_paused:
                resume_candidate = recovery
            existing = None
            release_checkpoint_memory()
        stdout_path = state / f"train.{time.strftime('%Y%m%d_%H%M%S')}.stdout.log"
        stderr_path = state / f"train.{time.strftime('%Y%m%d_%H%M%S')}.stderr.log"
        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        command = [sys.executable, str(SCRIPT), "--config", str(config_path), "--output-dir", str(run), "--device", args.device]
        if args.allow_gpu:
            command.append("--allow-gpu")
        if resume_candidate is not None:
            try:
                if not resume_candidate.exists() or resume_candidate.name.endswith(".tmp"):
                    raise ValueError("resume checkpoint changed after verification")
                command.extend(["--resume", str(resume_candidate)])
            except Exception as exc:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "resume_rejected", "error": str(exc)}) + "\n")
                resume_candidate = None
        try:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=stdout_handle, stderr=stderr_handle, text=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "launch", "pid": process.pid, "restarts": restarts, "command": command, "stdout": str(stdout_path), "stderr": str(stderr_path)}) + "\n")
            while process.poll() is None:
                time.sleep(max(args.poll_seconds, 1.0))
            return_code = int(process.returncode or 0)
        finally:
            stdout_handle.close()
            stderr_handle.close()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "exit", "return_code": return_code, "restarts": restarts}) + "\n")
        if not recovery.exists():
            raise RuntimeError("Trainer stopped without an atomic recovery checkpoint")
        payload = verified_recovery(recovery, run)
        status = str(payload.get("stage_status", ""))
        if status in {"PAUSED", "SAFETY_STOP"} or (run / "STOP_REQUESTED").exists():
            destination = write_handoff(
                raw,
                run,
                recovery,
                status,
                severity="critical" if status == "SAFETY_STOP" else "warning",
                classification="abnormality_or_safety_stop",
                trigger="trainer_safety_state",
                terra_instruction="diagnose_only_no_luna_redesign",
            )
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "terra_handoff", "path": str(destination), "status": status, "severity": "critical" if status == "SAFETY_STOP" else "warning"}) + "\n")
            return 0
        if status == "COMPLETE":
            destination = write_handoff(raw, run, recovery, status)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "terra_handoff", "path": str(destination), "status": status}) + "\n")
            return 0
        if return_code != 0:
            # The existing recovery is retained, but Luna must not improvise after an abnormal exit.
            atomic_marker(run / "STOP_REQUESTED", "LUNA_ESCALATION_PROCESS_INTERRUPTION\n")
            destination = write_handoff(
                raw,
                run,
                recovery,
                status or "INTERRUPTED",
                severity="critical",
                classification="process_interruption",
                trigger="unexpected_trainer_exit",
                terra_instruction="diagnose_only_no_luna_redesign",
            )
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "terra_handoff", "path": str(destination), "classification": "process_interruption"}) + "\n")
            return 0
        restarts += 1
        if args.max_restarts > 0 and restarts > args.max_restarts:
            raise RuntimeError("Maximum unattended trainer restarts exceeded")
        time.sleep(max(args.restart_delay, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unattended FDT v4 supervisor with Terra handoff")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, help="Separate run-owned directory for supervisor logs and PID")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--resume-paused", action="store_true", help="Explicitly consume a verified PAUSED recovery and resume training")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--restart-delay", type=float, default=2.0)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 means unlimited only across clean return-code-0 interruptions")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
