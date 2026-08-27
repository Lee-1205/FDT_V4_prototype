from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
STATE_DIR = ROOT / "runs" / "fdt_v4_console"
STATE_PATH = STATE_DIR / "state.json"
CONFIG = ROOT / "configs" / "fdt_v4_main_426m.yaml"
SUPERVISOR = ROOT / "scripts" / "luna_fdt_v4_supervisor.py"
GENERATOR = ROOT / "scripts" / "generate_fdt_v4.py"
TOKENIZER = ROOT / "tokenizers" / "fdt_v3_bpe_24k"
RUNS = ROOT / "runs"
_LOCK = threading.RLock()
_CHECKPOINT_CACHE: dict[str, tuple[int, int, dict]] = {}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_state() -> dict:
    with _LOCK:
        if not STATE_PATH.exists():
            return {}
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def write_state(payload: dict) -> None:
    with _LOCK:
        atomic_json(STATE_PATH, payload)


def c_run(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:" or RUNS.resolve() not in resolved.parents:
        raise ValueError("FDT v4 runs must stay under the C: workspace runs directory")
    if resolved.is_symlink():
        raise ValueError("A run directory may not be a symlink")
    return resolved


def alive(pid: int | None) -> bool:
    try:
        process = psutil.Process(int(pid))
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (TypeError, ValueError, psutil.Error):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_info(path: Path, hash_file: bool = False) -> dict:
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        return {}
    if list(path.parent.glob(path.name + ".tmp*")):
        return {}
    identity = (path.stat().st_mtime_ns, path.stat().st_size)
    cached = _CHECKPOINT_CACHE.get(str(path))
    if cached and cached[:2] == identity and (not hash_file or cached[2].get("sha256")):
        return cached[2]
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = payload.get("model_config", {})
    if config.get("model_type") != "fdt_v4":
        return {}
    result = {
        "path": str(path),
        "name": path.parent.name,
        "stage_status": payload.get("stage_status"),
        "optimizer_step": int(payload.get("optimizer_step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "optimizer_state_included": bool(payload.get("optimizer_state_included", False)),
        "parameters_class": "426M",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path) if hash_file else None,
    }
    _CHECKPOINT_CACHE[str(path)] = (*identity, result)
    return result


def last_json_event(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 256 * 1024))
            rows = handle.read().decode("utf-8", errors="replace").splitlines()
        for row in reversed(rows):
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            if payload.get("event") in {"train", "final", "start", "resume"}:
                return payload
    except OSError:
        pass
    return {}


def current_metrics(run: Path) -> dict:
    live = run / "live_metrics.json"
    if live.exists():
        try:
            return json.loads(live.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return last_json_event(run / "training_log.jsonl")


def discover_checkpoints() -> list[dict]:
    rows = []
    for path in sorted(RUNS.glob("fdt_v4*/latest.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            info = checkpoint_info(path)
            if info:
                rows.append(info)
        except Exception:
            continue
    return rows[:64]


def status_payload() -> dict:
    state = read_state()
    run_value = state.get("run_dir")
    run = c_run(Path(run_value)) if run_value else None
    supervisor_pid = state.get("supervisor_pid")
    trainer_pid = None
    if run and (run / "train.pid").exists():
        try:
            trainer_pid = int((run / "train.pid").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            trainer_pid = None
    running = alive(trainer_pid) or alive(supervisor_pid)
    metrics = current_metrics(run) if run else {}
    additional = int(metrics.get("additional_tokens", 0) or 0)
    target = int(metrics.get("target_additional_tokens", 0) or 0)
    if not target and run and (run / "run_manifest.json").exists():
        try:
            target = int(json.loads((run / "run_manifest.json").read_text(encoding="utf-8")).get("target_additional_tokens", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            target = 0
    rate = float(metrics.get("tokens_per_sec", 0.0) or 0.0)
    remaining_seconds = max(target - additional, 0) / rate if target and rate > 0 else None
    recovery = run / "latest_recovery.pt" if run else None
    can_resume = bool(not running and recovery and recovery.exists() and not list(run.glob("latest_recovery.pt.tmp*")))
    final = checkpoint_info(run / "latest.pt") if run and (run / "latest.pt").exists() else {}
    status = "TRAINING" if running else final.get("stage_status") or ("PAUSED" if can_resume else "IDLE")
    return {
        "status": status,
        "running": running,
        "run_dir": str(run) if run else None,
        "metrics": {
            "loss": metrics.get("loss"),
            "entropy_normalized": metrics.get("entropy_normalized"),
            "remaining_seconds": remaining_seconds,
            "additional_tokens": additional,
            "target_additional_tokens": target,
        },
        "checkpoint": final,
        "can_emergency_stop": running and bool(run),
        "can_resume": can_resume,
        "can_test": not running,
    }


def launch_supervisor(run: Path, resume: bool) -> dict:
    run = c_run(run)
    if alive(read_state().get("supervisor_pid")):
        raise RuntimeError("Luna supervision is already running")
    if resume and not (run / "latest_recovery.pt").exists():
        raise RuntimeError("No verified recovery checkpoint is available")
    if (run / "STOP_REQUESTED").exists():
        (run / "STOP_REQUESTED").unlink()
    stdout_path = run / f"console.{datetime.now():%Y%m%d_%H%M%S}.stdout.log"
    stderr_path = run / f"console.{datetime.now():%Y%m%d_%H%M%S}.stderr.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(SUPERVISOR),
        "--config",
        str(CONFIG),
        "--run-dir",
        str(run),
        "--device",
        "cuda",
        "--allow-gpu",
    ]
    if resume:
        command.append("--resume-paused")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_MODULE_LOADING"] = "LAZY"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stdout.close()
    stderr.close()
    payload = {
        "run_dir": str(run),
        "supervisor_pid": process.pid,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "updated_at": time.time(),
    }
    write_state(payload)
    return payload


class Handler(BaseHTTPRequestHandler):
    def json_response(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.json_response(status_payload())
            return
        if path == "/api/checkpoints":
            self.json_response({"checkpoints": discover_checkpoints()})
            return
        target = STATIC / ("index.html" if path == "/" else "models.html" if path == "/models.html" else path.lstrip("/"))
        if not target.is_file() or STATIC.resolve() not in target.resolve().parents:
            self.send_error(404)
            return
        mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            snapshot = status_payload()
            if path == "/api/emergency-stop":
                if not snapshot.get("can_emergency_stop"):
                    raise RuntimeError("No active training process")
                run = c_run(Path(snapshot["run_dir"]))
                marker = run / "STOP_REQUESTED"
                temporary = marker.with_name(marker.name + ".tmp")
                temporary.write_text("USER_EMERGENCY_STOP\n", encoding="ascii")
                os.replace(temporary, marker)
                self.json_response(status_payload())
                return
            if path == "/api/resume":
                if not snapshot.get("can_resume"):
                    raise RuntimeError("No verified paused recovery is available")
                self.json_response({"launched": launch_supervisor(Path(snapshot["run_dir"]), True)})
                return
            if path == "/api/generate":
                if snapshot.get("running"):
                    raise RuntimeError("Model tests are disabled during training")
                checkpoint = c_run(Path(str(data.get("checkpoint", ""))))
                if checkpoint.name != "latest.pt" or not checkpoint_info(checkpoint, hash_file=True):
                    raise ValueError("Select a verified FDT v4 model checkpoint")
                prompt = str(data.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("Prompt is empty")
                result = subprocess.run(
                    [sys.executable, str(GENERATOR), "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZER), "--prompt", prompt, "--max-new-tokens", "128"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=1800,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode:
                    raise RuntimeError(result.stderr[-2000:] or "Generation failed")
                self.json_response(json.loads(result.stdout.strip().splitlines()[-1]))
                return
            self.send_error(404)
        except Exception as error:
            self.json_response({"error": str(error)}, 400)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure FDT v4 Training Console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir:
        write_state({"run_dir": str(c_run(args.run_dir)), "supervisor_pid": None, "updated_at": time.time()})
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
