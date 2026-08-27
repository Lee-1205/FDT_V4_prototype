from __future__ import annotations

import importlib.util
import shutil
from unittest.mock import patch
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEST_ROOT if (TEST_ROOT / "apps").is_dir() else TEST_ROOT.parent
APP_ROOT = ROOT / ("apps" if (ROOT / "apps").is_dir() else "app")


def load_console():
    path = APP_ROOT / "fdt_v4_console" / "server.py"
    spec = importlib.util.spec_from_file_location("fdt_v4_console_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_console_exposes_only_emergency_stop_and_resume_controls():
    html = (APP_ROOT / "fdt_v4_console" / "static" / "index.html").read_text(encoding="utf-8")
    assert "비상 정지" in html
    assert "재개" in html
    assert ">시작<" not in html


def test_console_poll_interval_is_three_seconds():
    script = (APP_ROOT / "fdt_v4_console" / "static" / "app.js").read_text(encoding="utf-8")
    assert "setInterval(poll,3000)" in script


def test_console_rejects_paths_outside_workspace_runs(tmp_path):
    server = load_console()
    try:
        server.c_run(tmp_path / "outside")
    except ValueError:
        pass
    else:
        raise AssertionError("console accepted a non-owned path")


def test_console_resume_passes_explicit_resume_flag(tmp_path):
    server = load_console()
    run = server.RUNS / "fdt_v4_console_resume_test"
    run.mkdir(parents=True, exist_ok=True)
    (run / "latest_recovery.pt").write_bytes(b"recovery")

    class Process:
        pid = 12345

    try:
        with patch.object(server, "read_state", return_value={}), patch.object(
            server.subprocess, "Popen", return_value=Process()
        ) as popen, patch.object(server, "write_state"):
            server.launch_supervisor(run, True)
        command = popen.call_args.args[0]
        assert "--resume-paused" in command
    finally:
        shutil.rmtree(run, ignore_errors=True)
