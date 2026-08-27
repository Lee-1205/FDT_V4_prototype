from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_profiler():
    path = ROOT / "scripts" / "profile_fdt_v4_current_gpu.py"
    spec = importlib.util.spec_from_file_location("profile_fdt_v4_current_gpu", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profiler_atomic_outputs_are_machine_readable(tmp_path):
    module = load_profiler()
    json_path = tmp_path / "profile.json"
    csv_path = tmp_path / "profile.csv"
    module.atomic_json(json_path, {"dtype": "float32", "quantization": "none"})
    module.atomic_csv(csv_path, [{"context": 512, "status": "PASS"}])
    assert json.loads(json_path.read_text(encoding="utf-8"))["dtype"] == "float32"
    assert "context,status" in csv_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_profiler_source_does_not_enable_quantization():
    source = (ROOT / "scripts" / "profile_fdt_v4_current_gpu.py").read_text(encoding="utf-8")
    assert '"quantization": "none"' in source
    assert "dtype=torch.float32" in source
    assert "warmup(model, config)" in source
    assert '"PERFORMANCE"' in source
    assert "outputs are immutable" in source
