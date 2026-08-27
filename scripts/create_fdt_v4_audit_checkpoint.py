from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT.parent / "source" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdt_rlm.models import build_model  # noqa: E402
from train_fdt_v4_curriculum import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    convert_v20_state_dict,
    load_payload,
    metadata_model_preflight,
    model_config_and_settings,
    require_c_path,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an untrained FDT v4 warm-start checkpoint for audit only")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = require_c_path(args.config, "config")
    parent_path = require_c_path(args.parent, "parent checkpoint")
    output_dir = require_c_path(args.output_dir, "audit output")
    runs_root = (ROOT / "runs").resolve()
    if output_dir.resolve() == runs_root or runs_root not in output_dir.resolve().parents:
        raise ValueError("audit checkpoint must be written to a fresh child of runs")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"audit output is not fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config, _, _, _ = model_config_and_settings(config_path, output_dir)
    preflight = metadata_model_preflight(config)
    parent_sha = sha256_file(parent_path)
    parent = load_payload(parent_path)
    if parent.get("stage_status") != "COMPLETE":
        raise ValueError("audit warm-start parent must be a COMPLETE checkpoint")

    model = build_model(config)
    conversion = convert_v20_state_dict(model, parent)
    conversion.update(
        {
            "source_checkpoint": str(parent_path),
            "source_checkpoint_sha256": parent_sha,
            "target_parameter_count": preflight["parameter_count"],
            "audit_only": True,
            "new_v4_parameters_trained": False,
        }
    )
    checkpoint = output_dir / "latest.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model_type": "fdt_v4",
            "model_config": asdict(config),
            "stage_status": "AUDIT_UNTRAINED_WARM_START",
            "optimizer_state_included": False,
            "model_state_dict": model.state_dict(),
            "audit_metadata": conversion,
        },
    )
    conversion["checkpoint"] = str(checkpoint)
    conversion["checkpoint_sha256"] = sha256_file(checkpoint)
    atomic_json(output_dir / "conversion_manifest.json", conversion)
    print(json.dumps(conversion, ensure_ascii=False))


if __name__ == "__main__":
    main()
