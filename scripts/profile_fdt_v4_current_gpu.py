from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT.parent / "source" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def cuda_ms(operation) -> tuple[Any, float]:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = operation()
    stop.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(stop))


def driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def profile_context(model, config: ModelConfig, context: int, decode_steps: int) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = ((torch.arange(context, device="cuda", dtype=torch.long) * 104729 + 17) % config.vocab_size).unsqueeze(0)
    mask = torch.ones_like(ids)
    record: dict[str, Any] = {"context": context, "dtype": "float32", "device": "cuda", "status": "PASS"}
    try:
        with torch.inference_mode():
            (output, cache), prefill_ms = cuda_ms(lambda: model.prefill(ids, mask))
            record["prefill_ms"] = prefill_ms
            record["prefill_tokens_per_sec"] = context / max(prefill_ms / 1000.0, 1e-12)

            memory, store_ms = cuda_ms(
                lambda: model.build_exact_memory(output["hidden"], ids, mask, source_length=context)
            )
            record["exact_store_ms"] = store_ms
            if model.exact_pointer is not None and memory is not None:
                state = LexicalPointerDecodeState(source_length=context, max_activation_steps=1, max_copy_tokens=1)
                query = model.exact_route_indices(output["hidden"])[:, -1]
                (_, diagnostics), retrieve_ms = cuda_ms(
                    lambda: state.prepare_logits(
                        model.exact_pointer,
                        output["logits"][:, -1].float(),
                        output["hidden"],
                        ids,
                        mask,
                        anchor_memory=memory,
                        query_anchor_ids=query,
                        max_candidate_chunks=config.exact_pointer_candidate_chunks,
                    )
                )
                record["exact_first_lookup_ms"] = retrieve_ms
                record["exact_candidate_count"] = diagnostics.get("candidate_count")
                record["exact_full_scan_used"] = diagnostics.get("used_full_scan_fallback")

            decode_times = []
            next_id = ids[:, -1:]
            for _ in range(decode_steps):
                (output, cache), elapsed = cuda_ms(lambda: model.decode_step(next_id, cache))
                decode_times.append(elapsed)
                next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            record["decode_steps"] = decode_steps
            record["decode_ms_per_token"] = sum(decode_times) / max(len(decode_times), 1)
            record["decode_tokens_per_sec"] = 1000.0 / max(record["decode_ms_per_token"], 1e-12)
            record["decode_ms_samples"] = decode_times
        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    except torch.cuda.OutOfMemoryError as error:
        record.update(
            {
                "status": "FAIL",
                "failure": "CUDA_OUT_OF_MEMORY",
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )
        torch.cuda.empty_cache()
    except Exception as error:
        record.update({"status": "FAIL", "failure": type(error).__name__, "error": str(error)})
        torch.cuda.empty_cache()
    return record


def warmup(model, config: ModelConfig) -> None:
    length = min(128, config.max_seq_len - 1)
    ids = ((torch.arange(length, device="cuda", dtype=torch.long) * 8191 + 3) % config.vocab_size).unsqueeze(0)
    with torch.inference_mode():
        output, cache = model.prefill(ids, torch.ones_like(ids))
        model.decode_step(output["logits"][:, -1].argmax(dim=-1, keepdim=True), cache)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile original unquantized FP32 FDT v4 on the current CUDA GPU")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for current-hardware profiling")
    if args.output_json.exists() or args.output_csv.exists():
        raise FileExistsError("profiler outputs are immutable and must use fresh paths")

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v4":
        raise ValueError("Profiler requires an FDT v4 checkpoint")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32).eval()
    torch.cuda.synchronize()
    warmup(model, config)

    environment = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_stage_status": payload.get("stage_status"),
        "official_operation_dtype": "float32",
        "quantization": "none",
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver": driver_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "os": platform.platform(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    rows = [profile_context(model, config, int(context), int(args.decode_steps)) for context in args.contexts]
    passed = [row for row in rows if row.get("status") == "PASS" and row.get("decode_ms_per_token") is not None]
    slope = None
    if len(passed) >= 2:
        first, last = passed[0], passed[-1]
        slope = (last["decode_ms_per_token"] - first["decode_ms_per_token"]) / max(last["context"] - first["context"], 1)
    report = {
        "environment": environment,
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "git_commit": args.git_commit or "UNKNOWN"},
        "contexts": rows,
        "decode_latency_slope_ms_per_token_per_context_token": slope,
        "audit_axes": {
            "PERFORMANCE": {
                "status": "PARTIAL" if rows and all(row.get("status") == "PASS" for row in rows) else "FAIL",
                "reason": "Current-GPU FP32 prefill/decode/VRAM are measured, but kernel-count, ATen breakdown, optimization A/B, and dense-control evidence are not supplied.",
            }
        },
        "limitations": [
            "The checkpoint may be an explicitly untrained warm-start audit checkpoint.",
            "This script profiles eager scatter_add_ code only; compile, CUDA Graph, and custom kernels are not inferred.",
            "Prefill and decode are measured separately with original FP32 model operations and no quantization.",
            "One 128-token prefill/decode warm-up runs before timed contexts.",
        ],
    }
    atomic_json(args.output_json.resolve(), report)
    atomic_csv(args.output_csv.resolve(), rows)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
