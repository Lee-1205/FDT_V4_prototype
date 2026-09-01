from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fdt_v4 as evaluator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only unquantized FP32 penalty-off repetition audit"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ngram-loop-penalty", type=float, default=0.0)
    parser.add_argument("--exact-mode", choices=evaluator.EXACT_MODES, default="off")
    args = parser.parse_args()

    started = time.time()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    tokenizer, tokenizer_info = evaluator.tokenizer_metadata(args.tokenizer)
    rows = evaluator.load_dataset(dataset, tokenizer, args.limit)
    evaluator.require_nonempty_rows(rows, dataset, "repetition")
    model, config, metadata = evaluator.load_checkpoint(checkpoint)
    model = model.to(device=args.device, dtype=torch.float32).eval()

    records = []
    for row_index, row in enumerate(rows):
        prompt = row.get("prompt_ids") or row.get("input_ids")
        if not isinstance(prompt, list) or not prompt:
            continue
        prompt_ids = [int(value) for value in prompt[: config.max_seq_len - 1]]
        max_new_tokens = min(int(row.get("max_new_tokens", 128)), 128)
        generated, _ = evaluator.generate(
            model,
            config,
            prompt_ids,
            max_new_tokens,
            args.exact_mode,
            ngram_loop_penalty=float(args.ngram_loop_penalty),
        )
        loop = evaluator.loop_metrics(generated)
        records.append(
            {
                "row_index": row_index,
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "completion": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_ids": generated,
                **loop,
            }
        )

    rates = [float(row["trigram_repetition_rate"]) for row in records]
    loop_free = sum(bool(row["loop_free"]) for row in records)
    report = {
        "schema": "fdt_v4_repetition_development_v1",
        "status": "PASS" if records else "FAIL",
        "official_evaluation": False,
        "quantization": "none",
        "dtype": "float32",
        "device": args.device,
        "generation": {
            "decoder": "greedy",
            "repetition_penalty": 1.0,
            "penalty_enabled": False,
            "ngram_loop_penalty": float(args.ngram_loop_penalty),
            "ngram_loop_control_enabled": float(args.ngram_loop_penalty) > 0.0,
            "exact_mode": args.exact_mode,
            "transition_decode_contract": "full_recompute_for_intermediate_output_blend",
        },
        "checkpoint": metadata,
        "tokenizer": tokenizer_info,
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "rows": len(records),
        },
        "summary": {
            "count": len(records),
            "loop_free_count": loop_free,
            "loop_free_rate": loop_free / len(records) if records else None,
            "mean_repetition_rate": sum(rates) / len(rates) if rates else None,
        },
        "rows": records,
        "elapsed_seconds": time.time() - started,
    }
    evaluator.atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output.resolve()),
                "summary": report["summary"],
            }
        )
    )


if __name__ == "__main__":
    main()
