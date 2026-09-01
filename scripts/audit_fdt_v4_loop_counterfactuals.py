from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fdt_v4 as evaluator
import prepare_fdt_v4_1_generated_prefix_recovery as builder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_raw_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


@torch.inference_mode()
def audit_checkpoint(
    checkpoint: Path,
    payload: dict[str, torch.Tensor],
    raw_rows: list[dict],
    *,
    device: str,
    batch_size: int,
) -> dict:
    model, config, metadata = evaluator.load_checkpoint(checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()
    rows = []
    for start in range(0, len(raw_rows), batch_size):
        stop = min(start + batch_size, len(raw_rows))
        boundaries = payload["recovery_boundary"][start:stop].long()
        active_end = int(boundaries.max().item())
        input_ids = payload["input_ids"][start:stop, :active_end].long().to(device)
        attention_mask = payload["attention_mask"][start:stop, :active_end].long().to(device)
        output = model(input_ids, attention_mask=attention_mask)
        logits = output["logits"].float()
        for local_index, boundary_value in enumerate(boundaries.tolist()):
            row_index = start + local_index
            boundary = int(boundary_value)
            negative = int(payload["loop_negative_ids"][row_index, boundary].item())
            prediction = logits[local_index, boundary - 1]
            negative_logit = prediction[negative]
            top1 = int(prediction.argmax().item())
            probability = float(torch.softmax(prediction, dim=-1)[negative].item())
            rank = int((prediction > negative_logit).sum().item()) + 1
            prompt_length = int(raw_rows[row_index]["prompt_length"])
            generated = [
                int(value)
                for value in payload["input_ids"][
                    row_index, prompt_length:boundary
                ].tolist()
            ]
            top1_closes, top1_occurrences = builder.closes_degenerate_ngram(
                generated,
                top1,
                order=3,
                prior_occurrences=2,
                window=96,
            )
            rows.append(
                {
                    "row_index": row_index,
                    "negative_token": negative,
                    "negative_probability": probability,
                    "negative_rank": rank,
                    "top1_token": top1,
                    "top1_is_negative": top1 == negative,
                    "top1_closes_loop": bool(top1_closes),
                    "top1_prior_occurrences": int(top1_occurrences),
                    "top1_is_eos": top1 == int(config.eos_token_id),
                }
            )
        del output, logits, input_ids, attention_mask
    count = len(rows)
    report = {
        "checkpoint": metadata,
        "summary": {
            "count": count,
            "negative_top1_count": sum(row["top1_is_negative"] for row in rows),
            "top1_loop_closure_count": sum(row["top1_closes_loop"] for row in rows),
            "top1_eos_count": sum(row["top1_is_eos"] for row in rows),
            "mean_negative_probability": sum(
                row["negative_probability"] for row in rows
            )
            / count,
            "mean_negative_rank": sum(row["negative_rank"] for row in rows) / count,
        },
        "rows": rows,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether loop unlikelihood changes the actual failure state"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    shard = dataset / "shards" / "train" / "shard_00000.pt"
    raw_path = dataset / "raw_failure_rows.jsonl"
    payload = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
    raw_rows = load_raw_rows(raw_path, args.limit)
    if not raw_rows:
        raise ValueError("counterfactual audit dataset is empty")
    report = {
        "schema": "fdt_v4_loop_counterfactual_audit_v1",
        "status": "PASS",
        "quantization": "none",
        "dtype": "float32",
        "device": args.device,
        "dataset": {
            "path": str(dataset),
            "shard_sha256": sha256_file(shard),
            "raw_rows_sha256": sha256_file(raw_path),
            "rows": len(raw_rows),
        },
        "baseline": audit_checkpoint(
            args.baseline.resolve(),
            payload,
            raw_rows,
            device=args.device,
            batch_size=args.batch_size,
        ),
        "candidate": audit_checkpoint(
            args.candidate.resolve(),
            payload,
            raw_rows,
            device=args.device,
            batch_size=args.batch_size,
        ),
    }
    baseline_rows = report["baseline"]["rows"]
    candidate_rows = report["candidate"]["rows"]
    report["comparison"] = {
        "negative_probability_delta": report["candidate"]["summary"][
            "mean_negative_probability"
        ]
        - report["baseline"]["summary"]["mean_negative_probability"],
        "negative_top1_delta": report["candidate"]["summary"][
            "negative_top1_count"
        ]
        - report["baseline"]["summary"]["negative_top1_count"],
        "top1_loop_closure_delta": report["candidate"]["summary"][
            "top1_loop_closure_count"
        ]
        - report["baseline"]["summary"]["top1_loop_closure_count"],
        "top1_changed_count": sum(
            before["top1_token"] != after["top1_token"]
            for before, after in zip(baseline_rows, candidate_rows, strict=True)
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "baseline": report["baseline"]["summary"],
                "candidate": report["candidate"]["summary"],
                "comparison": report["comparison"],
            }
        )
    )


if __name__ == "__main__":
    main()
