from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.data.filters import normalize_web_text  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
from prepare_balanced_novel_pilot import SOURCE_REVISIONS, stream_hf_texts  # noqa: E402
from prepare_capability_completion_v15 import (  # noqa: E402
    active_rows,
    file_sha256,
    reference_hashes,
    require_c_path,
    token_hash,
)
from prepare_capability_completion_v20 import (  # noqa: E402
    add_json_grounded,
    add_python_grounded,
    category_token_report,
)
from prepare_fdt_v21_knowledge_scale import (  # noqa: E402
    CATEGORY_NAMES,
    MAX_NATURAL_REPEATED_4GRAM_RATE,
    NATURAL_ANSWER_TOKENS,
    V21Writer,
    add_natural_language,
    add_varied_retrieval,
    repeated_ngram_rate,
)


ROWS_PER_CATEGORY = 20_000
CATEGORY_COUNTS = {name: ROWS_PER_CATEGORY for name in CATEGORY_NAMES}


def add_wikipedia_knowledge(
    writer: V21Writer,
    rows: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    stream = stream_hf_texts("wikipedia", seed)
    accepted = Counter()
    for _ in range(600_000):
        if writer.categories["factual_qa"] >= rows:
            break
        text, _ = next(stream)
        text = normalize_web_text(text)
        ids = (
            list(writer.tokenizer.encode(text, add_special_tokens=False))
            if text and "\x00" not in text
            else []
        )
        if len(ids) < 560:
            writer.rejections["factual_knowledge_wikipedia_short"] += 1
            continue
        prompt_tokens = min(300, len(ids) - NATURAL_ANSWER_TOKENS)
        span = prompt_tokens + NATURAL_ANSWER_TOKENS
        offset = rng.randrange(0, len(ids) - span + 1)
        active = ids[offset : offset + span]
        completion = active[-NATURAL_ANSWER_TOKENS:]
        if repeated_ngram_rate(completion, 4) > MAX_NATURAL_REPEATED_4GRAM_RATE:
            writer.rejections["factual_knowledge_wikipedia_repetitive"] += 1
            continue
        if writer.add_ids(
            active[:-NATURAL_ANSWER_TOKENS],
            completion,
            "factual_qa",
        ):
            accepted["wikipedia_continuation"] += 1
    if writer.categories["factual_qa"] != rows:
        raise RuntimeError(
            f"could not fill Wikipedia knowledge rows: {dict(accepted)}"
        )
    return dict(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh natural-language and Wikipedia-knowledge FDT v4 base"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=ROOT / "prepared_data" / "capability_curriculum_5k",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k"
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--stage", default="fdt_v4_natural_knowledge_base_v1")
    parser.add_argument(
        "--experiment-nonce", default="fdt-v4-natural-knowledge-v1-20260826"
    )
    args = parser.parse_args()

    output = require_c_path(args.output_dir)
    reference_root = require_c_path(args.reference_root)
    validation_source = require_c_path(args.validation_source)
    tokenizer_path = require_c_path(args.tokenizer)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(tokenizer_path))
    references, reference_report = reference_hashes(
        reference_root, output, int(tokenizer.eos_token_id)
    )
    writer = V21Writer(output, tokenizer, args.max_length, references)

    factual = add_wikipedia_knowledge(
        writer, CATEGORY_COUNTS["factual_qa"], args.seed + 1
    )
    retrieval = add_varied_retrieval(
        writer, CATEGORY_COUNTS["retrieval"], args.seed + 2
    )
    python = add_python_grounded(
        writer, CATEGORY_COUNTS["python_code"], args.seed + 3
    )
    json_families = add_json_grounded(
        writer, CATEGORY_COUNTS["json"], args.seed + 4
    )
    natural = add_natural_language(
        writer, CATEGORY_COUNTS["natural_lm"], args.seed + 5
    )
    token_report = category_token_report(writer)
    train = writer.save()

    validation_dir = output / "shards" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    source_validation = (
        validation_source / "shards" / "validation" / "shard_000000.pt"
    )
    target_validation = validation_dir / "shard_000000.pt"
    shutil.copy2(source_validation, target_validation)
    validation_manifest = json.loads(
        (validation_source / "manifest.json").read_text(encoding="utf-8")
    )

    post_rows = list(active_rows(output))
    if len({token_hash(row) for row in post_rows}) != len(post_rows):
        raise RuntimeError("FDT v4 source base has a local exact-row collision")
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"FDT v4 source base exact overlap is nonzero: {overlap}")

    report = {
        "stage": args.stage,
        "format": "fdt_v4_natural_knowledge_source_v1",
        "purpose": "Fresh natural language and Wikipedia knowledge for FDT v4",
        "conversational_sft": False,
        "experiment_nonce": args.experiment_nonce,
        "nonce_inserted_into_model_text": False,
        "historical_payload_limitation": (
            "All remaining payloads were scanned; hash-journaled pruned rows cannot be rescanned."
        ),
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "category_names": CATEGORY_NAMES,
        "category_objective_intent": {
            "natural_lm": "primary",
            "factual_qa": "primary Wikipedia knowledge continuation",
            "retrieval": "substantial auxiliary",
            "python_code": "retention-only auxiliary",
            "json": "retention-only auxiliary",
        },
        "source_revisions": {
            "wikipedia_knowledge": SOURCE_REVISIONS["wikipedia"],
            **{name: SOURCE_REVISIONS[name] for name in natural},
        },
        "source_provenance": {
            "factual_qa": {
                **SOURCE_REVISIONS["wikipedia"],
                "construction": "clean Wikipedia context continuation; no QA template",
            },
            "natural_lm": {name: SOURCE_REVISIONS[name] for name in natural},
        },
        "train": train,
        "category_token_report": token_report,
        "factual_knowledge_sources": factual,
        "retrieval_families": retrieval,
        "python_families": python,
        "json_families": json_families,
        "natural_sources": natural,
        "validation": {
            **validation_manifest["validation"],
            "source": str(validation_source),
            "source_manifest_sha256": file_sha256(
                validation_source / "manifest.json"
            ),
            "shard_sha256": file_sha256(target_validation),
            "unchanged_from_fixed_strict_validation": True,
        },
        "reference_root": str(reference_root),
        "reference_datasets_scanned": len(reference_report),
        "reference_audit": reference_report,
        "post_build_exact_overlap": overlap,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_sha256": file_sha256(manifest),
                "train_shard_sha256": train["shard_sha256"],
                "validation_shard_sha256": file_sha256(target_validation),
                "rows": train["rows"],
                "active_tokens": train["active_tokens"],
                "categories": train["categories"],
                "post_build_exact_overlap": overlap,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
