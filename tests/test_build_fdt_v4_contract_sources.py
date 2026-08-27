from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_source_builder_emits_exact_loop_and_long_context_contracts(tmp_path):
    builder = load_script("build_fdt_v4_contract_sources.py")
    trainer = load_script("train_fdt_v4_curriculum.py")
    base = tmp_path / "base"
    validation = tmp_path / "validation"
    output = tmp_path / "output"
    (base / "shards" / "train").mkdir(parents=True)
    (validation / "shards" / "validation").mkdir(parents=True)

    rows, length = 24, 512
    ids = torch.zeros(rows, length, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    categories = torch.tensor([0] * 12 + [4] * 12)
    for row in range(rows):
        active = 480
        values = (torch.arange(active) * (row + 3) + row * 17 + 5) % 24571 + 3
        ids[row, :active] = values
        mask[row, :active] = 1
        labels[row, 400:active] = values[400:active]
    torch.save(
        {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": mask,
            "category_ids": categories,
        },
        base / "shards" / "train" / "shard_000000.pt",
    )
    (base / "manifest.json").write_text(
        json.dumps({"post_build_exact_overlap": 0}), encoding="utf-8"
    )
    validation_ids = ((ids[:4] + 10007) % 24571) + 3
    validation_payload = {
        "input_ids": validation_ids,
        "labels": validation_ids.clone(),
        "attention_mask": mask[:4].clone(),
    }
    torch.save(
        validation_payload,
        validation / "shards" / "validation" / "shard_000000.pt",
    )
    (validation / "manifest.json").write_text("{}", encoding="utf-8")

    result = builder.build(
        argparse.Namespace(
            base_source=base,
            validation_source=validation,
            output_dir=output,
            validation_split="validation",
            seed=41,
            natural_category_id=4,
            factual_category_id=0,
            natural_rows=8,
            factual_rows=8,
            exact_rows=8,
            generated_prefix_rows=8,
            long_8k_rows=1,
            long_16k_rows=1,
        )
    )
    assert result["status"] == "PASS"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cross_source_exact_overlap"] == 0
    assert manifest["train_validation_exact_overlap"] == 0

    directories = {
        "natural": output / "natural",
        "factual": output / "factual",
        "exact_copy": output / "exact_copy",
        "generated_prefix": output / "generated_prefix",
        "long_context": output / "long_context",
        "validation": output / "validation",
    }
    report = trainer.preflight_dataset_contract(
        directories,
        {"split": "train", "validation_split": "validation"},
    )
    exact = report["sources"]["exact_copy"]["shards"][0]
    prefix = report["sources"]["generated_prefix"]["shards"][0]
    long_lengths = {
        row["sequence_length"]
        for row in report["sources"]["long_context"]["shards"]
    }
    assert exact["exact_label_contract"] is True
    assert prefix["generated_prefix_contract"] is True
    assert long_lengths == {8192, 16384}
    stored_natural = torch.load(
        output / "natural" / "shards" / "train" / "shard_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert stored_natural["input_ids"].dtype == torch.int32
    assert stored_natural["attention_mask"].dtype == torch.uint8
