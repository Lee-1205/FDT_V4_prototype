from pathlib import Path

import importlib.util
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "build_fdt_v4_context_bridge.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_context_bridge", SOURCE)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)


def test_pack_after_offset_preserves_order_and_disjoint_lengths(tmp_path):
    source = tmp_path / "natural" / "shards" / "train"
    source.mkdir(parents=True)
    ids = torch.arange(48, dtype=torch.int32).reshape(4, 12)
    torch.save(
        {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids, dtype=torch.uint8)},
        source / "shard.pt",
    )

    packed = bridge.pack_after_offset(
        [source / "shard.pt"], skip_tokens=8, rows_by_length=[(4, 3), (8, 2)]
    )

    assert [length for length, _ in packed] == [4, 8]
    assert packed[0][1]["input_ids"].flatten().tolist() == list(range(8, 20))
    assert packed[1][1]["input_ids"].flatten().tolist() == list(range(20, 36))
    assert packed[0][1]["labels"].data_ptr() != packed[0][1]["input_ids"].data_ptr()


def test_bridge_runtime_requires_all_four_context_levels():
    trainer = (ROOT / "scripts" / "train_fdt_v4_curriculum_bridge.py").read_text(encoding="utf-8")
    config = (ROOT / "configs" / "fdt_v4_main_426m_speed_r3_bridge.yaml").read_text(encoding="utf-8")

    assert "context bridge requires an exact 2K" in trainer
    assert "context bridge requires an exact 4K" in trainer
    assert "bridge_context: 0.02" in config
    assert "activation_checkpointing_min_sequence_length: 2048" in config


def test_available_payload_audit_hashes_only_matching_lengths(tmp_path):
    audit_root = tmp_path / "prepared"
    audit_root.mkdir()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.int32)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.uint8)
    torch.save({"input_ids": ids, "attention_mask": mask}, audit_root / "rows.pt")
    digest = bridge.row_hash(ids[0], mask[0])

    report = bridge.audit_available_payloads(
        audit_root, tmp_path / "fresh_output", {4: {digest}, 8: set()}
    )

    assert report["same_length_rows_hashed"] == 1
    assert report["post_build_exact_overlap"] == 1
