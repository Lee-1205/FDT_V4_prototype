from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig, load_yaml_like  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402

MAIN_ARCHITECTURE = {
    "model_type": "fdt_v4",
    "dim": 1216,
    "n_layers": 20,
    "n_heads": 19,
    "max_seq_len": 16384,
    "tie_embeddings": True,
    "use_rope": True,
    "local_attention_window": 64,
    "num_anchors": 256,
    "top_k": 8,
    "router_dim": 256,
    "anchor_layer_indices": tuple(range(0, 20, 2)),
    "aggregation_impl": "sparse_chunked_scan",
    "anchor_scan_chunk_size": 64,
    "exact_memory_enabled": True,
    "exact_memory_mode": "copy",
    "exact_memory_copy_cursor": True,
    "exact_memory_commit_threshold": 0.5,
    "exact_memory_full_scan_fallback": True,
    "exact_memory_candidate_cap": 16,
    "exact_pointer_chunk_size": 32,
    "exact_pointer_chunk_anchors": 4,
    "exact_pointer_candidate_chunks": 4,
    "generation_repetition_scope": "generated",
    "generation_ngram_order": 3,
    "generation_ngram_penalty": 8.0,
    "generation_ngram_window": 96,
    "generation_ngram_hard_block_after": 2,
}
EXPECTED_PARAMETER_MIN = 420_000_000
EXPECTED_PARAMETER_MAX = 430_000_000
V20_PARENT_MODEL_TYPE = "fdt_v3"
WARM_START_ALLOWLIST = ("token_embedding.weight", "lm_head.weight", "blocks.", "norm.weight")
WARM_START_DENYLIST = (
    "position_embedding", "rope", "exact_pointer", "exact_memory",
)


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def owned_run_path(path: Path) -> Path:
    resolved = require_c_path(path, "output directory")
    runs_root = (ROOT / "runs").resolve()
    if resolved == runs_root or runs_root not in resolved.parents or resolved.is_symlink():
        raise ValueError(f"output directory must be a non-symlink child of {runs_root}")
    return resolved


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_c_path(path, "JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.flush()
        os.fsync(handle.fileno())
    durable_replace(temporary, path)


def durable_replace(source: Path, destination: Path) -> None:
    """Atomically replace and request write-through for crash durability."""
    if os.name == "nt":
        move_file_ex = ctypes.windll.kernel32.MoveFileExW
        move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file_ex.restype = ctypes.c_int
        movefile_replace_existing = 0x1
        movefile_write_through = 0x8
        if not move_file_ex(
            str(source),
            str(destination),
            movefile_replace_existing | movefile_write_through,
        ):
            raise ctypes.WinError()
    else:
        os.replace(source, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path = require_c_path(path, "checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    durable_replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def disk_free_gib() -> float:
    return shutil.disk_usage("C:\\").free / (1024**3)


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms), warn_only=bool(deterministic_algorithms)
    )


def validate_main_architecture(config: ModelConfig) -> None:
    for key, expected in MAIN_ARCHITECTURE.items():
        actual = tuple(getattr(config, key)) if key == "anchor_layer_indices" else getattr(config, key)
        if actual != expected:
            raise ValueError(f"FDT v4 main architecture mismatch for {key}: {actual!r} != {expected!r}")
    if (config.vocab_size, config.pad_token_id, config.eos_token_id) != (24576, 0, 2):
        raise ValueError("FDT v4 main requires vocab_size=24576, pad_token_id=0, eos_token_id=2")
    if config.exact_memory_candidate_cap < 1:
        raise ValueError("FDT v4 main requires a positive exact-memory candidate cap")


def model_config_from_yaml(raw: dict[str, Any]) -> ModelConfig:
    values = dict(raw.get("model", {}))
    values.update(raw.get("fdt", {}))
    values["model_type"] = "fdt_v4"
    fields = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: value for key, value in values.items() if key in fields})


def metadata_model_preflight(config: ModelConfig) -> dict[str, Any]:
    """Check the one allowed model on meta before allocating model weights."""
    validate_main_architecture(config)
    with torch.device("meta"):
        model = build_model(config)
    count = sum(parameter.numel() for parameter in model.parameters())
    if not EXPECTED_PARAMETER_MIN <= count <= EXPECTED_PARAMETER_MAX:
        raise ValueError(f"FDT v4 parameter count is outside 426M class: {count}")
    return {
        "model_type": config.model_type,
        "parameter_count": int(count),
        "parameter_class": "426M",
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "meta_device": True,
    }


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(require_c_path(path, "checkpoint or shard"), map_location="cpu", weights_only=False)


def manifest_hash(dataset_dir: Path) -> str:
    manifest = require_c_path(dataset_dir, "dataset") / "manifest.json"
    return sha256_file(manifest) if manifest.exists() else "MISSING"


def shard_paths(dataset_dir: Path, split: str) -> list[Path]:
    root = require_c_path(dataset_dir, "dataset")
    paths = sorted((root / "shards" / split).glob("*.pt"))
    if not paths:
        paths = sorted((root / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards under {root}")
    return paths


def _validate_tensor_field(payload: dict[str, Any], name: str, shape: tuple[int, ...], path: Path) -> None:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"Shard {path} lacks tensor field {name} with shape {shape}")


def validate_shard_contract(
    payload: dict[str, Any],
    path: Path,
    exact: bool,
    generated_prefix: bool = False,
    real_loop_contract: bool = False,
) -> dict[str, Any]:
    ids = payload.get("input_ids")
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
        raise ValueError(f"Shard {path} requires input_ids [rows, sequence]")
    shape = tuple(ids.shape)
    _validate_tensor_field(payload, "attention_mask", shape, path)
    _validate_tensor_field(payload, "labels", shape, path)
    labels = payload["labels"]
    if not bool((labels.eq(-100) | labels.ge(0)).all()):
        raise ValueError(f"labels must contain token ids or -100 only: {path}")
    if exact:
        for name in ("prompt_mask", "copy_source_positions", "copy_target_mask"):
            _validate_tensor_field(payload, name, shape, path)
        boundary = payload.get("source_boundary")
        if not isinstance(boundary, torch.Tensor) or tuple(boundary.shape) not in {(ids.size(0),), shape}:
            raise ValueError(f"Shard {path} lacks source_boundary with shape [rows] or [rows, sequence]")
        if not bool(payload["prompt_mask"].bool().any()) or not bool(payload["copy_target_mask"].bool().any()):
            raise ValueError(f"exact shard must contain prompt and copy-target rows: {path}")
        if bool(labels.masked_select(payload["prompt_mask"].bool()).ne(-100).any()):
            raise ValueError(f"exact labels must be -100 throughout prompt_mask: {path}")
        if bool(labels.masked_select(payload["copy_target_mask"].bool()).eq(-100).any()):
            raise ValueError(f"copy_target_mask must select explicit labels: {path}")
        prompt_mask = payload["prompt_mask"].bool()
        target_mask = payload["copy_target_mask"].bool()
        if bool((target_mask & ~payload["attention_mask"].bool()).any()):
            raise ValueError(f"exact copy targets must be active tokens: {path}")
        source_positions = payload["copy_source_positions"].long()
        boundaries = (
            boundary.long().view(-1, 1).expand_as(ids)
            if boundary.ndim == 1
            else boundary.long()
        )
        rows, targets = target_mask.nonzero(as_tuple=True)
        mapped = source_positions[rows, targets]
        if bool(
            (
                mapped.le(0)
                | mapped.ge(boundaries[rows, targets])
                | mapped.ge(targets)
                | targets.lt(boundaries[rows, targets])
                | boundaries[rows, targets].gt(ids.size(1))
            ).any()
        ):
            raise ValueError(f"exact source mapping is outside the prompt boundary: {path}")
        if bool((~prompt_mask[rows, mapped]).any()):
            raise ValueError(f"exact source mapping must select prompt_mask tokens: {path}")
        if bool(ids[rows, mapped].ne(labels[rows, targets]).any()):
            raise ValueError(f"exact mapped source tokens must equal target labels: {path}")
    if generated_prefix:
        for name in ("loop_negative_ids", "loop_negative_mask"):
            _validate_tensor_field(payload, name, shape, path)
        negative_ids = payload["loop_negative_ids"]
        negative_mask = payload["loop_negative_mask"].bool()
        unlikelihood_only = payload.get("loop_unlikelihood_only")
        if unlikelihood_only is not None:
            if not isinstance(unlikelihood_only, torch.Tensor) or tuple(
                unlikelihood_only.shape
            ) != (ids.size(0),):
                raise ValueError(
                    f"loop_unlikelihood_only must have shape [rows]: {path}"
                )
            if not bool(unlikelihood_only.bool().all()):
                raise ValueError(
                    f"mixed generated-prefix objective contracts are forbidden: {path}"
                )
        if not bool(negative_mask.any()):
            raise ValueError(f"generated-prefix shard has no loop-negative positions: {path}")
        if not bool((negative_ids.masked_select(negative_mask) >= 0).all()):
            raise ValueError(f"generated-prefix loop-negative token ids must be nonnegative: {path}")
        if unlikelihood_only is None and bool(
            labels.masked_select(negative_mask).eq(-100).any()
        ):
            raise ValueError(f"generated-prefix negatives require supervised recovery labels: {path}")
        if unlikelihood_only is None and bool(
            negative_ids.masked_select(negative_mask).eq(
                labels.masked_select(negative_mask)
            ).any()
        ):
            raise ValueError(f"loop-negative ids must differ from clean labels: {path}")
        if unlikelihood_only is not None and bool(
            negative_ids.masked_select(negative_mask).ne(
                ids.masked_select(negative_mask)
            ).any()
        ):
            raise ValueError(
                f"trajectory loop negatives must equal generated input tokens: {path}"
            )
        if real_loop_contract:
            _validate_tensor_field(
                payload,
                "loop_negative_prior_occurrences",
                shape,
                path,
            )
            per_row_negatives = negative_mask.sum(dim=1)
            if unlikelihood_only is None and bool(per_row_negatives.ne(1).any()):
                raise ValueError(
                    f"real generated-prefix rows require one loop closure each: {path}"
                )
            if unlikelihood_only is not None and bool(per_row_negatives.lt(1).any()):
                raise ValueError(
                    f"trajectory unlikelihood rows require loop closures: {path}"
                )
            occurrences = payload["loop_negative_prior_occurrences"]
            if bool(occurrences.masked_select(negative_mask).lt(2).any()):
                raise ValueError(
                    f"loop negatives must close at least the third n-gram occurrence: {path}"
                )
        if "loop_candidate_ids" in payload:
            candidates = payload["loop_candidate_ids"]
            candidate_mask = payload.get("loop_candidate_mask")
            escape_ids = payload.get("loop_escape_ids")
            positions = payload.get("loop_contrast_position")
            if (
                not isinstance(candidates, torch.Tensor)
                or candidates.ndim != 2
                or candidates.size(0) != ids.size(0)
                or not isinstance(candidate_mask, torch.Tensor)
                or tuple(candidate_mask.shape) != tuple(candidates.shape)
                or not isinstance(escape_ids, torch.Tensor)
                or tuple(escape_ids.shape) != (ids.size(0),)
                or not isinstance(positions, torch.Tensor)
                or tuple(positions.shape) != (ids.size(0),)
            ):
                raise ValueError(f"loop contrast fields have invalid shapes: {path}")
            if not bool(candidate_mask.bool().any(dim=1).all()):
                raise ValueError(f"loop contrast rows require candidates: {path}")
            if bool((escape_ids < 0).any()) or bool((positions <= 0).any()):
                raise ValueError(f"loop contrast escape/position values are invalid: {path}")
            if bool((positions >= ids.size(1)).any()):
                raise ValueError(f"loop contrast positions exceed the sequence: {path}")
            if bool(
                candidates.eq(escape_ids.unsqueeze(1)).logical_and(
                    candidate_mask.bool()
                ).any()
            ):
                raise ValueError(f"loop escape token conflicts with a candidate: {path}")
    return {
        "rows": int(ids.size(0)),
        "sequence_length": int(ids.size(1)),
        "fields": sorted(payload.keys()),
        "exact_label_contract": exact,
        "generated_prefix_contract": generated_prefix,
    }


def preflight_dataset_contract(dataset_dirs: dict[str, Path], data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate shard fields and hashes without concatenating shards into RAM."""
    report: dict[str, Any] = {"contract": "fdt_v4_main_426m_v2", "sources": {}}
    for name, path in dataset_dirs.items():
        split = data_cfg.get(f"{name}_split", data_cfg.get("split", "train"))
        if not path.is_dir():
            raise FileNotFoundError(f"{name} dataset directory is missing: {path}")
        manifest = path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{name} dataset manifest is missing: {manifest}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        files = shard_paths(path, split)
        exact = name == "exact_copy"
        generated_prefix = name == "generated_prefix"
        schema_version = manifest_payload.get("schema_version")
        real_loop_contract = bool(
            generated_prefix
            and schema_version
            in {
                "fdt_v4_1_model_failure_recovery_v1",
                "fdt_v4_1_model_failure_unlikelihood_v1",
            }
        )
        expected_construction = {
            "fdt_v4_1_model_failure_recovery_v1":
                "frozen_model_penalty_off_real_trigram_loop_closure",
            "fdt_v4_1_model_failure_unlikelihood_v1":
                "frozen_model_penalty_off_full_trajectory_trigram_unlikelihood",
        }.get(schema_version)
        if real_loop_contract and (
            manifest_payload.get("construction") != expected_construction
            or manifest_payload.get("penalty_scope")
            != "natural_and_factual_only_excludes_exact_copy_code_json"
        ):
            raise ValueError(
                f"generated-prefix v4.1 manifest has an unsafe penalty scope: {manifest}"
            )
        reports = []
        for item in files:
            payload = load_payload(item)
            reports.append({
                "path": str(item),
                "sha256": sha256_file(item),
                **validate_shard_contract(
                    payload,
                    item,
                    exact,
                    generated_prefix=generated_prefix,
                    real_loop_contract=real_loop_contract,
                ),
            })
        report["sources"][name] = {"directory": str(path), "split": split, "manifest_sha256": sha256_file(manifest), "shards": reports, "shard_count": len(files), "label_contract": "explicit_labels_with_minus_100_prompt" if exact else "explicit_labels"}
    return report


def deterministic_source_cycle(source_fractions: dict[str, float], slots: int = 100) -> list[str]:
    if not source_fractions or any(float(value) < 0.0 for value in source_fractions.values()):
        raise ValueError("source fractions must be nonnegative and nonempty")
    total = sum(float(value) for value in source_fractions.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"source fractions must sum to 1.0, got {total}")
    slots = max(int(slots), len(source_fractions))
    exact = {name: float(value) * slots for name, value in source_fractions.items()}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = slots - sum(counts.values())
    order = sorted(
        source_fractions,
        key=lambda name: (exact[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    cycle: list[str] = []
    for name in sorted(counts):
        cycle.extend([name] * counts[name])
    random.Random(0).shuffle(cycle)
    return cycle


class TokenBudgetSourcePlanner:
    """Deterministic weighted fair queue measured in active tokens."""

    def __init__(self, source_fractions: dict[str, float]):
        if not source_fractions or any(float(value) <= 0.0 for value in source_fractions.values()):
            raise ValueError("token source fractions must be positive and nonempty")
        total = sum(float(value) for value in source_fractions.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"token source fractions must sum to 1.0, got {total}")
        self.fractions = {str(name): float(value) for name, value in source_fractions.items()}
        self.virtual_finish = {name: 0.0 for name in self.fractions}
        self.active_tokens = {name: 0 for name in self.fractions}
        self.samples = {name: 0 for name in self.fractions}

    def choose_source(self) -> str:
        return min(self.fractions, key=lambda name: (self.virtual_finish[name], name))

    def record(self, source_name: str, active_tokens: int) -> None:
        tokens = int(active_tokens)
        if source_name not in self.fractions or tokens <= 0:
            raise ValueError("token planner received an invalid source observation")
        self.active_tokens[source_name] += tokens
        self.samples[source_name] += 1
        self.virtual_finish[source_name] += tokens / self.fractions[source_name]

    def state(self) -> dict[str, Any]:
        return {
            "schema_version": "fdt_token_budget_planner_v1",
            "fractions": dict(self.fractions),
            "virtual_finish": dict(self.virtual_finish),
            "active_tokens": dict(self.active_tokens),
            "samples": dict(self.samples),
        }

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "fdt_token_budget_planner_v1":
            raise ValueError("token planner resume schema mismatch")
        restored_fractions = {str(name): float(value) for name, value in state["fractions"].items()}
        if restored_fractions != self.fractions:
            raise ValueError("token planner source fractions changed during resume")
        for field, target, cast in (
            ("virtual_finish", self.virtual_finish, float),
            ("active_tokens", self.active_tokens, int),
            ("samples", self.samples, int),
        ):
            values = state.get(field, {})
            if set(values) != set(target):
                raise ValueError(f"token planner {field} keys changed during resume")
            target.update({name: cast(values[name]) for name in target})


class RowPool:
    """Deterministic shard streaming: only one .pt shard is resident at a time."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        seed: int,
        exact: bool = False,
        generated_prefix: bool = False,
    ):
        self.dataset_dir = require_c_path(dataset_dir, "dataset")
        self.split = split
        self.seed = int(seed)
        self.exact = bool(exact)
        self.generated_prefix = bool(generated_prefix)
        self.paths = shard_paths(self.dataset_dir, split)
        manifest_payload = json.loads(
            (self.dataset_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.total_rows = sum(
            int(item.get("rows", 0)) for item in manifest_payload.get("shards", [])
        )
        if self.total_rows <= 0:
            raise ValueError(f"dataset manifest has no rows: {self.dataset_dir}")
        self.epoch = 0
        self.shard_index = 0
        self.cursor = 0
        self.rows_consumed = 0
        self.payload: dict[str, torch.Tensor] = {}
        self.order: torch.Tensor | None = None
        self._load_shard()

    def _load_shard(self) -> None:
        path = self.paths[self.shard_index]
        payload = load_payload(path)
        validate_shard_contract(
            payload,
            path,
            self.exact,
            generated_prefix=self.generated_prefix,
        )
        self.payload = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        rows = int(self.payload["input_ids"].size(0))
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch * 1_000_003 + self.shard_index * 97)
        self.order = torch.randperm(rows, generator=generator)

    def state(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "shard_index": self.shard_index,
            "cursor": self.cursor,
            "seed": self.seed,
            "rows_consumed": self.rows_consumed,
            "unique_rows_seen": min(self.rows_consumed, self.total_rows),
            "total_rows": self.total_rows,
        }

    def restore(self, state: dict[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("Dataset seed mismatch during resume")
        self.epoch = int(state.get("epoch", 0))
        self.shard_index = int(state.get("shard_index", 0))
        self.cursor = int(state.get("cursor", 0))
        self.rows_consumed = int(state.get("rows_consumed", 0))
        if int(state.get("total_rows", self.total_rows)) != self.total_rows:
            raise ValueError("Dataset row count changed during resume")
        if not 0 <= self.shard_index < len(self.paths):
            raise ValueError("Invalid dataset shard cursor")
        self._load_shard()
        if not 0 <= self.cursor <= int(self.payload["input_ids"].size(0)):
            raise ValueError("Invalid dataset row cursor")

    def _advance(self) -> None:
        self.shard_index += 1
        self.cursor = 0
        if self.shard_index >= len(self.paths):
            self.shard_index = 0
            self.epoch += 1
        self._load_shard()

    def next_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        rows: list[dict[str, torch.Tensor]] = []
        for _ in range(int(batch_size)):
            if self.order is None or self.cursor >= self.order.numel():
                self._advance()
            index = int(self.order[self.cursor])
            rows.append({key: value[index] for key, value in self.payload.items()})
            self.cursor += 1
            self.rows_consumed += 1
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def weighted_lm_loss(logits, labels, attention_mask, pad_token_id, eos_token_id, eos_weight):
    shifted_logits = logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = attention_mask[:, 1:].bool() & shifted_labels.ne(-100) & shifted_labels.ne(int(pad_token_id))
    safe_labels = shifted_labels.masked_fill(shifted_labels.eq(-100), int(pad_token_id))
    weights = torch.where(safe_labels.eq(int(eos_token_id)), float(eos_weight), 1.0).to(shifted_logits.dtype)
    losses = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), safe_labels.view(-1), reduction="none").view_as(safe_labels)
    valid_float = valid.to(losses.dtype) * weights
    return (losses * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def chunked_weighted_lm_loss(
    model,
    hidden,
    labels,
    attention_mask,
    pad_token_id,
    eos_token_id,
    eos_weight,
    sequence_chunk_size,
    checkpoint_chunks=True,
):
    """Compute LM loss without retaining a full 16K-by-vocabulary tensor."""
    chunk_size = max(int(sequence_chunk_size), 1)
    numerator = hidden.new_zeros((hidden.size(0),), dtype=torch.float32)
    denominator = hidden.new_zeros((hidden.size(0),), dtype=torch.float32)
    for start in range(0, max(hidden.size(1) - 1, 0), chunk_size):
        stop = min(start + chunk_size, hidden.size(1) - 1)
        target = labels[:, start + 1 : stop + 1]
        target_attention = attention_mask[:, start + 1 : stop + 1]

        def chunk_terms(hidden_chunk, target_chunk, attention_chunk):
            logits = (
                model._language_logits(hidden_chunk)
                if hasattr(model, "_language_logits")
                else model.lm_head(hidden_chunk).clamp(
                    -model.config.lm_logit_clip,
                    model.config.lm_logit_clip,
                )
            ).float()
            valid = (
                attention_chunk.bool()
                & target_chunk.ne(-100)
                & target_chunk.ne(int(pad_token_id))
            )
            safe = target_chunk.masked_fill(target_chunk.eq(-100), int(pad_token_id))
            weights = torch.where(
                safe.eq(int(eos_token_id)), float(eos_weight), 1.0
            ).to(logits.dtype)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                safe.reshape(-1),
                reduction="none",
            ).reshape_as(safe)
            weighted_valid = valid.to(losses.dtype) * weights
            return (losses * weighted_valid).sum(dim=1), weighted_valid.sum(dim=1)

        if checkpoint_chunks:
            chunk_numerator, chunk_denominator = checkpoint(
                chunk_terms,
                hidden[:, start:stop],
                target,
                target_attention,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_numerator, chunk_denominator = chunk_terms(
                hidden[:, start:stop],
                target,
                target_attention,
            )
        numerator = numerator + chunk_numerator
        denominator = denominator + chunk_denominator
    return (numerator / denominator.clamp_min(1.0)).sum()


def set_sequence_gradient_checkpointing(model, sequence_length: int, train_cfg: dict[str, Any]) -> bool:
    enabled = bool(train_cfg.get("activation_checkpointing", True)) and int(
        sequence_length
    ) >= int(train_cfg.get("activation_checkpointing_min_sequence_length", 1))
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(enabled)
    return enabled


def lm_loss_checkpointing_enabled(sequence_length: int, train_cfg: dict[str, Any]) -> bool:
    return bool(train_cfg.get("activation_checkpointing", True)) and int(
        sequence_length
    ) >= int(train_cfg.get("lm_loss_checkpointing_min_sequence_length", 1))


def gradient_norms_are_finite(*values: torch.Tensor | float) -> bool:
    """Check scalar norms independently so empty CPU groups can coexist with CUDA groups."""
    return all(
        math.isfinite(
            float(value.detach().cpu())
            if isinstance(value, torch.Tensor)
            else float(value)
        )
        for value in values
    )


def merge_row_batches(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot merge an empty row batch")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("short-row batch fields changed")
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in keys}


def planned_base_forward_groups(
    pools: dict[str, "RowPool"],
    source_cycle: list[str],
    optimizer_step: int,
    effective_samples: int,
    short_batch_size: int,
    short_sequence_max_length: int,
    token_planner: TokenBudgetSourcePlanner | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Group only short rows while preserving the original eight-row source order."""
    groups: list[dict[str, torch.Tensor]] = []
    pending_short: list[dict[str, torch.Tensor]] = []

    def flush_short() -> None:
        nonlocal pending_short
        if pending_short:
            groups.append(merge_row_batches(pending_short))
            pending_short = []

    for micro_index in range(int(effective_samples)):
        source_name = (
            token_planner.choose_source()
            if token_planner is not None
            else source_cycle[
                (int(optimizer_step) * int(effective_samples) + micro_index)
                % len(source_cycle)
            ]
        )
        row = pools[source_name].next_batch(1)
        if token_planner is not None:
            token_planner.record(source_name, int(row["attention_mask"].sum()))
        sequence_length = int(row["input_ids"].size(1))
        if sequence_length <= int(short_sequence_max_length):
            pending_short.append(row)
            if len(pending_short) >= int(short_batch_size):
                flush_short()
        else:
            flush_short()
            groups.append(row)
    flush_short()
    return groups


def move_batch(payload: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.long) for key, value in payload.items()}


def exact_curriculum_cap(progress: float, caps: tuple[int, ...] = (4, 8, 16, 32, 64)) -> int:
    """Advance exact-copy difficulty without changing source order or model structure."""
    boundaries = (0.0, 0.10, 0.25, 0.50, 0.75)
    bounded = min(max(float(progress), 0.0), 1.0)
    selected = caps[0]
    for boundary, cap in zip(boundaries, caps):
        if bounded >= boundary:
            selected = int(cap)
    return selected


def cap_copy_targets(batch: dict[str, torch.Tensor], target_cap: int) -> dict[str, torch.Tensor]:
    """Keep the first N explicit copy targets per row; all other fields stay unchanged."""
    mask = batch["copy_target_mask"].bool()
    ranks = mask.long().cumsum(dim=1)
    batch["copy_target_mask"] = mask & ranks.le(max(int(target_cap), 1))
    return batch


def linear_ramp(progress: float, start: float, end: float) -> float:
    if end <= start:
        return float(progress >= end)
    return min(max((float(progress) - start) / (end - start), 0.0), 1.0)


def apply_architecture_transition(
    model,
    train_cfg: dict[str, Any],
    additional_tokens: int,
) -> dict[str, float]:
    transition_tokens = int(train_cfg.get("architecture_transition_tokens", 0))
    if transition_tokens <= 0:
        return {
            "transition_progress": 1.0,
            "rope_transition_alpha": float(model.config.rope_transition_alpha),
            "anchor_recency_bias": float(model.config.anchor_recency_bias),
            "routing_logit_quantization": float(
                model.config.routing_logit_quantization
            ),
        }
    progress = min(max(int(additional_tokens) / transition_tokens, 0.0), 1.0)
    rope_progress = linear_ramp(
        progress,
        float(train_cfg.get("rope_transition_start_fraction", 0.0)),
        float(train_cfg.get("rope_transition_end_fraction", 1.0)),
    )
    alpha_start = float(train_cfg.get("rope_transition_alpha_start", 0.0))
    alpha_end = float(train_cfg.get("rope_transition_alpha_end", 1.0))
    recency_start = float(
        train_cfg.get("anchor_recency_bias_start", model.config.anchor_recency_bias)
    )
    recency_end = float(
        train_cfg.get("anchor_recency_bias_end", model.config.anchor_recency_bias)
    )
    alpha = alpha_start + (alpha_end - alpha_start) * rope_progress
    recency_bias = recency_start + (recency_end - recency_start) * progress
    routing_quantization_start = float(
        train_cfg.get(
            "routing_logit_quantization_start",
            model.config.routing_logit_quantization,
        )
    )
    routing_quantization_end = float(
        train_cfg.get(
            "routing_logit_quantization_end",
            model.config.routing_logit_quantization,
        )
    )
    routing_quantization = routing_quantization_start + (
        routing_quantization_end - routing_quantization_start
    ) * progress
    model.set_transition_alpha(alpha)
    if "legacy_position_scale_start" in train_cfg:
        legacy_progress = linear_ramp(
            progress,
            float(train_cfg.get("legacy_position_transition_start_fraction", 0.0)),
            float(train_cfg.get("legacy_position_transition_end_fraction", 1.0)),
        )
        legacy_start = float(train_cfg["legacy_position_scale_start"])
        legacy_end = float(train_cfg.get("legacy_position_scale_end", 0.0))
        model.set_legacy_position_scale(
            legacy_start + (legacy_end - legacy_start) * legacy_progress
        )
    model.set_anchor_recency_bias(recency_bias)
    model.config.routing_logit_quantization = routing_quantization
    return {
        "transition_progress": progress,
        "rope_transition_alpha": alpha,
        "legacy_position_scale": float(model.legacy_position_scale),
        "anchor_recency_bias": recency_bias,
        "routing_logit_quantization": routing_quantization,
    }


def exact_copy_objective(
    model,
    batch: dict[str, torch.Tensor],
    weight: float,
    *,
    detach_hidden: bool = True,
    measure_proposal_recall: bool = True,
):
    required = ("input_ids", "labels", "attention_mask", "prompt_mask", "source_boundary", "copy_source_positions", "copy_target_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"exact batch lacks explicit fields: {missing}")
    if not bool(batch["prompt_mask"].bool().any()) or not bool(batch["copy_target_mask"].bool().any()):
        raise ValueError("exact batch cannot use all-token labels; prompt labels must include -100")
    if detach_hidden:
        # The base hidden state is an immutable feature for this objective.
        # Avoid constructing a full graph that would be detached immediately.
        with torch.no_grad():
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_logits=False,
            )
        hidden = output["hidden"]
    else:
        output = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_logits=False,
        )
        hidden = output["hidden"]
    result = model.exact_memory_loss(
        hidden,
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        copy_source_positions=batch["copy_source_positions"],
        copy_target_mask=batch["copy_target_mask"],
        source_boundary=batch["source_boundary"],
        measure_proposal_recall=measure_proposal_recall,
    )
    return result.loss * float(weight), result


def generated_prefix_recovery_objective(
    model,
    batch: dict[str, torch.Tensor],
    *,
    pad_token_id: int,
    eos_token_id: int,
    eos_weight: float,
    recovery_weight: float,
    unlikelihood_weight: float,
    logit_margin_weight: float = 0.0,
    logit_margin: float = 1.0,
    force_unlikelihood_only: bool = False,
):
    """Recover a clean continuation and suppress recorded loop closures."""
    required = (
        "input_ids",
        "labels",
        "attention_mask",
        "loop_negative_ids",
        "loop_negative_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"generated-prefix batch lacks explicit fields: {missing}")
    unlikelihood_only = batch.get("loop_unlikelihood_only")
    trajectory_only = bool(force_unlikelihood_only or unlikelihood_only is not None)
    if trajectory_only:
        if unlikelihood_only is not None and not bool(unlikelihood_only.bool().all()):
            raise ValueError("mixed generated-prefix objective contracts are forbidden")
        negative_positions = batch["loop_negative_mask"].bool().nonzero(as_tuple=False)
        if negative_positions.numel() == 0:
            raise ValueError("generated-prefix batch has no loop-negative positions")
        active_end = int(negative_positions[:, 1].max().item()) + 1
        objective_batch = {
            name: value[:, :active_end]
            if isinstance(value, torch.Tensor) and value.ndim == 2
            else value
            for name, value in batch.items()
        }
    else:
        objective_batch = batch
    output = model(
        objective_batch["input_ids"],
        attention_mask=objective_batch["attention_mask"],
    )
    if trajectory_only:
        recovery = output["logits"].sum() * 0.0
    else:
        recovery = weighted_lm_loss(
            output["logits"],
            objective_batch["labels"],
            objective_batch["attention_mask"],
            pad_token_id,
            eos_token_id,
            eos_weight,
        )
    shifted_logits = output["logits"][:, :-1].float()
    if "loop_candidate_ids" in objective_batch:
        positions = objective_batch["loop_contrast_position"].long()
        row_indices = torch.arange(positions.size(0), device=positions.device)
        state_logits = output["logits"].float()[row_indices, positions - 1]
        candidate_ids = objective_batch["loop_candidate_ids"].long()
        candidate_mask = objective_batch["loop_candidate_mask"].bool()
        escape_ids = objective_batch["loop_escape_ids"].long()
        safe_candidates = candidate_ids.masked_fill(~candidate_mask, 0)
        candidate_logits = state_logits.gather(-1, safe_candidates)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
        probabilities = torch.softmax(state_logits, dim=-1)
        candidate_mass = probabilities.gather(-1, safe_candidates).masked_fill(
            ~candidate_mask, 0.0
        ).sum(dim=-1)
        unlikelihood = -torch.log1p(
            -candidate_mass.clamp(max=1.0 - 1e-6)
        ).mean()
        escape_logits = state_logits.gather(-1, escape_ids.unsqueeze(-1)).squeeze(-1)
        max_candidate_logits = candidate_logits.max(dim=-1).values
        clean_minus_negative = escape_logits - max_candidate_logits
        margin_loss = torch.relu(float(logit_margin) - clean_minus_negative).mean()
        total = (
            float(unlikelihood_weight) * unlikelihood
            + float(logit_margin_weight) * margin_loss
        )
        return total, {
            "recovery_lm": total.detach() * 0.0,
            "loop_unlikelihood": unlikelihood.detach(),
            "loop_logit_margin": margin_loss.detach(),
            "loop_clean_minus_negative_logit": clean_minus_negative.mean().detach(),
            "loop_candidate_probability_mass": candidate_mass.mean().detach(),
            "loop_escape_top1_rate": state_logits.argmax(dim=-1)
            .eq(escape_ids)
            .float()
            .mean()
            .detach(),
        }
    negative_ids = objective_batch["loop_negative_ids"][:, 1:].long()
    negative_mask = (
        objective_batch["loop_negative_mask"][:, 1:].bool()
        & objective_batch["attention_mask"][:, 1:].bool()
    )
    if not trajectory_only:
        negative_mask = negative_mask & objective_batch["labels"][:, 1:].ne(-100)
    if not bool(negative_mask.any()):
        raise ValueError("generated-prefix batch has no supervised loop-negative positions")
    selected_negative_ids = negative_ids.masked_select(negative_mask)
    if bool((selected_negative_ids < 0).any()) or bool(
        (selected_negative_ids >= shifted_logits.size(-1)).any()
    ):
        raise ValueError("generated-prefix loop-negative token id is outside the vocabulary")
    if not trajectory_only and bool(
        selected_negative_ids.eq(
            objective_batch["labels"][:, 1:].masked_select(negative_mask)
        ).any()
    ):
        raise ValueError("loop-negative token id conflicts with its clean target")
    if "loop_negative_prior_occurrences" in objective_batch:
        occurrences = objective_batch["loop_negative_prior_occurrences"][:, 1:]
        if bool(occurrences.masked_select(negative_mask).lt(2).any()):
            raise ValueError("unlikelihood requires a real third-or-later n-gram closure")
    safe_negative_ids = negative_ids.masked_fill(~negative_mask, 0)
    negative_probs = torch.softmax(shifted_logits, dim=-1).gather(
        -1,
        safe_negative_ids.unsqueeze(-1),
    ).squeeze(-1)
    unlikelihood = -torch.log1p(-negative_probs.clamp(max=1.0 - 1e-6))
    unlikelihood = unlikelihood.masked_select(negative_mask).mean()
    negative_logits = shifted_logits.gather(
        -1,
        safe_negative_ids.unsqueeze(-1),
    ).squeeze(-1)
    if trajectory_only:
        clean_minus_negative = negative_logits.masked_select(negative_mask) * 0.0
        margin_loss = negative_logits.sum() * 0.0
    else:
        clean_ids = objective_batch["labels"][:, 1:].long().masked_fill(~negative_mask, 0)
        clean_logits = shifted_logits.gather(-1, clean_ids.unsqueeze(-1)).squeeze(-1)
        clean_minus_negative = (clean_logits - negative_logits).masked_select(negative_mask)
        margin_loss = torch.relu(float(logit_margin) - clean_minus_negative).mean()
    total = (
        float(recovery_weight) * recovery
        + float(unlikelihood_weight) * unlikelihood
        + float(logit_margin_weight) * margin_loss
    )
    return total, {
        "recovery_lm": recovery.detach(),
        "loop_unlikelihood": unlikelihood.detach(),
        "loop_logit_margin": margin_loss.detach(),
        "loop_clean_minus_negative_logit": clean_minus_negative.mean().detach(),
    }


def optimizer_parameter_groups(model, train_cfg: dict[str, Any]):
    base_parameters = []
    exact_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("exact_pointer."):
            exact_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    if (
        model.exact_pointer is not None
        and not exact_parameters
        and not bool(train_cfg.get("freeze_base_for_loop_controller", False))
    ):
        raise RuntimeError("enabled exact pointer has no trainable parameters")
    groups = []
    if base_parameters:
        groups.append(
            {
                "params": base_parameters,
                "lr": float(train_cfg.get("learning_rate", 2.5e-9)),
                "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
                "group_name": "base_model",
            }
        )
    if exact_parameters:
        groups.append(
            {
                "params": exact_parameters,
                "lr": float(train_cfg.get("exact_pointer_learning_rate", 1e-4)),
                "weight_decay": float(train_cfg.get("exact_pointer_weight_decay", 0.0)),
                "group_name": "exact_pointer",
            }
        )
    return groups, base_parameters, exact_parameters


def checkpoint_payload(model, optimizer, config, settings, status, step, tokens_seen, source_states, include_optimizer):
    save_mode = str(settings.get("checkpoint_save_mode", "full"))
    payload: dict[str, Any] = {"model_type": config.model_type, "model_config": asdict(config), "train_config": settings, "stage_status": status, "optimizer_step": int(step), "tokens_seen": int(tokens_seen), "source_states": source_states, "optimizer_state_included": bool(include_optimizer)}
    if save_mode == "adapter_overlay":
        if include_optimizer:
            raise ValueError("adapter overlay pilots do not support optimizer checkpoints")
        parent = require_c_path(ROOT / str(settings["parent_checkpoint"]), "adapter parent")
        adapter_state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name.startswith("loop_controller.")
        }
        if not adapter_state:
            raise ValueError("adapter overlay requested without loop-controller weights")
        payload.update(
            {
                "checkpoint_format": "fdt_v4_adapter_overlay_v1",
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": sha256_file(parent),
                "adapter_state_dict": adapter_state,
            }
        )
    elif save_mode == "full":
        payload["model_state_dict"] = model.state_dict()
    else:
        raise ValueError(f"unsupported checkpoint_save_mode: {save_mode}")
    if include_optimizer:
        payload.update({"optimizer_state_dict": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "python_random_state": random.getstate()})
    return payload


def save_immutable_milestone(
    milestone_path,
    model,
    optimizer,
    config,
    settings,
    status,
    step,
    tokens_seen,
    source_states,
):
    digest_path = milestone_path.with_suffix(milestone_path.suffix + ".sha256.json")
    if milestone_path.exists():
        actual = sha256_file(milestone_path)
        if digest_path.exists():
            recorded = json.loads(digest_path.read_text(encoding="utf-8"))
            metadata_matches = (
                actual == str(recorded.get("sha256", "")).upper()
                and int(recorded.get("optimizer_step", -1)) == int(step)
                and int(recorded.get("tokens_seen", -1)) == int(tokens_seen)
            )
        else:
            # A reboot may occur after the atomic model write but before its
            # small digest sidecar. Recover only a complete, metadata-identical
            # milestone; never overwrite or accept a different immutable file.
            existing = load_payload(milestone_path)
            metadata_matches = (
                int(existing.get("optimizer_step", -1)) == int(step)
                and int(existing.get("tokens_seen", -1)) == int(tokens_seen)
                and existing.get("model_config") == asdict(config)
                and not bool(existing.get("optimizer_state_included"))
            )
        if not metadata_matches:
            raise FileExistsError(
                f"Immutable milestone does not match retry transaction: {milestone_path}"
            )
    else:
        atomic_torch_save(
            milestone_path,
            checkpoint_payload(
                model,
                optimizer,
                config,
                settings,
                status,
                step,
                tokens_seen,
                source_states,
                False,
            ),
        )
        actual = sha256_file(milestone_path)
    atomic_json(
        digest_path,
        {
            "sha256": actual,
            "optimizer_step": int(step),
            "tokens_seen": int(tokens_seen),
        },
    )
    return actual


def save_checkpoints(output_dir, model, optimizer, config, settings, status, step, tokens_seen, source_states, milestone_path=None):
    model_payload = checkpoint_payload(model, optimizer, config, settings, status, step, tokens_seen, source_states, False)
    if milestone_path is not None:
        save_immutable_milestone(
            milestone_path,
            model,
            optimizer,
            config,
            settings,
            status,
            step,
            tokens_seen,
            source_states,
        )
    # The immutable milestone is durable before mutable latest files advertise
    # the advanced milestone cursor. A retry can verify and reuse it safely.
    save_optimizer_recovery = bool(settings.get("save_optimizer_recovery", True))
    recovery_hash = None
    if save_optimizer_recovery:
        atomic_torch_save(output_dir / "latest_recovery.pt", checkpoint_payload(model, optimizer, config, settings, status, step, tokens_seen, source_states, True))
    atomic_torch_save(output_dir / "latest.pt", model_payload)
    if save_optimizer_recovery:
        recovery_hash = sha256_file(output_dir / "latest_recovery.pt")
    return sha256_file(output_dir / "latest.pt"), recovery_hash


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def validate_resume_checkpoint(path: Path, output_dir: Path) -> dict[str, Any]:
    resolved = require_c_path(path, "resume checkpoint")
    if resolved.parent != output_dir.resolve() or resolved.name.endswith(".tmp") or not resolved.exists():
        raise ValueError("Resume checkpoint is not an owned, complete file")
    payload = load_payload(resolved)
    required = ("model_state_dict", "optimizer_state_dict", "model_config", "train_config", "source_states", "torch_rng_state", "python_random_state")
    if payload.get("stage_status") not in {"PAUSED", "INTERRUPTED", "SAFETY_STOP", "RUNNING"} or not payload.get("optimizer_state_included"):
        raise ValueError("Resume requires a full optimizer-bearing recovery checkpoint")
    if any(key not in payload for key in required) or payload["model_config"].get("model_type") != "fdt_v4":
        raise ValueError("Resume checkpoint lacks FDT v4 resumability state")
    if bool(payload.get("train_config", {}).get("overfit_triggered", False)):
        raise ValueError("Resume refused: checkpoint records a validated overfitting stop")
    if bool(
        payload.get("train_config", {}).get(
            "transition_regression_triggered", False
        )
    ):
        raise ValueError(
            "Resume refused: checkpoint records a transition-regression stop"
        )
    return payload


TRANSITION_RUNTIME_CONFIG_FIELDS = {
    "rope_transition_alpha",
    "legacy_position_scale",
    "anchor_recency_bias",
    "routing_logit_quantization",
}


def resume_model_configs_compatible(
    checkpoint_config: dict[str, Any], configured: dict[str, Any]
) -> bool:
    left = dict(checkpoint_config)
    right = dict(configured)
    for name in TRANSITION_RUNTIME_CONFIG_FIELDS:
        left.pop(name, None)
        right.pop(name, None)
    return left == right


def _key_allowed(name: str) -> bool:
    return (name.startswith(WARM_START_ALLOWLIST) and not any(token in name for token in WARM_START_DENYLIST))


def convert_v20_state_dict(model, parent_payload: dict[str, Any]) -> dict[str, Any]:
    source_config = parent_payload.get("model_config", {})
    if source_config.get("model_type") != V20_PARENT_MODEL_TYPE:
        raise ValueError("Warm-start parent must be a V20 FDT v3 checkpoint")
    if (source_config.get("vocab_size"), source_config.get("pad_token_id"), source_config.get("eos_token_id")) != (24576, 0, 2):
        raise ValueError("V20 warm-start tokenizer contract does not match FDT v4")
    source, target = parent_payload.get("model_state_dict", {}), model.state_dict()
    converted, skipped, mismatched = [], [], []
    for name, target_tensor in target.items():
        if not _key_allowed(name):
            skipped.append(name)
            continue
        source_tensor = source.get(name)
        if source_tensor is None or tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched.append(name)
            continue
        target_tensor.copy_(source_tensor)
        converted.append(name)
    legacy_position_key = "legacy_position_embedding.weight"
    if legacy_position_key in target:
        source_position = source.get("position_embedding.weight")
        target_position = target[legacy_position_key]
        if source_position is None or tuple(source_position.shape) != tuple(
            target_position.shape
        ):
            raise ValueError(
                "Warm-start refused: legacy position table was not shape-compatible"
            )
        target_position.copy_(source_position)
        converted.append(legacy_position_key)
        if legacy_position_key in skipped:
            skipped.remove(legacy_position_key)
    anchor_keys = [name for name in target if ".anchor." in name]
    if not anchor_keys or not set(anchor_keys).issubset(converted):
        raise ValueError("Warm-start refused: inherited anchor transfer was incomplete")
    return {"schema_version": "fdt_v4_1_v20_conversion_v2", "source_model_type": source_config.get("model_type"), "allowlist": list(WARM_START_ALLOWLIST), "denylist": list(WARM_START_DENYLIST), "converted_keys": converted, "skipped_new_or_denied_keys": skipped, "shape_mismatches": mismatched, "anchor_transfer_verified": True, "legacy_position_transfer_verified": legacy_position_key not in target or legacy_position_key in converted, "new_random_components": [name for name in target if name not in converted], "rope_and_exact_memory_initialized_new": True}


def load_fdt_v4_continuation_state(model, parent_payload: dict[str, Any]) -> dict[str, Any]:
    source_config = parent_payload.get("model_config", {})
    if source_config.get("model_type") != "fdt_v4":
        raise ValueError("Continuation parent must be an FDT v4 checkpoint")
    if (
        source_config.get("vocab_size"),
        source_config.get("pad_token_id"),
        source_config.get("eos_token_id"),
    ) != (24576, 0, 2):
        raise ValueError("FDT v4 continuation tokenizer contract does not match")
    source = parent_payload.get("model_state_dict")
    target = model.state_dict()
    if not isinstance(source, dict):
        raise ValueError("FDT v4 continuation state is missing")
    added_loop_keys = {
        name for name in target if name.startswith("loop_controller.") and name not in source
    }
    if set(source) | added_loop_keys != set(target):
        raise ValueError("FDT v4 continuation state keys do not match exactly")
    mismatched = [
        name
        for name, tensor in target.items()
        if name not in added_loop_keys
        and tuple(source[name].shape) != tuple(tensor.shape)
    ]
    if mismatched:
        raise ValueError(
            f"FDT v4 continuation state shapes differ: {mismatched[:8]}"
        )
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != added_loop_keys or incompatible.unexpected_keys:
        raise ValueError("FDT v4 loop-controller continuation migration failed")
    if added_loop_keys:
        up = model.state_dict().get("loop_controller.up.weight")
        if up is None or bool(up.ne(0).any()):
            raise ValueError("new loop controller must initialize as an exact no-op")
    return {
        "schema_version": "fdt_v4_1_fresh_objective_continuation_v1",
        "source_model_type": "fdt_v4",
        "state_keys_exact": True,
        "state_shapes_exact": True,
        "fresh_optimizer": True,
        "fresh_optimizer_reason": "isolated_objective_pilot",
        "added_zero_initialized_loop_controller_keys": sorted(added_loop_keys),
    }


def model_config_and_settings(config_path: Path, output_dir: Path):
    raw = load_yaml_like(config_path)
    config = model_config_from_yaml(raw)
    validate_main_architecture(config)
    return config, {**dict(raw.get("train", {})), "output_dir": str(output_dir), "stage": raw.get("stage", "fdt_v4_main_426m_curriculum")}, dict(raw.get("data", {})), raw


def routing_diagnostics(
    output: dict[str, Any], *, device_resident: bool = False
) -> dict[str, Any]:
    stats = output.get("anchor_stats") or []
    entropy_values = []
    dead_values = []
    top1_values = []
    active_anchor_masks = []
    for item in stats:
        if isinstance(item, dict) and isinstance(item.get("entropy_normalized"), torch.Tensor):
            entropy = item["entropy_normalized"].detach().float()
            entropy_values.append(entropy if device_resident else float(entropy.cpu()))
            if isinstance(item.get("dead_anchor_fraction"), torch.Tensor):
                dead = item["dead_anchor_fraction"].detach().float()
                dead_values.append(dead if device_resident else float(dead.cpu()))
            if isinstance(item.get("active_anchor_mask"), torch.Tensor):
                mask = item["active_anchor_mask"].detach().bool()
                active_anchor_masks.append(mask if device_resident else mask.cpu())
            if isinstance(item.get("top1_membership"), torch.Tensor):
                top1 = item["top1_membership"].detach().float()
                top1_values.append(top1 if device_resident else float(top1.cpu()))
        elif hasattr(item, "entropy") and isinstance(item.entropy, torch.Tensor):
            entropy = item.entropy.detach().float() / math.log(
                max(int(getattr(item, "membership", torch.empty(0)).size(-1)), 2)
            )
            entropy_values.append(entropy if device_resident else float(entropy.cpu()))
            if isinstance(getattr(item, "load_prob", None), torch.Tensor):
                load_prob = item.load_prob.detach()
                dead = load_prob.le(0).float().mean()
                dead_values.append(dead if device_resident else float(dead.cpu()))
                mask = load_prob.gt(0).bool()
                active_anchor_masks.append(mask if device_resident else mask.cpu())
            if isinstance(getattr(item, "top1_membership", None), torch.Tensor):
                top1 = item.top1_membership.detach().float()
                top1_values.append(top1 if device_resident else float(top1.cpu()))
        elif isinstance(item, dict) and isinstance(item.get("membership"), torch.Tensor):
            membership = item["membership"].detach().float()
            probs = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = (
                -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                / math.log(max(probs.size(-1), 2))
            ).mean()
            entropy_values.append(entropy if device_resident else float(entropy.cpu()))
    if device_resident:
        entropy_value = torch.stack(entropy_values).mean() if entropy_values else None
        dead_value = torch.stack(dead_values).max() if dead_values else None
        top1_value = torch.stack(top1_values).mean() if top1_values else None
    else:
        entropy_value = sum(entropy_values) / len(entropy_values) if entropy_values else None
        dead_value = max(dead_values) if dead_values else None
        top1_value = sum(top1_values) / len(top1_values) if top1_values else None
    return {
        "entropy_normalized": entropy_value,
        "dead_anchor_fraction": dead_value,
        "top1_membership": top1_value,
        "active_anchor_masks": active_anchor_masks,
    }


def merge_active_anchor_masks(
    accumulated: list[torch.Tensor] | None,
    current: list[torch.Tensor],
) -> list[torch.Tensor]:
    if not current:
        return [] if accumulated is None else accumulated
    if accumulated is None:
        return [mask.clone() for mask in current]
    if len(accumulated) != len(current) or any(
        left.shape != right.shape for left, right in zip(accumulated, current)
    ):
        raise RuntimeError("routing active-anchor mask shape changed")
    return [left | right for left, right in zip(accumulated, current)]


def dead_anchor_fraction_from_masks(
    masks: list[torch.Tensor] | None,
    *,
    device_resident: bool = False,
) -> float | torch.Tensor | None:
    if not masks:
        return None
    values = [(~mask).float().mean() for mask in masks]
    if device_resident:
        return torch.stack(values).max()
    return max(float(value.cpu()) for value in values)


def routing_entropy(output: dict[str, Any]) -> float | None:
    return routing_diagnostics(output)["entropy_normalized"]


def validation_snapshot(
    model,
    pool: RowPool,
    config: ModelConfig,
    device: torch.device,
    eos_weight: float,
    *,
    batches: int = 1,
    fixed_state: dict[str, int] | None = None,
) -> float:
    current_state = pool.state()
    if fixed_state is not None:
        pool.restore(fixed_state)
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.no_grad():
            for _ in range(max(int(batches), 1)):
                batch = move_batch(pool.next_batch(1), device)
                output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                value = weighted_lm_loss(output["logits"], batch["labels"], batch["attention_mask"], config.pad_token_id, config.eos_token_id, eos_weight)
                values.append(float(value.detach().float().cpu()))
    finally:
        pool.restore(current_state)
        if was_training:
            model.train()
    return sum(values) / len(values)


def transition_control_state(model) -> dict[str, float]:
    return {
        "rope_transition_alpha": float(model.transition_alpha),
        "legacy_position_scale": float(model.legacy_position_scale),
        "anchor_recency_bias": float(model.config.anchor_recency_bias),
        "routing_logit_quantization": float(model.config.routing_logit_quantization),
    }


def restore_transition_control_state(model, state: dict[str, float]) -> None:
    model.set_transition_alpha(state["rope_transition_alpha"])
    model.set_legacy_position_scale(state["legacy_position_scale"])
    model.set_anchor_recency_bias(state["anchor_recency_bias"])
    model.config.routing_logit_quantization = state["routing_logit_quantization"]


def train(args: argparse.Namespace) -> int:
    config_path, output_dir = require_c_path(args.config, "config"), owned_run_path(args.output_dir)
    if output_dir.exists() and not args.resume:
        allowed = {"train.pid", "active_logs.json", "stdout.log", "stderr.log"}
        leftovers = [path.name for path in output_dir.iterdir() if path.name not in allowed]
        if leftovers:
            raise FileExistsError(f"Output directory is not fresh: {leftovers}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config, train_cfg, data_cfg, raw = model_config_and_settings(config_path, output_dir)
    atomic_json(output_dir / "model_preflight.json", metadata_model_preflight(config))
    if disk_free_gib() < float(train_cfg.get("min_free_gib", 10.0)):
        raise RuntimeError("C: free space is below the configured safety floor")
    seed = int(train_cfg.get("seed", 20260823))
    deterministic_algorithms = bool(train_cfg.get("deterministic_algorithms", False))
    seed_everything(seed, deterministic_algorithms=deterministic_algorithms)
    device = torch.device(args.device)
    if device.type == "cuda" and (not args.allow_gpu or not torch.cuda.is_available()):
        raise RuntimeError("GPU requires --allow-gpu and an available CUDA device")
    dataset_dirs = {"natural": ROOT / data_cfg["natural_lm_dir"], "factual": ROOT / data_cfg["factual_dir"], "exact_copy": ROOT / data_cfg["exact_copy_dir"]}
    long_context_path = str(data_cfg.get("long_context_dir", "")).strip()
    if bool(train_cfg.get("require_long_context_curriculum", False)) and not long_context_path:
        raise ValueError("required long-context curriculum needs a dataset path")
    if long_context_path:
        dataset_dirs["long_context"] = ROOT / data_cfg["long_context_dir"]
    generated_prefix_path = str(data_cfg.get("generated_prefix_dir", "")).strip()
    generated_prefix_weight = float(train_cfg.get("generated_prefix_recovery_weight", 0.0))
    generated_prefix_enabled = any(
        float(train_cfg.get(name, 0.0)) > 0.0
        for name in (
            "generated_prefix_recovery_weight",
            "generated_prefix_unlikelihood_weight",
            "generated_prefix_logit_margin_weight",
        )
    )
    require_generated_prefix = bool(train_cfg.get("require_generated_prefix_recovery", False))
    if require_generated_prefix and (not generated_prefix_path or not generated_prefix_enabled):
        raise ValueError(
            "required generated-prefix recovery needs a dataset path and positive weight"
        )
    if generated_prefix_path:
        dataset_dirs["generated_prefix"] = ROOT / data_cfg["generated_prefix_dir"]
    bridge_context_path = str(data_cfg.get("bridge_context_dir", "")).strip()
    if bool(train_cfg.get("require_context_bridge", False)) and not bridge_context_path:
        raise ValueError("required context bridge needs a dataset path")
    if bridge_context_path:
        dataset_dirs["bridge_context"] = ROOT / data_cfg["bridge_context_dir"]
    for name, path in dataset_dirs.items():
        require_c_path(path, f"{name} dataset")
    validation_path = None
    validation_dir = str(data_cfg.get("validation_dir", "")).strip()
    audited_dataset_dirs = dict(dataset_dirs)
    if validation_dir:
        validation_path = require_c_path(ROOT / validation_dir, "validation dataset")
        audited_dataset_dirs["validation"] = validation_path
    data_preflight = preflight_dataset_contract(audited_dataset_dirs, data_cfg)
    for source in data_preflight["sources"].values():
        if any(int(shard["sequence_length"]) > config.max_seq_len for shard in source["shards"]):
            raise ValueError("dataset sequence length exceeds the FDT v4 16K contract")
    if "long_context" in data_preflight["sources"]:
        long_lengths = [
            int(shard["sequence_length"])
            for shard in data_preflight["sources"]["long_context"]["shards"]
        ]
        required_length = int(train_cfg.get("long_context_min_sequence_length", 8192))
        if max(long_lengths, default=0) < required_length:
            raise ValueError(
                f"long-context curriculum must contain a shard of at least {required_length} tokens"
            )
        if bool(train_cfg.get("long_context_require_8k_bucket", True)) and not any(
            8192 <= length < 16384 for length in long_lengths
        ):
            raise ValueError("long-context curriculum requires a distinct 8K-16K training shard")
        if bool(train_cfg.get("long_context_require_16k_shard", True)) and not any(
            length == 16384 for length in long_lengths
        ):
            raise ValueError("long-context curriculum requires an exact 16K training shard")
    if "bridge_context" in data_preflight["sources"]:
        bridge_lengths = {
            int(shard["sequence_length"])
            for shard in data_preflight["sources"]["bridge_context"]["shards"]
        }
        if bool(train_cfg.get("bridge_require_2k_bucket", True)) and 2048 not in bridge_lengths:
            raise ValueError("context bridge requires an exact 2K training shard")
        if bool(train_cfg.get("bridge_require_4k_bucket", True)) and 4096 not in bridge_lengths:
            raise ValueError("context bridge requires an exact 4K training shard")
        if not bridge_lengths.issubset({2048, 4096}):
            raise ValueError("context bridge may contain only exact 2K and 4K shards")
    atomic_json(output_dir / "data_preflight.json", data_preflight)
    tokenizer_dir = require_c_path(ROOT / data_cfg["tokenizer_dir"], "tokenizer directory")
    tokenizer_path = require_c_path(ROOT / data_cfg["tokenizer_json"], "tokenizer.json")
    tokenizer_sha = sha256_file(tokenizer_path)
    if not tokenizer_dir.is_dir() or tokenizer_sha != str(data_cfg["tokenizer_json_sha256"]).upper():
        raise ValueError("Tokenizer directory or pinned tokenizer.json hash is invalid")
    tokenizer = load_tokenizer(str(tokenizer_dir))
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model config")
    parent_path = str(train_cfg.get("parent_checkpoint", "")).strip()
    parent = require_c_path(ROOT / parent_path, "parent checkpoint") if parent_path else None
    resume_payload = validate_resume_checkpoint(args.resume, output_dir) if args.resume else None
    is_resume = resume_payload is not None
    if resume_payload is not None:
        if str(resume_payload["train_config"].get("config_sha256", "")).upper() != sha256_file(config_path) or not resume_model_configs_compatible(resume_payload["model_config"], asdict(config)):
            raise ValueError("Resume config/model configuration mismatch")
    model = build_model(config)
    warm_started = False
    if parent is not None and resume_payload is None:
        parent_payload = load_payload(parent)
        if parent_payload.get("model_config", {}).get("model_type") == "fdt_v4":
            conversion = load_fdt_v4_continuation_state(model, parent_payload)
            manifest_name = "fdt_v4_continuation_manifest.json"
            warm_started = bool(
                parent_payload.get("train_config", {}).get(
                    "warm_started_from_v20", True
                )
            )
        else:
            conversion = convert_v20_state_dict(model, parent_payload)
            manifest_name = "v20_conversion_manifest.json"
            warm_started = True
        conversion.update({"source_checkpoint": str(parent), "source_checkpoint_sha256": sha256_file(parent)})
        atomic_json(output_dir / manifest_name, conversion)
    elif resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        warm_started = bool(resume_payload.get("train_config", {}).get("warm_started_from_v20", False))
    if warm_started:
        for name, parameter in model.named_parameters():
            if ".anchor." in name or ".anchor_norm." in name:
                parameter.requires_grad_(False)
    if bool(train_cfg.get("freeze_base_for_loop_controller", False)):
        if getattr(model, "loop_controller", None) is None:
            raise ValueError("loop-controller-only training requires an enabled controller")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.loop_controller.parameters():
            parameter.requires_grad_(True)
        if str(train_cfg.get("checkpoint_save_mode", "full")) != "adapter_overlay":
            raise ValueError("loop-controller-only training requires adapter_overlay saving")
        if bool(train_cfg.get("save_optimizer_recovery", True)):
            raise ValueError("adapter_overlay saving cannot include an optimizer checkpoint")
    if args.preflight_only:
        unique_parameters = sum(parameter.numel() for parameter in model.parameters())
        report = {
            "status": "PASS",
            "mode": "PREFLIGHT_ONLY_NO_TRAINING",
            "model_type": config.model_type,
            "parameter_count": unique_parameters,
            "parameter_class": "400M",
            "max_seq_len": config.max_seq_len,
            "exact_memory_enabled": config.exact_memory_enabled,
            "generation_repetition_scope": config.generation_repetition_scope,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "tokenizer_json_sha256": tokenizer_sha,
            "parent_checkpoint": str(parent) if parent is not None else None,
            "parent_checkpoint_sha256": sha256_file(parent) if parent is not None else None,
            "dataset_manifest_sha256": {
                name: manifest_hash(path) for name, path in audited_dataset_dirs.items()
            },
            "data_preflight": data_preflight,
            "warm_started_from_v20": warm_started,
            "training_launched": False,
        }
        atomic_json(output_dir / "training_preflight.json", report)
        print(json.dumps(report))
        return 0
    optimizer_groups, base_trainable, exact_trainable = optimizer_parameter_groups(model, train_cfg)
    trainable = base_trainable + exact_trainable
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(bool(train_cfg.get("activation_checkpointing", True)))
    model.to(device=device)  # FP32 parameters/master semantics; CUDA compute uses BF16 autocast.
    optimizer = torch.optim.AdamW(optimizer_groups)
    pools = {
        name: RowPool(
            path,
            data_cfg.get(f"{name}_split", data_cfg.get("split", "train")),
            seed + index * 1009,
            exact=name == "exact_copy",
            generated_prefix=name == "generated_prefix",
        )
        for index, (name, path) in enumerate(dataset_dirs.items())
    }
    configured_fractions = train_cfg.get("source_token_fractions") or train_cfg.get(
        "source_batch_fractions"
    ) or {
        "natural": float(train_cfg.get("natural_batch_fraction", 0.60)),
        "factual": float(train_cfg.get("factual_batch_fraction", 0.40)),
    }
    source_fractions = {
        str(name): float(value) for name, value in configured_fractions.items()
    }
    missing_sources = set(source_fractions) - set(pools)
    if missing_sources:
        raise ValueError(
            f"source mixture references missing datasets: {sorted(missing_sources)}"
        )
    source_mixture_unit = str(train_cfg.get("source_mixture_unit", "rows"))
    if source_mixture_unit not in {"rows", "tokens"}:
        raise ValueError("source_mixture_unit must be rows or tokens")
    token_planner = (
        TokenBudgetSourcePlanner(source_fractions)
        if source_mixture_unit == "tokens"
        else None
    )
    validation_pool = None
    validation_fixed_state = None
    if validation_path is not None:
        validation_seed = int(train_cfg.get("validation_seed", seed + 7001))
        validation_pool = RowPool(
            validation_path,
            data_cfg.get("validation_split", "validation"),
            validation_seed,
        )
        validation_fixed_state = validation_pool.state()
    source_states = {name: pool.state() for name, pool in pools.items()}
    if token_planner is not None:
        source_states["__token_planner__"] = token_planner.state()
    if validation_pool is not None:
        source_states["__validation__"] = validation_pool.state()
    step = tokens_seen = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for name, state in resume_payload.get("source_states", {}).items():
            if name in pools:
                pools[name].restore(state)
        if token_planner is not None:
            planner_state = resume_payload.get("source_states", {}).get(
                "__token_planner__"
            )
            if planner_state is None:
                raise ValueError("Resume checkpoint lacks token planner state")
            token_planner.restore(planner_state)
        if validation_pool is not None:
            if "__validation__" not in resume_payload.get("source_states", {}):
                raise ValueError("Resume checkpoint lacks validation cursor state")
            validation_pool.restore(resume_payload["source_states"]["__validation__"])
        source_states = {name: pool.state() for name, pool in pools.items()}
        if token_planner is not None:
            source_states["__token_planner__"] = token_planner.state()
        if validation_pool is not None:
            source_states["__validation__"] = validation_pool.state()
        step, tokens_seen = int(resume_payload.get("optimizer_step", 0)), int(resume_payload.get("tokens_seen", 0))
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])
        random.setstate(resume_payload["python_random_state"])
    fingerprints = {name: manifest_hash(path) for name, path in audited_dataset_dirs.items() if path.exists()}
    if resume_payload is not None and resume_payload["train_config"].get("dataset_manifest_sha256", {}) != fingerprints:
        raise ValueError("Resume dataset manifest hash mismatch")
    if resume_payload is not None and resume_payload["train_config"].get("data_preflight") != data_preflight:
        raise ValueError("Resume dataset shard hashes or contracts changed")
    resume_settings = dict(resume_payload.get("train_config", {})) if resume_payload is not None else {}
    run_start_tokens = int(resume_settings.get("run_start_tokens", tokens_seen))
    settings = {**train_cfg, "model_config": asdict(config), "config_path": str(config_path), "config_sha256": sha256_file(config_path), "tokenizer_dir": str(tokenizer_dir), "tokenizer_json": str(tokenizer_path), "tokenizer_json_sha256": tokenizer_sha, "dataset_dirs": {name: str(path) for name, path in audited_dataset_dirs.items()}, "dataset_manifest_sha256": fingerprints, "data_preflight": data_preflight, "exact_copy_supervision": "explicit_source_token_mapping_and_target_mask", "generic_all_token_pointer_loss": False, "fp32_optimizer_master": True, "activation_checkpointing": bool(train_cfg.get("activation_checkpointing", True)), "deterministic_algorithms": deterministic_algorithms, "cuda_backend_contract": "verified_fast_equivalent_v1" if not deterministic_algorithms else "strict_deterministic", "output_dir": str(output_dir), "run_owner": "fdt_v4_curriculum", "run_start_tokens": run_start_tokens, "warm_started_from_v20": warm_started}
    for durable_key in (
        "overfit_train_reference",
        "validation_history",
        "next_milestone_tokens",
        "overfit_triggered",
        "transition_regression_triggered",
        "last_validation_loss",
        "last_gate_validation_loss",
        "last_validation_step",
    ):
        if durable_key in resume_settings:
            settings[durable_key] = resume_settings[durable_key]
    gate_version = str(train_cfg.get("overfit_gate_version", "fixed_validation_v2"))
    if str(resume_settings.get("overfit_gate_version", "moving_cursor_v1")) != gate_version:
        settings["invalidated_validation_history"] = list(
            resume_settings.get("validation_history", [])
        )
        settings["validation_history"] = []
        settings["overfit_triggered"] = False
    settings["overfit_gate_version"] = gate_version
    atomic_json(output_dir / "run_manifest.json", settings)
    log_path = output_dir / "training_log.jsonl"
    append_jsonl(log_path, {"event": "resume" if is_resume else "start", "step": step, "tokens_seen": tokens_seen, **settings})
    stop_path = output_dir / str(raw.get("safety", {}).get("stop_file", "STOP_REQUESTED"))
    target_tokens = int(train_cfg.get("target_additional_tokens", 1_000_000_000))
    start_tokens = run_start_tokens
    source_cycle = (
        deterministic_source_cycle(source_fractions, slots=100)
        if token_planner is None
        else []
    )
    next_milestone = int(resume_payload["train_config"].get("next_milestone_tokens", train_cfg.get("model_checkpoint_every_tokens", 100_000_000))) if is_resume else int(train_cfg.get("model_checkpoint_every_tokens", 100_000_000))
    # The model, optimizer, cursors, and RNG states now own independent copies.
    # Releasing the deserialized recovery prevents a full CPU-side checkpoint duplicate.
    resume_payload = None
    gc.collect()
    started, status = time.perf_counter(), "RUNNING"
    session_start_tokens = tokens_seen
    last_log_time = started
    last_log_tokens = tokens_seen
    pending_generated_prefix_observation = None
    checkpoint_consistent = True
    autocast_enabled = device.type == "cuda"
    model.train()
    routing_window_masks: list[torch.Tensor] | None = None
    routing_window_observed_steps = 0
    routing_window_steps = max(
        int(train_cfg.get("routing_safety_window_steps", 25)), 1
    )
    max_epochs_per_source = {
        str(name): float(value)
        for name, value in dict(train_cfg.get("max_epochs_per_source", {})).items()
    }
    max_tokens_per_source = {
        str(name): int(value)
        for name, value in dict(train_cfg.get("max_tokens_per_source", {})).items()
    }
    try:
        while tokens_seen - start_tokens < target_tokens:
            exhausted_sources = [
                name
                for name, limit in max_epochs_per_source.items()
                if name in pools
                and pools[name].rows_consumed
                >= math.floor(pools[name].total_rows * float(limit))
            ]
            if token_planner is not None:
                exhausted_sources.extend(
                    name
                    for name, limit in max_tokens_per_source.items()
                    if token_planner.active_tokens.get(name, 0) >= int(limit)
                )
            if exhausted_sources:
                status = "DATA_DIVERSITY_STOP"
                settings["data_diversity_exhausted_sources"] = sorted(
                    set(exhausted_sources)
                )
                break
            if disk_free_gib() < float(train_cfg.get("min_free_gib", 10.0)):
                status = "SAFETY_STOP"
                break
            if stop_path.exists():
                status = "PAUSED"
                break
            checkpoint_consistent = False
            transition_metrics = apply_architecture_transition(
                model,
                train_cfg,
                tokens_seen - start_tokens,
            )
            optimizer.zero_grad(set_to_none=True)
            accumulation = max(
                int(train_cfg.get("grad_accum_steps", 1))
                * int(train_cfg.get("batch_size", 1)),
                1,
            )
            log_every = int(train_cfg.get("log_every", 10))
            validation_every = int(train_cfg.get("validation_every_steps", 0))
            log_due = (step + 1) % log_every == 0
            validation_due = (
                validation_pool is not None
                and validation_every > 0
                and (step + 1) % validation_every == 0
            )
            metrics_due = (
                log_due
                or validation_due
                or "overfit_train_reference" not in settings
            )
            objective_values, diagnostic_values, batch_tokens = {}, {}, 0
            if metrics_due:
                diagnostic_values.update(transition_metrics)
            finite_loss_flags: list[torch.Tensor] = []
            entropy_value = top1_membership_value = None
            entropy_tensor = top1_membership_tensor = None
            dead_anchor_value = 0.0
            optimizer_step_masks: list[torch.Tensor] | None = None
            routing_diagnostics_seen = 0
            forward_groups = planned_base_forward_groups(
                pools,
                source_cycle,
                step,
                accumulation,
                max(int(train_cfg.get("short_sequence_batch_size", 1)), 1),
                int(train_cfg.get("short_sequence_max_length", 512)),
                token_planner=token_planner,
            )
            for micro_index, host_batch in enumerate(forward_groups):
                batch = move_batch(host_batch, device)
                sequence_length = int(batch["input_ids"].size(1))
                set_sequence_gradient_checkpointing(model, sequence_length, train_cfg)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    output = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_logits=False,
                    )
                    micro_losses = {"base_lm": chunked_weighted_lm_loss(
                        model,
                        output["hidden"],
                        batch["labels"],
                        batch["attention_mask"],
                        config.pad_token_id,
                        config.eos_token_id,
                        float(train_cfg.get("eos_loss_weight", 2.0)),
                        int(train_cfg.get("lm_loss_sequence_chunk_size", 256)),
                        lm_loss_checkpointing_enabled(sequence_length, train_cfg),
                    ) * float(train_cfg.get("base_lm_weight", 1.0)) / accumulation}
                    route_metrics = routing_diagnostics(output, device_resident=True)
                    current_dead = route_metrics["dead_anchor_fraction"]
                    if current_dead is None:
                        if bool(train_cfg.get("require_routing_diagnostics", True)):
                            raise RuntimeError("required dead-anchor diagnostics are missing")
                    else:
                        routing_diagnostics_seen += 1
                        optimizer_step_masks = merge_active_anchor_masks(
                            optimizer_step_masks,
                            route_metrics["active_anchor_masks"],
                        )
                    if log_due:
                        entropy_tensor = route_metrics["entropy_normalized"]
                        top1_membership_tensor = route_metrics["top1_membership"]
                    batch_tokens += int(host_batch["attention_mask"].sum().item())
                    micro_total = sum(micro_losses.values())
                finite_loss_flags.append(torch.isfinite(micro_total.detach()))
                micro_total.backward()
                if metrics_due:
                    for name, value in micro_losses.items():
                        objective_values[name] = objective_values.get(name, 0.0) + float(value.detach().float().cpu())
            del output, micro_losses, micro_total, batch, host_batch
            if bool(train_cfg.get("require_routing_diagnostics", True)) and (
                routing_diagnostics_seen != len(forward_groups)
            ):
                raise RuntimeError("dead-anchor diagnostics were incomplete for the optimizer step")
            routing_window_masks = merge_active_anchor_masks(
                routing_window_masks,
                optimizer_step_masks or [],
            )
            routing_window_observed_steps += 1
            current_window_dead_tensor = dead_anchor_fraction_from_masks(
                routing_window_masks, device_resident=True
            )
            if current_window_dead_tensor is None:
                raise RuntimeError("routing active-anchor masks are missing")
            window_due = routing_window_observed_steps >= routing_window_steps
            if log_due:
                dead_anchor_value = float(current_window_dead_tensor.detach().cpu())
                entropy_value = (
                    float(entropy_tensor.detach().cpu())
                    if isinstance(entropy_tensor, torch.Tensor)
                    else None
                )
                top1_membership_value = (
                    float(top1_membership_tensor.detach().cpu())
                    if isinstance(top1_membership_tensor, torch.Tensor)
                    else None
                )
            if window_due:
                if not log_due:
                    dead_anchor_value = float(current_window_dead_tensor.detach().cpu())
                if dead_anchor_value > float(train_cfg.get("dead_anchor_fraction_limit", 0.01)):
                    raise FloatingPointError(
                        f"{routing_window_steps}-step dead-anchor fraction "
                        f"{dead_anchor_value:.6f} exceeds safety limit"
                    )
                routing_window_masks = None
                routing_window_observed_steps = 0
            # The exact batch runs only after every base-LM microbatch has
            # backpropagated. This prevents a second full-model graph from
            # overlapping the retained LM graph and reproducing prior OOM/TDR
            # failure conditions.
            exact_copy_weight = float(train_cfg.get("exact_copy_weight", 0.10))
            if exact_copy_weight > 0.0 and step % int(
                train_cfg.get("exact_batch_every_steps", 1)
            ) == 0:
                exact_batch = move_batch(
                    pools["exact_copy"].next_batch(int(train_cfg.get("batch_size", 1))),
                    device,
                )
                curriculum_progress = (tokens_seen - start_tokens) / max(target_tokens, 1)
                exact_cap = exact_curriculum_cap(curriculum_progress)
                cap_copy_targets(exact_batch, exact_cap)
                set_sequence_gradient_checkpointing(
                    model, int(exact_batch["input_ids"].size(1)), train_cfg
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    exact_loss, exact_result = exact_copy_objective(
                        model,
                        exact_batch,
                        exact_copy_weight,
                        detach_hidden=bool(train_cfg.get("detach_exact_hidden", True)),
                        measure_proposal_recall=metrics_due,
                    )
                finite_loss_flags.append(torch.isfinite(exact_loss.detach()))
                exact_loss.backward()
                if metrics_due:
                    objective_values["exact_copy"] = float(exact_loss.detach().float().cpu())
                    diagnostic_values["exact_curriculum_target_cap"] = float(exact_cap)
                    for name in (
                        "pointer_loss",
                        "gate_loss",
                        "commit_loss",
                        "copyable_rate",
                        "pointer_accuracy",
                        "proposal_recall",
                        "cursor_continuation_rate",
                        "hard_negative_loss",
                        "max_copy_distance",
                        "scanned_source_tokens",
                    ):
                        value = getattr(exact_result, name, None)
                        if isinstance(value, torch.Tensor):
                            diagnostic_values[f"exact_{name}"] = float(value.float().cpu())
                del exact_loss, exact_result, exact_batch
            generated_prefix_due = (
                "generated_prefix" in pools
                and generated_prefix_enabled
                and step >= int(train_cfg.get("generated_prefix_min_step", 0))
                and step % int(train_cfg.get("generated_prefix_every_steps", 1)) == 0
            )
            if generated_prefix_due:
                curriculum_progress = (tokens_seen - start_tokens) / max(target_tokens, 1)
                prefix_scale = linear_ramp(
                    curriculum_progress,
                    float(train_cfg.get("generated_prefix_ramp_start", 0.02)),
                    float(train_cfg.get("generated_prefix_ramp_end", 0.20)),
                )
                # Advance the deterministic data cursor even while the ramp is
                # exactly zero, but do not build a graph whose total gradient is zero.
                host_prefix_batch = pools["generated_prefix"].next_batch(
                    int(
                        train_cfg.get(
                            "generated_prefix_batch_size",
                            train_cfg.get("batch_size", 1),
                        )
                    )
                )
                if prefix_scale > 0.0:
                    prefix_batch = move_batch(host_prefix_batch, device)
                    set_sequence_gradient_checkpointing(
                        model, int(prefix_batch["input_ids"].size(1)), train_cfg
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        prefix_loss, prefix_metrics = generated_prefix_recovery_objective(
                            model,
                            prefix_batch,
                            pad_token_id=config.pad_token_id,
                            eos_token_id=config.eos_token_id,
                            eos_weight=float(train_cfg.get("eos_loss_weight", 2.0)),
                            recovery_weight=generated_prefix_weight * prefix_scale,
                            unlikelihood_weight=float(
                                train_cfg.get("generated_prefix_unlikelihood_weight", 0.01)
                            ) * prefix_scale,
                            logit_margin_weight=float(
                                train_cfg.get("generated_prefix_logit_margin_weight", 0.0)
                            ) * prefix_scale,
                            logit_margin=float(
                                train_cfg.get("generated_prefix_logit_margin", 1.0)
                            ),
                            force_unlikelihood_only=bool(
                                train_cfg.get(
                                    "generated_prefix_counterfactual_unlikelihood_only",
                                    False,
                                )
                            ),
                        )
                    finite_loss_flags.append(torch.isfinite(prefix_loss.detach()))
                    prefix_loss.backward()
                    pending_generated_prefix_observation = {
                        "optimizer_step": step + 1,
                        "loss": prefix_loss.detach(),
                        "curriculum_scale": float(prefix_scale),
                        "metrics": {
                            name: value.detach()
                            for name, value in prefix_metrics.items()
                        },
                    }
                    if metrics_due:
                        objective_values["generated_prefix"] = float(
                            prefix_loss.detach().float().cpu()
                        )
                        diagnostic_values.update({
                            f"generated_prefix_{name}": float(value.float().cpu())
                            for name, value in prefix_metrics.items()
                        })
                    del prefix_loss, prefix_metrics, prefix_batch
                elif metrics_due:
                    objective_values["generated_prefix"] = 0.0
                    diagnostic_values["generated_prefix_zero_weight_skipped"] = 1.0
                elif prefix_scale <= 0.0:
                    pending_generated_prefix_observation = {
                        "optimizer_step": step + 1,
                        "loss": None,
                        "curriculum_scale": 0.0,
                        "metrics": {"zero_weight_skipped": None},
                    }
                if metrics_due:
                    diagnostic_values["generated_prefix_curriculum_scale"] = float(prefix_scale)
                del host_prefix_batch
            if finite_loss_flags and not bool(torch.stack(finite_loss_flags).all()):
                raise FloatingPointError("non-finite FDT v4 curriculum loss")
            base_grad_norm = torch.nn.utils.clip_grad_norm_(
                base_trainable,
                float(train_cfg.get("grad_clip", 0.7)),
            ) if base_trainable else torch.zeros((), device=device)
            exact_grad_norm = torch.nn.utils.clip_grad_norm_(
                exact_trainable,
                float(train_cfg.get("exact_pointer_grad_clip", 1.0)),
            ) if exact_trainable else torch.zeros((), device=device)
            if not gradient_norms_are_finite(base_grad_norm, exact_grad_norm):
                raise FloatingPointError("non-finite FDT v4 gradient")
            optimizer.step()
            tokens_seen += batch_tokens
            step += 1
            source_states = {name: pool.state() for name, pool in pools.items()}
            if token_planner is not None:
                source_states["__token_planner__"] = token_planner.state()
            if validation_pool is not None:
                source_states["__validation__"] = validation_pool.state()
            checkpoint_consistent = True
            total_loss = sum(objective_values.values()) if metrics_due else None
            if total_loss is not None:
                settings.setdefault("overfit_train_reference", total_loss)
            if step % log_every == 0:
                additional = tokens_seen - start_tokens
                session_tokens = tokens_seen - session_start_tokens
                now = time.perf_counter()
                session_elapsed = max(now - started, 1e-9)
                tps = session_tokens / session_elapsed
                interval_tokens = tokens_seen - last_log_tokens
                interval_elapsed = max(now - last_log_time, 1e-9)
                interval_tps = interval_tokens / interval_elapsed
                recent_generated_prefix = None
                if pending_generated_prefix_observation is not None:
                    pending = pending_generated_prefix_observation
                    pending_loss = pending["loss"]
                    recent_generated_prefix = {
                        "optimizer_step": int(pending["optimizer_step"]),
                        "loss": (
                            float(pending_loss.float().cpu())
                            if isinstance(pending_loss, torch.Tensor)
                            else None
                        ),
                        "curriculum_scale": float(pending["curriculum_scale"]),
                        "metrics": {
                            name: (
                                float(value.float().cpu())
                                if isinstance(value, torch.Tensor)
                                else value
                            )
                            for name, value in pending["metrics"].items()
                        },
                    }
                    pending_generated_prefix_observation = None
                append_jsonl(log_path, {"event": "train", "step": step, "tokens_seen": tokens_seen, "additional_tokens": additional, "target_additional_tokens": target_tokens, "effective_tokens": batch_tokens, "base_forward_groups": len(forward_groups), "effective_samples_per_step": accumulation, "loss": total_loss, "finite_loss": math.isfinite(total_loss), "entropy_normalized": entropy_value, "dead_anchor_fraction": dead_anchor_value, "top1_membership": top1_membership_value, "tokens_per_sec": tps, "throughput_contract": "session_local_v2", "session_tokens": session_tokens, "interval_tokens_per_sec": interval_tps, "interval_throughput_contract": "wall_clock_between_train_logs_v1", "recent_generated_prefix": recent_generated_prefix, "eta_seconds": max(target_tokens - additional, 0) / max(tps, 1e-9), "objective_losses": objective_values, "diagnostic_metrics": diagnostic_values, "base_grad_norm": float(base_grad_norm.detach().cpu()), "exact_pointer_grad_norm": float(exact_grad_norm.detach().cpu()), "free_gib": disk_free_gib()})
                last_log_time = now
                last_log_tokens = tokens_seen
            if validation_pool is not None and validation_every > 0 and step % validation_every == 0:
                checkpoint_consistent = False
                scheduled_controls = transition_control_state(model)
                validation_loss = validation_snapshot(
                    model,
                    validation_pool,
                    config,
                    device,
                    float(train_cfg.get("eos_loss_weight", 2.0)),
                    batches=int(train_cfg.get("validation_batches", 8)),
                    fixed_state=validation_fixed_state,
                )
                gate_validation_loss = validation_loss
                gate_control = str(
                    train_cfg.get("overfit_monitor_controls", "scheduled")
                )
                if gate_control == "legacy":
                    model.set_transition_alpha(
                        float(train_cfg.get("rope_transition_alpha_start", 0.0))
                    )
                    model.set_legacy_position_scale(
                        float(train_cfg.get("legacy_position_scale_start", 1.0))
                    )
                    model.set_anchor_recency_bias(
                        float(train_cfg.get("anchor_recency_bias_start", 4.0))
                    )
                    model.config.routing_logit_quantization = float(
                        train_cfg.get("routing_logit_quantization_start", 0.0)
                    )
                    gate_validation_loss = validation_snapshot(
                        model,
                        validation_pool,
                        config,
                        device,
                        float(train_cfg.get("eos_loss_weight", 2.0)),
                        batches=int(train_cfg.get("validation_batches", 8)),
                        fixed_state=validation_fixed_state,
                    )
                    restore_transition_control_state(model, scheduled_controls)
                elif gate_control != "scheduled":
                    raise ValueError(
                        "overfit_monitor_controls must be scheduled or legacy"
                    )
                source_states["__validation__"] = validation_pool.state()
                append_jsonl(log_path, {"event": "validation", "step": step, "validation_loss": validation_loss, "gate_validation_loss": gate_validation_loss, "gate_control": gate_control, "transition_controls": scheduled_controls, "finite_loss": math.isfinite(validation_loss) and math.isfinite(gate_validation_loss), "validation_batches": int(train_cfg.get("validation_batches", 8)), "validation_contract": gate_version})
                previous = settings.setdefault("validation_history", [])
                previous.append(gate_validation_loss)
                del previous[:-int(train_cfg.get("overfit_validation_checks", 3))]
                train_improved = total_loss <= float(settings.get("overfit_train_reference", total_loss)) * (1.0 - float(train_cfg.get("overfit_min_train_improvement", 0.005)))
                regression = 1.0 + float(train_cfg.get("overfit_relative_regression", 0.01))
                consecutive_regression = len(previous) >= int(train_cfg.get("overfit_validation_checks", 3)) and all(
                    right > left * regression for left, right in zip(previous, previous[1:])
                )
                overfit_triggered = (
                    bool(train_cfg.get("stop_on_overfit", True))
                    and consecutive_regression
                    and train_improved
                )
                baseline = float(train_cfg.get("transition_parent_validation_loss", 0.0))
                max_transition_regression = float(
                    train_cfg.get("transition_max_relative_regression", -1.0)
                )
                transition_regression_triggered = (
                    baseline > 0.0
                    and max_transition_regression >= 0.0
                    and validation_loss
                    > baseline * (1.0 + max_transition_regression)
                )
                validation_nonfinite_triggered = not (
                    math.isfinite(validation_loss)
                    and math.isfinite(gate_validation_loss)
                )
                settings.update({
                    "source_states": source_states,
                    "validation_history": previous,
                    "last_validation_loss": validation_loss,
                    "last_gate_validation_loss": gate_validation_loss,
                    "last_validation_step": step,
                    "overfit_triggered": overfit_triggered,
                    "transition_regression_triggered": transition_regression_triggered,
                    "validation_nonfinite_triggered": validation_nonfinite_triggered,
                })
                checkpoint_consistent = True
                if (
                    overfit_triggered
                    or transition_regression_triggered
                    or validation_nonfinite_triggered
                ):
                    status = "SAFETY_STOP"
                    gate_event = (
                        "validation_nonfinite_gate"
                        if validation_nonfinite_triggered
                        else "overfit_gate"
                        if overfit_triggered
                        else "transition_regression_gate"
                    )
                    append_jsonl(log_path, {"event": gate_event, "step": step, "validation_history": previous, "scheduled_validation_loss": validation_loss, "gate_validation_loss": gate_validation_loss, "baseline": baseline, "max_transition_regression": max_transition_regression, "action": "atomic_stop"})
                    break
            recovery_every = int(train_cfg.get("recovery_every_steps", 250))
            initial_recovery_step = int(
                train_cfg.get("initial_recovery_step", recovery_every)
            )
            reached_milestone = tokens_seen - start_tokens >= next_milestone
            if (
                step == initial_recovery_step
                or step % recovery_every == 0
                or reached_milestone
            ):
                milestone_value = next_milestone
                if reached_milestone:
                    pending_next_milestone = next_milestone + int(
                        train_cfg.get("model_checkpoint_every_tokens", 100_000_000)
                    )
                    milestone_settings = {
                        **settings,
                        "optimizer_step": step,
                        "tokens_seen": tokens_seen,
                        "source_states": source_states,
                        "next_milestone_tokens": pending_next_milestone,
                    }
                    save_immutable_milestone(
                        output_dir / f"milestone_{milestone_value:012d}_tokens.pt",
                        model,
                        optimizer,
                        config,
                        milestone_settings,
                        "RUNNING",
                        step,
                        tokens_seen,
                        source_states,
                    )
                    next_milestone = pending_next_milestone
                settings.update({"optimizer_step": step, "tokens_seen": tokens_seen, "source_states": source_states, "next_milestone_tokens": next_milestone})
                save_checkpoints(output_dir, model, optimizer, config, settings, "RUNNING", step, tokens_seen, source_states)
                atomic_json(output_dir / "run_manifest.json", settings)
        if status == "RUNNING":
            apply_architecture_transition(
                model,
                train_cfg,
                max(tokens_seen - start_tokens, target_tokens),
            )
            status = "COMPLETE"
    except KeyboardInterrupt:
        status = "PAUSED"
        settings["failure"] = "keyboard_interrupt"
        raise
    except Exception:
        status = "SAFETY_STOP"
        settings["failure"] = "exception_or_interruption"
        raise
    finally:
        settings.update({"optimizer_step": step, "tokens_seen": tokens_seen, "source_states": source_states, "stage_status": status, "next_milestone_tokens": next_milestone})
        model_hash = recovery_hash = None
        if checkpoint_consistent:
            model_hash, recovery_hash = save_checkpoints(output_dir, model, optimizer, config, settings, status, step, tokens_seen, source_states)
        else:
            settings["stage_status"] = "INTERRUPTED_UNCHECKPOINTED_STEP"
            atomic_json(
                output_dir / "interrupted_uncheckpointed_step.json",
                {
                    "reason": "optimizer/data cursor transaction did not reach a committed boundary",
                    "last_committed_step": step,
                    "last_committed_tokens": tokens_seen,
                    "recovery_policy": "retain the previous latest_recovery.pt without overwrite",
                },
            )
        append_jsonl(log_path, {"event": "final", "stage_status": settings["stage_status"], "optimizer_step": step, "tokens_seen": tokens_seen, "additional_tokens": tokens_seen - start_tokens, "target_additional_tokens": target_tokens, "model_sha256": model_hash, "recovery_sha256": recovery_hash, "free_gib": disk_free_gib()})
        atomic_json(output_dir / "run_manifest.json", settings)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDT v4 main 426M curriculum trainer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
