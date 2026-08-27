from __future__ import annotations

import argparse
import ctypes
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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
        if not bool(negative_mask.any()):
            raise ValueError(f"generated-prefix shard has no loop-negative positions: {path}")
        if not bool((negative_ids.masked_select(negative_mask) >= 0).all()):
            raise ValueError(f"generated-prefix loop-negative token ids must be nonnegative: {path}")
        if bool(labels.masked_select(negative_mask).eq(-100).any()):
            raise ValueError(f"generated-prefix negatives require supervised recovery labels: {path}")
        if bool(
            negative_ids.masked_select(negative_mask).eq(
                labels.masked_select(negative_mask)
            ).any()
        ):
            raise ValueError(f"loop-negative ids must differ from clean labels: {path}")
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
        files = shard_paths(path, split)
        exact = name == "exact_copy"
        generated_prefix = name == "generated_prefix"
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
        self.epoch = 0
        self.shard_index = 0
        self.cursor = 0
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
        return {"epoch": self.epoch, "shard_index": self.shard_index, "cursor": self.cursor, "seed": self.seed}

    def restore(self, state: dict[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("Dataset seed mismatch during resume")
        self.epoch = int(state.get("epoch", 0))
        self.shard_index = int(state.get("shard_index", 0))
        self.cursor = int(state.get("cursor", 0))
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
):
    """Compute LM loss without retaining a full 16K-by-vocabulary tensor."""
    chunk_size = max(int(sequence_chunk_size), 1)
    numerator = hidden.new_zeros((), dtype=torch.float32)
    denominator = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, max(hidden.size(1) - 1, 0), chunk_size):
        stop = min(start + chunk_size, hidden.size(1) - 1)
        target = labels[:, start + 1 : stop + 1]
        target_attention = attention_mask[:, start + 1 : stop + 1]

        def chunk_terms(hidden_chunk, target_chunk, attention_chunk):
            logits = model.lm_head(hidden_chunk).clamp(
                -model.config.lm_logit_clip,
                model.config.lm_logit_clip,
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
            return (losses * weighted_valid).sum(), weighted_valid.sum()

        chunk_numerator, chunk_denominator = checkpoint(
            chunk_terms,
            hidden[:, start:stop],
            target,
            target_attention,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        numerator = numerator + chunk_numerator
        denominator = denominator + chunk_denominator
    return numerator / denominator.clamp_min(1.0)


def move_batch(payload: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.long) for key, value in payload.items()}


def exact_copy_objective(
    model,
    batch: dict[str, torch.Tensor],
    weight: float,
    *,
    detach_hidden: bool = True,
):
    required = ("input_ids", "labels", "attention_mask", "prompt_mask", "source_boundary", "copy_source_positions", "copy_target_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"exact batch lacks explicit fields: {missing}")
    if not bool(batch["prompt_mask"].bool().any()) or not bool(batch["copy_target_mask"].bool().any()):
        raise ValueError("exact batch cannot use all-token labels; prompt labels must include -100")
    output = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_logits=False,
    )
    hidden = output["hidden"].detach() if detach_hidden else output["hidden"]
    result = model.exact_memory_loss(
        hidden,
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        copy_source_positions=batch["copy_source_positions"],
        copy_target_mask=batch["copy_target_mask"],
        source_boundary=batch["source_boundary"],
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
    output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    recovery = weighted_lm_loss(
        output["logits"],
        batch["labels"],
        batch["attention_mask"],
        pad_token_id,
        eos_token_id,
        eos_weight,
    )
    shifted_logits = output["logits"][:, :-1].float()
    negative_ids = batch["loop_negative_ids"][:, 1:].long()
    negative_mask = (
        batch["loop_negative_mask"][:, 1:].bool()
        & batch["attention_mask"][:, 1:].bool()
        & batch["labels"][:, 1:].ne(-100)
    )
    if not bool(negative_mask.any()):
        raise ValueError("generated-prefix batch has no supervised loop-negative positions")
    selected_negative_ids = negative_ids.masked_select(negative_mask)
    if bool((selected_negative_ids < 0).any()) or bool(
        (selected_negative_ids >= shifted_logits.size(-1)).any()
    ):
        raise ValueError("generated-prefix loop-negative token id is outside the vocabulary")
    if bool(
        selected_negative_ids.eq(
            batch["labels"][:, 1:].masked_select(negative_mask)
        ).any()
    ):
        raise ValueError("loop-negative token id conflicts with its clean target")
    safe_negative_ids = negative_ids.masked_fill(~negative_mask, 0)
    negative_probs = torch.softmax(shifted_logits, dim=-1).gather(
        -1,
        safe_negative_ids.unsqueeze(-1),
    ).squeeze(-1)
    unlikelihood = -torch.log1p(-negative_probs.clamp(max=1.0 - 1e-6))
    unlikelihood = unlikelihood.masked_select(negative_mask).mean()
    total = float(recovery_weight) * recovery + float(unlikelihood_weight) * unlikelihood
    return total, {
        "recovery_lm": recovery.detach(),
        "loop_unlikelihood": unlikelihood.detach(),
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
    if model.exact_pointer is not None and not exact_parameters:
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
    payload: dict[str, Any] = {"model_type": config.model_type, "model_config": asdict(config), "train_config": settings, "stage_status": status, "optimizer_step": int(step), "tokens_seen": int(tokens_seen), "source_states": source_states, "optimizer_state_included": bool(include_optimizer), "model_state_dict": model.state_dict()}
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
    atomic_torch_save(output_dir / "latest_recovery.pt", checkpoint_payload(model, optimizer, config, settings, status, step, tokens_seen, source_states, True))
    atomic_torch_save(output_dir / "latest.pt", model_payload)
    return sha256_file(output_dir / "latest.pt"), sha256_file(output_dir / "latest_recovery.pt")


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
    return payload


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
    anchor_keys = [name for name in target if ".anchor." in name]
    if not anchor_keys or not set(anchor_keys).issubset(converted):
        raise ValueError("Warm-start refused: inherited anchor transfer was incomplete")
    return {"schema_version": "fdt_v4_v20_conversion_v1", "source_model_type": source_config.get("model_type"), "allowlist": list(WARM_START_ALLOWLIST), "denylist": list(WARM_START_DENYLIST), "converted_keys": converted, "skipped_new_or_denied_keys": skipped, "shape_mismatches": mismatched, "anchor_transfer_verified": True, "new_random_components": [name for name in target if name not in converted], "rope_and_exact_memory_initialized_new": True}


def model_config_and_settings(config_path: Path, output_dir: Path):
    raw = load_yaml_like(config_path)
    config = model_config_from_yaml(raw)
    validate_main_architecture(config)
    return config, {**dict(raw.get("train", {})), "output_dir": str(output_dir), "stage": raw.get("stage", "fdt_v4_main_426m_curriculum")}, dict(raw.get("data", {})), raw


def routing_diagnostics(output: dict[str, Any]) -> dict[str, float | None]:
    stats = output.get("anchor_stats") or []
    entropy_values = []
    dead_values = []
    top1_values = []
    active_anchor_masks = []
    for item in stats:
        if isinstance(item, dict) and isinstance(item.get("entropy_normalized"), torch.Tensor):
            entropy_values.append(float(item["entropy_normalized"].detach().float().cpu()))
            if isinstance(item.get("dead_anchor_fraction"), torch.Tensor):
                dead_values.append(float(item["dead_anchor_fraction"].detach().float().cpu()))
            if isinstance(item.get("active_anchor_mask"), torch.Tensor):
                active_anchor_masks.append(item["active_anchor_mask"].detach().bool().cpu())
            if isinstance(item.get("top1_membership"), torch.Tensor):
                top1_values.append(float(item["top1_membership"].detach().float().cpu()))
        elif hasattr(item, "entropy") and isinstance(item.entropy, torch.Tensor):
            entropy_values.append(
                float(item.entropy.detach().float().cpu())
                / math.log(max(int(getattr(item, "membership", torch.empty(0)).size(-1)), 2))
            )
            if isinstance(getattr(item, "load_prob", None), torch.Tensor):
                dead_values.append(float(item.load_prob.detach().le(0).float().mean().cpu()))
            if isinstance(getattr(item, "top1_membership", None), torch.Tensor):
                top1_values.append(float(item.top1_membership.detach().float().cpu()))
        elif isinstance(item, dict) and isinstance(item.get("membership"), torch.Tensor):
            membership = item["membership"].detach().float()
            probs = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy_values.append((-(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(max(probs.size(-1), 2))).mean().item())
    return {
        "entropy_normalized": (
            sum(entropy_values) / len(entropy_values) if entropy_values else None
        ),
        "dead_anchor_fraction": max(dead_values) if dead_values else None,
        "top1_membership": (
            sum(top1_values) / len(top1_values) if top1_values else None
        ),
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
) -> float | None:
    if not masks:
        return None
    return max(float((~mask).float().mean()) for mask in masks)


def routing_entropy(output: dict[str, Any]) -> float | None:
    return routing_diagnostics(output)["entropy_normalized"]


def validation_snapshot(model, pool: RowPool, config: ModelConfig, device: torch.device, eos_weight: float) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = move_batch(pool.next_batch(1), device)
        output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        value = weighted_lm_loss(output["logits"], batch["labels"], batch["attention_mask"], config.pad_token_id, config.eos_token_id, eos_weight)
    if was_training:
        model.train()
    return float(value.detach().float().cpu())


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
    seed_everything(seed)
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
    require_generated_prefix = bool(train_cfg.get("require_generated_prefix_recovery", False))
    if require_generated_prefix and (not generated_prefix_path or generated_prefix_weight <= 0.0):
        raise ValueError(
            "required generated-prefix recovery needs a dataset path and positive weight"
        )
    if generated_prefix_path:
        dataset_dirs["generated_prefix"] = ROOT / data_cfg["generated_prefix_dir"]
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
    if resume_payload is not None:
        if str(resume_payload["train_config"].get("config_sha256", "")).upper() != sha256_file(config_path) or resume_payload["model_config"] != asdict(config):
            raise ValueError("Resume config/model configuration mismatch")
    model = build_model(config)
    warm_started = False
    if parent is not None and resume_payload is None:
        parent_payload = load_payload(parent)
        conversion = convert_v20_state_dict(model, parent_payload)
        conversion.update({"source_checkpoint": str(parent), "source_checkpoint_sha256": sha256_file(parent)})
        atomic_json(output_dir / "v20_conversion_manifest.json", conversion)
        warm_started = True
    elif resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        warm_started = bool(resume_payload.get("train_config", {}).get("warm_started_from_v20", False))
    if warm_started:
        for name, parameter in model.named_parameters():
            if ".anchor." in name or ".anchor_norm." in name:
                parameter.requires_grad_(False)
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
    validation_pool = None
    if validation_path is not None:
        validation_pool = RowPool(validation_path, data_cfg.get("validation_split", "validation"), seed + 7001)
    source_states = {name: pool.state() for name, pool in pools.items()}
    if validation_pool is not None:
        source_states["__validation__"] = validation_pool.state()
    step = tokens_seen = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for name, state in resume_payload.get("source_states", {}).items():
            if name in pools:
                pools[name].restore(state)
        if validation_pool is not None:
            if "__validation__" not in resume_payload.get("source_states", {}):
                raise ValueError("Resume checkpoint lacks validation cursor state")
            validation_pool.restore(resume_payload["source_states"]["__validation__"])
        source_states = {name: pool.state() for name, pool in pools.items()}
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
    settings = {**train_cfg, "model_config": asdict(config), "config_path": str(config_path), "config_sha256": sha256_file(config_path), "tokenizer_dir": str(tokenizer_dir), "tokenizer_json": str(tokenizer_path), "tokenizer_json_sha256": tokenizer_sha, "dataset_dirs": {name: str(path) for name, path in audited_dataset_dirs.items()}, "dataset_manifest_sha256": fingerprints, "data_preflight": data_preflight, "exact_copy_supervision": "explicit_source_token_mapping_and_target_mask", "generic_all_token_pointer_loss": False, "fp32_optimizer_master": True, "activation_checkpointing": bool(train_cfg.get("activation_checkpointing", True)), "output_dir": str(output_dir), "run_owner": "fdt_v4_curriculum", "run_start_tokens": run_start_tokens, "warm_started_from_v20": warm_started}
    for durable_key in (
        "overfit_train_reference",
        "validation_history",
        "next_milestone_tokens",
        "overfit_triggered",
        "last_validation_loss",
        "last_validation_step",
    ):
        if durable_key in resume_settings:
            settings[durable_key] = resume_settings[durable_key]
    atomic_json(output_dir / "run_manifest.json", settings)
    log_path = output_dir / "training_log.jsonl"
    append_jsonl(log_path, {"event": "start" if resume_payload is None else "resume", "step": step, "tokens_seen": tokens_seen, **settings})
    stop_path = output_dir / str(raw.get("safety", {}).get("stop_file", "STOP_REQUESTED"))
    target_tokens = int(train_cfg.get("target_additional_tokens", 1_000_000_000))
    start_tokens = run_start_tokens
    configured_fractions = train_cfg.get("source_batch_fractions") or {
        "natural": float(train_cfg.get("natural_batch_fraction", 0.60)),
        "factual": float(train_cfg.get("factual_batch_fraction", 0.40)),
    }
    source_fractions = {
        str(name): float(value) for name, value in configured_fractions.items()
    }
    missing_sources = set(source_fractions) - set(pools)
    if missing_sources:
        raise ValueError(f"source cycle references missing datasets: {sorted(missing_sources)}")
    source_cycle = deterministic_source_cycle(source_fractions, slots=100)
    next_milestone = int(resume_payload["train_config"].get("next_milestone_tokens", train_cfg.get("model_checkpoint_every_tokens", 100_000_000))) if resume_payload is not None else int(train_cfg.get("model_checkpoint_every_tokens", 100_000_000))
    started, status = time.perf_counter(), "RUNNING"
    checkpoint_consistent = True
    autocast_enabled = device.type == "cuda"
    model.train()
    routing_window_masks: list[torch.Tensor] | None = None
    routing_window_steps = max(
        int(train_cfg.get("routing_safety_window_steps", 25)), 1
    )
    try:
        while tokens_seen - start_tokens < target_tokens:
            if disk_free_gib() < float(train_cfg.get("min_free_gib", 10.0)):
                status = "SAFETY_STOP"
                break
            if stop_path.exists():
                status = "PAUSED"
                break
            checkpoint_consistent = False
            optimizer.zero_grad(set_to_none=True)
            accumulation = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
            objective_values, diagnostic_values, batch_tokens = {}, {}, 0
            entropy_value = top1_membership_value = None
            dead_anchor_value = 0.0
            optimizer_step_masks: list[torch.Tensor] | None = None
            routing_diagnostics_seen = 0
            for micro_index in range(accumulation):
                source_name = source_cycle[(step * accumulation + micro_index) % len(source_cycle)]
                batch = move_batch(pools[source_name].next_batch(int(train_cfg.get("batch_size", 1))), device)
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
                    ) * float(train_cfg.get("base_lm_weight", 1.0)) / accumulation}
                    route_metrics = routing_diagnostics(output)
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
                    if step % int(train_cfg.get("log_every", 10)) == 0:
                        entropy_value = route_metrics["entropy_normalized"]
                        top1_membership_value = route_metrics["top1_membership"]
                    batch_tokens += int(batch["attention_mask"].sum().item())
                    micro_total = sum(micro_losses.values())
                if not torch.isfinite(micro_total):
                    raise FloatingPointError("non-finite FDT v4 curriculum loss")
                micro_total.backward()
                for name, value in micro_losses.items():
                    objective_values[name] = objective_values.get(name, 0.0) + float(value.detach().float().cpu())
            del output, micro_losses, micro_total, batch
            if bool(train_cfg.get("require_routing_diagnostics", True)) and (
                routing_diagnostics_seen != accumulation
            ):
                raise RuntimeError("dead-anchor diagnostics were incomplete for the optimizer step")
            routing_window_masks = merge_active_anchor_masks(
                routing_window_masks,
                optimizer_step_masks or [],
            )
            current_window_dead = dead_anchor_fraction_from_masks(routing_window_masks)
            if current_window_dead is None:
                raise RuntimeError("routing active-anchor masks are missing")
            dead_anchor_value = current_window_dead
            if (step + 1) % routing_window_steps == 0:
                if dead_anchor_value > float(train_cfg.get("dead_anchor_fraction_limit", 0.01)):
                    raise FloatingPointError(
                        f"{routing_window_steps}-step dead-anchor fraction "
                        f"{dead_anchor_value:.6f} exceeds safety limit"
                    )
                routing_window_masks = None
            # The exact batch runs only after every base-LM microbatch has
            # backpropagated. This prevents a second full-model graph from
            # overlapping the retained LM graph and reproducing prior OOM/TDR
            # failure conditions.
            if step % int(train_cfg.get("exact_batch_every_steps", 1)) == 0:
                exact_batch = move_batch(
                    pools["exact_copy"].next_batch(int(train_cfg.get("batch_size", 1))),
                    device,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    exact_loss, exact_result = exact_copy_objective(
                        model,
                        exact_batch,
                        float(train_cfg.get("exact_copy_weight", 0.10)),
                        detach_hidden=bool(train_cfg.get("detach_exact_hidden", True)),
                    )
                if not torch.isfinite(exact_loss):
                    raise FloatingPointError("non-finite FDT v4 exact-memory loss")
                exact_loss.backward()
                objective_values["exact_copy"] = float(exact_loss.detach().float().cpu())
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
                and generated_prefix_weight > 0.0
                and step >= int(train_cfg.get("generated_prefix_min_step", 0))
                and step % int(train_cfg.get("generated_prefix_every_steps", 1)) == 0
            )
            if generated_prefix_due:
                prefix_batch = move_batch(
                    pools["generated_prefix"].next_batch(int(train_cfg.get("batch_size", 1))),
                    device,
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
                        recovery_weight=generated_prefix_weight,
                        unlikelihood_weight=float(
                            train_cfg.get("generated_prefix_unlikelihood_weight", 0.01)
                        ),
                    )
                if not torch.isfinite(prefix_loss):
                    raise FloatingPointError("non-finite generated-prefix recovery loss")
                prefix_loss.backward()
                objective_values["generated_prefix"] = float(
                    prefix_loss.detach().float().cpu()
                )
                diagnostic_values.update({
                    f"generated_prefix_{name}": float(value.float().cpu())
                    for name, value in prefix_metrics.items()
                })
                del prefix_loss, prefix_metrics, prefix_batch
            base_grad_norm = torch.nn.utils.clip_grad_norm_(
                base_trainable,
                float(train_cfg.get("grad_clip", 0.7)),
            ) if base_trainable else torch.zeros((), device=device)
            exact_grad_norm = torch.nn.utils.clip_grad_norm_(
                exact_trainable,
                float(train_cfg.get("exact_pointer_grad_clip", 1.0)),
            ) if exact_trainable else torch.zeros((), device=device)
            if not torch.isfinite(base_grad_norm) or not torch.isfinite(exact_grad_norm):
                raise FloatingPointError("non-finite FDT v4 gradient")
            optimizer.step()
            tokens_seen += batch_tokens
            step += 1
            source_states = {name: pool.state() for name, pool in pools.items()}
            if validation_pool is not None:
                source_states["__validation__"] = validation_pool.state()
            checkpoint_consistent = True
            total_loss = sum(objective_values.values())
            settings.setdefault("overfit_train_reference", total_loss)
            if step % int(train_cfg.get("log_every", 10)) == 0:
                additional = tokens_seen - start_tokens
                tps = additional / max(time.perf_counter() - started, 1e-9)
                append_jsonl(log_path, {"event": "train", "step": step, "tokens_seen": tokens_seen, "additional_tokens": additional, "target_additional_tokens": target_tokens, "effective_tokens": batch_tokens, "loss": total_loss, "finite_loss": math.isfinite(total_loss), "entropy_normalized": entropy_value, "dead_anchor_fraction": dead_anchor_value, "top1_membership": top1_membership_value, "tokens_per_sec": tps, "eta_seconds": max(target_tokens - additional, 0) / max(tps, 1e-9), "objective_losses": objective_values, "diagnostic_metrics": diagnostic_values, "base_grad_norm": float(base_grad_norm.detach().cpu()), "exact_pointer_grad_norm": float(exact_grad_norm.detach().cpu()), "free_gib": disk_free_gib()})
            validation_every = int(train_cfg.get("validation_every_steps", 0))
            if validation_pool is not None and validation_every > 0 and step % validation_every == 0:
                checkpoint_consistent = False
                validation_loss = validation_snapshot(model, validation_pool, config, device, float(train_cfg.get("eos_loss_weight", 2.0)))
                source_states["__validation__"] = validation_pool.state()
                append_jsonl(log_path, {"event": "validation", "step": step, "validation_loss": validation_loss, "finite_loss": math.isfinite(validation_loss)})
                previous = settings.setdefault("validation_history", [])
                previous.append(validation_loss)
                del previous[:-int(train_cfg.get("overfit_validation_checks", 3))]
                train_improved = total_loss <= float(settings.get("overfit_train_reference", total_loss)) * (1.0 - float(train_cfg.get("overfit_min_train_improvement", 0.005)))
                overfit_triggered = len(previous) >= int(train_cfg.get("overfit_validation_checks", 3)) and train_improved and previous[-1] > previous[0] * (1.0 + float(train_cfg.get("overfit_relative_regression", 0.01)))
                settings.update({
                    "source_states": source_states,
                    "validation_history": previous,
                    "last_validation_loss": validation_loss,
                    "last_validation_step": step,
                    "overfit_triggered": overfit_triggered,
                })
                checkpoint_consistent = True
                if overfit_triggered:
                    status = "SAFETY_STOP"
                    append_jsonl(log_path, {"event": "overfit_gate", "step": step, "validation_history": previous, "action": "atomic_stop"})
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
