from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig, load_yaml_like
from fdt_rlm.models import build_model
from fdt_rlm.models.causal_lm import RMSNorm
from fdt_rlm.tokenization import load_tokenizer


NEXT_MODEL_TYPES = (
    "transformer",
    "fdt",
    "fdt_optimized",
    "fdt_hybrid",
    "fdt_anchor_mixer",
    "fdt_v3",
    "fdt_v4",
)


def enable_optimized_fp32_runtime(model: torch.nn.Module) -> int:
    """Enable mathematically equivalent FP32 RMSNorm kernels for interactive inference."""
    count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.runtime_fused = True
            count += 1
    return count


def choose_onk_cache(
    config: ModelConfig,
    prompt_len: int,
    precision: str = "fp32",
    memory_priority: bool = False,
) -> bool:
    """Select a true incremental cache when the model exposes one."""
    if config.model_type == "transformer":
        return True
    pure_anchor = config.model_type in {"fdt", "fdt_optimized", "fdt_anchor_mixer", "fdt_v3", "fdt_v4"} and not config.use_self_attention
    if not pure_anchor or not config.use_incremental_anchor_state:
        return False
    if memory_priority:
        return True
    long_context_threshold = min(256, max(config.max_seq_len // 2, 1))
    return precision == "bf16" and prompt_len >= long_context_threshold


def safe_torch_load(path: str | Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def flatten_next_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("data", "model", "fdt", "train", "evaluation", "benchmark"):
        value = payload.get(section, {})
        if isinstance(value, Mapping):
            result.update(value)
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            result[key] = value
    return result


def load_next_config(path: str | Path) -> dict[str, Any]:
    return flatten_next_config(load_yaml_like(path))


def make_model_config(
    values: Mapping[str, Any],
    vocab_size: int,
    pad_token_id: int,
    eos_token_id: int,
) -> ModelConfig:
    allowed = {field.name for field in fields(ModelConfig)}
    payload = {key: value for key, value in values.items() if key in allowed}
    payload.update(
        vocab_size=int(vocab_size),
        pad_token_id=int(pad_token_id),
        eos_token_id=int(eos_token_id),
    )
    return ModelConfig(**payload)


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    device: str | torch.device,
    model_type: Optional[str] = None,
    tokenizer_name: str = "",
):
    checkpoint = safe_torch_load(checkpoint_path, device)
    config_values = dict(checkpoint["model_config"])
    if model_type:
        config_values["model_type"] = model_type
    config = ModelConfig(**config_values)
    tokenizer_path = tokenizer_name or checkpoint.get("train_config", {}).get("tokenizer_name")
    if not tokenizer_path and checkpoint.get("base_checkpoint"):
        base_path = Path(checkpoint["base_checkpoint"])
        if not base_path.is_absolute():
            base_path = Path(checkpoint_path).resolve().parent / base_path
        if base_path.exists():
            base_checkpoint = safe_torch_load(base_path)
            tokenizer_path = base_checkpoint.get("train_config", {}).get("tokenizer_name")
    tokenizer_path = tokenizer_path or "gpt2"
    tokenizer = load_tokenizer(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary "
            f"({config.vocab_size}); tokenizer={tokenizer_path!s}"
        )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, config, checkpoint


def filter_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0 < top_p < 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    return logits.scatter(-1, sorted_indices, sorted_logits.masked_fill(remove, -float("inf")))


def apply_repetition_penalty_(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply the standard sign-aware token penalty in place.

    The caller chooses the history scope. Exact-copy cursor logits should not
    pass through this function because repeated source tokens may be correct.
    """
    if penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if penalty == 1.0 or token_ids.numel() == 0:
        return logits
    if logits.ndim != 2 or token_ids.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("logits and token_ids must have matching batch dimensions")
    for batch_index in range(logits.size(0)):
        seen = torch.unique(token_ids[batch_index])
        seen_logits = logits[batch_index, seen]
        logits[batch_index, seen] = torch.where(
            seen_logits < 0,
            seen_logits * penalty,
            seen_logits / penalty,
        )
    return logits


def apply_ngram_loop_penalty_(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_order: int = 3,
    penalty: float = 8.0,
    window: int | None = None,
    hard_block_after: int = 0,
) -> torch.Tensor:
    """Suppress tokens that would close repeated generated-only n-grams.

    Repeated closures accumulate in logit space. A positive hard-block threshold
    turns sufficiently persistent loops into an impossible transition. Callers
    must bypass this helper while an exact-copy cursor is active.
    """
    order = max(int(ngram_order), 2)
    if generated_ids.size(1) < order - 1 or penalty <= 0:
        return logits
    for batch_index in range(logits.size(0)):
        sequence = generated_ids[batch_index].tolist()
        if window is not None and int(window) > 0:
            sequence = sequence[-max(int(window), order) :]
        prefix = tuple(sequence[-(order - 1) :])
        closure_counts: dict[int, int] = {}
        for start in range(0, len(sequence) - order + 1):
            if tuple(sequence[start : start + order - 1]) == prefix:
                token_id = int(sequence[start + order - 1])
                closure_counts[token_id] = closure_counts.get(token_id, 0) + 1
        for token_id, count in closure_counts.items():
            if int(hard_block_after) > 0 and count >= int(hard_block_after):
                logits[batch_index, token_id] = -float("inf")
            else:
                logits[batch_index, token_id] -= float(penalty) * count
    return logits


@torch.no_grad()
def generate_ids(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    repetition_scope: str = "generated",
    greedy: bool = False,
    use_cache: bool = True,
    ngram_loop_penalty: float = 0.0,
) -> torch.Tensor:
    if repetition_scope not in {"all", "generated"}:
        raise ValueError("repetition_scope must be 'all' or 'generated'")
    prompt_length = input_ids.size(1)
    max_length = min(model.config.max_seq_len, input_ids.size(1) + max_new_tokens)
    generated = input_ids.new_empty((input_ids.size(0), max_length))
    generated[:, : input_ids.size(1)].copy_(input_ids)
    cursor = input_ids.size(1)
    output = None
    cache = None
    if use_cache:
        output, cache = model.prefill(input_ids, torch.ones_like(input_ids))
    for _ in range(max_new_tokens):
        if cursor >= max_length:
            break
        if use_cache:
            logits = output["logits"][:, -1].float()
        else:
            current = generated[:, :cursor]
            logits = model(current, attention_mask=torch.ones_like(current))["logits"][:, -1].float()
        ngram_penalty = float(ngram_loop_penalty)
        # Prefill/decode may return an inference tensor. Clone only when an
        # in-place generated-history transform is requested.
        if repetition_penalty > 1.0 or ngram_penalty > 0.0:
            logits = logits.clone()
            history_start = 0 if repetition_scope == "all" else prompt_length
            history = generated[:, history_start:cursor]
            if repetition_penalty > 1.0:
                apply_repetition_penalty_(logits, history, repetition_penalty)
            if ngram_penalty > 0.0:
                apply_ngram_loop_penalty_(
                    logits,
                    history,
                    ngram_order=int(getattr(model.config, "generation_ngram_order", 3)),
                    penalty=ngram_penalty,
                    window=int(getattr(model.config, "generation_ngram_window", 96)),
                    hard_block_after=int(
                        getattr(model.config, "generation_ngram_hard_block_after", 2)
                    ),
                )
        if 0 < top_k < logits.size(-1):
            cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, -float("inf"))
        if greedy or temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = filter_logits(logits / max(temperature, 1e-5), top_p)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        generated[:, cursor].copy_(next_id.squeeze(1))
        cursor += 1
        if bool((next_id == model.config.eos_token_id).all()):
            break
        if use_cache:
            output, cache = model.decode_step(next_id, cache)
    return generated[:, :cursor]
