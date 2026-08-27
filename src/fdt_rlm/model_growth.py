from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping

import torch

from fdt_rlm.config import ModelConfig


def expanded_fdt_v3_config(
    source: ModelConfig,
    *,
    target_dim: int,
    target_layers: int,
    target_heads: int,
    target_router_dim: int,
) -> ModelConfig:
    if source.model_type != "fdt_v3" or source.use_self_attention:
        raise ValueError("Only pure fdt_v3 checkpoints can be expanded")
    if target_dim < source.dim or target_layers < source.n_layers:
        raise ValueError("Expansion cannot shrink model width or depth")
    if target_dim % target_heads:
        raise ValueError("target_dim must be divisible by target_heads")
    if source.dim % source.n_heads:
        raise ValueError("Source attention head geometry is invalid")
    if target_dim // target_heads != source.dim // source.n_heads:
        raise ValueError("Head dimension must remain unchanged for function-preserving growth")
    if target_router_dim < source.router_dim:
        raise ValueError("Router dimension cannot shrink")

    old_anchor_layers = set(source.anchor_layer_indices)
    anchor_layers = sorted(
        old_anchor_layers
        | {
            index
            for index in range(source.n_layers, target_layers)
            if index % 2 == 0
        }
    )
    return replace(
        source,
        dim=target_dim,
        n_layers=target_layers,
        n_heads=target_heads,
        router_dim=target_router_dim,
        anchor_layer_indices=anchor_layers,
    )


def _copy_qkv(target: torch.Tensor, source: torch.Tensor, noise_std: float, generator) -> torch.Tensor:
    source_dim = source.shape[1]
    target_dim = target.shape[1]
    if source.shape[0] != 3 * source_dim or target.shape[0] != 3 * target_dim:
        raise ValueError("Unexpected qkv tensor geometry")
    result = torch.zeros_like(target)
    for part in range(3):
        source_part = source[part * source_dim : (part + 1) * source_dim]
        target_start = part * target_dim
        result[target_start : target_start + source_dim, :source_dim].copy_(source_part)
        if noise_std > 0 and target_dim > source_dim:
            extra = result[target_start + source_dim : target_start + target_dim]
            extra.normal_(mean=0.0, std=noise_std, generator=generator)
    return result


def _copy_matrix(target: torch.Tensor, source: torch.Tensor, noise_std: float, generator) -> torch.Tensor:
    if target.ndim != 2 or source.ndim != 2:
        raise ValueError("Matrix growth requires rank-2 tensors")
    rows = min(target.shape[0], source.shape[0])
    cols = min(target.shape[1], source.shape[1])
    result = torch.zeros_like(target)
    result[:rows, :cols].copy_(source[:rows, :cols])
    if noise_std > 0 and target.shape[0] > rows:
        result[rows:].normal_(mean=0.0, std=noise_std, generator=generator)
    if noise_std > 0 and target.shape[1] > cols and target.shape[0] == source.shape[0]:
        result[:, cols:].normal_(mean=0.0, std=noise_std, generator=generator)
    return result


def _copy_norm(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(target)
    copied = min(target.numel(), source.numel())
    scale = math.sqrt(source.numel() / target.numel())
    result[:copied].copy_(source[:copied] * scale)
    return result


def _grow_existing_tensor(
    name: str,
    target: torch.Tensor,
    source: torch.Tensor,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if target.shape == source.shape:
        return source.detach().clone()
    if name.endswith("local_attention.qkv.weight"):
        return _copy_qkv(target, source, noise_std, generator)
    if target.ndim == 1 and source.ndim == 1 and name.endswith("norm.weight"):
        return _copy_norm(target, source)
    if target.ndim == 2 and source.ndim == 2:
        return _copy_matrix(target, source, noise_std, generator)
    raise ValueError(f"No safe growth rule for {name}: {tuple(source.shape)} -> {tuple(target.shape)}")


def grow_fdt_v3_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_model: torch.nn.Module,
    *,
    source_layers: int,
    noise_std: float = 1e-4,
    seed: int = 20260803,
) -> dict[str, torch.Tensor]:
    """Grow width/depth while preserving the source path at initialization.

    Existing output channels are copied exactly. New channels receive a tiny
    symmetry-breaking initialization, while newly appended residual blocks are
    exact identities because their branch output projections start at zero.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target_state = target_model.state_dict()
    grown: dict[str, torch.Tensor] = {}

    for name, target in target_state.items():
        source = source_state.get(name)
        if source is not None:
            grown[name] = _grow_existing_tensor(
                name,
                target.detach().cpu(),
                source.detach().cpu(),
                noise_std,
                generator,
            )
            continue

        if not name.startswith("blocks."):
            raise ValueError(f"Unexpected new non-block tensor: {name}")
        layer = int(name.split(".", 2)[1])
        if layer < source_layers:
            raise ValueError(f"Source tensor missing inside an existing layer: {name}")
        grown[name] = target.detach().cpu().clone()

    for layer in range(source_layers, target_model.config.n_layers):
        for suffix in (
            "local_attention.out_proj.weight",
            "mlp.down.weight",
            "anchor.out_proj.weight",
        ):
            name = f"blocks.{layer}.{suffix}"
            if name in grown:
                grown[name].zero_()

    missing_target = sorted(set(source_state) - set(target_state))
    if missing_target:
        raise ValueError(f"Target model dropped source tensors: {missing_target}")
    return grown


def unique_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
