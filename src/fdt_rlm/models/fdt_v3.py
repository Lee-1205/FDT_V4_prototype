from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.causal_lm import (
    AnchorStats,
    BaseCausalLM,
    CausalFuzzyAnchorLayer,
    RMSNorm,
    masked_mean,
)


def _segmented_prefix_sum(
    values: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Prefix sum independent groups that are already contiguous."""
    if values.size(0) == 0:
        return values
    starts = torch.ones(groups.size(0), device=groups.device, dtype=torch.bool)
    starts[1:] = groups[1:] != groups[:-1]
    segment_ids = starts.long().cumsum(0) - 1
    accumulation_values = values if values.dtype == torch.float64 else values.float()
    prefix = accumulation_values.cumsum(0)
    bases = prefix[starts] - accumulation_values[starts]
    return prefix - bases.index_select(0, segment_ids)


def _deterministic_sparse_chunk_totals(
    chunk_indices: torch.Tensor,
    chunk_weights: torch.Tensor,
    chunk_values: torch.Tensor,
    num_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce each sparse chunk without colliding CUDA atomic additions."""
    bsz, chunk_count, chunk_size, top_k = chunk_indices.shape
    channels = chunk_weights.size(-1)
    dim = chunk_values.size(-1)
    lanes = bsz * chunk_count
    records = chunk_size * top_k
    flat_indices = chunk_indices.reshape(lanes, records)
    lane_ids = torch.arange(lanes, device=chunk_indices.device).view(lanes, 1)
    groups = lane_ids * int(num_anchors) + flat_indices
    positions = torch.arange(records, device=chunk_indices.device).view(1, records)
    order = torch.argsort((groups * max(records, 1) + positions).reshape(-1))
    sorted_groups = groups.reshape(-1).index_select(0, order)
    contributions = (
        chunk_weights.unsqueeze(-1)
        * chunk_values[:, :, :, None, None, :]
    ).reshape(-1, channels, dim)
    sorted_numerator = _segmented_prefix_sum(
        contributions.index_select(0, order), sorted_groups
    )
    sorted_mass = _segmented_prefix_sum(
        chunk_weights.reshape(-1, channels).index_select(0, order), sorted_groups
    )
    ends = torch.ones_like(sorted_groups, dtype=torch.bool)
    ends[:-1] = sorted_groups[:-1] != sorted_groups[1:]
    end_groups = sorted_groups[ends]
    flat_numerator = chunk_values.new_zeros(
        lanes * int(num_anchors), channels, dim, dtype=torch.float32
    )
    flat_mass = chunk_values.new_zeros(
        lanes * int(num_anchors), channels, dtype=torch.float32
    )
    flat_numerator.index_copy_(0, end_groups, sorted_numerator[ends])
    flat_mass.index_copy_(0, end_groups, sorted_mass[ends])
    return (
        flat_numerator.reshape(bsz, chunk_count, num_anchors, channels, dim),
        flat_mass.reshape(bsz, chunk_count, num_anchors, channels),
    )


def _deterministic_sparse_final_mass(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_anchors: int,
) -> torch.Tensor:
    """Accumulate the small inference cache mass in FP64 and token order."""
    bsz, seq_len, top_k = indices.shape
    records = seq_len * top_k
    flat_indices = indices.reshape(bsz, records)
    batch_ids = torch.arange(bsz, device=indices.device).view(bsz, 1)
    groups = batch_ids * int(num_anchors) + flat_indices
    positions = torch.arange(records, device=indices.device).view(1, records)
    order = torch.argsort((groups * max(records, 1) + positions).reshape(-1))
    sorted_groups = groups.reshape(-1).index_select(0, order)
    sorted_mass = _segmented_prefix_sum(
        weights.reshape(-1, weights.size(-1)).double().index_select(0, order),
        sorted_groups,
    )
    ends = torch.ones_like(sorted_groups, dtype=torch.bool)
    ends[:-1] = sorted_groups[:-1] != sorted_groups[1:]
    flat_mass = torch.zeros(
        bsz * int(num_anchors),
        weights.size(-1),
        device=weights.device,
        dtype=torch.float64,
    )
    flat_mass.index_copy_(0, sorted_groups[ends], sorted_mass[ends])
    return flat_mass.reshape(bsz, num_anchors, weights.size(-1))


def sparse_segmented_prefix_summaries(
    membership: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    num_anchors: int,
    max_seq_len: int,
    recency_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return long-term and recency-weighted summaries in O(N top_k D) storage.

    Records are sorted by (batch, anchor, position), scanned within each anchor,
    and restored to token/top-k order. No [B,N,A,D] tensor is materialized.
    """
    bsz, seq_len, top_k = membership.shape
    dim = values.size(-1)
    device = membership.device
    dtype = membership.dtype

    positions = torch.arange(seq_len, device=device)
    batch_ids = torch.arange(bsz, device=device).view(bsz, 1, 1)
    groups = (batch_ids * num_anchors + indices).reshape(-1)
    record_positions = positions.view(1, seq_len, 1).expand(bsz, -1, top_k).reshape(-1)
    keys = groups * max(seq_len, 1) + record_positions
    order = torch.argsort(keys)
    sorted_groups = groups.index_select(0, order)

    base_weights = membership.float() * token_mask[:, :, None].float()
    denominator = max(int(max_seq_len) - 1, 1)
    recency = torch.exp(
        membership.new_tensor(float(recency_bias), dtype=torch.float32)
        * positions.float()
        / denominator
    )
    weights = torch.stack(
        (base_weights, base_weights * recency.view(1, seq_len, 1)),
        dim=-1,
    )
    contributions = weights.unsqueeze(-1) * values.float().unsqueeze(2).unsqueeze(3)
    flat_contributions = contributions.reshape(-1, 2, dim).index_select(0, order)
    flat_mass = weights.reshape(-1, 2).index_select(0, order)

    sorted_numerator = _segmented_prefix_sum(flat_contributions, sorted_groups)
    sorted_mass = _segmented_prefix_sum(flat_mass, sorted_groups).clamp_min(1e-6)
    sorted_summary = sorted_numerator / sorted_mass.unsqueeze(-1)

    flat_summary = torch.empty_like(sorted_summary)
    flat_summary.index_copy_(0, order, sorted_summary)
    summary = flat_summary.reshape(bsz, seq_len, top_k, 2, dim)
    summary = summary.to(dtype=dtype)
    return summary[..., 0, :], summary[..., 1, :]


def _sparse_chunked_prefix_forward(
    membership: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    num_anchors: int,
    max_seq_len: int,
    recency_bias: float,
    chunk_size: int,
    return_slab_carries: bool = False,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Exact causal sparse summaries without a global record sort.

    Chunks are independent batch lanes. Dense anchor prefix state is bounded to
    a small slab of chunks, while token outputs retain only selected anchors.
    Peak storage is independent of total chunk count, never O(B * N * A * D).
    """
    bsz, seq_len, top_k = membership.shape
    dim = values.size(-1)
    chunk_size = max(int(chunk_size), 1)
    chunk_count = (seq_len + chunk_size - 1) // chunk_size
    padded_len = chunk_count * chunk_size
    right_pad = padded_len - seq_len
    if right_pad:
        membership = F.pad(membership, (0, 0, 0, right_pad))
        indices = F.pad(indices, (0, 0, 0, right_pad))
        values = F.pad(values, (0, 0, 0, right_pad))
        token_mask = F.pad(token_mask, (0, right_pad))

    positions = torch.arange(padded_len, device=values.device, dtype=torch.float32)
    base_weights = membership.float() * token_mask[:, :, None].float()
    recency = torch.exp(
        membership.new_tensor(float(recency_bias), dtype=torch.float32)
        * positions
        / max(int(max_seq_len) - 1, 1)
    )
    weights = torch.stack(
        (base_weights, base_weights * recency.view(1, padded_len, 1)),
        dim=-1,
    ).reshape(bsz, chunk_count, chunk_size, top_k, 2)
    chunk_values = values.float().reshape(bsz, chunk_count, chunk_size, dim)
    chunk_indices = indices.reshape(bsz, chunk_count, chunk_size, top_k)

    carry_numerator = torch.zeros(
        bsz, num_anchors, 2, dim, device=values.device, dtype=torch.float32
    )
    carry_mass = torch.zeros(
        bsz, num_anchors, 2, device=values.device, dtype=torch.float32
    )
    summary_slabs = []
    mass_slabs = []
    carry_numerator_starts = []
    carry_mass_starts = []
    # Keep chunk-parallel execution but bound the dense prefix to 16 chunks.
    # At 16K this cuts the anchor-state transient from 256 chunks to 16 while
    # avoiding a slow Python loop over every token.
    chunks_per_slab = 16
    for slab_start in range(0, chunk_count, chunks_per_slab):
        if return_slab_carries:
            carry_numerator_starts.append(carry_numerator.clone())
            carry_mass_starts.append(carry_mass.clone())
        slab_stop = min(slab_start + chunks_per_slab, chunk_count)
        slab_indices = chunk_indices[:, slab_start:slab_stop]
        slab_weights = weights[:, slab_start:slab_stop]
        slab_values = chunk_values[:, slab_start:slab_stop]
        slab_count = slab_stop - slab_start
        chunk_numerator, chunk_mass = _deterministic_sparse_chunk_totals(
            slab_indices,
            slab_weights,
            slab_values,
            num_anchors,
        )
        contributions = slab_weights.unsqueeze(-1) * slab_values[:, :, :, None, None, :]
        numerator = (
            carry_numerator[:, None]
            + chunk_numerator.cumsum(1)
            - chunk_numerator
        )
        mass = carry_mass[:, None] + chunk_mass.cumsum(1) - chunk_mass
        selected_summaries = []
        selected_masses = []
        for offset in range(chunk_size):
            current_indices = slab_indices[:, :, offset]
            current_numerator_index = current_indices[..., None, None].expand(
                -1, -1, -1, 2, dim
            )
            current_mass_index = current_indices[..., None].expand(-1, -1, -1, 2)
            numerator.scatter_add_(
                2,
                current_numerator_index,
                contributions[:, :, offset],
            )
            mass.scatter_add_(2, current_mass_index, slab_weights[:, :, offset])
            selected_numerator = torch.gather(numerator, 2, current_numerator_index)
            selected_mass = torch.gather(mass, 2, current_mass_index).clamp_min(1e-6)
            selected_summaries.append(selected_numerator / selected_mass.unsqueeze(-1))
            selected_masses.append(selected_mass)
        summary_slabs.append(torch.stack(selected_summaries, dim=2))
        mass_slabs.append(torch.stack(selected_masses, dim=2))
        carry_numerator = numerator[:, -1].clone()
        carry_mass = mass[:, -1].clone()

    summary = torch.cat(summary_slabs, dim=1)
    summary = summary.reshape(bsz, padded_len, top_k, 2, dim)[:, :seq_len]
    selected_mass = torch.cat(mass_slabs, dim=1)
    selected_mass = selected_mass.reshape(bsz, padded_len, top_k, 2)[:, :seq_len]
    if return_slab_carries:
        return (
            summary.to(dtype=values.dtype),
            selected_mass,
            torch.stack(carry_numerator_starts, dim=1),
            torch.stack(carry_mass_starts, dim=1),
            carry_numerator,
            carry_mass,
        )
    if return_final_state:
        final_mass = _deterministic_sparse_final_mass(
            chunk_indices.reshape(bsz, padded_len, top_k),
            weights.reshape(bsz, padded_len, top_k, 2),
            num_anchors,
        )
        return summary.to(dtype=values.dtype), selected_mass, carry_numerator, final_mass
    return summary.to(dtype=values.dtype), selected_mass


def _segmented_suffix_sum(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    reversed_sum = _segmented_prefix_sum(values.flip(0), groups.flip(0))
    return reversed_sum.flip(0)


class _SparseChunkedPrefixScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membership: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
        token_mask: torch.Tensor,
        num_anchors: int,
        max_seq_len: int,
        recency_bias: float,
        chunk_size: int,
    ):
        (
            summary,
            selected_mass,
            carry_numerator_starts,
            carry_mass_starts,
            _,
            _,
        ) = (
            _sparse_chunked_prefix_forward(
            membership,
            indices,
            values,
            token_mask,
            num_anchors,
            max_seq_len,
            recency_bias,
            chunk_size,
            return_slab_carries=True,
        ))
        ctx.num_anchors = int(num_anchors)
        ctx.max_seq_len = int(max_seq_len)
        ctx.recency_bias = float(recency_bias)
        ctx.chunk_size = max(int(chunk_size), 1)
        ctx.save_for_backward(
            membership,
            indices,
            values,
            token_mask,
            carry_numerator_starts,
            carry_mass_starts,
        )
        return summary[..., 0, :], summary[..., 1, :]

    @staticmethod
    def backward(ctx, grad_long: torch.Tensor, grad_recent: torch.Tensor):
        (
            membership,
            indices,
            values,
            token_mask,
            carry_numerator_starts,
            carry_mass_starts,
        ) = ctx.saved_tensors
        bsz, seq_len, top_k = membership.shape
        dim = values.size(-1)
        device = membership.device
        chunk_size = ctx.chunk_size
        chunk_count = (seq_len + chunk_size - 1) // chunk_size
        padded_len = chunk_count * chunk_size
        right_pad = padded_len - seq_len
        if right_pad:
            membership_padded = F.pad(membership, (0, 0, 0, right_pad))
            indices_padded = F.pad(indices, (0, 0, 0, right_pad))
            values_padded = F.pad(values, (0, 0, 0, right_pad))
            mask_padded = F.pad(token_mask, (0, right_pad))
            grad_long_padded = F.pad(grad_long, (0, 0, 0, 0, 0, right_pad))
            grad_recent_padded = F.pad(grad_recent, (0, 0, 0, 0, 0, right_pad))
        else:
            membership_padded, indices_padded = membership, indices
            values_padded, mask_padded = values, token_mask
            grad_long_padded, grad_recent_padded = grad_long, grad_recent

        positions = torch.arange(padded_len, device=device, dtype=torch.float32)
        denominator = max(ctx.max_seq_len - 1, 1)
        recency = torch.exp(
            membership.new_tensor(ctx.recency_bias, dtype=torch.float32)
            * positions
            / denominator
        )
        base_weights = membership_padded.float() * mask_padded[:, :, None].float()
        weights = torch.stack(
            (base_weights, base_weights * recency.view(1, padded_len, 1)),
            dim=-1,
        ).reshape(bsz, chunk_count, chunk_size, top_k, 2)
        chunk_values = values_padded.float().reshape(bsz, chunk_count, chunk_size, dim)
        chunk_indices = indices_padded.reshape(bsz, chunk_count, chunk_size, top_k)
        membership_grad = torch.zeros_like(membership_padded, dtype=torch.float32)
        value_grad = torch.zeros_like(values_padded, dtype=torch.float32)
        future_numerator_adjoint = torch.zeros(
            bsz, ctx.num_anchors, 2, dim, device=device, dtype=torch.float32
        )
        future_mass_adjoint = torch.zeros(
            bsz, ctx.num_anchors, 2, device=device, dtype=torch.float32
        )
        batch_ids = torch.arange(bsz, device=device).view(bsz, 1, 1)
        chunks_per_slab = 16
        slab_starts = list(range(0, chunk_count, chunks_per_slab))
        for slab_index in range(len(slab_starts) - 1, -1, -1):
            slab_start = slab_starts[slab_index]
            slab_stop = min(slab_start + chunks_per_slab, chunk_count)
            slab_count = slab_stop - slab_start
            slab_token_count = slab_count * chunk_size
            token_start = slab_start * chunk_size
            token_stop = token_start + slab_token_count
            slab_indices = chunk_indices[:, slab_start:slab_stop]
            slab_weights = weights[:, slab_start:slab_stop]
            slab_values = chunk_values[:, slab_start:slab_stop]
            chunk_numerator, chunk_mass = _deterministic_sparse_chunk_totals(
                slab_indices,
                slab_weights,
                slab_values,
                ctx.num_anchors,
            )
            contributions = (
                slab_weights.unsqueeze(-1)
                * slab_values[:, :, :, None, None, :]
            )
            numerator = (
                carry_numerator_starts[:, slab_index, None]
                + chunk_numerator.cumsum(1)
                - chunk_numerator
            )
            mass = (
                carry_mass_starts[:, slab_index, None]
                + chunk_mass.cumsum(1)
                - chunk_mass
            )
            selected_summaries = []
            selected_masses = []
            for offset in range(chunk_size):
                current_indices = slab_indices[:, :, offset]
                current_numerator_index = current_indices[..., None, None].expand(
                    -1, -1, -1, 2, dim
                )
                current_mass_index = current_indices[..., None].expand(
                    -1, -1, -1, 2
                )
                numerator.scatter_add_(
                    2, current_numerator_index, contributions[:, :, offset]
                )
                mass.scatter_add_(
                    2, current_mass_index, slab_weights[:, :, offset]
                )
                selected_numerator = torch.gather(
                    numerator, 2, current_numerator_index
                )
                selected_mass = torch.gather(
                    mass, 2, current_mass_index
                ).clamp_min(1e-6)
                selected_summaries.append(
                    selected_numerator / selected_mass.unsqueeze(-1)
                )
                selected_masses.append(selected_mass)
            slab_summary = torch.stack(selected_summaries, dim=2).reshape(
                bsz, slab_token_count, top_k, 2, dim
            )
            slab_mass = torch.stack(selected_masses, dim=2).reshape(
                bsz, slab_token_count, top_k, 2
            )
            grad_summary = torch.stack(
                (
                    grad_long_padded[:, token_start:token_stop],
                    grad_recent_padded[:, token_start:token_stop],
                ),
                dim=-2,
            ).float()
            source_numerator_adjoint = grad_summary / slab_mass.unsqueeze(-1)
            source_mass_adjoint = -(
                grad_summary * slab_summary
            ).sum(-1) / slab_mass

            slab_indices_flat = slab_indices.reshape(
                bsz, slab_token_count, top_k
            )
            groups = (batch_ids * ctx.num_anchors + slab_indices_flat).reshape(-1)
            record_positions = (
                torch.arange(slab_token_count, device=device)
                .view(1, slab_token_count, 1)
                .expand(bsz, -1, top_k)
                .reshape(-1)
            )
            order = torch.argsort(groups * max(slab_token_count, 1) + record_positions)
            sorted_groups = groups.index_select(0, order)
            sorted_source_numerator = source_numerator_adjoint.reshape(
                -1, 2, dim
            ).index_select(0, order)
            sorted_source_mass = source_mass_adjoint.reshape(-1, 2).index_select(
                0, order
            )
            future_numerator = future_numerator_adjoint.reshape(
                bsz * ctx.num_anchors, 2, dim
            ).index_select(0, sorted_groups)
            future_mass = future_mass_adjoint.reshape(
                bsz * ctx.num_anchors, 2
            ).index_select(0, sorted_groups)
            numerator_adjoint = (
                _segmented_suffix_sum(sorted_source_numerator, sorted_groups)
                + future_numerator
            )
            mass_adjoint = (
                _segmented_suffix_sum(sorted_source_mass, sorted_groups)
                + future_mass
            )
            sorted_values = (
                slab_values.reshape(bsz, slab_token_count, dim)
                .unsqueeze(2)
                .expand(-1, -1, top_k, -1)
                .reshape(-1, dim)
                .index_select(0, order)
            )
            sorted_weight_grad = (
                numerator_adjoint * sorted_values.unsqueeze(1)
            ).sum(-1) + mass_adjoint
            sorted_value_grad = (
                slab_weights.reshape(bsz, slab_token_count, top_k, 2)
                .reshape(-1, 2)
                .index_select(0, order)
                .unsqueeze(-1)
                * numerator_adjoint
            ).sum(1)
            weight_grad = torch.empty_like(sorted_weight_grad)
            weight_grad.index_copy_(0, order, sorted_weight_grad)
            weight_grad = weight_grad.reshape(
                bsz, slab_token_count, top_k, 2
            )
            slab_recency = recency[token_start:token_stop].view(
                1, slab_token_count, 1
            )
            membership_grad[:, token_start:token_stop] = (
                mask_padded[:, token_start:token_stop, None].float()
                * (weight_grad[..., 0] + slab_recency * weight_grad[..., 1])
            )
            unsorted_value_grad = torch.empty_like(sorted_value_grad)
            unsorted_value_grad.index_copy_(0, order, sorted_value_grad)
            value_grad[:, token_start:token_stop] = unsorted_value_grad.reshape(
                bsz, slab_token_count, top_k, dim
            ).sum(2)

            local_numerator_total = torch.zeros_like(future_numerator_adjoint)
            local_numerator_total.scatter_add_(
                1,
                slab_indices_flat.reshape(bsz, -1, 1, 1).expand(
                    -1, -1, 2, dim
                ),
                source_numerator_adjoint.reshape(bsz, -1, 2, dim),
            )
            local_mass_total = torch.zeros_like(future_mass_adjoint)
            local_mass_total.scatter_add_(
                1,
                slab_indices_flat.reshape(bsz, -1, 1).expand(-1, -1, 2),
                source_mass_adjoint.reshape(bsz, -1, 2),
            )
            future_numerator_adjoint = (
                future_numerator_adjoint + local_numerator_total
            )
            future_mass_adjoint = future_mass_adjoint + local_mass_total

        return (
            membership_grad[:, :seq_len].to(dtype=membership.dtype),
            None,
            value_grad[:, :seq_len].to(dtype=values.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def sparse_chunked_prefix_summaries(
    membership: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    num_anchors: int,
    max_seq_len: int,
    recency_bias: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _SparseChunkedPrefixScan.apply(
        membership,
        indices,
        values,
        token_mask,
        num_anchors,
        max_seq_len,
        recency_bias,
        chunk_size,
    )


def _fixed_token_linear(
    module: nn.Linear,
    x: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size <= 0 or x.size(1) <= 1:
        return module(x)
    rows = []
    for start in range(0, x.size(1), group_size):
        current = x[:, start : start + group_size]
        count = current.size(1)
        if count < group_size:
            current = F.pad(current, (0, 0, 0, group_size - count))
        rows.append(module(current)[:, :count])
    return torch.cat(rows, dim=1)


def _fixed_token_anchor_matmul(
    q: torch.Tensor,
    anchor_keys: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size <= 0 or q.size(1) <= 1:
        return torch.matmul(q, anchor_keys.t())
    rows = []
    for start in range(0, q.size(1), group_size):
        current = q[:, start : start + group_size]
        count = current.size(1)
        if count < group_size:
            current = F.pad(current, (0, 0, 0, group_size - count))
        rows.append(torch.matmul(current, anchor_keys.t())[:, :count])
    return torch.cat(rows, dim=1)


class CausalWindowAttention(nn.Module):
    """Chunked local causal attention with O(N W) score storage and compute."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.window = int(config.local_attention_window)
        if self.window < 1:
            raise ValueError("local_attention_window must be positive")
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.inference_group_size = int(
            getattr(config, "inference_prefix_stable_group_size", 0)
        )
        if self.inference_group_size < 0:
            raise ValueError("inference prefix stable group size cannot be negative")

    def _project(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        group_size = self.inference_group_size if not self.training else 0
        qkv = _fixed_token_linear(self.qkv, x, group_size).view(
            bsz, seq_len, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        bsz, seq_len, dim = x.shape
        q, k, v = self._project(x)
        state_k, state_v = k, v
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
        chunks = (seq_len + self.window - 1) // self.window
        padded_len = chunks * self.window
        right_pad = padded_len - seq_len
        q = F.pad(q, (0, 0, 0, right_pad))
        k = F.pad(k, (0, 0, self.window, right_pad))
        v = F.pad(v, (0, 0, self.window, right_pad))
        padded_mask = F.pad(attention_mask, (self.window, right_pad), value=False)

        q_chunks = q.view(bsz, self.n_heads, chunks, self.window, self.head_dim)
        # unfold appends the window dimension after head_dim.
        k_windows = k.unfold(2, 2 * self.window, self.window).permute(0, 1, 2, 4, 3)
        v_windows = v.unfold(2, 2 * self.window, self.window).permute(0, 1, 2, 4, 3)
        mask_windows = padded_mask.unfold(1, 2 * self.window, self.window)

        if self.inference_group_size > 0 and not self.training and seq_len > 1:
            scores = torch.cat(
                [
                    torch.matmul(
                        q_chunks[:, :, index : index + 1],
                        k_windows[:, :, index : index + 1].transpose(-2, -1),
                    )
                    for index in range(chunks)
                ],
                dim=2,
            ) * (self.head_dim ** -0.5)
        else:
            scores = torch.matmul(q_chunks, k_windows.transpose(-2, -1)) * (self.head_dim ** -0.5)
        query_offsets = torch.arange(self.window, device=x.device).view(self.window, 1)
        key_offsets = torch.arange(-self.window, self.window, device=x.device).view(1, 2 * self.window)
        local_causal = (key_offsets <= query_offsets) & ((query_offsets - key_offsets) < self.window)
        allowed = local_causal.view(1, 1, 1, self.window, 2 * self.window)
        allowed = allowed & mask_windows[:, None, :, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v_windows)
        query_mask = attention_mask.new_zeros(bsz, padded_len, dtype=torch.bool)
        query_mask[:, :seq_len] = attention_mask
        out = out * query_mask.view(bsz, 1, chunks, self.window, 1).to(dtype=out.dtype)
        out = out.reshape(bsz, self.n_heads, padded_len, self.head_dim)[:, :, :seq_len]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        output = _fixed_token_linear(
            self.out_proj,
            out,
            self.inference_group_size if not self.training else 0,
        )
        if return_state:
            return output, self._state_from_projected(state_k, state_v, attention_mask)
        return output

    def _state_from_projected(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> "WindowAttentionState":
        capacity = self.window
        count = min(int(k.size(2)), capacity)
        key_cache = k.new_zeros(k.size(0), k.size(1), capacity, k.size(3))
        value_cache = v.new_zeros(v.size(0), v.size(1), capacity, v.size(3))
        mask_cache = mask.new_zeros(mask.size(0), capacity, dtype=torch.bool)
        if count:
            key_cache[:, :, :count].copy_(k[:, :, -count:])
            value_cache[:, :, :count].copy_(v[:, :, -count:])
            mask_cache[:, :count].copy_(mask[:, -count:])
        return WindowAttentionState(
            key=key_cache,
            value=value_cache,
            mask=mask_cache,
            cursor=count % capacity,
            count=count,
        )

    def state_from_sequence(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> "WindowAttentionState":
        _, k, v = self._project(x)
        mask = (
            torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=x.device, dtype=torch.bool)
        )
        return self._state_from_projected(k, v, mask)

    def step(
        self,
        x: torch.Tensor,
        state: Optional["WindowAttentionState"],
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, "WindowAttentionState"]:
        if x.size(1) != 1:
            raise ValueError("window attention step expects one token")
        q, k, v = self._project(x)
        current_mask = token_mask.to(device=x.device, dtype=torch.bool)
        if state is None:
            empty_mask = current_mask.new_zeros(current_mask.size(0), 0)
            state = self._state_from_projected(k[:, :, :0], v[:, :, :0], empty_mask)
        cursor = state.cursor
        state.key[:, :, cursor : cursor + 1].copy_(k)
        state.value[:, :, cursor : cursor + 1].copy_(v)
        state.mask[:, cursor : cursor + 1].copy_(current_mask)
        state.count = min(state.count + 1, self.window)
        state.cursor = (cursor + 1) % self.window
        out = F.scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            attn_mask=state.mask[:, None, None, :],
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out * current_mask[:, None, :, None].to(dtype=out.dtype)
        out = out.transpose(1, 2).contiguous().view(x.size(0), 1, -1)
        return self.out_proj(out), state


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(2 * config.dim * config.mlp_ratio / 3)
        hidden = max(64, ((hidden + 63) // 64) * 64)
        self.gate = nn.Linear(config.dim, hidden, bias=False)
        self.up = nn.Linear(config.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.inference_group_size = int(
            getattr(config, "inference_prefix_stable_group_size", 0)
        )

    def forward(self, x: torch.Tensor):
        group_size = self.inference_group_size if not self.training else 0
        gate = _fixed_token_linear(self.gate, x, group_size)
        up = _fixed_token_linear(self.up, x, group_size)
        hidden = F.silu(gate) * up
        return self.dropout(_fixed_token_linear(self.down, hidden, group_size))


class V3FuzzyAnchorLayer(CausalFuzzyAnchorLayer):
    def set_aggregation_impl(self, implementation: str) -> None:
        if implementation not in {"sparse_segmented_scan", "sparse_chunked_scan"}:
            super().set_aggregation_impl(implementation)
            return
        self.aggregation_impl = implementation

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        router_dim = min(int(config.router_dim), int(config.dim))
        if router_dim < 16:
            raise ValueError("router_dim must be at least 16")
        self.router_dim = router_dim
        if router_dim != config.dim:
            self.q_proj = nn.Linear(config.dim, router_dim, bias=False)
            self.anchor_keys = nn.Parameter(torch.randn(config.num_anchors, router_dim) * 0.02)
        self.recent_gate = nn.Parameter(torch.tensor(0.0))
        self._normalized_anchor_cache: Optional[tuple[int, int, torch.Tensor, torch.Tensor]] = None
        self._runtime_gate_cache: Optional[tuple[int, int, torch.Tensor, torch.Tensor]] = None
        denominator = max(self.recency_reference_len() - 1, 1)
        recency = [
            math.exp(float(config.anchor_recency_bias) * position / denominator)
            for position in range(config.max_seq_len)
        ]
        self.register_buffer(
            "_decode_recency_scales",
            torch.tensor([[1.0, value] for value in recency], dtype=torch.float32),
            persistent=False,
        )

    def train(self, mode: bool = True):
        if mode:
            self._normalized_anchor_cache = None
            self._runtime_gate_cache = None
        return super().train(mode)

    def runtime_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            return torch.sigmoid(self.gate), torch.sigmoid(self.recent_gate)
        try:
            versions = (int(self.gate._version), int(self.recent_gate._version))
        except RuntimeError:
            versions = (-1, -1)
        cached = self._runtime_gate_cache
        if cached is None or cached[:2] != versions:
            cached = (*versions, torch.sigmoid(self.gate), torch.sigmoid(self.recent_gate))
            self._runtime_gate_cache = cached
        return cached[2], cached[3]

    def normalized_anchors(self):
        if self.training or not self.config.use_cached_normalized_anchors:
            return F.normalize(self.anchor_keys, dim=-1), F.normalize(self.anchor_values, dim=-1)
        try:
            versions = (int(self.anchor_keys._version), int(self.anchor_values._version))
        except RuntimeError:
            # Parameters created inside inference_mode do not expose version counters.
            # They are immutable, so a stable sentinel is sufficient for the eval cache.
            versions = (-1, -1)
        cached = self._normalized_anchor_cache
        if cached is None or cached[:2] != versions:
            cached = (*versions, F.normalize(self.anchor_keys, dim=-1), F.normalize(self.anchor_values, dim=-1))
            self._normalized_anchor_cache = cached
        return cached[2], cached[3]

    def route(self, x: torch.Tensor, return_distance: bool = True):
        bsz, seq_len, _ = x.shape
        group_size = int(
            getattr(self.config, "inference_prefix_stable_group_size", 0)
        ) if not self.training else 0
        q = F.normalize(_fixed_token_linear(self.q_proj, x, group_size), dim=-1)
        v = _fixed_token_linear(self.v_proj, x, group_size)
        anchor_keys, anchor_values = self.normalized_anchors()
        cosine = _fixed_token_anchor_matmul(q, anchor_keys, group_size)
        if self.config.routing_type == "cosine":
            logits = cosine / max(self.config.cosine_temperature, 1e-5)
            full_distance = 1.0 - cosine if return_distance else None
        elif self.config.routing_type == "gaussian":
            distance = (2.0 - 2.0 * cosine).clamp_min(0.0)
            sigma = self.log_sigma.exp().clamp(self.config.sigma_min, self.config.sigma_max)
            logits = -distance / (2.0 * sigma.view(1, 1, -1).pow(2))
            full_distance = distance if return_distance else None
        else:
            raise ValueError(f"Unknown routing_type: {self.config.routing_type}")
        logits = torch.nan_to_num(logits, nan=0.0).clamp(
            -self.config.membership_logit_clip,
            self.config.membership_logit_clip,
        )
        routing_logits = logits
        quantum = float(getattr(self.config, "routing_logit_quantization", 0.0))
        if quantum > 0.0:
            quantized = torch.round(logits / quantum) * quantum
            anchor_tie_break = -torch.arange(
                self.num_anchors,
                device=logits.device,
                dtype=logits.dtype,
            ) * (quantum / (2.0 * max(self.num_anchors, 1)))
            stable_forward = quantized + anchor_tie_break.view(1, 1, -1)
            routing_logits = logits + (stable_forward - logits).detach()
        smoothing = float(
            getattr(self.config, "routing_boundary_smoothing_epsilon", 0.0)
        )
        extra_candidates = int(
            getattr(self.config, "routing_boundary_extra_candidates", 0)
        )
        if smoothing < 0.0 or extra_candidates < 0:
            raise ValueError("routing boundary smoothing settings cannot be negative")
        smooth_boundary = (
            smoothing > 0.0
            and extra_candidates > 0
            and self.probe_mode == "learned"
        )
        candidate_count = min(
            self.num_anchors,
            self.top_k + extra_candidates if smooth_boundary else self.top_k,
        )
        ranked_logits, ranked_indices = torch.topk(
            routing_logits, k=candidate_count, dim=-1
        )
        boundary_logit = ranked_logits[..., self.top_k - 1 : self.top_k]
        if smooth_boundary:
            # Canonical anchor order makes reductions independent of tiny score-order
            # swaps. Candidates outside the original top-k enter continuously only
            # when their score lies inside the declared boundary band.
            indices = ranked_indices.sort(dim=-1).values
        else:
            indices = ranked_indices
        top_logits = torch.gather(logits, 2, indices)
        if self.probe_mode == "random_topk":
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).view(1, seq_len, 1)
            anchors = torch.arange(self.num_anchors, device=x.device, dtype=torch.float32).view(1, 1, -1)
            batches = torch.arange(bsz, device=x.device, dtype=torch.float32).view(bsz, 1, 1)
            scores = torch.sin((anchors + 1) * 12.9898 + (positions + 1) * 78.233 + (batches + 1) * 37.719)
            indices = torch.topk(scores, k=self.top_k, dim=-1).indices
            top_logits = torch.gather(logits, 2, indices)
        if smooth_boundary:
            candidate_gate = (
                (top_logits - boundary_logit + smoothing) / smoothing
            ).clamp(0.0, 1.0)
            candidate_gate = torch.where(
                top_logits >= boundary_logit,
                torch.ones_like(candidate_gate),
                candidate_gate,
            )
            shifted = top_logits.float() - top_logits.float().max(
                dim=-1, keepdim=True
            ).values
            membership = shifted.exp() * candidate_gate.float()
            membership = membership / membership.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            membership = membership.to(dtype=x.dtype)
        else:
            membership = F.softmax(top_logits.float(), dim=-1).to(dtype=x.dtype)
        if self.probe_mode == "uniform_topk":
            membership = torch.full_like(membership, 1.0 / self.top_k)
        elif self.probe_mode == "shuffle_membership":
            membership = membership.roll(1, dims=-1)
        elif self.probe_mode == "onehot_top1":
            membership = torch.zeros_like(membership)
            membership[..., 0] = 1.0
        membership_quantum = float(
            getattr(self.config, "routing_membership_quantization", 0.0)
        )
        if membership_quantum < 0.0:
            raise ValueError("routing membership quantization cannot be negative")
        if membership_quantum > 0.0:
            quantized_membership = (
                torch.round(membership / membership_quantum) * membership_quantum
            )
            residual = 1.0 - quantized_membership.sum(dim=-1, keepdim=True)
            maximum_index = membership.argmax(dim=-1, keepdim=True)
            quantized_membership = quantized_membership.scatter_add(
                -1,
                maximum_index,
                residual,
            )
            membership = membership + (
                quantized_membership - membership
            ).detach()
        if not smooth_boundary:
            membership = membership.clamp_min(1e-8)
        membership = membership / membership.sum(-1, keepdim=True).clamp_min(1e-8)
        if getattr(self, "capture_exact_route", False):
            self.last_exact_route_indices = indices.detach()
        return v, anchor_values, full_distance, indices, membership

    def recency_reference_len(self) -> int:
        configured = int(self.config.anchor_recency_reference_len)
        return configured if configured > 0 else int(self.config.max_seq_len)

    def set_recency_bias(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("anchor recency bias must be finite")
        self.config.anchor_recency_bias = value
        denominator = max(self.recency_reference_len() - 1, 1)
        recency = [
            math.exp(value * position / denominator)
            for position in range(self.config.max_seq_len)
        ]
        self._decode_recency_scales = torch.tensor(
            [[1.0, item] for item in recency],
            dtype=torch.float32,
            device=self._decode_recency_scales.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        bsz, seq_len, _ = x.shape
        token_mask = (
            torch.ones(bsz, seq_len, device=x.device, dtype=x.dtype)
            if attention_mask is None
            else attention_mask.to(device=x.device, dtype=x.dtype)
        )
        v, anchor_values, full_distance, indices, membership = self.route(x)
        state = None
        if self.aggregation_impl == "sparse_chunked_scan":
            if return_state:
                summary, _, final_numerator, final_mass = (
                    _sparse_chunked_prefix_forward(
                        membership,
                        indices,
                        v,
                        token_mask,
                        self.num_anchors,
                        self.recency_reference_len(),
                        self.config.anchor_recency_bias,
                        self.config.anchor_scan_chunk_size,
                        return_final_state=True,
                    )
                )
                long_summary, recent_summary = summary[..., 0, :], summary[..., 1, :]
                if self.config.model_type == "fdt_v4":
                    state = _normalized_anchor_state_from_route_online(
                        self,
                        v,
                        indices,
                        membership,
                        token_mask,
                    )
                else:
                    final_summary = final_numerator / final_mass.to(
                        dtype=final_numerator.dtype
                    ).clamp_min(1e-6).unsqueeze(-1)
                    if bool(getattr(self.config, "anchor_decode_state_fp32", False)):
                        final_summary = final_summary.float()
                        final_mass = final_mass.double()
                    state = V3AnchorState(final_summary, final_mass, normalized=True)
            else:
                long_summary, recent_summary = sparse_chunked_prefix_summaries(
                    membership,
                    indices,
                    v,
                    token_mask,
                    self.num_anchors,
                    self.recency_reference_len(),
                    self.config.anchor_recency_bias,
                    self.config.anchor_scan_chunk_size,
                )
        else:
            long_summary, recent_summary = sparse_segmented_prefix_summaries(
                membership,
                indices,
                v,
                token_mask,
                self.num_anchors,
                self.recency_reference_len(),
                self.config.anchor_recency_bias,
            )
        long_context = torch.einsum("bnk,bnkd->bnd", membership, long_summary)
        recent_context = torch.einsum("bnk,bnkd->bnd", membership, recent_summary)
        recent_mix = torch.sigmoid(self.recent_gate)
        structural = (1.0 - recent_mix) * long_context + recent_mix * recent_context
        latent = torch.einsum("bnk,bnkd->bnd", membership, anchor_values[indices])
        gate = torch.sigmoid(self.gate)
        group_size = int(
            getattr(self.config, "inference_prefix_stable_group_size", 0)
        ) if not self.training else 0
        routed = _fixed_token_linear(
            self.out_proj,
            gate * structural + (1.0 - gate) * latent,
            group_size,
        )
        if self.probe_mode == "zero_anchor":
            routed = torch.zeros_like(routed)

        entropy_per_token = -(
            membership.float() * membership.float().clamp_min(1e-8).log()
        ).sum(-1)
        selected_distance = torch.gather(full_distance, 2, indices)
        cluster_per_token = (membership.float() * selected_distance.float()).sum(-1)
        usage = torch.zeros(self.num_anchors, device=x.device)
        usage.scatter_add_(0, indices.reshape(-1), (membership.float() * token_mask[:, :, None]).reshape(-1))
        load = torch.zeros(self.num_anchors, device=x.device)
        load.scatter_add_(0, indices.reshape(-1), token_mask[:, :, None].float().expand_as(membership).reshape(-1))
        metrics_enabled = getattr(self, "runtime_anchor_metrics", self.config.enable_anchor_metrics)
        ordered_membership = membership.float().topk(
            min(2, membership.size(-1)), dim=-1
        ).values
        top1 = masked_mean(ordered_membership[..., 0], token_mask) if metrics_enabled else entropy_per_token.new_zeros(())
        margin_values = ordered_membership[..., 0] - ordered_membership[..., 1] if membership.size(-1) > 1 else ordered_membership[..., 0]
        margin = masked_mean(margin_values.float(), token_mask) if metrics_enabled else entropy_per_token.new_zeros(())
        diagnostic_indices = indices.detach() if self.config.enable_diagnostics else indices.new_empty((0, 0, indices.size(-1)))
        diagnostic_membership = (
            membership.detach() if self.config.enable_diagnostics and self.config.detach_diagnostics
            else membership if self.config.enable_diagnostics
            else membership.new_empty((0, 0, membership.size(-1)))
        )
        stats = AnchorStats(
            indices=diagnostic_indices,
            membership=diagnostic_membership,
            entropy=masked_mean(entropy_per_token, token_mask),
            usage_prob=usage / usage.sum().clamp_min(1e-8),
            load_prob=(load / load.sum().clamp_min(1e-8)).detach(),
            cluster_loss=masked_mean(cluster_per_token, token_mask),
            top1_membership=top1.detach(),
            membership_margin=margin.detach(),
        )
        routed = routed * token_mask.unsqueeze(-1)
        if return_state:
            if state is None:
                state = _anchor_state_from_route(
                    self,
                    v,
                    indices,
                    membership,
                    token_mask,
                )
                if self.config.model_type == "fdt_v4":
                    state.numerator = (
                        state.numerator
                        / state.mass.clamp_min(1e-6).unsqueeze(-1)
                    )
                    state.normalized = True
            return routed, stats, state
        return routed, stats


class FDTv3Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.local_norm = RMSNorm(config.dim)
        self.local_attention = CausalWindowAttention(config)
        self.anchor_norm = RMSNorm(config.dim)
        self.anchor = V3FuzzyAnchorLayer(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = x + self.local_attention(self.local_norm(x), attention_mask)
        stats = None
        if self.anchor is not None:
            anchor_out, stats = self.anchor(self.anchor_norm(x), attention_mask)
            x = x + anchor_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, [stats] if stats is not None else []


@dataclass
class WindowAttentionState:
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor
    cursor: int = 0
    count: int = 0


@dataclass
class V3AnchorState:
    numerator: torch.Tensor
    mass: torch.Tensor
    normalized: bool = False


def _full_anchor_state(
    layer: V3FuzzyAnchorLayer,
    x: torch.Tensor,
    token_mask: torch.Tensor,
) -> V3AnchorState:
    v, _, _, indices, membership = layer.route(x)
    return _anchor_state_from_route(layer, v, indices, membership, token_mask)


def _anchor_state_from_route(
    layer: V3FuzzyAnchorLayer,
    v: torch.Tensor,
    indices: torch.Tensor,
    membership: torch.Tensor,
    token_mask: torch.Tensor,
) -> V3AnchorState:
    bsz, seq_len, top_k = membership.shape
    dim = v.size(-1)
    weights = membership * token_mask[:, :, None].to(dtype=membership.dtype)
    positions = torch.arange(seq_len, device=v.device, dtype=torch.float32)
    recency = torch.exp(
        float(layer.config.anchor_recency_bias)
        * positions
        / max(layer.recency_reference_len() - 1, 1)
    ).to(dtype=weights.dtype)
    recent_weights = weights * recency.view(1, seq_len, 1)
    flat_indices = indices.reshape(bsz, -1)
    stacked_weights = torch.stack((weights, recent_weights), dim=-1)
    expanded = flat_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, dim)

    numerator = torch.zeros(bsz, layer.num_anchors, 2, dim, device=v.device, dtype=v.dtype)
    numerator.scatter_add_(
        1,
        expanded,
        (stacked_weights.unsqueeze(-1) * v.unsqueeze(2).unsqueeze(3)).reshape(bsz, -1, 2, dim),
    )
    mass = torch.zeros(bsz, layer.num_anchors, 2, device=v.device, dtype=v.dtype)
    mass.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, 2), stacked_weights.reshape(bsz, -1, 2))
    return V3AnchorState(numerator, mass)


def _normalized_anchor_state_from_route_online(
    layer: V3FuzzyAnchorLayer,
    values: torch.Tensor,
    indices: torch.Tensor,
    membership: torch.Tensor,
    token_mask: torch.Tensor,
) -> V3AnchorState:
    """Build the prefill cache with the exact recurrence used by decode."""
    bsz, seq_len, dim = values.shape
    summary_dtype = (
        torch.float32
        if bool(getattr(layer.config, "anchor_decode_state_fp32", False))
        else values.dtype
    )
    summary = torch.zeros(
        bsz,
        layer.num_anchors,
        2,
        dim,
        device=values.device,
        dtype=summary_dtype,
    )
    mass = torch.zeros(
        bsz,
        layer.num_anchors,
        2,
        device=values.device,
        dtype=torch.float64,
    )
    for position in range(seq_len):
        selected = indices[:, position]
        weights = membership[:, position] * token_mask[:, position, None].to(
            dtype=membership.dtype
        )
        scales = layer._decode_recency_scales[position].to(dtype=weights.dtype)
        stacked_weights = weights.unsqueeze(-1) * scales
        expanded = selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, dim)
        mass_index = selected.unsqueeze(-1).expand(-1, -1, 2)
        previous_summary = torch.gather(summary, 1, expanded)
        previous_mass = torch.gather(mass, 1, mass_index)
        mass_weights = stacked_weights.to(dtype=previous_mass.dtype)
        next_mass = previous_mass + mass_weights
        interpolation = (mass_weights / next_mass.clamp_min(1e-6)).to(
            dtype=previous_summary.dtype
        )
        incoming = values[:, position, None, None, :].to(
            dtype=previous_summary.dtype
        ).expand_as(previous_summary)
        next_summary = previous_summary + interpolation.unsqueeze(-1) * (
            incoming - previous_summary
        )
        summary.scatter_(1, expanded, next_summary)
        mass.scatter_(1, mass_index, next_mass)
    return V3AnchorState(summary, mass, normalized=True)


def _single_anchor_step(
    layer: V3FuzzyAnchorLayer,
    x: torch.Tensor,
    state: Optional[V3AnchorState],
    token_mask: torch.Tensor,
    position: int,
    compute_stats: bool = True,
):
    v, anchor_values, full_distance, indices, membership = layer.route(
        x,
        return_distance=compute_stats,
    )
    bsz, _, dim = x.shape
    if state is None:
        normalized = layer.config.model_type == "fdt_v4"
        state = V3AnchorState(
            torch.zeros(bsz, layer.num_anchors, 2, dim, device=x.device, dtype=x.dtype),
            torch.zeros(
                bsz,
                layer.num_anchors,
                2,
                device=x.device,
                dtype=torch.float64 if normalized else x.dtype,
            ),
            normalized=normalized,
        )
    index = indices[:, 0]
    weights = membership[:, 0] * token_mask[:, 0, None].to(dtype=x.dtype)
    scales = layer._decode_recency_scales[position].to(dtype=weights.dtype)
    stacked_weights = weights.unsqueeze(-1) * scales
    expanded = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, dim)
    mass_index = index.unsqueeze(-1).expand(-1, -1, 2)
    if state.normalized:
        previous_summary = torch.gather(state.numerator, 1, expanded)
        previous_mass = torch.gather(state.mass, 1, mass_index)
        mass_weights = stacked_weights.to(dtype=previous_mass.dtype)
        next_mass = previous_mass + mass_weights
        interpolation = (mass_weights / next_mass.clamp_min(1e-6)).to(
            dtype=previous_summary.dtype
        )
        incoming = v[:, 0, None, None, :].to(
            dtype=previous_summary.dtype
        ).expand_as(previous_summary)
        summaries = previous_summary + interpolation.unsqueeze(-1) * (
            incoming - previous_summary
        )
        state.numerator.scatter_(1, expanded, summaries)
        state.mass.scatter_(1, mass_index, next_mass)
    else:
        contribution = stacked_weights.unsqueeze(-1) * v[:, 0, None, None, :]
        state.numerator.scatter_add_(1, expanded, contribution)
        state.mass.scatter_add_(1, mass_index, stacked_weights)
        selected_num = torch.gather(state.numerator, 1, expanded)
        selected_mass = torch.gather(state.mass, 1, mass_index).clamp_min(1e-6)
        summaries = selected_num / selected_mass.unsqueeze(-1)
    context_membership = membership[:, 0].to(dtype=summaries.dtype)
    contexts = torch.einsum("bk,bkmd->bmd", context_membership, summaries).to(
        dtype=x.dtype
    )
    long_context, recent_context = contexts.unbind(dim=1)
    gate, recent_mix = layer.runtime_gates()
    structural = (1.0 - recent_mix) * long_context + recent_mix * recent_context
    latent = torch.einsum("bk,bkd->bd", membership[:, 0], anchor_values[index])
    routed = layer.out_proj(gate * structural.unsqueeze(1) + (1.0 - gate) * latent.unsqueeze(1))

    if not compute_stats:
        return routed, None, state

    mask = token_mask.to(dtype=x.dtype)
    entropy_values = -(membership.float() * membership.float().log()).sum(-1)
    selected_distance = torch.gather(full_distance, 2, indices)
    cluster_values = (membership.float() * selected_distance.float()).sum(-1)
    usage = torch.zeros(layer.num_anchors, device=x.device)
    usage.scatter_add_(0, index.reshape(-1), (membership[:, 0].float() * mask).reshape(-1))
    load = torch.zeros(layer.num_anchors, device=x.device)
    load.scatter_add_(0, index.reshape(-1), mask[:, :, None].float().expand_as(membership).reshape(-1))
    stats = AnchorStats(
        indices=indices.detach() if layer.config.enable_diagnostics else indices.new_empty((0, 0, layer.top_k)),
        membership=membership.detach() if layer.config.enable_diagnostics else membership.new_empty((0, 0, layer.top_k)),
        entropy=masked_mean(entropy_values, mask),
        usage_prob=usage / usage.sum().clamp_min(1e-8),
        load_prob=(load / load.sum().clamp_min(1e-8)).detach(),
        cluster_loss=masked_mean(cluster_values, mask),
        top1_membership=masked_mean(membership.float().max(dim=-1).values, mask).detach(),
        membership_margin=masked_mean(
            (
                membership.float().topk(2, dim=-1).values[..., 0]
                - membership.float().topk(2, dim=-1).values[..., 1]
            ) if membership.size(-1) > 1 else membership[..., 0].float(),
            mask,
        ).detach(),
    )
    return routed, stats, state


class CausalFDTv3LM(BaseCausalLM):
    block_cls = FDTv3Block

    def __init__(self, config: ModelConfig):
        if config.use_rope:
            raise ValueError("fdt_v3 currently requires use_rope=False and learned positions")
        super().__init__(config)
        from fdt_rlm.models.next_causal_lm import resolve_anchor_layer_indices

        anchor_indices = set(resolve_anchor_layer_indices(config))
        self.anchor_layer_indices = sorted(anchor_indices)
        for index, block in enumerate(self.blocks):
            if index not in anchor_indices:
                block.anchor = None
                block.anchor_norm = None

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError("prefill input exceeds max_seq_len")
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = self._position_ids[:, :seq_len]
            x = x + self.position_embedding(positions)
        x = self.drop(x)
        layer_caches = []
        stats = []
        for block in self.blocks:
            local_input = block.local_norm(x)
            local_out, local_state = block.local_attention(
                local_input,
                attention_mask,
                return_state=True,
            )
            x = x + local_out
            anchor_state = None
            if block.anchor is not None:
                anchor_input = block.anchor_norm(x)
                anchor_out, anchor_stats, anchor_state = block.anchor(
                    anchor_input,
                    attention_mask,
                    return_state=True,
                )
                x = x + anchor_out
                stats.append(anchor_stats)
            x = x + block.mlp(block.mlp_norm(x))
            layer_caches.append((local_state, anchor_state))
        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        output = {
            "logits": logits[:, -1:],
            "hidden": hidden[:, -1:],
            "pred_noise": None,
            "anchor_stats": stats,
        }
        return output, {"backend": "fdt_v3_incremental", "layers": layer_caches, "length": seq_len}

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: dict[str, Any],
        token_mask: Optional[torch.Tensor] = None,
    ):
        if cache.get("backend") != "fdt_v3_incremental":
            raise ValueError("invalid FDT v3 cache")
        if input_ids.size(1) != 1:
            raise ValueError("decode_step expects one token")
        if cache["length"] >= self.config.max_seq_len:
            raise ValueError("decode cache reached max_seq_len")
        token_mask = self._decode_token_mask.expand(input_ids.size(0), 1) if token_mask is None else token_mask
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            position = self._position_ids[:, cache["length"] : cache["length"] + 1]
            x = x + self.position_embedding(position)
        x = self.drop(x)
        stats = []
        for block, (local_state, anchor_state) in zip(self.blocks, cache["layers"]):
            local_out, local_state = block.local_attention.step(block.local_norm(x), local_state, token_mask)
            x = x + local_out
            if block.anchor is not None:
                anchor_out, anchor_stats, anchor_state = _single_anchor_step(
                    block.anchor,
                    block.anchor_norm(x),
                    anchor_state,
                    token_mask,
                    cache["length"],
                    compute_stats=False,
                )
                x = x + anchor_out
                if anchor_stats is not None:
                    stats.append(anchor_stats)
            x = x + block.mlp(block.mlp_norm(x))
        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        output = {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": None,
            "anchor_stats": stats,
        }
        cache["length"] += 1
        return output, cache


class CausalFDTDualMemoryLM(CausalFDTv3LM):
    """FDT v3 with an in-checkpoint lossless raw-token copy path."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from fdt_rlm.lexical_pointer import SparseLexicalPointer

        self.exact_pointer = SparseLexicalPointer(
            hidden_dim=config.dim,
            pointer_dim=config.exact_pointer_dim,
            window=config.exact_pointer_window,
            anchor_bias_init=config.exact_pointer_anchor_bias_init,
        )

    def exact_memory_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        return self.exact_pointer.training_loss(
            hidden,
            input_ids,
            labels,
            attention_mask,
            gate_weight=self.config.exact_pointer_gate_weight,
            min_copy_span=3,
            prompt_sources_only=False,
        )


class CausalFDTAnchorIndexedDualMemoryLM(CausalFDTDualMemoryLM):
    """Working + fuzzy semantic + anchor-indexed exact episodic memory."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not self.anchor_layer_indices:
            raise ValueError("anchor-indexed memory requires at least one anchor layer")
        self.blocks[self.anchor_layer_indices[-1]].anchor.capture_exact_route = True

    def exact_route_indices(self, hidden: torch.Tensor) -> torch.Tensor:
        anchor = self.blocks[self.anchor_layer_indices[-1]].anchor
        indices = getattr(anchor, "last_exact_route_indices", None)
        if indices is None or indices.shape[:2] != hidden.shape[:2]:
            raise RuntimeError("existing fuzzy route was not captured for exact-memory indexing")
        return indices

    def exact_memory_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        route_indices = self.exact_route_indices(hidden)
        return self.exact_pointer.training_loss(
            hidden,
            input_ids,
            labels,
            attention_mask,
            gate_weight=self.config.exact_pointer_gate_weight,
            min_copy_span=3,
            prompt_sources_only=False,
            route_indices=route_indices,
        )

    def build_exact_memory(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_length: int | None = None,
    ):
        from fdt_rlm.lexical_pointer import AnchorIndexedExactMemory

        return AnchorIndexedExactMemory.from_prompt(
            input_ids,
            self.exact_route_indices(hidden),
            attention_mask,
            source_length=source_length,
            chunk_size=self.config.exact_pointer_chunk_size,
            chunk_anchor_count=self.config.exact_pointer_chunk_anchors,
            commit_scores=self.exact_pointer.commit_scores(hidden).detach(),
        )
