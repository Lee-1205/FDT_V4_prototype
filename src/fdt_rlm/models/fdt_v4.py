from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.causal_lm import BaseCausalLM, RMSNorm, RotaryEmbedding
from fdt_rlm.models.fdt_v3 import (
    CausalWindowAttention,
    SwiGLU,
    V3FuzzyAnchorLayer,
    _full_anchor_state,
    _single_anchor_step,
)


class RotaryCausalWindowAttention(CausalWindowAttention):
    """FDT local working memory with RoPE applied only to local Q/K."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.transition_alpha = float(config.rope_transition_alpha)
        self.transition_mode = str(config.rope_transition_mode)
        if not 0.0 <= self.transition_alpha <= 1.0:
            raise ValueError("rope_transition_alpha must be in [0, 1]")
        if self.transition_mode not in {"lerp", "phase", "output_blend"}:
            raise ValueError(
                "rope_transition_mode must be lerp, phase, or output_blend"
            )
        base = torch.arange(self.window, dtype=torch.long)
        self.register_buffer(
            "_decode_ring_orders",
            torch.stack([(base + cursor) % self.window for cursor in range(self.window)]),
            persistent=False,
        )

    def _project(self, x: torch.Tensor, position_offset: int = 0):
        q, k, v = super()._project(x)
        stop = position_offset + q.size(-2)
        if position_offset < 0 or stop > self.rope.cos.size(2):
            raise ValueError("RoPE position exceeds configured max_seq_len")
        if self.transition_mode == "phase" and self.transition_alpha < 1.0:
            positions = torch.arange(
                position_offset,
                stop,
                device=q.device,
                dtype=self.rope.inv_freq.dtype,
            )
            frequencies = torch.einsum(
                "n,d->nd",
                positions,
                self.rope.inv_freq.to(device=q.device),
            )
            phase = torch.cat((frequencies, frequencies), dim=-1)
            phase = phase * self.transition_alpha
            cos = phase.cos()[None, None].to(dtype=q.dtype)
            sin = phase.sin()[None, None].to(dtype=q.dtype)
        else:
            cos = self.rope.cos[:, :, position_offset:stop].to(device=q.device, dtype=q.dtype)
            sin = self.rope.sin[:, :, position_offset:stop].to(device=q.device, dtype=q.dtype)
        rotated_q = (q * cos) + (self.rope.rotate_half(q) * sin)
        rotated_k = (k * cos) + (self.rope.rotate_half(k) * sin)
        if self.transition_mode == "phase" or self.transition_alpha >= 1.0:
            q, k = rotated_q, rotated_k
        elif self.transition_mode == "output_blend":
            # The intermediate blend is applied to attention outputs in forward.
            # Keep the projected legacy keys here for endpoint/state compatibility.
            pass
        elif self.transition_alpha > 0.0:
            q = torch.lerp(q, rotated_q, self.transition_alpha)
            k = torch.lerp(k, rotated_k, self.transition_alpha)
        return q, k, v

    def _window_attention_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        bsz = q.size(0)
        chunks = (seq_len + self.window - 1) // self.window
        padded_len = chunks * self.window
        right_pad = padded_len - seq_len
        q = F.pad(q, (0, 0, 0, right_pad))
        k = F.pad(k, (0, 0, self.window, right_pad))
        v = F.pad(v, (0, 0, self.window, right_pad))
        padded_mask = F.pad(attention_mask, (self.window, right_pad), value=False)
        q_chunks = q.view(
            bsz, self.n_heads, chunks, self.window, self.head_dim
        )
        k_windows = k.unfold(2, 2 * self.window, self.window).permute(
            0, 1, 2, 4, 3
        )
        v_windows = v.unfold(2, 2 * self.window, self.window).permute(
            0, 1, 2, 4, 3
        )
        mask_windows = padded_mask.unfold(1, 2 * self.window, self.window)
        query_offsets = torch.arange(self.window, device=q.device).view(
            self.window, 1
        )
        key_offsets = torch.arange(
            -self.window, self.window, device=q.device
        ).view(1, 2 * self.window)
        local_causal = (key_offsets <= query_offsets) & (
            (query_offsets - key_offsets) < self.window
        )
        allowed_base = local_causal.view(
            1, 1, 1, self.window, 2 * self.window
        )
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
            ) * (self.head_dim**-0.5)
        else:
            scores = torch.matmul(q_chunks, k_windows.transpose(-2, -1)) * (
                self.head_dim**-0.5
            )
        allowed = allowed_base & mask_windows[:, None, :, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v_windows)
        query_mask = attention_mask.new_zeros(
            bsz, padded_len, dtype=torch.bool
        )
        query_mask[:, :seq_len] = attention_mask
        out = out * query_mask.view(
            bsz, 1, chunks, self.window, 1
        ).to(dtype=out.dtype)
        return out.reshape(
            bsz, self.n_heads, padded_len, self.head_dim
        )[:, :, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        intermediate_output_blend = (
            self.transition_mode == "output_blend"
            and 0.0 < self.transition_alpha < 1.0
        )
        if not intermediate_output_blend:
            return super().forward(x, attention_mask, return_state)
        if return_state:
            raise ValueError(
                "intermediate output-blend transition has no single-KV cache state"
            )
        bsz, seq_len, dim = x.shape
        mask = (
            torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=x.device, dtype=torch.bool)
        )
        q, k, v = super()._project(x)
        cos = self.rope.cos[:, :, :seq_len].to(device=x.device, dtype=x.dtype)
        sin = self.rope.sin[:, :, :seq_len].to(device=x.device, dtype=x.dtype)
        rope_q = (q * cos) + (self.rope.rotate_half(q) * sin)
        rope_k = (k * cos) + (self.rope.rotate_half(k) * sin)
        legacy_out = self._window_attention_output(q, k, v, mask, seq_len)
        rope_out = self._window_attention_output(rope_q, rope_k, v, mask, seq_len)
        out = torch.lerp(legacy_out, rope_out, self.transition_alpha)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        from fdt_rlm.models.fdt_v3 import _fixed_token_linear

        return _fixed_token_linear(
            self.out_proj,
            out,
            self.inference_group_size if not self.training else 0,
        )

    def step(
        self,
        x: torch.Tensor,
        state,
        token_mask: torch.Tensor,
        position: int,
    ):
        if (
            self.transition_mode == "output_blend"
            and 0.0 < self.transition_alpha < 1.0
        ):
            raise ValueError(
                "intermediate output-blend transition is training/full-recompute only"
            )
        if x.size(1) != 1:
            raise ValueError("window attention step expects one token")
        q, k, v = self._project(x, position_offset=int(position))
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

        # Read the ring in causal time order. Besides making the state contract
        # explicit, this keeps the cached reduction order aligned with the
        # last-token window used by the full attention path after rollover.
        if state.count < self.window:
            key = state.key[:, :, : state.count]
            value = state.value[:, :, : state.count]
            key_mask = state.mask[:, : state.count]
        else:
            order = self._decode_ring_orders[state.cursor]
            key = state.key.index_select(2, order)
            value = state.value.index_select(2, order)
            key_mask = state.mask.index_select(1, order)

        # Match CausalWindowAttention.forward exactly at the operation level.
        # SDPA is mathematically equivalent, but may select a different fused
        # reduction path and perturb downstream top-k anchor boundaries.
        scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = key_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        probs = self.dropout(probs)
        out = torch.matmul(probs, value)
        out = out * current_mask[:, None, :, None].to(dtype=out.dtype)
        out = out.transpose(1, 2).contiguous().view(x.size(0), 1, -1)
        return self.out_proj(out), state


class FDTv4Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.local_norm = RMSNorm(config.dim)
        self.local_attention = RotaryCausalWindowAttention(config)
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


class LoopCorrectionAdapter(nn.Module):
    """Small hidden-state correction head initialized to exact no-op behavior."""

    def __init__(self, hidden_dim: int, rank: int, vocab_size: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden)))


class CausalFDTv4LM(BaseCausalLM):
    """Local RoPE + fuzzy semantic memory + optional exact episodic memory."""

    block_cls = FDTv4Block

    def __init__(self, config: ModelConfig):
        if not config.use_rope:
            raise ValueError("fdt_v4 requires use_rope=True")
        if config.exact_memory_mode not in {"off", "store", "retrieve", "copy"}:
            raise ValueError("exact_memory_mode must be off, store, retrieve, or copy")
        super().__init__(config)
        legacy_max_len = int(config.legacy_position_transition_max_len)
        if legacy_max_len < 0 or legacy_max_len > int(config.max_seq_len):
            raise ValueError("legacy_position_transition_max_len is invalid")
        self.legacy_position_embedding = (
            nn.Embedding(legacy_max_len, config.dim) if legacy_max_len else None
        )
        if self.legacy_position_embedding is not None:
            nn.init.normal_(self.legacy_position_embedding.weight, mean=0.0, std=0.02)
            self.legacy_position_embedding.weight.requires_grad_(False)
        self.transition_alpha = float(config.rope_transition_alpha)
        configured_legacy_scale = float(config.legacy_position_scale)
        self._legacy_scale_tracks_alpha = configured_legacy_scale < 0.0
        self.legacy_position_scale = (
            1.0 - self.transition_alpha
            if self._legacy_scale_tracks_alpha
            else configured_legacy_scale
        )
        if not 0.0 <= self.legacy_position_scale <= 1.0:
            raise ValueError("legacy_position_scale must be in [0, 1] or negative")
        self.gradient_checkpointing = False
        self.register_buffer(
            "_decode_token_mask_bool",
            torch.ones(1, 1, dtype=torch.bool),
            persistent=False,
        )
        from fdt_rlm.models.next_causal_lm import resolve_anchor_layer_indices

        anchor_indices = set(resolve_anchor_layer_indices(config))
        self.anchor_layer_indices = sorted(anchor_indices)
        for index, block in enumerate(self.blocks):
            if index not in anchor_indices:
                block.anchor = None
                block.anchor_norm = None

        self.exact_pointer = None
        exact_enabled = config.exact_memory_enabled or config.exact_memory_mode != "off"
        if exact_enabled:
            from fdt_rlm.lexical_pointer import SparseLexicalPointer

            if not self.anchor_layer_indices:
                raise ValueError("exact memory requires at least one fuzzy anchor layer")
            self.exact_pointer = SparseLexicalPointer(
                hidden_dim=config.dim,
                pointer_dim=config.exact_pointer_dim,
                window=config.exact_pointer_window,
                anchor_bias_init=config.exact_pointer_anchor_bias_init,
            )
            self.blocks[self.anchor_layer_indices[-1]].anchor.capture_exact_route = True
        loop_rank = int(config.loop_controller_rank)
        if loop_rank < 0:
            raise ValueError("loop_controller_rank must be non-negative")
        self.loop_controller = (
            LoopCorrectionAdapter(config.dim, loop_rank, config.vocab_size)
            if loop_rank > 0
            else None
        )

    def _language_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.loop_controller is not None:
            logits = logits + float(self.config.loop_controller_scale) * self.loop_controller(
                hidden
            )
        return logits.clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)

    def set_transition_alpha(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("transition alpha must be in [0, 1]")
        self.transition_alpha = value
        self.config.rope_transition_alpha = value
        for block in self.blocks:
            block.local_attention.transition_alpha = value
        if self._legacy_scale_tracks_alpha:
            self.legacy_position_scale = 1.0 - value

    def set_legacy_position_scale(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("legacy position scale must be in [0, 1]")
        self._legacy_scale_tracks_alpha = False
        self.legacy_position_scale = value
        self.config.legacy_position_scale = value

    def set_anchor_recency_bias(self, value: float) -> None:
        self.config.anchor_recency_bias = float(value)
        for block in self.blocks:
            if block.anchor is not None:
                block.anchor.set_recency_bias(value)

    def _transition_embeddings(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor] = None,
        *,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            x = self.token_embedding(input_ids)
        else:
            x = inputs_embeds
        stop = int(position_offset) + x.size(1)
        if stop > self.config.max_seq_len:
            raise ValueError(
                f"sequence end {stop} > max_seq_len {self.config.max_seq_len}"
            )
        if self.legacy_position_embedding is not None and self.legacy_position_scale > 0.0:
            if stop > self.legacy_position_embedding.num_embeddings:
                raise ValueError(
                    "legacy position transition cannot exceed its preserved table"
                )
            positions = self._position_ids[:, position_offset:stop]
            x = x + self.legacy_position_embedding(positions) * self.legacy_position_scale
        return self.drop(x)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> dict[str, object]:
        if not self.gradient_checkpointing or not self.training:
            x = self._transition_embeddings(input_ids, inputs_embeds)
            anchor_stats = []
            for block in self.blocks:
                x, stats = block(x, attention_mask=attention_mask)
                anchor_stats.extend(stats)
            hidden = self.norm(x)
            logits = None
            if return_logits:
                logits = self._language_logits(hidden)
            return {
                "logits": logits,
                "hidden": hidden,
                "pred_noise": self.noise_head(hidden) if return_logits else None,
                "anchor_stats": anchor_stats,
            }
        x = self._transition_embeddings(input_ids, inputs_embeds)
        anchor_stats: list[dict[str, torch.Tensor]] = []
        entropy_denominator = math.log(max(self.config.top_k, 2))
        for block in self.blocks:
            has_anchor = block.anchor is not None

            def block_forward(state: torch.Tensor, current_block=block):
                updated, stats = current_block(state, attention_mask=attention_mask)
                entropy = (
                    stats[0].entropy
                    if stats
                    else updated.new_zeros((), dtype=torch.float32)
                )
                load_prob = (
                    stats[0].load_prob
                    if stats
                    else updated.new_zeros((self.config.num_anchors,), dtype=torch.float32)
                )
                top1_membership = (
                    stats[0].top1_membership
                    if stats
                    else updated.new_zeros((), dtype=torch.float32)
                )
                return updated, entropy, load_prob, top1_membership

            x, entropy, load_prob, top1_membership = checkpoint(
                block_forward,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            if has_anchor:
                anchor_stats.append({
                    "entropy_normalized": (entropy.float() / entropy_denominator).detach(),
                    "dead_anchor_fraction": load_prob.le(0).float().mean().detach(),
                    "active_anchor_mask": load_prob.gt(0).detach(),
                    "top1_membership": top1_membership.float().detach(),
                })
        hidden = self.norm(x)
        logits = None
        if return_logits:
            logits = self._language_logits(hidden)
        return {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": self.noise_head(hidden) if return_logits else None,
            "anchor_stats": anchor_stats,
        }

    def exact_route_indices(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.exact_pointer is None:
            raise RuntimeError("exact episodic memory is disabled")
        anchor = self.blocks[self.anchor_layer_indices[-1]].anchor
        indices = getattr(anchor, "last_exact_route_indices", None)
        if indices is None or indices.shape[:2] != hidden.shape[:2]:
            raise RuntimeError("fuzzy route was not captured for exact-memory indexing")
        return indices

    def exact_memory_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mode: str = "global_prompt",
        source_chunk_size: int = 256,
        query_chunk_size: int = 16,
        max_global_queries: int = 64,
        copy_source_positions: torch.Tensor | None = None,
        copy_target_mask: torch.Tensor | None = None,
        source_boundary: torch.Tensor | None = None,
        measure_proposal_recall: bool = True,
    ):
        if self.exact_pointer is None:
            raise RuntimeError("exact episodic memory is disabled")
        return self.exact_pointer.training_loss(
            hidden,
            input_ids,
            labels,
            attention_mask,
            gate_weight=self.config.exact_pointer_gate_weight,
            min_copy_span=3,
            prompt_sources_only=True,
            route_indices=self.exact_route_indices(hidden),
            mode=mode,
            source_chunk_size=source_chunk_size,
            query_chunk_size=query_chunk_size,
            max_global_queries=max_global_queries,
            proposal_chunk_size=self.config.exact_pointer_chunk_size,
            proposal_chunk_anchors=self.config.exact_pointer_chunk_anchors,
            max_candidate_chunks=self.config.exact_pointer_candidate_chunks,
            measure_proposal_recall=measure_proposal_recall,
            copy_source_positions=copy_source_positions,
            copy_target_mask=copy_target_mask,
            source_boundary=source_boundary,
        )

    def build_exact_memory(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_length: int | None = None,
        span_end_positions: torch.Tensor | None = None,
        registered_key_mask: torch.Tensor | None = None,
        registered_key_positions: torch.Tensor | None = None,
        registered_payload_ids: torch.Tensor | None = None,
        registered_payload_lengths: torch.Tensor | None = None,
    ):
        if self.exact_pointer is None:
            raise RuntimeError("exact episodic memory is disabled")
        from fdt_rlm.lexical_pointer import AnchorIndexedExactMemory

        key_vectors = F.normalize(self.exact_pointer.project_keys(hidden), dim=-1).to(
            dtype=torch.float16
        )
        return AnchorIndexedExactMemory.from_prompt(
            input_ids,
            self.exact_route_indices(hidden),
            attention_mask,
            source_length=source_length,
            chunk_size=self.config.exact_pointer_chunk_size,
            chunk_anchor_count=self.config.exact_pointer_chunk_anchors,
            commit_scores=self.exact_pointer.commit_scores(hidden).detach(),
            key_vectors=key_vectors,
            span_end_positions=span_end_positions,
            registered_key_mask=registered_key_mask,
            registered_key_positions=registered_key_positions,
            registered_payload_ids=registered_payload_ids,
            registered_payload_lengths=registered_payload_lengths,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError("prefill input exceeds max_seq_len")
        x = self._transition_embeddings(input_ids)
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
        logits = self._language_logits(hidden[:, -1:])
        return {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": None,
            "anchor_stats": stats,
        }, {"backend": "fdt_v4_incremental", "layers": layer_caches, "length": seq_len}

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: dict[str, Any],
        token_mask: Optional[torch.Tensor] = None,
    ):
        if cache.get("backend") != "fdt_v4_incremental":
            raise ValueError("invalid FDT v4 cache")
        if input_ids.size(1) != 1:
            raise ValueError("decode_step expects one token")
        if cache["length"] >= self.config.max_seq_len:
            raise ValueError("decode cache reached max_seq_len")
        token_mask = self._decode_token_mask_bool.expand(input_ids.size(0), 1) if token_mask is None else token_mask
        x = self._transition_embeddings(
            input_ids,
            position_offset=int(cache["length"]),
        )
        stats = []
        for layer_index, (block, (local_state, anchor_state)) in enumerate(zip(self.blocks, cache["layers"])):
            local_out, local_state = block.local_attention.step(
                block.local_norm(x),
                local_state,
                token_mask,
                position=cache["length"],
            )
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
            cache["layers"][layer_index] = (local_state, anchor_state)
        hidden = self.norm(x)
        logits = self._language_logits(hidden)
        cache["length"] += 1
        return {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": None,
            "anchor_stats": stats,
        }, cache
