from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.causal_lm import (
    AnchorStats,
    BaseCausalLM,
    CausalDepthwiseConvMixer,
    CausalFDTLM,
    CausalFuzzyAnchorLayer,
    CausalTransformerLM,
    FDTBlock,
    MLP,
    RMSNorm,
    TransformerBlock,
    masked_mean,
)


@dataclass
class IncrementalAnchorState:
    numerator: torch.Tensor
    mass: torch.Tensor


def resolve_anchor_layer_indices(config: ModelConfig) -> list[int]:
    if config.anchor_layer_indices:
        indices = sorted(set(int(index) for index in config.anchor_layer_indices))
    else:
        n = config.n_layers
        pattern = config.anchor_layer_pattern
        if pattern == "alternate":
            indices = list(range(1, n, 2))
        elif pattern == "last_half":
            indices = list(range(n // 2, n))
        elif pattern == "middle":
            width = max(1, n // 3)
            start = (n - width) // 2
            indices = list(range(start, start + width))
        elif pattern == "all_but_first_last":
            indices = list(range(1, max(1, n - 1)))
        elif pattern == "all":
            indices = list(range(n))
        elif pattern == "none":
            indices = []
        else:
            raise ValueError(f"Unknown anchor_layer_pattern: {pattern}")
    if any(index < 0 or index >= config.n_layers for index in indices):
        raise ValueError("anchor_layer_indices contains an out-of-range layer")
    return indices


class OptimizedCausalFuzzyAnchorLayer(CausalFuzzyAnchorLayer):
    """Routing-compatible layer with an explicit distance backend."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._normalized_anchor_cache: Optional[tuple[int, int, torch.Tensor, torch.Tensor]] = None

    def train(self, mode: bool = True):
        if mode:
            self._normalized_anchor_cache = None
        return super().train(mode)

    def normalized_anchors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache normalized anchors only in eval mode; training must keep gradients live."""
        if self.training or not self.config.use_cached_normalized_anchors:
            return F.normalize(self.anchor_keys, dim=-1), F.normalize(self.anchor_values, dim=-1)
        key_version = int(self.anchor_keys._version)
        value_version = int(self.anchor_values._version)
        cached = self._normalized_anchor_cache
        if cached is None or cached[0] != key_version or cached[1] != value_version:
            cached = (
                key_version,
                value_version,
                F.normalize(self.anchor_keys, dim=-1),
                F.normalize(self.anchor_values, dim=-1),
            )
            self._normalized_anchor_cache = cached
        return cached[2], cached[3]

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # The historical path remains authoritative during training. This keeps
        # checkpoint behavior exactly compatible while incremental decode uses
        # the cached normalization path below.
        return super().forward(x, attention_mask=attention_mask)

    def membership_logits(self, q: torch.Tensor, anchor_keys: torch.Tensor):
        backend = self.config.routing_backend
        if backend == "matmul":
            return super().membership_logits(q, anchor_keys)
        if backend != "cdist":
            raise ValueError(f"Unknown routing_backend: {backend}")

        distance = torch.cdist(q.float(), anchor_keys.float(), p=2).square()
        cosine = (1.0 - 0.5 * distance).to(dtype=q.dtype)
        distance = distance.to(dtype=q.dtype).clamp_min(0.0)
        if self.config.routing_type == "cosine":
            logits = cosine / max(self.config.cosine_temperature, 1e-5)
        elif self.config.routing_type == "gaussian":
            sigma = self.log_sigma.exp().clamp(self.config.sigma_min, self.config.sigma_max)
            logits = -distance / (2.0 * sigma.view(1, 1, -1).pow(2))
        else:
            raise ValueError(f"Unknown routing_type: {self.config.routing_type}")
        logits = torch.nan_to_num(logits, nan=0.0).clamp(
            -self.config.membership_logit_clip,
            self.config.membership_logit_clip,
        )
        return logits, distance


class OptimizedFDTBlock(FDTBlock):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.use_self_attention = config.use_self_attention
        self.attn_norm = RMSNorm(config.dim)
        self.attn = None
        if self.use_self_attention:
            from fdt_rlm.models.causal_lm import CausalSelfAttention

            self.attn = CausalSelfAttention(config)
        self.local_norm = RMSNorm(config.dim) if config.use_local_mixer else None
        self.local_mixer = CausalDepthwiseConvMixer(config) if config.use_local_mixer else None
        self.anchor_norm = RMSNorm(config.dim)
        self.anchor = OptimizedCausalFuzzyAnchorLayer(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = MLP(config)


def _single_token_anchor_step(
    layer: CausalFuzzyAnchorLayer,
    x: torch.Tensor,
    state: Optional[IncrementalAnchorState],
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, AnchorStats, IncrementalAnchorState]:
    if x.size(1) != 1:
        raise ValueError("incremental anchor update expects exactly one token")
    bsz, _, dim = x.shape
    dtype = x.dtype
    device = x.device
    q = F.normalize(layer.q_proj(x), dim=-1)
    v = layer.v_proj(x)
    if hasattr(layer, "normalized_anchors"):
        anchor_keys, anchor_values = layer.normalized_anchors()
    else:
        anchor_keys = F.normalize(layer.anchor_keys, dim=-1)
        anchor_values = F.normalize(layer.anchor_values, dim=-1)
    logits, full_distance = layer.membership_logits(q, anchor_keys)
    top_logits, indices = torch.topk(logits, k=layer.top_k, dim=-1)
    membership = F.softmax(top_logits.float(), dim=-1).to(dtype=dtype)
    if layer.probe_mode == "uniform_topk":
        membership = torch.full_like(membership, 1.0 / layer.top_k)
    elif layer.probe_mode == "shuffle_membership":
        membership = membership.roll(shifts=1, dims=-1)
    elif layer.probe_mode == "onehot_top1":
        membership = torch.zeros_like(membership)
        membership[..., 0] = 1.0
    membership = membership.clamp_min(1e-8)
    membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    if state is None:
        state = IncrementalAnchorState(
            numerator=torch.zeros(bsz, layer.num_anchors, dim, device=device, dtype=dtype),
            mass=torch.zeros(bsz, layer.num_anchors, device=device, dtype=dtype),
        )
    weights = membership[:, 0] * token_mask[:, 0, None].to(dtype=dtype)
    index = indices[:, 0]
    contribution = weights.unsqueeze(-1) * v[:, 0, None, :]
    # Inference owns this cache, so updating it in-place avoids allocating two
    # dense BxK state tensors for every generated token and every layer.
    state.numerator.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, dim), contribution)
    state.mass.scatter_add_(1, index, weights)
    next_state = state
    selected_numerator = torch.gather(
        next_state.numerator,
        1,
        index.unsqueeze(-1).expand(-1, -1, dim),
    )
    selected_mass = torch.gather(next_state.mass, 1, index).clamp_min(1e-6)
    structural = torch.einsum(
        "bk,bkd->bd",
        membership[:, 0],
        selected_numerator / selected_mass.unsqueeze(-1),
    ).unsqueeze(1)
    latent = torch.einsum("bnk,bnkd->bnd", membership, anchor_values[indices])
    routed = torch.sigmoid(layer.gate) * structural + (1.0 - torch.sigmoid(layer.gate)) * latent
    routed = layer.out_proj(routed)
    if layer.probe_mode == "zero_anchor":
        routed = torch.zeros_like(routed)

    mask = token_mask.to(device=device, dtype=dtype)
    entropy_per_token = -(membership.float() * membership.float().clamp_min(1e-8).log()).sum(-1)
    selected_distance = torch.gather(full_distance, 2, indices)
    cluster_per_token = (membership.float() * selected_distance.float()).sum(-1)
    usage = torch.zeros(layer.num_anchors, device=device)
    usage.scatter_add_(0, indices.reshape(-1), (membership.float() * mask[:, :, None]).reshape(-1))
    load = torch.zeros(layer.num_anchors, device=device)
    load.scatter_add_(
        0,
        indices.reshape(-1),
        mask[:, :, None].float().expand_as(membership).reshape(-1),
    )
    if layer.config.enable_diagnostics:
        diagnostic_indices = indices.detach()
        diagnostic_membership = membership.detach() if layer.config.detach_diagnostics else membership
    else:
        diagnostic_indices = indices.new_empty((0, 0, layer.top_k))
        diagnostic_membership = membership.new_empty((0, 0, layer.top_k))
    stats = AnchorStats(
        indices=diagnostic_indices,
        membership=diagnostic_membership,
        entropy=masked_mean(entropy_per_token, mask),
        usage_prob=usage / usage.sum().clamp_min(1e-8),
        load_prob=(load / load.sum().clamp_min(1e-8)).detach(),
        cluster_loss=masked_mean(cluster_per_token, mask),
        top1_membership=masked_mean(membership[..., 0].float(), mask).detach(),
        membership_margin=masked_mean(
            (membership[..., 0] - membership[..., 1]).float()
            if layer.top_k > 1
            else membership[..., 0].float(),
            mask,
        ).detach(),
    )
    return routed, stats, next_state


def _full_anchor_state(
    layer: CausalFuzzyAnchorLayer,
    x: torch.Tensor,
    token_mask: torch.Tensor,
) -> IncrementalAnchorState:
    """Build the final causal anchor prefix state in one vectorized O(nK) pass."""
    bsz, seq_len, dim = x.shape
    q = F.normalize(layer.q_proj(x), dim=-1)
    v = layer.v_proj(x)
    if hasattr(layer, "normalized_anchors"):
        anchor_keys, _ = layer.normalized_anchors()
    else:
        anchor_keys = F.normalize(layer.anchor_keys, dim=-1)
    logits, _ = layer.membership_logits(q, anchor_keys)
    top_logits, indices = torch.topk(logits, k=layer.top_k, dim=-1)
    if layer.probe_mode == "random_topk":
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).view(1, seq_len, 1)
        anchors = torch.arange(layer.num_anchors, device=x.device, dtype=torch.float32).view(1, 1, -1)
        batches = torch.arange(bsz, device=x.device, dtype=torch.float32).view(bsz, 1, 1)
        random_scores = torch.sin(
            (anchors + 1.0) * 12.9898
            + (positions + 1.0) * 78.233
            + (batches + 1.0) * 37.719
        )
        indices = torch.topk(random_scores, k=layer.top_k, dim=-1).indices
        top_logits = torch.gather(logits, 2, indices)
    membership = F.softmax(top_logits.float(), dim=-1).to(dtype=x.dtype)
    if layer.probe_mode == "uniform_topk":
        membership = torch.full_like(membership, 1.0 / layer.top_k)
    elif layer.probe_mode == "shuffle_membership":
        membership = membership.roll(shifts=1, dims=-1)
    elif layer.probe_mode == "onehot_top1":
        membership = torch.zeros_like(membership)
        membership[..., 0] = 1.0
    if layer.probe_mode != "onehot_top1":
        membership = membership.clamp_min(1e-8)
        membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    mask = token_mask.to(device=x.device, dtype=x.dtype)
    weights = membership * mask[:, :, None]
    flat_indices = indices.reshape(bsz, -1)
    flat_weights = weights.reshape(bsz, -1)
    contributions = (weights.unsqueeze(-1) * v.unsqueeze(2)).reshape(bsz, -1, dim)
    numerator = torch.zeros(bsz, layer.num_anchors, dim, device=x.device, dtype=x.dtype)
    numerator.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, dim), contributions)
    mass = torch.zeros(bsz, layer.num_anchors, device=x.device, dtype=x.dtype)
    mass.scatter_add_(1, flat_indices, flat_weights)
    return IncrementalAnchorState(numerator=numerator, mass=mass)


class InferenceAPIMixin:
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        output = self(input_ids=input_ids, attention_mask=attention_mask)
        return output, {
            "backend": "recompute",
            "input_ids": input_ids.detach(),
            "attention_mask": attention_mask.detach(),
            "length": int(input_ids.size(1)),
        }

    @torch.inference_mode()
    def decode_step(self, input_ids: torch.Tensor, cache: dict[str, Any]):
        if cache["length"] >= self.config.max_seq_len:
            raise ValueError("decode cache reached max_seq_len")
        all_ids = torch.cat((cache["input_ids"], input_ids), dim=1)
        all_mask = torch.ones_like(all_ids)
        output = self(input_ids=all_ids, attention_mask=all_mask)
        cache = {
            "backend": "recompute",
            "input_ids": all_ids.detach(),
            "attention_mask": all_mask.detach(),
            "length": int(all_ids.size(1)),
        }
        return {**output, "logits": output["logits"][:, -1:]}, cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        generated = input_ids
        if use_cache:
            output, cache = self.prefill(generated, torch.ones_like(generated))
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.max_seq_len:
                break
            if use_cache:
                logits = output["logits"][:, -1].float()
            else:
                logits = self(generated, attention_mask=torch.ones_like(generated))["logits"][:, -1].float()
            if temperature > 0:
                next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_id), dim=1)
            if bool((next_id == self.config.eos_token_id).all()):
                break
            if use_cache:
                output, cache = self.decode_step(next_id, cache)
        return generated


class AnchorIncrementalMixin(InferenceAPIMixin):
    def _supports_incremental_anchor_state(self) -> bool:
        return bool(self.config.use_incremental_anchor_state) and all(
            isinstance(block, FDTBlock) and block.attn is None for block in self.blocks
        )

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if not self._supports_incremental_anchor_state():
            if inputs_embeds is not None:
                raise ValueError("conditioned prefill requires incremental anchor routing")
            return super().prefill(input_ids, attention_mask)
        seq_len = int(inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1))
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        if seq_len > self.config.max_seq_len:
            raise ValueError("prefill input exceeds max_seq_len")
        x = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        x = self.drop(x)
        stats = []
        states = []
        local_states = []
        for block in self.blocks:
            if block.local_mixer is not None:
                local_input = block.local_norm(x)
                local_states.append(block.local_mixer.state_from_sequence(local_input))
                x = x + block.local_mixer(local_input, attention_mask=attention_mask)
            else:
                local_states.append(None)
            normalized = block.anchor_norm(x)
            anchor_out, anchor_stats = block.anchor(normalized, attention_mask=attention_mask)
            states.append(_full_anchor_state(block.anchor, normalized, attention_mask))
            x = x + anchor_out
            x = x + block.mlp(block.mlp_norm(x))
            stats.append(anchor_stats)
        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        output = {
            "logits": logits[:, -1:],
            "hidden": hidden[:, -1:],
            "pred_noise": self.noise_head(hidden[:, -1:]),
            "anchor_stats": stats,
        }
        return output, {
            "backend": "anchor_incremental",
            "layers": states,
            "local_layers": local_states,
            "length": seq_len,
        }

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: dict[str, Any],
        token_mask: Optional[torch.Tensor] = None,
    ):
        if cache.get("backend") != "anchor_incremental":
            return super().decode_step(input_ids, cache)
        if input_ids.size(1) != 1:
            raise ValueError("decode_step expects one token")
        if cache["length"] >= self.config.max_seq_len:
            raise ValueError("decode cache reached max_seq_len")
        token_mask = torch.ones_like(input_ids) if token_mask is None else token_mask
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            position = torch.tensor([[cache["length"]]], device=input_ids.device)
            x = x + self.position_embedding(position)
        x = self.drop(x)
        stats = []
        next_layers = []
        next_local_layers = []
        local_layers = cache.get("local_layers", [None] * len(self.blocks))
        for block, state, local_state in zip(self.blocks, cache["layers"], local_layers):
            if block.local_mixer is not None:
                local_out, next_local_state = block.local_mixer.step(
                    block.local_norm(x),
                    local_state,
                    token_mask=token_mask,
                )
                x = x + local_out
            else:
                next_local_state = None
            anchor_out, anchor_stats, next_state = _single_token_anchor_step(
                block.anchor,
                block.anchor_norm(x),
                state,
                token_mask,
            )
            x = x + anchor_out
            x = x + block.mlp(block.mlp_norm(x))
            stats.append(anchor_stats)
            next_layers.append(next_state)
            next_local_layers.append(next_local_state)
        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        output = {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": self.noise_head(hidden),
            "anchor_stats": stats,
        }
        return output, {
            "backend": "anchor_incremental",
            "layers": next_layers,
            "local_layers": next_local_layers,
            "length": cache["length"] + 1,
        }


class NextTransformerLM(InferenceAPIMixin, CausalTransformerLM):
    """Transformer baseline with a real per-layer KV cache for decoding."""

    @staticmethod
    def _apply_rope_at_position(attn, q: torch.Tensor, k: torch.Tensor, position: int):
        rope = attn.rope
        if rope is None:
            return q, k
        cos = rope.cos[:, :, position : position + 1].to(device=q.device, dtype=q.dtype)
        sin = rope.sin[:, :, position : position + 1].to(device=q.device, dtype=q.dtype)
        q = (q * cos) + (rope.rotate_half(q) * sin)
        k = (k * cos) + (rope.rotate_half(k) * sin)
        return q, k

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("prefill input exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
        x = self.drop(x)

        layer_caches = []
        for block in self.blocks:
            normalized = block.norm1(x)
            attn = block.attn
            qkv = attn.qkv(normalized).view(bsz, seq_len, 3, attn.n_heads, attn.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if attn.rope is not None:
                q, k = attn.rope(q, k)

            key_cache = k.new_empty(bsz, attn.n_heads, self.config.max_seq_len, attn.head_dim)
            value_cache = v.new_empty(bsz, attn.n_heads, self.config.max_seq_len, attn.head_dim)
            key_cache[:, :, :seq_len].copy_(k)
            value_cache[:, :, :seq_len].copy_(v)
            layer_caches.append({"key": key_cache, "value": value_cache})

            scores = torch.matmul(q, k.transpose(-2, -1)) / (attn.head_dim**0.5)
            causal = attn.causal_mask[:seq_len, :seq_len].to(device=x.device)
            scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
            key_mask = attention_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            probs = attn.drop(probs)
            attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            x = x + attn.out_proj(attn_out)
            x = x + block.mlp(block.norm2(x))

        hidden = self.norm(x)
        logits = self.lm_head(hidden[:, -1:]).clamp(
            -self.config.lm_logit_clip,
            self.config.lm_logit_clip,
        )
        cached_mask = attention_mask.new_zeros(bsz, self.config.max_seq_len)
        cached_mask[:, :seq_len].copy_(attention_mask)
        output = {
            "logits": logits,
            "hidden": hidden[:, -1:],
            "pred_noise": None,
            "anchor_stats": [],
        }
        return output, {
            "backend": "transformer_kv",
            "layers": layer_caches,
            "attention_mask": cached_mask,
            "length": seq_len,
        }

    @torch.inference_mode()
    def decode_step(self, input_ids: torch.Tensor, cache: dict[str, Any]):
        if cache.get("backend") != "transformer_kv":
            return super().decode_step(input_ids, cache)
        if input_ids.size(1) != 1:
            raise ValueError("decode_step expects one token")
        position = int(cache["length"])
        if position >= self.config.max_seq_len:
            raise ValueError("decode cache reached max_seq_len")

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.full(
                (input_ids.size(0), 1),
                position,
                device=input_ids.device,
                dtype=torch.long,
            )
            x = x + self.position_embedding(positions)
        x = self.drop(x)
        cache["attention_mask"][:, position] = 1

        for block, layer_cache in zip(self.blocks, cache["layers"]):
            normalized = block.norm1(x)
            attn = block.attn
            qkv = attn.qkv(normalized).view(
                input_ids.size(0), 1, 3, attn.n_heads, attn.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = self._apply_rope_at_position(attn, q, k, position)
            layer_cache["key"][:, :, position : position + 1].copy_(k)
            layer_cache["value"][:, :, position : position + 1].copy_(v)

            keys = layer_cache["key"][:, :, : position + 1]
            values = layer_cache["value"][:, :, : position + 1]
            scores = torch.matmul(q, keys.transpose(-2, -1)) / (attn.head_dim**0.5)
            key_mask = cache["attention_mask"][:, None, None, : position + 1].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            probs = attn.drop(probs)
            attn_out = torch.matmul(probs, values).transpose(1, 2).contiguous().view(
                input_ids.size(0), 1, -1
            )
            x = x + attn.out_proj(attn_out)
            x = x + block.mlp(block.norm2(x))

        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        cache["length"] = position + 1
        return {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": None,
            "anchor_stats": [],
        }, cache


class CachedCausalFDTLM(AnchorIncrementalMixin, CausalFDTLM):
    pass


class CausalOptimizedFDTLM(AnchorIncrementalMixin, BaseCausalLM):
    block_cls = OptimizedFDTBlock


class CausalAnchorMixerLM(CausalOptimizedFDTLM):
    pass


class CausalHybridFDTLM(InferenceAPIMixin, BaseCausalLM):
    block_cls = TransformerBlock

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        anchor_indices = set(resolve_anchor_layer_indices(config))
        self.anchor_layer_indices = sorted(anchor_indices)
        self.blocks = nn.ModuleList(
            OptimizedFDTBlock(config) if index in anchor_indices else TransformerBlock(config)
            for index in range(config.n_layers)
        )


def build_next_model(config: ModelConfig) -> BaseCausalLM:
    if config.model_type == "transformer":
        return NextTransformerLM(config)
    if config.model_type == "fdt":
        return CachedCausalFDTLM(config)
    if config.model_type == "fdt_optimized":
        return CausalOptimizedFDTLM(config)
    if config.model_type == "fdt_hybrid":
        return CausalHybridFDTLM(config)
    if config.model_type == "fdt_anchor_mixer":
        return CausalAnchorMixerLM(config)
    if config.model_type == "fdt_v3":
        from fdt_rlm.models.fdt_v3 import CausalFDTv3LM

        return CausalFDTv3LM(config)
    if config.model_type == "fdt_v3_dual_memory":
        from fdt_rlm.models.fdt_v3 import CausalFDTDualMemoryLM

        return CausalFDTDualMemoryLM(config)
    if config.model_type == "fdt_v3_anchor_indexed_dual_memory":
        from fdt_rlm.models.fdt_v3 import CausalFDTAnchorIndexedDualMemoryLM

        return CausalFDTAnchorIndexedDualMemoryLM(config)
    if config.model_type == "fdt_v4":
        from fdt_rlm.models.fdt_v4 import CausalFDTv4LM

        return CausalFDTv4LM(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
