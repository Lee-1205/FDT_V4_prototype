from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig


@dataclass
class AnchorStats:
    indices: torch.Tensor
    membership: torch.Tensor
    entropy: torch.Tensor
    usage_prob: torch.Tensor
    load_prob: torch.Tensor
    cluster_loss: torch.Tensor
    top1_membership: torch.Tensor
    membership_margin: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.runtime_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_fused:
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("n,d->nd", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads.")
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len) if config.use_rope else None
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:seq_len, :seq_len].to(device=x.device)
        attn = attn.masked_fill(~causal[None, None, :, :], torch.finfo(attn.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
            attn = attn.masked_fill(~key_mask, torch.finfo(attn.dtype).min)
        probs = F.softmax(attn.float(), dim=-1).to(dtype=x.dtype)
        probs = self.drop(probs)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.dim * config.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(config.dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.dim, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalDepthwiseConvMixer(nn.Module):
    """Fixed-window causal token mixer with O(NDk) cost.

    The channel projection starts at zero, so adding this module to an existing
    checkpoint preserves its function until continued training updates it.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        kernel_size = int(config.local_mixer_kernel_size)
        if kernel_size < 2:
            raise ValueError("local_mixer_kernel_size must be at least 2")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            config.dim,
            config.dim,
            kernel_size,
            groups=config.dim,
            bias=False,
        )
        self.channel_mix = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        with torch.no_grad():
            self.depthwise.weight.zero_()
            self.depthwise.weight[:, 0, -1] = 1.0
            self.channel_mix.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x * attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.depthwise(y).transpose(1, 2)
        y = self.channel_mix(F.silu(y))
        if attention_mask is not None:
            y = y * attention_mask.to(dtype=y.dtype, device=y.device).unsqueeze(-1)
        return self.dropout(y)

    def state_from_sequence(self, x: torch.Tensor) -> torch.Tensor:
        keep = self.kernel_size - 1
        if x.size(1) >= keep:
            return x[:, -keep:].detach()
        padding = x.new_zeros(x.size(0), keep - x.size(1), x.size(2))
        return torch.cat((padding, x), dim=1).detach()

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) != 1:
            raise ValueError("local mixer step expects exactly one token")
        if token_mask is not None:
            x = x * token_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        if state is None:
            state = x.new_zeros(x.size(0), self.kernel_size - 1, x.size(2))
        window = torch.cat((state, x), dim=1)
        y = self.depthwise(window.transpose(1, 2)).transpose(1, 2)
        y = self.channel_mix(F.silu(y))
        if token_mask is not None:
            y = y * token_mask.to(dtype=y.dtype, device=y.device).unsqueeze(-1)
        return self.dropout(y), window[:, 1:].detach()


class CausalFuzzyAnchorLayer(nn.Module):
    """Causal sparse token-anchor routing.

    Position t receives only prefix anchor summaries accumulated from positions <= t.
    This preserves the FDT interpretability idea without leaking future tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_anchors = config.num_anchors
        self.top_k = min(config.top_k, config.num_anchors)
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.anchor_keys = nn.Parameter(torch.randn(config.num_anchors, config.dim) * 0.02)
        self.anchor_values = nn.Parameter(torch.randn(config.num_anchors, config.dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(config.num_anchors))
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.probe_mode = "learned"
        self.set_aggregation_impl(config.aggregation_impl)

    def set_probe_mode(self, mode: str) -> None:
        """Select a deterministic inference-only routing counterfactual."""
        valid = {
            "learned",
            "uniform_topk",
            "random_topk",
            "shuffle_membership",
            "onehot_top1",
            "zero_anchor",
        }
        if mode not in valid:
            raise ValueError(f"Unknown probe mode {mode!r}; expected one of {sorted(valid)}")
        self.probe_mode = mode

    def set_aggregation_impl(self, implementation: str) -> None:
        if implementation not in {"dense_reference", "dense_gather_first"}:
            raise ValueError("aggregation implementation must be 'dense_reference' or 'dense_gather_first'")
        self.aggregation_impl = implementation

    def membership_logits(
        self,
        q: torch.Tensor,
        anchor_keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cosine = torch.einsum("bnd,kd->bnk", q, anchor_keys)
        if self.config.routing_type == "cosine":
            logits = cosine / max(self.config.cosine_temperature, 1e-5)
            distance = 1.0 - cosine
        elif self.config.routing_type == "gaussian":
            distance = (2.0 - 2.0 * cosine).clamp_min(0.0)
            sigma = self.log_sigma.exp().clamp(self.config.sigma_min, self.config.sigma_max)
            logits = -distance / (2.0 * sigma.view(1, 1, -1).pow(2))
        else:
            raise ValueError(f"Unknown routing_type: {self.config.routing_type}")
        logits = torch.nan_to_num(logits, nan=0.0).clamp(
            -self.config.membership_logit_clip,
            self.config.membership_logit_clip,
        )
        return logits, distance

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AnchorStats]:
        bsz, seq_len, dim = x.shape
        dtype = x.dtype
        device = x.device

        q = F.normalize(self.q_proj(x), dim=-1)
        v = self.v_proj(x)
        anchor_keys = F.normalize(self.anchor_keys, dim=-1)
        anchor_values = F.normalize(self.anchor_values, dim=-1)

        logits, full_distance = self.membership_logits(q, anchor_keys)
        top_logits, indices = torch.topk(logits, k=self.top_k, dim=-1)
        if self.probe_mode == "random_topk":
            positions = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
            anchors = torch.arange(self.num_anchors, device=device, dtype=torch.float32).view(1, 1, -1)
            batches = torch.arange(bsz, device=device, dtype=torch.float32).view(bsz, 1, 1)
            random_scores = torch.sin(
                (anchors + 1.0) * 12.9898
                + (positions + 1.0) * 78.233
                + (batches + 1.0) * 37.719
            )
            indices = torch.topk(random_scores, k=self.top_k, dim=-1).indices
            top_logits = torch.gather(logits, 2, indices)
        membership = F.softmax(top_logits.float(), dim=-1).to(dtype=dtype)
        if self.probe_mode == "uniform_topk":
            membership = torch.full_like(membership, 1.0 / self.top_k)
        elif self.probe_mode == "shuffle_membership":
            membership = membership.roll(shifts=1, dims=-1)
        elif self.probe_mode == "onehot_top1":
            membership = torch.zeros_like(membership)
            membership[..., 0] = 1.0
        if self.probe_mode != "onehot_top1":
            membership = membership.clamp_min(1e-8)
            membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if attention_mask is None:
            token_mask = torch.ones(bsz, seq_len, device=device, dtype=dtype)
        else:
            token_mask = attention_mask.to(device=device, dtype=dtype)

        selected_values = anchor_values[indices]
        latent_context = torch.einsum("bnk,bnkd->bnd", membership, selected_values)

        if self.aggregation_impl == "dense_reference":
            structural_context = dense_causal_prefix_aggregate_reference(
                membership,
                indices,
                v,
                token_mask,
                self.num_anchors,
            )
        else:
            structural_context = dense_causal_prefix_aggregate(
                membership,
                indices,
                v,
                token_mask,
                self.num_anchors,
            )

        gate = torch.sigmoid(self.gate)
        routed = gate * structural_context + (1.0 - gate) * latent_context
        routed = self.out_proj(routed)
        if self.probe_mode == "zero_anchor":
            routed = torch.zeros_like(routed)

        entropy_per_token = -(membership.float() * membership.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = masked_mean(entropy_per_token, token_mask)
        selected_distance = torch.gather(full_distance, 2, indices)
        cluster_per_token = (membership.float() * selected_distance.float()).sum(dim=-1)
        cluster_loss = masked_mean(cluster_per_token, token_mask)
        metrics_enabled = getattr(self, "runtime_anchor_metrics", self.config.enable_anchor_metrics)
        if metrics_enabled:
            top1_membership = masked_mean(membership[..., 0].float(), token_mask)
            if self.top_k > 1:
                membership_margin = masked_mean(
                    (membership[..., 0] - membership[..., 1]).float(), token_mask
                )
            else:
                membership_margin = top1_membership
        else:
            top1_membership = entropy.new_zeros(())
            membership_margin = entropy.new_zeros(())
        usage = torch.zeros(self.num_anchors, device=device, dtype=torch.float32)
        usage.scatter_add_(0, indices.reshape(-1), (membership.float() * token_mask[:, :, None].float()).reshape(-1))
        usage_prob = usage / usage.sum().clamp_min(1e-8)
        load = torch.zeros(self.num_anchors, device=device, dtype=torch.float32)
        load.scatter_add_(
            0,
            indices.reshape(-1),
            token_mask[:, :, None].float().expand_as(membership).reshape(-1),
        )
        load_prob = load / load.sum().clamp_min(1e-8)

        if self.config.enable_diagnostics:
            diagnostic_indices = indices.detach()
            diagnostic_membership = membership.detach() if self.config.detach_diagnostics else membership
        else:
            diagnostic_indices = indices.new_empty((0, 0, self.top_k))
            diagnostic_membership = membership.new_empty((0, 0, self.top_k))
        stats = AnchorStats(
            indices=diagnostic_indices,
            membership=diagnostic_membership,
            entropy=entropy,
            usage_prob=usage_prob,
            load_prob=load_prob.detach(),
            cluster_loss=cluster_loss,
            top1_membership=top1_membership.detach(),
            membership_margin=membership_margin.detach(),
        )
        return routed, stats


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x, []


class FDTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_self_attention = config.use_self_attention
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config) if self.use_self_attention else None
        self.local_norm = RMSNorm(config.dim) if config.use_local_mixer else None
        self.local_mixer = CausalDepthwiseConvMixer(config) if config.use_local_mixer else None
        self.anchor_norm = RMSNorm(config.dim)
        self.anchor = CausalFuzzyAnchorLayer(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        stats: List[AnchorStats] = []
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        if self.local_mixer is not None:
            x = x + self.local_mixer(self.local_norm(x), attention_mask=attention_mask)
        anchor_out, anchor_stats = self.anchor(self.anchor_norm(x), attention_mask=attention_mask)
        x = x + anchor_out
        x = x + self.mlp(self.mlp_norm(x))
        stats.append(anchor_stats)
        return x, stats


class BaseCausalLM(nn.Module):
    block_cls = TransformerBlock

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[config.pad_token_id].zero_()
        self.position_embedding = None if config.use_rope else nn.Embedding(config.max_seq_len, config.dim)
        if self.position_embedding is not None:
            # PyTorch's default Embedding initialization has unit variance,
            # which overwhelms an already-trained token embedding when a
            # position table is introduced during checkpoint migration.
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([self.block_cls(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.noise_head = nn.Linear(config.dim, config.dim, bias=False)
        self.register_buffer(
            "_position_ids",
            torch.arange(config.max_seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer("_decode_token_mask", torch.ones(1, 1, dtype=torch.long), persistent=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required.")
            x = self.token_embedding(input_ids)
        else:
            x = inputs_embeds
        bsz, seq_len, _ = x.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.config.max_seq_len}")
        if self.position_embedding is not None:
            pos = self._position_ids[:, :seq_len]
            x = x + self.position_embedding(pos)
        x = self.drop(x)

        anchor_stats: List[AnchorStats] = []
        for block in self.blocks:
            x, stats = block(x, attention_mask=attention_mask)
            anchor_stats.extend(stats)
        hidden = self.norm(x)
        logits = self.lm_head(hidden).clamp(-self.config.lm_logit_clip, self.config.lm_logit_clip)
        return {
            "logits": logits,
            "hidden": hidden,
            "pred_noise": self.noise_head(hidden),
            "anchor_stats": anchor_stats,
        }

    def anchor_diversity_loss(self) -> torch.Tensor:
        losses = []
        for block in self.blocks:
            anchor = getattr(block, "anchor", None)
            if anchor is None:
                continue
            for params in (anchor.anchor_keys, anchor.anchor_values):
                normed = F.normalize(params, dim=-1)
                sim = normed @ normed.t()
                eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                off_diag = sim.masked_select(~eye)
                squared = off_diag.pow(2).mean()
                margin_penalty = F.relu(
                    off_diag.abs() - self.config.diversity_margin
                ).pow(2).mean()
                max_pair_penalty = F.relu(
                    off_diag.abs().max() - self.config.diversity_margin
                ).pow(2)
                losses.append(
                    squared
                    + margin_penalty
                    + self.config.diversity_max_weight * max_pair_penalty
                )
        if not losses:
            return self.token_embedding.weight.new_zeros(())
        return torch.stack(losses).mean()


class CausalTransformerLM(BaseCausalLM):
    block_cls = TransformerBlock


class CausalFDTLM(BaseCausalLM):
    block_cls = FDTBlock


def dense_causal_prefix_aggregate_reference(
    membership: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    num_anchors: int,
) -> torch.Tensor:
    """Historical implementation that divides every token-anchor prefix state."""
    bsz, seq_len, _ = membership.shape
    dim = values.size(-1)
    dtype = membership.dtype
    device = membership.device
    contribution = membership.unsqueeze(-1) * values.unsqueeze(2) * token_mask[:, :, None, None]
    denom = membership * token_mask[:, :, None]
    numer = torch.zeros(bsz, seq_len, num_anchors, dim, device=device, dtype=dtype)
    mass = torch.zeros(bsz, seq_len, num_anchors, device=device, dtype=dtype)
    expanded = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    numer.scatter_add_(2, expanded, contribution)
    mass.scatter_add_(2, indices, denom)
    prefix_numer = numer.cumsum(dim=1)
    prefix_mass = mass.cumsum(dim=1).clamp_min(1e-6)
    selected_summary = torch.gather(prefix_numer / prefix_mass.unsqueeze(-1), 2, expanded)
    return torch.einsum("bnk,bnkd->bnd", membership, selected_summary)


def dense_causal_prefix_aggregate(
    membership: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    token_mask: torch.Tensor,
    num_anchors: int,
) -> torch.Tensor:
    """Equivalent dense prefix scan with gather before pointwise division."""
    bsz, seq_len, _ = membership.shape
    dim = values.size(-1)
    dtype = membership.dtype
    device = membership.device
    contribution = membership.unsqueeze(-1) * values.unsqueeze(2) * token_mask[:, :, None, None]
    denom = membership * token_mask[:, :, None]
    numer = torch.zeros(bsz, seq_len, num_anchors, dim, device=device, dtype=dtype)
    mass = torch.zeros(bsz, seq_len, num_anchors, device=device, dtype=dtype)
    expanded = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    numer.scatter_add_(2, expanded, contribution)
    mass.scatter_add_(2, indices, denom)
    prefix_numer = numer.cumsum(dim=1)
    prefix_mass = mass.cumsum(dim=1).clamp_min(1e-6)
    selected_numer = torch.gather(prefix_numer, 2, expanded)
    selected_mass = torch.gather(prefix_mass, 2, indices)
    selected_summary = selected_numer / selected_mass.unsqueeze(-1)
    return torch.einsum("bnk,bnkd->bnd", membership, selected_summary)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def causal_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pad_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is not None:
        label_mask = attention_mask[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(label_mask == 0, -100)
    if pad_token_id is not None:
        shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def anchor_regularization_losses(
    model: BaseCausalLM,
    anchor_stats: List[AnchorStats],
    entropy_target_ratio_start: float = 0.65,
    entropy_target_ratio_end: float = 0.45,
    compute_diversity: bool = True,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    zero = next(model.parameters()).new_zeros(())
    if not anchor_stats:
        return {
            "cluster": zero,
            "usage": zero,
            "entropy": zero,
            "entropy_reward": zero,
            "entropy_over": zero,
            "entropy_target": zero,
            "entropy_normalized": zero,
            "effective_anchors": zero,
            "top1_membership": zero,
            "membership_margin": zero,
            "diversity": zero,
        }

    cluster = torch.stack([s.cluster_loss for s in anchor_stats]).mean()
    entropy_values = torch.stack([s.entropy for s in anchor_stats])
    entropy = entropy_values.mean()

    layer_count = len(anchor_stats)
    entropy_targets = []
    entropy_rewards = []
    entropy_over_losses = []
    normalized_entropies = []
    for layer_index, stats in enumerate(anchor_stats):
        progress = layer_index / max(layer_count - 1, 1)
        target_ratio = (
            entropy_target_ratio_start
            + progress
            * (
                entropy_target_ratio_end
                - entropy_target_ratio_start
            )
        )
        target_ratio = min(max(float(target_ratio), 0.0), 1.0)
        max_entropy = math.log(max(int(stats.membership.size(-1)), 2))
        target = stats.entropy.new_tensor(max_entropy * target_ratio)
        entropy_targets.append(target)
        entropy_rewards.append(torch.minimum(stats.entropy, target))
        entropy_over_losses.append(F.relu(stats.entropy - target).pow(2))
        normalized_entropies.append(stats.entropy / max(max_entropy, 1e-8))

    entropy_target = torch.stack(entropy_targets).mean()
    entropy_reward = torch.stack(entropy_rewards).mean()
    entropy_over = torch.stack(entropy_over_losses).mean()
    entropy_normalized = torch.stack(normalized_entropies).mean()

    usage_losses = []
    for stats in anchor_stats:
        n = stats.usage_prob.numel()
        # Switch-style balancing separates hard Top-K load from fuzzy mass. It
        # balances anchors across the batch without rewarding uniform weights
        # inside every token's selected set.
        usage_losses.append(
            (n * torch.sum(stats.load_prob * stats.usage_prob) - 1.0).clamp_min(0.0)
        )
    usage = torch.stack(usage_losses).mean() if usage_losses else torch.zeros((), device=device)

    top1_membership = torch.stack([s.top1_membership for s in anchor_stats]).mean()
    membership_margin = torch.stack([s.membership_margin for s in anchor_stats]).mean()
    return {
        "cluster": cluster,
        "usage": usage,
        "entropy": entropy,
        "entropy_reward": entropy_reward,
        "entropy_over": entropy_over,
        "entropy_target": entropy_target,
        "entropy_normalized": entropy_normalized,
        "effective_anchors": entropy_values.exp().mean(),
        "top1_membership": top1_membership,
        "membership_margin": membership_margin,
        "diversity": model.anchor_diversity_loss() if compute_diversity else zero,
    }


def build_model(config: ModelConfig) -> BaseCausalLM:
    from fdt_rlm.models.next_causal_lm import build_next_model

    return build_next_model(config)
