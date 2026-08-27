from __future__ import annotations

import math
from typing import Dict, Iterable

import torch


QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def entropy_from_membership(membership: torch.Tensor) -> torch.Tensor:
    probs = membership.float().clamp_min(1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def normalized_entropy(entropy: torch.Tensor, top_k: int) -> torch.Tensor:
    return entropy.float() / math.log(max(int(top_k), 2))


def summarize(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0}
    result: Dict[str, float] = {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }
    q_values = torch.quantile(values, torch.tensor(QUANTILES, dtype=values.dtype))
    names = ("p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99")
    result.update({name: float(value.item()) for name, value in zip(names, q_values)})
    return result


def membership_gini(membership: torch.Tensor) -> torch.Tensor:
    probs = membership.float().clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs = probs.sort(dim=-1).values
    n = sorted_probs.size(-1)
    weights = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype)
    return (2.0 * (sorted_probs * weights).sum(dim=-1) / n) - ((n + 1.0) / n)


def usage_metrics(
    indices: torch.Tensor,
    membership: torch.Tensor,
    valid_mask: torch.Tensor,
    num_anchors: int,
) -> Dict[str, object]:
    indices = indices.detach().long()
    membership = membership.detach().float()
    valid = valid_mask.detach().bool()
    valid_slots = valid.unsqueeze(-1).expand_as(indices)
    token_count = int(valid.sum().item())
    slot_count = int(valid_slots.sum().item())

    top1_counts = torch.bincount(indices[..., 0][valid], minlength=num_anchors).float()
    inclusion_counts = torch.bincount(indices[valid_slots], minlength=num_anchors).float()
    mass = torch.zeros(num_anchors, device=membership.device, dtype=torch.float32)
    mass.scatter_add_(0, indices[valid_slots], membership[valid_slots])

    coverage = torch.zeros(num_anchors, device=membership.device, dtype=torch.float32)
    valid_indices = indices[valid]
    if valid_indices.numel():
        rows = torch.arange(valid_indices.size(0), device=indices.device).unsqueeze(1)
        marker = torch.zeros(valid_indices.size(0), num_anchors, device=indices.device, dtype=torch.float32)
        marker[rows, valid_indices] = 1.0
        coverage = marker.sum(dim=0)

    top1_prob = top1_counts / max(token_count, 1)
    inclusion_slot_prob = inclusion_counts / max(slot_count, 1)
    mass_share = mass / mass.sum().clamp_min(1e-12)
    coverage_prob = coverage / max(token_count, 1)
    active = mass > 1e-8
    return {
        "valid_tokens": token_count,
        "top1_selection_distribution": top1_prob.cpu().tolist(),
        "topk_slot_distribution": inclusion_slot_prob.cpu().tolist(),
        "membership_mass_distribution": mass_share.cpu().tolist(),
        "token_coverage_distribution": coverage_prob.cpu().tolist(),
        "top1_selection_frequency": float(top1_prob.max().item()),
        "topk_inclusion_frequency": float(inclusion_slot_prob.max().item()),
        "membership_mass_share": float(mass_share.max().item()),
        "token_coverage_frequency": float(coverage_prob.max().item()),
        "global_active_anchors": int(active.sum().item()),
        "dead_anchor_ratio": float((~active).float().mean().item()),
        "mean_active_anchors_per_token": float(valid_indices.size(-1)) if token_count else 0.0,
    }


def mean_pairwise_distances(distributions: Dict[str, Iterable[float]]) -> Dict[str, float]:
    keys = sorted(distributions)
    js_values = []
    cosine_values = []
    tv_values = []
    eps = 1e-12
    for i, left in enumerate(keys):
        p = torch.tensor(list(distributions[left]), dtype=torch.float64).clamp_min(eps)
        p = p / p.sum()
        for right in keys[i + 1 :]:
            q = torch.tensor(list(distributions[right]), dtype=torch.float64).clamp_min(eps)
            q = q / q.sum()
            midpoint = 0.5 * (p + q)
            js_values.append(float((0.5 * ((p * (p / midpoint).log()).sum() + (q * (q / midpoint).log()).sum())).item()))
            cosine_values.append(float((1.0 - torch.dot(p, q) / (p.norm() * q.norm()).clamp_min(eps)).item()))
            tv_values.append(float((0.5 * (p - q).abs().sum()).item()))
    def avg(items):
        return float(sum(items) / len(items)) if items else 0.0
    return {
        "mean_jensen_shannon_divergence": avg(js_values),
        "mean_cosine_distance": avg(cosine_values),
        "mean_total_variation_distance": avg(tv_values),
        "pair_count": len(js_values),
    }
