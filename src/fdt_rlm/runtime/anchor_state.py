from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AnchorStateMetrics:
    state_norm: float
    state_change: float
    state_conflict: float
    active_anchors: int


class AnchorState:
    def __init__(
        self,
        num_anchors: int,
        state_dim: int,
        alpha: float = 0.85,
        norm_clip: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        self.num_anchors = num_anchors
        self.state_dim = state_dim
        self.alpha = alpha
        self.norm_clip = norm_clip
        self.device = torch.device(device)
        self.state = torch.zeros(num_anchors, state_dim, device=self.device)
        self.confidence = torch.zeros(num_anchors, device=self.device)
        self.age = torch.zeros(num_anchors, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, summary: torch.Tensor, usage: torch.Tensor) -> AnchorStateMetrics:
        summary = summary.to(self.device, dtype=self.state.dtype)
        usage = usage.to(self.device, dtype=self.state.dtype).clamp_min(0)
        if summary.shape != self.state.shape:
            raise ValueError(f"summary shape {tuple(summary.shape)} != {tuple(self.state.shape)}")
        if usage.shape != self.confidence.shape:
            raise ValueError(f"usage shape {tuple(usage.shape)} != {tuple(self.confidence.shape)}")
        old = self.state.clone()
        active = usage > 0
        conflict = torch.zeros((), device=self.device)
        if active.any() and old[active].norm(dim=-1).gt(1e-8).any():
            valid = active & old.norm(dim=-1).gt(1e-8) & summary.norm(dim=-1).gt(1e-8)
            if valid.any():
                conflict = (1.0 - F.cosine_similarity(old[valid], summary[valid], dim=-1)).mean()
        blend = (1.0 - self.alpha) * usage.clamp_max(1.0).unsqueeze(-1)
        self.state = old * (1.0 - blend) + summary * blend
        norms = self.state.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.state = self.state * (self.norm_clip / norms).clamp_max(1.0)
        self.confidence = self.alpha * self.confidence + (1.0 - self.alpha) * usage
        self.age += 1
        self.age[active] = 0
        return AnchorStateMetrics(
            state_norm=float(self.state.norm().item()),
            state_change=float((self.state - old).norm().item()),
            state_conflict=float(conflict.item()),
            active_anchors=int(active.sum().item()),
        )

    def reset(self) -> None:
        self.state.zero_()
        self.confidence.zero_()
        self.age.zero_()

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "state": self.state.clone(),
            "confidence": self.confidence.clone(),
            "age": self.age.clone(),
        }
