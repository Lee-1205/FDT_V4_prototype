from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from .anchor_state import AnchorState
from .recursive_controller import ModelStep


class LocalCausalBackend:
    def __init__(self, model, config, tokenizer, device: torch.device, max_new_tokens: int = 64, state_scale: float = 0.10):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.state_scale = state_scale

    def _forward(self, input_ids: torch.Tensor, anchor_state: AnchorState | None):
        attention_mask = torch.ones_like(input_ids)
        if anchor_state is None or not hasattr(self.model, "token_embedding"):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        confidence = anchor_state.confidence.to(self.device)
        if float(confidence.sum().item()) <= 1e-8:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        state = anchor_state.state.to(self.device)
        pooled = (state * confidence.unsqueeze(-1)).sum(dim=0) / confidence.sum().clamp_min(1e-8)
        pooled = F.normalize(pooled.float(), dim=-1).to(self.model.token_embedding.weight.dtype)
        embeddings = self.model.token_embedding(input_ids)
        embeddings[:, 0, :] = embeddings[:, 0, :] + self.state_scale * pooled
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask)

    def _prefill(self, input_ids: torch.Tensor, anchor_state: AnchorState | None):
        attention_mask = torch.ones_like(input_ids)
        if anchor_state is None or not hasattr(self.model, "token_embedding"):
            return self.model.prefill(input_ids, attention_mask)
        confidence = anchor_state.confidence.to(self.device)
        if float(confidence.sum().item()) <= 1e-8:
            return self.model.prefill(input_ids, attention_mask)
        state = anchor_state.state.to(self.device)
        pooled = (state * confidence.unsqueeze(-1)).sum(dim=0) / confidence.sum().clamp_min(1e-8)
        pooled = F.normalize(pooled.float(), dim=-1).to(self.model.token_embedding.weight.dtype)
        embeddings = self.model.token_embedding(input_ids)
        embeddings[:, 0, :] = embeddings[:, 0, :] + self.state_scale * pooled
        return self.model.prefill(input_ids, attention_mask, inputs_embeds=embeddings)

    @torch.inference_mode()
    def generate_action(self, prompt: str, anchor_state: AnchorState | None) -> ModelStep:
        ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        prompt_budget = max(self.config.max_seq_len - self.max_new_tokens, 1)
        ids = ids[-prompt_budget:]
        generated: List[int] = []
        token_entropies: List[float] = []
        logprobs: List[float] = []
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        use_cache = hasattr(self.model, "prefill") and hasattr(self.model, "decode_step")
        if use_cache:
            try:
                last_out, cache = self._prefill(input_ids, anchor_state)
            except (TypeError, ValueError):
                use_cache = False
                last_out, cache = self._forward(input_ids, anchor_state), None
        else:
            last_out, cache = self._forward(input_ids, anchor_state), None
        for _ in range(self.max_new_tokens):
            logits = last_out["logits"][0, -1].float()
            probs = logits.softmax(dim=-1)
            next_id = int(logits.argmax().item())
            token_entropies.append(float(-(probs * probs.clamp_min(1e-9).log()).sum().item()))
            logprobs.append(float(probs[next_id].clamp_min(1e-9).log().item()))
            if next_id in {self.config.eos_token_id, self.config.pad_token_id}:
                break
            generated.append(next_id)
            ids.append(next_id)
            if self.tokenizer.decode(generated, skip_special_tokens=True).count("}") > 0:
                break
            if use_cache:
                if cache["length"] >= self.config.max_seq_len:
                    break
                token = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                last_out, cache = self.model.decode_step(token, cache)
            else:
                context = ids[-self.config.max_seq_len :]
                input_ids = torch.tensor([context], device=self.device, dtype=torch.long)
                last_out = self._forward(input_ids, anchor_state)

        signals = {
            "token_entropy": sum(token_entropies) / max(len(token_entropies), 1),
            "mean_logprob": sum(logprobs) / max(len(logprobs), 1),
        }
        anchor_summary = None
        anchor_usage = None
        if last_out is not None and last_out["anchor_stats"]:
            stats = last_out["anchor_stats"][-1]
            membership = stats.membership[0].float()
            indices = stats.indices[0]
            hidden = last_out["hidden"][0].float()
            entropy = -(membership * membership.clamp_min(1e-8).log()).sum(dim=-1).mean()
            signals.update(
                {
                    "anchor_entropy": float(entropy.item()),
                    "normalized_anchor_entropy": float(entropy.item() / math.log(max(membership.size(-1), 2))),
                    "effective_anchor_count": float(entropy.exp().item()),
                    "top1_membership": float(membership[:, 0].mean().item()),
                    "top1_top2_margin": float((membership[:, 0] - membership[:, 1]).mean().item()),
                }
            )
            anchors = self.config.num_anchors
            dim = hidden.size(-1)
            numer = torch.zeros(anchors, dim, device=self.device)
            mass = torch.zeros(anchors, device=self.device)
            expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
            contribution = membership.unsqueeze(-1) * hidden.unsqueeze(1)
            numer.scatter_add_(0, expanded.reshape(-1, dim), contribution.reshape(-1, dim))
            mass.scatter_add_(0, indices.reshape(-1), membership.reshape(-1))
            anchor_summary = numer / mass.unsqueeze(-1).clamp_min(1e-8)
            anchor_usage = mass / mass.max().clamp_min(1e-8)
        return ModelStep(
            self.tokenizer.decode(generated, skip_special_tokens=True),
            generated_tokens=len(generated),
            signals=signals,
            anchor_summary=anchor_summary,
            anchor_usage=anchor_usage,
        )
