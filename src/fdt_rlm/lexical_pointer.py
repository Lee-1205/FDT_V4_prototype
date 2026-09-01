from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PointerLoss:
    loss: torch.Tensor
    pointer_loss: torch.Tensor
    gate_loss: torch.Tensor
    commit_loss: torch.Tensor
    copyable_rate: torch.Tensor
    pointer_accuracy: torch.Tensor
    contract_valid: torch.Tensor | None = None
    proposal_recall: torch.Tensor | None = None
    cursor_continuation_rate: torch.Tensor | None = None
    hard_negative_loss: torch.Tensor | None = None
    max_copy_distance: torch.Tensor | None = None
    scanned_source_tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class ExactTokenMemory:
    """Lossless prompt-token storage for the exact-copy path.

    Positions are implicit tensor indices, so the persistent payload is only
    one int32 token ID and one validity bit per source position.
    """

    token_ids: torch.Tensor
    valid_mask: torch.Tensor

    @classmethod
    def from_prompt(
        cls,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        source_length: int | None = None,
    ) -> "ExactTokenMemory":
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        end = input_ids.size(1) if source_length is None else int(source_length)
        if end < 1 or end > input_ids.size(1):
            raise ValueError("source_length must be within the input sequence")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        return cls(
            token_ids=input_ids[:, :end].to(dtype=torch.int32).contiguous().clone(),
            valid_mask=attention_mask[:, :end].to(dtype=torch.bool).contiguous().clone(),
        )

    @property
    def source_length(self) -> int:
        return int(self.token_ids.size(1))

    def token_at(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2 or positions.size(0) != self.token_ids.size(0):
            raise ValueError("positions must have shape [batch, candidates]")
        if bool((positions < 0).any()) or bool((positions >= self.source_length).any()):
            raise IndexError("exact-memory position is out of bounds")
        return self.token_ids.gather(1, positions).long()

    def storage_bytes(self) -> int:
        return self.token_ids.numel() * self.token_ids.element_size() + self.valid_mask.numel()


@dataclass(frozen=True)
class AnchorIndexedExactMemory:
    """Raw-token memory indexed by the model's existing fuzzy anchors."""

    token_ids: torch.Tensor
    valid_mask: torch.Tensor
    anchor_ids: torch.Tensor
    chunk_anchor_ids: torch.Tensor
    chunk_valid: torch.Tensor
    commit_scores: torch.Tensor
    chunk_commit_score: torch.Tensor
    chunk_size: int
    key_vectors: torch.Tensor | None = None
    span_end_positions: torch.Tensor | None = None
    registered_key_mask: torch.Tensor | None = None
    registered_key_positions: torch.Tensor | None = None
    registered_payload_ids: torch.Tensor | None = None
    registered_payload_lengths: torch.Tensor | None = None

    @classmethod
    def from_prompt(
        cls,
        input_ids: torch.Tensor,
        route_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        source_length: int | None = None,
        chunk_size: int = 16,
        chunk_anchor_count: int = 4,
        commit_scores: torch.Tensor | None = None,
        key_vectors: torch.Tensor | None = None,
        span_end_positions: torch.Tensor | None = None,
        registered_key_mask: torch.Tensor | None = None,
        registered_key_positions: torch.Tensor | None = None,
        registered_payload_ids: torch.Tensor | None = None,
        registered_payload_lengths: torch.Tensor | None = None,
    ) -> "AnchorIndexedExactMemory":
        base = ExactTokenMemory.from_prompt(input_ids, attention_mask, source_length)
        end = base.source_length
        if route_indices.ndim != 3 or route_indices.shape[:2] != input_ids.shape:
            raise ValueError("route_indices must have shape [batch, sequence, top_k]")
        chunk_size = max(int(chunk_size), 1)
        anchor_ids = route_indices[:, :end].to(dtype=torch.int16).contiguous().clone()
        chunk_rows = []
        chunk_masks = []
        chunk_commits = []
        if commit_scores is None:
            commit_scores = torch.zeros_like(input_ids, dtype=torch.float32)
        if commit_scores.shape != input_ids.shape:
            raise ValueError("commit_scores must match input_ids")
        stored_commit_scores = (
            commit_scores[:, :end].to(dtype=torch.float16).contiguous().clone()
        )
        for start in range(0, end, chunk_size):
            stop = min(start + chunk_size, end)
            valid = base.valid_mask[:, start:stop]
            anchors = anchor_ids[:, start:stop].long()
            per_batch = []
            for batch_index in range(anchors.size(0)):
                values = anchors[batch_index][valid[batch_index]].reshape(-1)
                counts = torch.bincount(values, minlength=max(int(route_indices.max()) + 1, 1))
                per_batch.append(counts.topk(min(max(int(chunk_anchor_count), 1), counts.numel())).indices)
            chunk_rows.append(torch.stack(per_batch).to(dtype=torch.int16))
            chunk_masks.append(valid.any(dim=1))
            masked_commit = stored_commit_scores[:, start:stop].float() * valid.float()
            chunk_commits.append(masked_commit.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0))
        stored_keys = None
        if key_vectors is not None:
            if key_vectors.ndim != 3 or key_vectors.shape[:2] != input_ids.shape:
                raise ValueError("key_vectors must have shape [batch, sequence, pointer_dim]")
            stored_keys = key_vectors[:, : max(end - 1, 0)].contiguous().clone()
        stored_span_ends = None
        if span_end_positions is not None:
            if span_end_positions.shape != input_ids.shape:
                raise ValueError("span_end_positions must match input_ids")
            stored_span_ends = span_end_positions[:, :end].to(dtype=torch.int32).contiguous().clone()
            positions = torch.arange(end, device=stored_span_ends.device).view(1, -1)
            if bool((stored_span_ends < positions).any()) or bool((stored_span_ends >= end).any()):
                raise ValueError("span_end_positions must be monotonic bounds within the source")
        stored_registered_keys = None
        if registered_key_mask is not None:
            if registered_key_mask.shape != input_ids.shape:
                raise ValueError("registered_key_mask must match input_ids")
            stored_registered_keys = (
                registered_key_mask[:, : max(end - 1, 0)]
                .bool()
                .contiguous()
                .clone()
            )
            transition_valid = base.valid_mask[:, :-1] & base.valid_mask[:, 1:]
            if bool((stored_registered_keys & ~transition_valid).any()):
                raise ValueError("registered Exact Memory keys must precede valid values")
            if not bool(stored_registered_keys.any()):
                raise ValueError("registered_key_mask must declare at least one Exact Memory span")
        payload_fields = (
            registered_key_positions,
            registered_payload_ids,
            registered_payload_lengths,
        )
        if any(value is not None for value in payload_fields):
            if not all(value is not None for value in payload_fields):
                raise ValueError("registered payload positions, ids, and lengths are required together")
            if registered_key_positions.ndim != 2:
                raise ValueError("registered_key_positions must have shape [batch, records]")
            if registered_payload_ids.ndim != 3:
                raise ValueError("registered_payload_ids must have shape [batch, records, tokens]")
            if registered_payload_lengths.shape != registered_key_positions.shape:
                raise ValueError("registered_payload_lengths must match registered_key_positions")
            if registered_payload_ids.shape[:2] != registered_key_positions.shape:
                raise ValueError("registered payload record dimensions must match")
            if registered_key_positions.size(0) != input_ids.size(0):
                raise ValueError("registered payload batch dimension must match input_ids")
            if bool((registered_key_positions < 0).any()) or bool(
                (registered_key_positions >= max(end - 1, 1)).any()
            ):
                raise ValueError("registered payload key position is out of bounds")
            if bool((registered_payload_lengths < 1).any()) or bool(
                (registered_payload_lengths > registered_payload_ids.size(-1)).any()
            ):
                raise ValueError("registered payload length is invalid")
            declared = torch.zeros_like(base.valid_mask[:, :-1])
            declared.scatter_(1, registered_key_positions.long(), True)
            if stored_registered_keys is None:
                stored_registered_keys = declared
            elif not torch.equal(stored_registered_keys, declared):
                raise ValueError("registered payload keys must equal registered_key_mask")
            stored_key_positions = registered_key_positions.to(dtype=torch.int32).contiguous().clone()
            stored_payload_ids = registered_payload_ids.long().contiguous().clone()
            stored_payload_lengths = registered_payload_lengths.to(dtype=torch.int32).contiguous().clone()
        else:
            stored_key_positions = None
            stored_payload_ids = None
            stored_payload_lengths = None
        return cls(
            token_ids=base.token_ids,
            valid_mask=base.valid_mask,
            anchor_ids=anchor_ids,
            chunk_anchor_ids=torch.stack(chunk_rows, dim=1),
            chunk_valid=torch.stack(chunk_masks, dim=1),
            commit_scores=stored_commit_scores,
            chunk_commit_score=torch.stack(chunk_commits, dim=1),
            chunk_size=chunk_size,
            key_vectors=stored_keys,
            span_end_positions=stored_span_ends,
            registered_key_mask=stored_registered_keys,
            registered_key_positions=stored_key_positions,
            registered_payload_ids=stored_payload_ids,
            registered_payload_lengths=stored_payload_lengths,
        )

    @property
    def source_length(self) -> int:
        return int(self.token_ids.size(1))

    def candidate_key_positions(
        self,
        query_anchor_ids: torch.Tensor,
        max_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_anchor_ids.ndim != 2 or query_anchor_ids.size(0) != self.token_ids.size(0):
            raise ValueError("query_anchor_ids must have shape [batch, top_k]")
        shared = self.chunk_anchor_ids.long().unsqueeze(-1).eq(
            query_anchor_ids.long()[:, None, None, :]
        )
        scores = shared.any(dim=-1).sum(dim=-1).float() + self.chunk_commit_score
        scores = scores.masked_fill(~self.chunk_valid, -1.0)
        count = min(max(int(max_chunks), 1), scores.size(1))
        selected = scores.topk(count, dim=1).indices
        offsets = torch.arange(self.chunk_size, device=selected.device)
        positions = selected.unsqueeze(-1) * self.chunk_size + offsets.view(1, 1, -1)
        positions = positions.reshape(selected.size(0), -1)
        valid = positions.lt(self.source_length - 1)
        positions = positions.clamp(max=max(self.source_length - 2, 0))
        valid &= self.valid_mask.gather(1, positions)
        valid &= self.valid_mask.gather(1, positions + 1)
        if self.registered_key_mask is not None:
            valid &= self.registered_key_mask.gather(1, positions)
        return positions, valid

    def storage_bytes(self) -> int:
        tensors = (
            self.token_ids,
            self.valid_mask,
            self.anchor_ids,
            self.chunk_anchor_ids,
            self.chunk_valid,
            self.commit_scores,
            self.chunk_commit_score,
        )
        total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if self.key_vectors is not None:
            total += self.key_vectors.numel() * self.key_vectors.element_size()
        if self.span_end_positions is not None:
            total += self.span_end_positions.numel() * self.span_end_positions.element_size()
        if self.registered_key_mask is not None:
            total += self.registered_key_mask.numel() * self.registered_key_mask.element_size()
        for tensor in (
            self.registered_key_positions,
            self.registered_payload_ids,
            self.registered_payload_lengths,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    def payload_first_tokens(
        self, key_positions: torch.Tensor, fallback: torch.Tensor
    ) -> torch.Tensor:
        if self.registered_key_positions is None:
            return fallback
        matches = key_positions.unsqueeze(-1).eq(
            self.registered_key_positions.long().unsqueeze(1)
        )
        found = matches.any(dim=-1)
        record_indices = matches.to(dtype=torch.int32).argmax(dim=-1)
        batch = torch.arange(key_positions.size(0), device=key_positions.device).view(-1, 1)
        first = self.registered_payload_ids[batch, record_indices, 0].long()
        return torch.where(found, first, fallback)

    def payloads_for_key_positions(
        self, key_positions: torch.Tensor
    ) -> list[list[list[int]]]:
        if self.registered_key_positions is None:
            return []
        result: list[list[list[int]]] = []
        for batch_index in range(key_positions.size(0)):
            batch_payloads = []
            declared = self.registered_key_positions[batch_index].long()
            for position in key_positions[batch_index].long():
                match = declared.eq(position).nonzero(as_tuple=False).flatten()
                if match.numel() == 0:
                    batch_payloads.append([])
                    continue
                record = int(match[0])
                length = int(self.registered_payload_lengths[batch_index, record])
                batch_payloads.append(
                    self.registered_payload_ids[batch_index, record, :length]
                    .detach()
                    .cpu()
                    .tolist()
                )
            result.append(batch_payloads)
        return result

    def inferred_span_end_positions(
        self,
        key_positions: torch.Tensor,
        min_commit: float = 0.5,
    ) -> torch.Tensor:
        """Infer a copy endpoint from the explicitly trained commit head."""
        if key_positions.ndim != 2 or key_positions.size(0) != self.token_ids.size(0):
            raise ValueError("key_positions must have shape [batch, candidates]")
        if bool((key_positions < 0).any()) or bool(
            (key_positions >= max(self.source_length - 1, 1)).any()
        ):
            raise IndexError("exact-memory key position is out of bounds")
        offsets = torch.arange(self.source_length, device=key_positions.device)
        keys = key_positions.long().unsqueeze(-1) + offsets.view(1, 1, -1)
        in_range = keys.lt(self.source_length - 2)
        safe_keys = keys.clamp(min=0, max=max(self.source_length - 1, 0))
        batch = torch.arange(key_positions.size(0), device=key_positions.device).view(-1, 1, 1)
        continuation = (
            in_range
            & self.valid_mask[batch, (safe_keys + 2).clamp(max=self.source_length - 1)]
            & self.commit_scores[batch, safe_keys].ge(float(min_commit))
        )
        first_stop_offset = (~continuation).to(dtype=torch.int32).argmax(dim=-1)
        return key_positions.long() + 1 + first_stop_offset


@dataclass
class LexicalPointerDecodeState:
    source_length: int
    max_activation_steps: int = 4
    max_copy_tokens: int = 32
    source_search_window: int | None = None
    max_cursor_source_occurrences: int = 1
    source_token_position: int | None = None
    source_span_end_position: int | None = None
    copied_tokens: int = 0
    generated_tokens: int = 0
    completed: bool = False
    exact_memory: ExactTokenMemory | None = None
    previous_anchor_ids: tuple[int, ...] | None = None
    cached_indexed_candidates: tuple[torch.Tensor, torch.Tensor] | None = None
    full_scan_attempted: bool = False
    full_scan_count: int = 0
    payload_token_ids: list[int] | None = None
    payload_cursor: int = 0

    def candidate_commit_eligible(self, diagnostics: dict, index: int = 0) -> bool:
        candidates = diagnostics.get("candidate_ids") or []
        positions = diagnostics.get("source_positions") or []
        if not candidates or not positions:
            return False
        occurrences = diagnostics.get("candidate_occurrences") or []
        occurrence_ok = (
            not occurrences
            or int(occurrences[0][index]) <= self.max_cursor_source_occurrences
        )
        continuation = diagnostics.get("cursor_continuation_supported") or []
        if continuation and bool(continuation[0][index]):
            occurrence_ok = True
        return occurrence_ok

    def prepare_logits(
        self,
        pointer,
        base_logits,
        hidden,
        input_ids,
        attention_mask,
        min_gate: float = 0.8,
        anchor_memory: AnchorIndexedExactMemory | None = None,
        query_anchor_ids: torch.Tensor | None = None,
        max_candidate_chunks: int = 4,
        full_scan_fallback: bool = True,
        fallback_margin: float = 0.0,
        candidate_cap: int = 16,
        commit_threshold: float = 0.5,
        hard_copy: bool = False,
        hard_copy_gate_threshold: float = 0.9,
        hard_copy_pointer_threshold: float = 0.9,
        hard_copy_margin_threshold: float = 1.0,
    ):
        if input_ids.size(0) != 1:
            raise ValueError("LexicalPointerDecodeState currently requires batch size 1")
        if self.exact_memory is None:
            if anchor_memory is not None:
                self.exact_memory = ExactTokenMemory(
                    token_ids=anchor_memory.token_ids,
                    valid_mask=anchor_memory.valid_mask,
                )
            else:
                self.exact_memory = ExactTokenMemory.from_prompt(
                    input_ids,
                    attention_mask,
                    source_length=self.source_length,
                )
        elif self.exact_memory.source_length != self.source_length:
            raise ValueError("exact-memory source length changed during decoding")

        if self.payload_token_ids is not None:
            if self.payload_cursor < len(self.payload_token_ids):
                candidate_id = int(self.payload_token_ids[self.payload_cursor])
                forced_logits = torch.full_like(base_logits, -1e4)
                forced_logits[0, candidate_id] = 0.0
                return forced_logits, {
                    "mode": "cursor",
                    "gate": 1.0,
                    "mix_gate": 1.0,
                    "source_positions": [[]],
                    "candidate_ids": [[candidate_id]],
                    "candidate_payload_ids": [[self.payload_token_ids]],
                    "span_end_source": "explicit_payload",
                    "full_scan_attempted": False,
                    "full_scan_count": self.full_scan_count,
                }
            self.payload_token_ids = None
            self.payload_cursor = 0
            self.completed = True

        if self.source_token_position is not None:
            next_position = self.source_token_position + 1
            span_end = (
                self.source_length - 1
                if self.source_span_end_position is None
                else self.source_span_end_position
            )
            if next_position < self.source_length and next_position <= span_end:
                candidate_id = int(self.exact_memory.token_ids[0, next_position])
                forced_logits = torch.full_like(base_logits, -1e4)
                forced_logits[0, candidate_id] = 0.0
                return forced_logits, {
                    "mode": "cursor",
                    "gate": 1.0,
                    "mix_gate": 1.0,
                    "source_positions": [[next_position - 1]],
                    "candidate_ids": [[candidate_id]],
                    "span_end_positions": [[span_end]],
                    "full_scan_attempted": False,
                    "full_scan_count": self.full_scan_count,
                }
            self.source_token_position = None
            self.source_span_end_position = None
            self.completed = True

        if self.completed or self.generated_tokens >= self.max_activation_steps:
            return base_logits, {
                "mode": "base",
                "gate": 0.0,
                "mix_gate": 0.0,
                "source_positions": [],
                "candidate_ids": [],
                "full_scan_attempted": False,
                "full_scan_count": self.full_scan_count,
            }

        indexed_candidates = None
        reused_candidates = False
        exact_needed = (
            min_gate <= 0.0
            or bool(torch.sigmoid(pointer.project_gate(hidden[:, -1])).ge(min_gate).any())
        )
        if exact_needed and anchor_memory is not None and query_anchor_ids is not None:
            current = tuple(sorted(int(value) for value in query_anchor_ids[0].detach().cpu().tolist()))
            previous = self.previous_anchor_ids
            if previous == current:
                indexed_candidates = self.cached_indexed_candidates
                reused_candidates = indexed_candidates is not None
            if indexed_candidates is None:
                indexed_candidates = anchor_memory.candidate_key_positions(
                    query_anchor_ids,
                    max_candidate_chunks,
                )
                self.cached_indexed_candidates = indexed_candidates
            self.previous_anchor_ids = current

        allow_full_scan = (
            bool(full_scan_fallback)
            and exact_needed
            and anchor_memory is not None
            and anchor_memory.key_vectors is not None
        )
        if allow_full_scan:
            self.full_scan_attempted = True
            self.full_scan_count += 1

        mixed_logits, diagnostics = pointer.mix_next_logits(
            base_logits,
            hidden,
            input_ids,
            attention_mask,
            source_length=self.source_length,
            search_window=(
                self.source_length
                if self.source_search_window is None
                else self.source_search_window
            ),
            min_gate=min_gate,
            anchor_memory=anchor_memory,
            query_anchor_ids=query_anchor_ids,
            max_candidate_chunks=max_candidate_chunks,
            indexed_candidates=indexed_candidates,
            full_scan_fallback=allow_full_scan,
            fallback_margin=fallback_margin,
            candidate_cap=candidate_cap,
            commit_threshold=commit_threshold,
        )
        diagnostics["reused_candidates"] = reused_candidates
        diagnostics["full_scan_attempted"] = allow_full_scan
        diagnostics["full_scan_count"] = self.full_scan_count
        diagnostics["mode"] = "mixed"
        explicit_span = diagnostics.get("span_end_source") in {
            "explicit",
            "explicit_payload",
        }
        hard_copy_eligible = (
            bool(hard_copy)
            and float(diagnostics.get("gate", 0.0))
            >= float(hard_copy_gate_threshold)
            and float(diagnostics.get("pointer_confidence", 0.0))
            >= float(hard_copy_pointer_threshold)
            and float(diagnostics.get("score_margin", 0.0))
            >= float(hard_copy_margin_threshold)
            and (
                explicit_span
                or float(diagnostics.get("commit_confidence", 0.0))
                >= float(commit_threshold)
            )
            and (explicit_span or self.candidate_commit_eligible(diagnostics))
        )
        if hard_copy_eligible:
            candidate_id = int(diagnostics["candidate_ids"][0][0])
            forced_logits = torch.full_like(base_logits, -1e4)
            forced_logits[0, candidate_id] = 0.0
            diagnostics["mode"] = "hard_copy"
            diagnostics["mix_gate"] = 1.0
            diagnostics["hard_copy_eligible"] = True
            return forced_logits, diagnostics
        diagnostics["hard_copy_eligible"] = False
        return mixed_logits, diagnostics

    def commit(self, selected_id: int, diagnostics: dict, boundary: bool = False):
        mode = diagnostics.get("mode", "base")
        candidates = diagnostics.get("candidate_ids") or []
        positions = diagnostics.get("source_positions") or []
        if mode == "cursor" and self.payload_token_ids is not None:
            expected = int(self.payload_token_ids[self.payload_cursor])
            if int(selected_id) == expected:
                self.payload_cursor += 1
                self.copied_tokens += 1
            else:
                self.payload_token_ids = None
                self.payload_cursor = 0
                self.completed = True
            self.generated_tokens += 1
            if (
                self.payload_token_ids is not None
                and self.payload_cursor >= len(self.payload_token_ids)
            ):
                self.payload_token_ids = None
                self.payload_cursor = 0
                self.completed = True
            return
        occurrence_ok = (
            diagnostics.get("span_end_source") in {"explicit", "explicit_payload"}
            or self.candidate_commit_eligible(diagnostics)
        )
        followed_pointer = (
            mode in {"mixed", "hard_copy", "cursor"}
            and float(diagnostics.get("mix_gate", 0.0)) > 0.0
            and candidates
            and positions
            and occurrence_ok
            and int(selected_id) == int(candidates[0][0])
        )
        if followed_pointer:
            payloads = diagnostics.get("candidate_payload_ids") or []
            selected_payload = payloads[0][0] if payloads and payloads[0] else []
            if selected_payload:
                self.payload_token_ids = [int(value) for value in selected_payload]
                self.payload_cursor = 1
            else:
                self.source_token_position = int(positions[0][0]) + 1
                span_ends = diagnostics.get("span_end_positions") or []
                if span_ends:
                    self.source_span_end_position = int(span_ends[0][0])
            self.copied_tokens += 1
        elif mode == "cursor":
            self.source_token_position = None
            self.source_span_end_position = None
            self.completed = True

        self.generated_tokens += 1
        span_complete = (
            self.source_token_position is not None
            and self.source_span_end_position is not None
            and self.source_token_position >= self.source_span_end_position
        )
        if (
            self.payload_token_ids is not None
            and self.payload_cursor >= len(self.payload_token_ids)
        ):
            self.payload_token_ids = None
            self.payload_cursor = 0
            self.completed = True
        heuristic_boundary = (
            self.source_span_end_position is None
            and boundary
            and self.copied_tokens >= 2
        )
        if self.source_token_position is not None and (
            self.copied_tokens >= self.max_copy_tokens
            or span_complete
            or heuristic_boundary
        ):
            self.source_token_position = None
            self.source_span_end_position = None
            self.completed = True


def causal_pointer_windows(hidden, input_ids, attention_mask, window: int):
    hidden = hidden.float()
    batch, length, dim = hidden.shape
    prediction_length = max(length - 1, 0)
    keys = hidden[:, :prediction_length]
    candidate_tokens = input_ids[:, 1:]
    key_windows = hidden.new_zeros(batch, prediction_length, window, dim)
    token_windows = input_ids.new_zeros(batch, prediction_length, window)
    valid_windows = torch.zeros(batch, prediction_length, window, dtype=torch.bool, device=hidden.device)
    for offset in range(1, window + 1):
        if offset >= prediction_length + 1:
            break
        count = prediction_length - offset
        if count <= 0:
            continue
        key_windows[:, offset:, offset - 1] = keys[:, :count]
        token_windows[:, offset:, offset - 1] = candidate_tokens[:, :count]
        valid_windows[:, offset:, offset - 1] = (
            attention_mask[:, :count].bool() & attention_mask[:, 1 : count + 1].bool()
        )
    return key_windows, token_windows, valid_windows


def sequential_prompt_copy_mask(
    input_ids,
    labels,
    attention_mask,
    source_positions,
    valid_candidates,
    min_span: int = 2,
    prompt_sources_only: bool = True,
):
    """Return supervised tokens that belong to a prompt-originating copied span."""
    targets = labels[:, 1:]
    supervised = targets.ne(-100)
    batch, prediction_length = targets.shape
    span = max(int(min_span), 1)
    starts = max(prediction_length - span + 1, 0)
    positive = torch.zeros_like(supervised)
    if starts == 0:
        return positive

    run_matches = valid_candidates[:, :starts].clone()
    for offset in range(span):
        target = targets[:, offset : offset + starts].unsqueeze(-1)
        positions = source_positions[:, :starts] + offset
        in_bounds = positions.lt(input_ids.size(1))
        safe_positions = positions.clamp(min=0, max=input_ids.size(1) - 1)
        flat_positions = safe_positions.reshape(batch, -1)
        source_tokens = input_ids.gather(1, flat_positions).reshape_as(positions)
        source_labels = labels.gather(1, flat_positions).reshape_as(positions)
        source_mask = attention_mask.gather(1, flat_positions).reshape_as(positions).bool()
        source_allowed = source_labels.eq(-100) if prompt_sources_only else torch.ones_like(source_mask)
        run_matches &= (
            in_bounds
            & source_mask
            & source_allowed
            & source_tokens.eq(target)
            & supervised[:, offset : offset + starts].unsqueeze(-1)
        )

    run_starts = run_matches.any(dim=-1)
    for offset in range(span):
        positive[:, offset : offset + starts] |= run_starts
    return positive & supervised


class SparseLexicalPointer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pointer_dim: int = 64,
        window: int = 64,
        anchor_bias_init: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pointer_dim = pointer_dim
        self.window = window
        self.q_proj = nn.Linear(hidden_dim, pointer_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, pointer_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.commit_proj = nn.Linear(hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.anchor_match_bias = nn.Parameter(torch.tensor(float(anchor_bias_init)))
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.constant_(self.commit_proj.bias, -2.0)

    @staticmethod
    def _project_fp32(layer: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        """Run a projection in its parameter dtype and expose stable FP32 scores."""
        return layer(hidden.to(dtype=layer.weight.dtype)).float()

    def project_queries(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._project_fp32(self.q_proj, hidden)

    def project_keys(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._project_fp32(self.k_proj, hidden)

    def project_gate(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._project_fp32(self.gate_proj, hidden)

    def project_commit(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._project_fp32(self.commit_proj, hidden)

    def _zero_loss(self, hidden: torch.Tensor, contract_valid: float = 0.0) -> PointerLoss:
        loss = hidden.float().sum() * 0.0
        zero = loss.detach().new_zeros(())
        return PointerLoss(
            loss=loss,
            pointer_loss=zero,
            gate_loss=zero,
            commit_loss=zero,
            copyable_rate=zero,
            pointer_accuracy=zero,
            contract_valid=zero.new_tensor(float(contract_valid)),
            proposal_recall=zero,
            cursor_continuation_rate=zero,
            hard_negative_loss=zero,
            max_copy_distance=zero,
            scanned_source_tokens=zero,
        )

    @staticmethod
    def _bounded_query_positions(
        query_positions: torch.Tensor,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        limit: int,
    ) -> torch.Tensor:
        limit = max(int(limit), 1)
        if query_positions.numel() <= limit:
            return query_positions
        sorted_source = source_tokens.sort().values
        offsets = torch.searchsorted(sorted_source, target_tokens)
        safe_offsets = offsets.clamp(max=max(sorted_source.numel() - 1, 0))
        appears = offsets.lt(sorted_source.numel()) & sorted_source.gather(0, safe_offsets).eq(target_tokens)
        likely_copy = query_positions[appears]
        other = query_positions[~appears]
        return torch.cat((likely_copy, other), dim=0)[:limit]

    def _global_prompt_training_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_weight: float,
        min_copy_span: int,
        route_indices: torch.Tensor | None,
        source_chunk_size: int,
        query_chunk_size: int,
        max_global_queries: int,
        proposal_chunk_size: int,
        max_candidate_chunks: int,
        hard_negative_weight: float,
        hard_negative_margin: float,
    ) -> PointerLoss:
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError("hidden must have shape [batch, sequence, hidden_dim]")
        if labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
            raise ValueError("labels and attention_mask must match input_ids")
        if route_indices is not None and route_indices.shape[:2] != input_ids.shape:
            raise ValueError("route_indices must match the input sequence")

        active = attention_mask.bool()
        supervised_tokens = active & labels.ne(-100)
        if bool((supervised_tokens & labels.ne(input_ids)).any()):
            raise ValueError("supervised labels must equal their aligned input token IDs")
        prompt_tokens = active & labels.eq(-100)
        source_transitions = (
            prompt_tokens[:, :-1]
            & prompt_tokens[:, 1:]
            & active[:, :-1]
            & active[:, 1:]
        )
        query_transitions = supervised_tokens[:, 1:] & active[:, :-1]
        valid_rows = source_transitions.any(dim=1) & query_transitions.any(dim=1)
        if not bool(valid_rows.any()):
            return self._zero_loss(hidden, contract_valid=0.0)

        source_chunk_size = max(int(source_chunk_size), 1)
        query_chunk_size = max(int(query_chunk_size), 1)
        proposal_chunk_size = max(int(proposal_chunk_size), 1)
        min_copy_span = max(int(min_copy_span), 1)
        prediction_length = input_ids.size(1) - 1
        queries = F.normalize(self.project_queries(hidden[:, :prediction_length]), dim=-1)
        keys = F.normalize(self.project_keys(hidden[:, :prediction_length]), dim=-1)
        gate_logits = self.project_gate(hidden[:, :prediction_length]).squeeze(-1)
        commit_logits = self.project_commit(hidden[:, :prediction_length]).squeeze(-1)
        scale = self.logit_scale.exp().clamp(max=50.0)
        route_bias = F.softplus(self.anchor_match_bias)

        examined = torch.zeros_like(query_transitions)
        activation_positive = torch.zeros_like(query_transitions)
        cursor_positive = torch.zeros_like(query_transitions)
        nll_terms = []
        hard_terms = []
        accuracy_terms = []
        proposal_terms = []
        distance_terms = []
        scanned_source_tokens = 0

        for batch_index in range(input_ids.size(0)):
            if not bool(valid_rows[batch_index]):
                continue
            source_positions = source_transitions[batch_index].nonzero(as_tuple=False).flatten()
            query_positions = query_transitions[batch_index].nonzero(as_tuple=False).flatten()
            query_positions = query_positions[query_positions.gt(source_positions.min())]
            if query_positions.numel() == 0:
                continue
            source_tokens = input_ids[batch_index, source_positions + 1]
            query_targets = labels[batch_index, query_positions + 1]
            query_positions = self._bounded_query_positions(
                query_positions,
                query_targets,
                source_tokens,
                max_global_queries,
            )
            scanned_source_tokens += int(source_positions.numel())
            source_chunk_ids, source_chunk_slots = torch.unique(
                torch.div(source_positions, proposal_chunk_size, rounding_mode="floor"),
                sorted=True,
                return_inverse=True,
            )

            for query_start in range(0, query_positions.numel(), query_chunk_size):
                query_stop = min(query_start + query_chunk_size, query_positions.numel())
                q_positions = query_positions[query_start:query_stop]
                q_vectors = queries[batch_index, q_positions]
                targets = labels[batch_index, q_positions + 1]
                examined[batch_index, q_positions] = True
                count = q_positions.numel()
                denominator = q_vectors.new_full((count,), float("-inf"))
                numerator = q_vectors.new_full((count,), float("-inf"))
                positive_max = q_vectors.new_full((count,), float("-inf"))
                negative_max = q_vectors.new_full((count,), float("-inf"))
                best_scores = q_vectors.new_full((count,), float("-inf"))
                best_tokens = targets.new_zeros((count,))
                proposal_scores = q_vectors.new_zeros((count, source_chunk_ids.numel()))
                positive_chunk_counts = q_vectors.new_zeros((count, source_chunk_ids.numel()))
                activation_any = torch.zeros(count, dtype=torch.bool, device=input_ids.device)
                cursor_any = torch.zeros_like(activation_any)

                for source_start in range(0, source_positions.numel(), source_chunk_size):
                    source_stop = min(source_start + source_chunk_size, source_positions.numel())
                    s_positions = source_positions[source_start:source_stop]
                    candidate_tokens = input_ids[batch_index, s_positions + 1]
                    scores = torch.matmul(q_vectors, keys[batch_index, s_positions].transpose(0, 1)) * scale
                    pair_valid = s_positions.view(1, -1).lt(q_positions.view(-1, 1))
                    if route_indices is not None:
                        query_routes = route_indices[batch_index, q_positions].long()
                        source_top1 = route_indices[batch_index, s_positions, 0].long()
                        shared_route = query_routes.unsqueeze(-1).eq(
                            source_top1.view(1, 1, -1)
                        ).any(dim=1)
                        scores = scores + route_bias * shared_route.to(scores.dtype)
                    else:
                        shared_route = torch.zeros_like(pair_valid)
                    scores = scores.masked_fill(~pair_valid, float("-inf"))

                    def offset_matches(offset: int) -> torch.Tensor:
                        query_token_positions = q_positions + 1 + offset
                        source_token_positions = s_positions + 1 + offset
                        query_in_bounds = query_token_positions.lt(input_ids.size(1))
                        source_in_bounds = source_token_positions.lt(input_ids.size(1))
                        safe_queries = query_token_positions.clamp(max=input_ids.size(1) - 1)
                        safe_sources = source_token_positions.clamp(max=input_ids.size(1) - 1)
                        query_ok = (
                            query_in_bounds
                            & active[batch_index, safe_queries]
                            & labels[batch_index, safe_queries].ne(-100)
                        )
                        source_ok = (
                            source_in_bounds
                            & active[batch_index, safe_sources]
                            & labels[batch_index, safe_sources].eq(-100)
                        )
                        return (
                            query_ok.view(-1, 1)
                            & source_ok.view(1, -1)
                            & input_ids[batch_index, safe_queries].view(-1, 1).eq(
                                input_ids[batch_index, safe_sources].view(1, -1)
                            )
                        )

                    span_matches = pair_valid & offset_matches(0)
                    for offset in range(1, min_copy_span):
                        span_matches &= offset_matches(offset)
                    continuation_matches = pair_valid & offset_matches(0) & offset_matches(1)
                    activation_any |= span_matches.any(dim=1)
                    cursor_any |= continuation_matches.any(dim=1)

                    chunk_denominator = torch.logsumexp(scores, dim=1)
                    chunk_numerator = torch.logsumexp(
                        scores.masked_fill(~span_matches, float("-inf")), dim=1
                    )
                    denominator = torch.logaddexp(denominator, chunk_denominator)
                    numerator = torch.logaddexp(numerator, chunk_numerator)
                    positive_max = torch.maximum(
                        positive_max,
                        scores.masked_fill(~span_matches, float("-inf")).max(dim=1).values,
                    )
                    negative_max = torch.maximum(
                        negative_max,
                        scores.masked_fill(~pair_valid | span_matches, float("-inf")).max(dim=1).values,
                    )
                    chunk_best, chunk_best_offsets = scores.max(dim=1)
                    better = chunk_best.gt(best_scores)
                    best_scores = torch.maximum(best_scores, chunk_best)
                    best_tokens = torch.where(
                        better,
                        candidate_tokens.gather(0, chunk_best_offsets),
                        best_tokens,
                    )
                    chunk_slots = source_chunk_slots[source_start:source_stop].view(1, -1).expand(count, -1)
                    proposal_scores.scatter_add_(1, chunk_slots, shared_route.float())
                    positive_chunk_counts.scatter_add_(1, chunk_slots, span_matches.float())
                    positive_distances = q_positions.view(-1, 1) - s_positions.view(1, -1)
                    if bool(span_matches.any()):
                        distance_terms.append(
                            positive_distances.masked_fill(~span_matches, 0).max().detach()
                        )

                activation_positive[batch_index, q_positions] = activation_any
                cursor_positive[batch_index, q_positions] = cursor_any
                valid_positive = torch.isfinite(numerator) & torch.isfinite(denominator)
                if bool(valid_positive.any()):
                    nll_terms.append((denominator - numerator)[valid_positive])
                    accuracy_terms.append(best_tokens[valid_positive].eq(targets[valid_positive]).float())
                    hard_valid = valid_positive & torch.isfinite(negative_max)
                    if bool(hard_valid.any()):
                        hard_terms.append(
                            F.relu(
                                float(hard_negative_margin)
                                + negative_max[hard_valid]
                                - positive_max[hard_valid]
                            )
                        )
                    if route_indices is None:
                        proposal_terms.append(torch.zeros_like(valid_positive[valid_positive], dtype=torch.float32))
                    else:
                        proposal_count = min(
                            max(int(max_candidate_chunks), 1), proposal_scores.size(1)
                        )
                        proposed = proposal_scores.topk(proposal_count, dim=1).indices
                        proposal_hit = positive_chunk_counts.gt(0).gather(1, proposed).any(dim=1)
                        proposal_terms.append(proposal_hit[valid_positive].float())

        pointer_nll = (
            torch.cat(nll_terms).mean()
            if nll_terms
            else queries.sum() * 0.0
        )
        hard_negative_loss = (
            torch.cat(hard_terms).mean()
            if hard_terms
            else queries.sum() * 0.0
        )
        pointer_loss = pointer_nll + float(hard_negative_weight) * hard_negative_loss
        gate_positive = examined & activation_positive
        gate_negative = examined & ~activation_positive
        gate_terms = []
        if bool(gate_positive.any()):
            gate_terms.append(
                F.binary_cross_entropy_with_logits(
                    gate_logits[gate_positive], torch.ones_like(gate_logits[gate_positive])
                )
            )
        if bool(gate_negative.any()):
            gate_terms.append(
                F.binary_cross_entropy_with_logits(
                    gate_logits[gate_negative], torch.zeros_like(gate_logits[gate_negative])
                )
            )
        gate_loss = torch.stack(gate_terms).mean() if gate_terms else gate_logits.sum() * 0.0

        commit_positive = examined & cursor_positive
        commit_negative = examined & ~cursor_positive
        commit_terms = []
        if bool(commit_positive.any()):
            commit_terms.append(
                F.binary_cross_entropy_with_logits(
                    commit_logits[commit_positive], torch.ones_like(commit_logits[commit_positive])
                )
            )
        if bool(commit_negative.any()):
            commit_terms.append(
                F.binary_cross_entropy_with_logits(
                    commit_logits[commit_negative], torch.zeros_like(commit_logits[commit_negative])
                )
            )
        commit_loss = torch.stack(commit_terms).mean() if commit_terms else commit_logits.sum() * 0.0
        loss = pointer_loss + float(gate_weight) * (gate_loss + commit_loss)
        examined_count = examined.sum().clamp_min(1)
        return PointerLoss(
            loss=loss,
            pointer_loss=pointer_loss.detach(),
            gate_loss=gate_loss.detach(),
            commit_loss=commit_loss.detach(),
            copyable_rate=gate_positive.sum().float().div(examined_count).detach(),
            pointer_accuracy=(
                torch.cat(accuracy_terms).mean().detach()
                if accuracy_terms
                else loss.detach().new_zeros(())
            ),
            contract_valid=valid_rows.float().mean().detach(),
            proposal_recall=(
                torch.cat(proposal_terms).mean().detach()
                if proposal_terms
                else loss.detach().new_zeros(())
            ),
            cursor_continuation_rate=commit_positive.sum().float().div(examined_count).detach(),
            hard_negative_loss=hard_negative_loss.detach(),
            max_copy_distance=(
                torch.stack(distance_terms).max().to(dtype=loss.dtype)
                if distance_terms
                else loss.detach().new_zeros(())
            ),
            scanned_source_tokens=loss.detach().new_tensor(float(scanned_source_tokens)),
        )

    def _explicit_copy_training_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        copy_source_positions: torch.Tensor,
        copy_target_mask: torch.Tensor,
        source_boundary: torch.Tensor,
        gate_weight: float,
        route_indices: torch.Tensor | None,
        source_chunk_size: int,
        query_chunk_size: int,
        proposal_chunk_size: int,
        proposal_chunk_anchors: int,
        max_candidate_chunks: int,
        measure_proposal_recall: bool,
        hard_negative_weight: float,
        hard_negative_margin: float,
    ) -> PointerLoss:
        """Train exact selection from the dataset's lossless source mapping.

        ``copy_source_positions`` contains the source *token* position for each
        supervised copied target. Position zero cannot be selected because an
        autoregressive key must precede the copied token.
        """
        shape = input_ids.shape
        if hidden.ndim != 3 or hidden.shape[:2] != shape:
            raise ValueError("hidden must have shape [batch, sequence, hidden_dim]")
        for name, value in (
            ("labels", labels),
            ("attention_mask", attention_mask),
            ("copy_source_positions", copy_source_positions),
            ("copy_target_mask", copy_target_mask),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must match input_ids")
        if source_boundary.shape == (shape[0],):
            boundaries = source_boundary.long().view(-1, 1).expand_as(input_ids)
        elif source_boundary.shape == shape:
            boundaries = source_boundary.long()
        else:
            raise ValueError("source_boundary must have shape [batch] or [batch, sequence]")
        if route_indices is not None and route_indices.shape[:2] != shape:
            raise ValueError("route_indices must match input_ids")

        active = attention_mask.bool()
        declared_targets = copy_target_mask.bool()
        if not bool(declared_targets.any()):
            raise ValueError("explicit exact-copy contract has no target tokens")
        if bool((declared_targets & ~active).any()):
            raise ValueError("explicit copy targets must be active tokens")
        if bool(labels[declared_targets].eq(-100).any()):
            raise ValueError("explicit copy targets require supervised labels")
        targets = declared_targets
        target_rows, target_positions = targets.nonzero(as_tuple=True)
        if bool(target_positions.eq(0).any()):
            raise ValueError("copy targets require a preceding autoregressive query token")
        mapped_tokens = copy_source_positions[target_rows, target_positions].long()
        target_boundaries = boundaries[target_rows, target_positions]
        valid_mapping = (
            mapped_tokens.gt(0)
            & mapped_tokens.lt(target_boundaries)
            & mapped_tokens.lt(input_ids.size(1))
            & mapped_tokens.lt(target_positions)
            & target_positions.ge(target_boundaries)
        )
        if not bool(valid_mapping.all()):
            raise ValueError("copy_source_positions must map targets to earlier prompt tokens")
        mapped_ids = input_ids[target_rows, mapped_tokens]
        if bool(mapped_ids.ne(labels[target_rows, target_positions]).any()):
            raise ValueError("explicit copy source token does not equal its target label")
        if bool((~active[target_rows, mapped_tokens]).any()) or bool(
            labels[target_rows, mapped_tokens].ne(-100).any()
        ):
            raise ValueError("explicit copy sources must be active unsupervised prompt tokens")

        prediction_length = input_ids.size(1) - 1
        queries = F.normalize(self.project_queries(hidden[:, :prediction_length]), dim=-1)
        keys = F.normalize(self.project_keys(hidden[:, :prediction_length]), dim=-1)
        gate_logits = self.project_gate(hidden[:, :prediction_length]).squeeze(-1)
        commit_logits = self.project_commit(hidden).squeeze(-1)
        scale = self.logit_scale.exp().clamp(max=50.0)
        route_bias = F.softplus(self.anchor_match_bias)
        source_chunk_size = max(int(source_chunk_size), 1)
        query_chunk_size = max(int(query_chunk_size), 1)

        nll_terms: list[torch.Tensor] = []
        hard_terms: list[torch.Tensor] = []
        accuracy_terms: list[torch.Tensor] = []
        proposal_terms: list[torch.Tensor] = []
        distance_terms: list[torch.Tensor] = []
        scanned_source_tokens = 0
        gate_positive = torch.zeros_like(gate_logits, dtype=torch.bool)
        supervised_predictions = active[:, 1:] & labels[:, 1:].ne(-100)
        gate_positive |= copy_target_mask[:, 1:].bool() & supervised_predictions
        commit_positive = torch.zeros_like(commit_logits, dtype=torch.bool)
        commit_examined = torch.zeros_like(commit_logits, dtype=torch.bool)

        for batch_index in range(input_ids.size(0)):
            row_targets = targets[batch_index].nonzero(as_tuple=False).flatten()
            if row_targets.numel() == 0:
                continue
            boundary_values = boundaries[batch_index, row_targets]
            if not bool(boundary_values.eq(boundary_values[0]).all()):
                raise ValueError("all exact-copy targets in a row must share one source boundary")
            boundary = int(boundary_values[0])
            if boundary < 2 or boundary > input_ids.size(1):
                raise ValueError("source_boundary is outside the valid prompt range")
            source_token_positions = torch.arange(1, boundary, device=input_ids.device)
            source_key_positions = source_token_positions - 1
            source_valid = (
                active[batch_index, source_token_positions]
                & active[batch_index, source_key_positions]
                & labels[batch_index, source_token_positions].eq(-100)
            )
            source_token_positions = source_token_positions[source_valid]
            source_key_positions = source_key_positions[source_valid]
            if source_key_positions.numel() == 0:
                raise ValueError("exact-copy row has no valid prompt source transitions")
            scanned_source_tokens += int(source_key_positions.numel())
            proposal_chunk_size = max(int(proposal_chunk_size), 1)
            proposal_chunk_anchors = max(int(proposal_chunk_anchors), 1)
            proposal_anchor_rows: list[torch.Tensor] = []
            proposal_commit_rows: list[torch.Tensor] = []
            if route_indices is not None and measure_proposal_recall:
                for chunk_start in range(0, boundary, proposal_chunk_size):
                    chunk_stop = min(chunk_start + proposal_chunk_size, boundary)
                    chunk_positions = torch.arange(
                        chunk_start, chunk_stop, device=input_ids.device
                    )
                    chunk_valid = active[batch_index, chunk_positions]
                    route_values = route_indices[
                        batch_index, chunk_positions
                    ][chunk_valid].long().reshape(-1)
                    if route_values.numel() == 0:
                        raise ValueError("exact-copy proposal chunk has no active routes")
                    anchor_count = max(int(route_indices.max()) + 1, 1)
                    counts = torch.bincount(route_values, minlength=anchor_count)
                    proposal_anchor_rows.append(
                        counts.topk(min(proposal_chunk_anchors, counts.numel())).indices
                    )
                    proposal_commit_rows.append(
                        torch.sigmoid(
                            commit_logits[batch_index, chunk_positions].detach()
                        )[chunk_valid].mean()
                    )
                proposal_anchor_ids = torch.stack(proposal_anchor_rows)
                proposal_commit_scores = torch.stack(proposal_commit_rows)

            for query_start in range(0, row_targets.numel(), query_chunk_size):
                selected_targets = row_targets[query_start : query_start + query_chunk_size]
                query_positions = selected_targets - 1
                mapped_source_tokens = copy_source_positions[
                    batch_index, selected_targets
                ].long()
                mapped_keys = mapped_source_tokens - 1
                q_vectors = queries[batch_index, query_positions]
                if route_indices is not None and measure_proposal_recall:
                    query_routes = route_indices[batch_index, query_positions].long()
                    shared_route = proposal_anchor_ids.unsqueeze(0).unsqueeze(-1).eq(
                        query_routes.unsqueeze(1).unsqueeze(1)
                    )
                    chunk_scores = shared_route.any(dim=-1).sum(dim=-1).float()
                    chunk_scores = chunk_scores + proposal_commit_scores.unsqueeze(0)
                    proposal_count = min(
                        max(int(max_candidate_chunks), 1), chunk_scores.size(1)
                    )
                    proposed_chunks = chunk_scores.topk(proposal_count, dim=1).indices
                    mapped_chunks = torch.div(
                        mapped_keys,
                        proposal_chunk_size,
                        rounding_mode="floor",
                    )
                    proposal_terms.append(
                        proposed_chunks.eq(mapped_chunks.unsqueeze(1)).any(dim=1).float()
                    )
                denominator = q_vectors.new_full((q_vectors.size(0),), float("-inf"))
                positive_scores = q_vectors.new_full((q_vectors.size(0),), float("-inf"))
                negative_max = q_vectors.new_full((q_vectors.size(0),), float("-inf"))
                best_scores = q_vectors.new_full((q_vectors.size(0),), float("-inf"))
                best_tokens = labels[batch_index, selected_targets].new_zeros(
                    (q_vectors.size(0),)
                )

                for source_start in range(0, source_key_positions.numel(), source_chunk_size):
                    source_stop = min(
                        source_start + source_chunk_size, source_key_positions.numel()
                    )
                    key_positions = source_key_positions[source_start:source_stop]
                    token_positions = source_token_positions[source_start:source_stop]
                    scores = torch.matmul(
                        q_vectors,
                        keys[batch_index, key_positions].transpose(0, 1),
                    ) * scale
                    causal = token_positions.view(1, -1).lt(selected_targets.view(-1, 1))
                    if route_indices is not None:
                        query_routes = route_indices[batch_index, query_positions].long()
                        source_top1 = route_indices[batch_index, key_positions, 0].long()
                        shared_route = query_routes.unsqueeze(-1).eq(
                            source_top1.view(1, 1, -1)
                        ).any(dim=1)
                        scores = scores + route_bias * shared_route.to(scores.dtype)
                    scores = scores.masked_fill(~causal, float("-inf"))
                    positives = key_positions.view(1, -1).eq(mapped_keys.view(-1, 1))
                    positives &= causal
                    denominator = torch.logaddexp(denominator, torch.logsumexp(scores, dim=1))
                    positive_scores = torch.maximum(
                        positive_scores,
                        scores.masked_fill(~positives, float("-inf")).max(dim=1).values,
                    )
                    negative_max = torch.maximum(
                        negative_max,
                        scores.masked_fill(~causal | positives, float("-inf")).max(dim=1).values,
                    )
                    chunk_best, chunk_offsets = scores.max(dim=1)
                    better = chunk_best.gt(best_scores)
                    best_scores = torch.maximum(best_scores, chunk_best)
                    best_tokens = torch.where(
                        better,
                        input_ids[batch_index, token_positions].gather(0, chunk_offsets),
                        best_tokens,
                    )

                if not bool(torch.isfinite(positive_scores).all()):
                    raise ValueError("explicit copy mapping was not present in the source candidates")
                nll_terms.append(denominator - positive_scores)
                accuracy_terms.append(
                    best_tokens.eq(labels[batch_index, selected_targets]).float()
                )
                hard_valid = torch.isfinite(negative_max)
                if bool(hard_valid.any()):
                    hard_terms.append(
                        F.relu(
                            float(hard_negative_margin)
                            + negative_max[hard_valid]
                            - positive_scores[hard_valid]
                        )
                    )
                distance_terms.append((query_positions - mapped_keys).max().detach())

            mapped = copy_source_positions[batch_index, row_targets].long()
            next_is_copy = torch.zeros_like(row_targets, dtype=torch.bool)
            if row_targets.numel() > 1:
                next_is_copy[:-1] = (
                    row_targets[1:].eq(row_targets[:-1] + 1)
                    & mapped[1:].eq(mapped[:-1] + 1)
                )
            source_keys = mapped - 1
            # Repeated prompt tokens may map several targets to the same key.
            # Advanced-index ``|=`` does not define a reduction for duplicate
            # indices, so reduce explicitly and preserve any positive span link.
            unique_source_keys = source_keys.unique()
            commit_examined[batch_index, unique_source_keys] = True
            for source_key in unique_source_keys.tolist():
                source_matches = source_keys.eq(int(source_key))
                commit_positive[batch_index, int(source_key)] = bool(
                    next_is_copy[source_matches].any()
                )

        pointer_nll = torch.cat(nll_terms).mean()
        hard_negative_loss = (
            torch.cat(hard_terms).mean()
            if hard_terms
            else pointer_nll.detach().new_zeros(())
        )
        pointer_loss = pointer_nll + float(hard_negative_weight) * hard_negative_loss
        gate_negative = supervised_predictions & ~gate_positive
        gate_terms = []
        if bool(gate_positive.any()):
            gate_terms.append(
                F.binary_cross_entropy_with_logits(
                    gate_logits[gate_positive], torch.ones_like(gate_logits[gate_positive])
                )
            )
        if bool(gate_negative.any()):
            gate_terms.append(
                F.binary_cross_entropy_with_logits(
                    gate_logits[gate_negative], torch.zeros_like(gate_logits[gate_negative])
                )
            )
        gate_loss = torch.stack(gate_terms).mean()
        commit_terms = []
        if bool((commit_examined & commit_positive).any()):
            mask = commit_examined & commit_positive
            commit_terms.append(
                F.binary_cross_entropy_with_logits(
                    commit_logits[mask], torch.ones_like(commit_logits[mask])
                )
            )
        if bool((commit_examined & ~commit_positive).any()):
            mask = commit_examined & ~commit_positive
            commit_terms.append(
                F.binary_cross_entropy_with_logits(
                    commit_logits[mask], torch.zeros_like(commit_logits[mask])
                )
            )
        commit_loss = (
            torch.stack(commit_terms).mean()
            if commit_terms
            else commit_logits.sum() * 0.0
        )
        loss = pointer_loss + float(gate_weight) * (gate_loss + commit_loss)
        return PointerLoss(
            loss=loss,
            pointer_loss=pointer_loss.detach(),
            gate_loss=gate_loss.detach(),
            commit_loss=commit_loss.detach(),
            copyable_rate=gate_positive.float().mean().detach(),
            pointer_accuracy=torch.cat(accuracy_terms).mean().detach(),
            contract_valid=loss.detach().new_ones(()),
            proposal_recall=(
                torch.cat(proposal_terms).mean().detach()
                if proposal_terms
                else loss.detach().new_zeros(())
            ),
            cursor_continuation_rate=(
                commit_positive[commit_examined].float().mean().detach()
                if bool(commit_examined.any())
                else loss.detach().new_zeros(())
            ),
            hard_negative_loss=hard_negative_loss.detach(),
            max_copy_distance=torch.stack(distance_terms).max().to(loss.dtype),
            scanned_source_tokens=loss.detach().new_tensor(float(scanned_source_tokens)),
        )

    def scores(self, hidden, input_ids, attention_mask, route_indices=None):
        queries = F.normalize(self.project_queries(hidden[:, :-1]), dim=-1)
        key_windows, token_windows, valid_windows = causal_pointer_windows(
            hidden, input_ids, attention_mask, self.window
        )
        keys = F.normalize(self.project_keys(key_windows), dim=-1)
        scores = torch.einsum("bnd,bnwd->bnw", queries, keys) * self.logit_scale.exp().clamp(max=50.0)
        if route_indices is not None:
            if route_indices.shape[:2] != input_ids.shape:
                raise ValueError("route_indices must match the input sequence")
            top1 = route_indices[:, :, 0]
            prediction_length = max(input_ids.size(1) - 1, 0)
            padded_routes = F.pad(top1[:, :prediction_length], (self.window, 0), value=-1)
            route_windows = padded_routes.unfold(1, self.window, 1)[:, :prediction_length]
            shared_anchor = route_windows.eq(top1[:, :prediction_length].unsqueeze(-1))
            scores = scores + F.softplus(self.anchor_match_bias) * shared_anchor.to(scores.dtype)
        scores = scores.masked_fill(~valid_windows, -1e4)
        gate_logits = self.project_gate(hidden[:, :-1]).squeeze(-1)
        return scores, gate_logits, token_windows, valid_windows

    def training_loss(
        self,
        hidden,
        input_ids,
        labels,
        attention_mask,
        gate_weight: float = 0.25,
        min_copy_span: int = 2,
        prompt_sources_only: bool = True,
        route_indices: torch.Tensor | None = None,
        mode: str = "local",
        source_chunk_size: int = 256,
        query_chunk_size: int = 16,
        max_global_queries: int = 64,
        proposal_chunk_size: int = 32,
        proposal_chunk_anchors: int = 4,
        max_candidate_chunks: int = 4,
        measure_proposal_recall: bool = True,
        hard_negative_weight: float = 0.1,
        hard_negative_margin: float = 0.2,
        copy_source_positions: torch.Tensor | None = None,
        copy_target_mask: torch.Tensor | None = None,
        source_boundary: torch.Tensor | None = None,
    ):
        explicit = (copy_source_positions, copy_target_mask, source_boundary)
        if any(value is not None for value in explicit):
            if not all(value is not None for value in explicit):
                raise ValueError("explicit exact-copy fields must be provided together")
            return self._explicit_copy_training_loss(
                hidden,
                input_ids,
                labels,
                attention_mask,
                copy_source_positions,
                copy_target_mask,
                source_boundary,
                gate_weight,
                route_indices,
                source_chunk_size,
                query_chunk_size,
                proposal_chunk_size,
                proposal_chunk_anchors,
                max_candidate_chunks,
                measure_proposal_recall,
                hard_negative_weight,
                hard_negative_margin,
            )
        if mode == "global_prompt":
            if not prompt_sources_only:
                raise ValueError("global_prompt mode requires prompt_sources_only=True")
            return self._global_prompt_training_loss(
                hidden,
                input_ids,
                labels,
                attention_mask,
                gate_weight,
                min_copy_span,
                route_indices,
                source_chunk_size,
                query_chunk_size,
                max_global_queries,
                proposal_chunk_size,
                max_candidate_chunks,
                hard_negative_weight,
                hard_negative_margin,
            )
        if mode != "local":
            raise ValueError("mode must be local or global_prompt")
        if prompt_sources_only and not bool(
            (labels.eq(-100) & attention_mask.bool()).any()
        ):
            return self._zero_loss(hidden, contract_valid=0.0)
        scores, gate_logits, candidates, valid_candidates = self.scores(
            hidden, input_ids, attention_mask, route_indices=route_indices
        )
        targets = labels[:, 1:]
        supervised = targets.ne(-100)

        prediction_length = targets.size(1)
        prediction_positions = torch.arange(
            prediction_length, device=input_ids.device
        ).view(1, -1, 1)
        candidate_offsets = torch.arange(self.window, device=input_ids.device).view(1, 1, -1)
        source_positions = (prediction_positions - candidate_offsets).expand(
            input_ids.size(0), -1, -1
        )
        safe_positions = source_positions.clamp(min=0)
        source_labels = labels.gather(
            1, safe_positions.reshape(input_ids.size(0), -1)
        ).reshape_as(source_positions)
        valid_candidates = valid_candidates & source_positions.ge(0)
        if prompt_sources_only:
            valid_candidates = valid_candidates & source_labels.eq(-100)
        scores = scores.masked_fill(~valid_candidates, -1e4)
        matches = candidates.eq(targets.unsqueeze(-1)) & valid_candidates & supervised.unsqueeze(-1)
        copyable = matches.any(dim=-1)
        valid_source = valid_candidates.any(dim=-1)

        pointer_positions = copyable & valid_source
        if pointer_positions.any():
            log_denominator = torch.logsumexp(scores, dim=-1)
            log_numerator = torch.logsumexp(scores.masked_fill(~matches, -1e4), dim=-1)
            pointer_loss = (log_denominator - log_numerator)[pointer_positions].mean()
            selected = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
            pointer_accuracy = selected[pointer_positions].eq(targets[pointer_positions]).float().mean()
        else:
            pointer_loss = scores.sum() * 0.0
            pointer_accuracy = scores.new_zeros(())

        positive = sequential_prompt_copy_mask(
            input_ids,
            labels,
            attention_mask,
            source_positions,
            valid_candidates,
            min_span=min_copy_span,
            prompt_sources_only=prompt_sources_only,
        )
        negative = supervised & ~positive
        gate_terms = []
        if positive.any():
            gate_terms.append(F.binary_cross_entropy_with_logits(gate_logits[positive], torch.ones_like(gate_logits[positive])))
        if negative.any():
            gate_terms.append(F.binary_cross_entropy_with_logits(gate_logits[negative], torch.zeros_like(gate_logits[negative])))
        gate_loss = torch.stack(gate_terms).mean() if gate_terms else gate_logits.sum() * 0.0
        commit_logits = self.project_commit(hidden[:, :-1]).squeeze(-1)
        commit_terms = []
        if positive.any():
            commit_terms.append(
                F.binary_cross_entropy_with_logits(commit_logits[positive], torch.ones_like(commit_logits[positive]))
            )
        if negative.any():
            commit_terms.append(
                F.binary_cross_entropy_with_logits(commit_logits[negative], torch.zeros_like(commit_logits[negative]))
            )
        commit_loss = torch.stack(commit_terms).mean() if commit_terms else commit_logits.sum() * 0.0
        loss = pointer_loss + gate_weight * (gate_loss + commit_loss)
        return PointerLoss(
            loss=loss,
            pointer_loss=pointer_loss.detach(),
            gate_loss=gate_loss.detach(),
            commit_loss=commit_loss.detach(),
            copyable_rate=copyable[supervised].float().mean() if supervised.any() else scores.new_zeros(()),
            pointer_accuracy=pointer_accuracy.detach(),
            contract_valid=scores.new_ones(()),
            proposal_recall=scores.new_zeros(()),
            cursor_continuation_rate=positive[supervised].float().mean() if supervised.any() else scores.new_zeros(()),
            hard_negative_loss=scores.new_zeros(()),
            max_copy_distance=scores.new_zeros(()),
            scanned_source_tokens=scores.new_tensor(float(min(input_ids.size(1), self.window))),
        )

    def commit_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.project_commit(hidden).squeeze(-1))

    def mix_next_logits(
        self,
        base_logits,
        hidden,
        input_ids,
        attention_mask,
        max_mix: float = 0.95,
        source_length: int | None = None,
        search_window: int | None = None,
        min_gate: float = 0.0,
        anchor_memory: AnchorIndexedExactMemory | None = None,
        query_anchor_ids: torch.Tensor | None = None,
        max_candidate_chunks: int = 4,
        indexed_candidates: tuple[torch.Tensor, torch.Tensor] | None = None,
        full_scan_fallback: bool = False,
        fallback_margin: float = 0.0,
        candidate_cap: int = 16,
        commit_threshold: float = 0.5,
    ):
        if input_ids.size(1) < 2:
            return base_logits, {
                "gate": 0.0,
                "mix_gate": 0.0,
                "source_positions": [],
                "candidate_ids": [],
                "search_span": 0,
            }
        source_end = (
            input_ids.size(1)
            if source_length is None
            else min(max(int(source_length), 2), input_ids.size(1))
        )
        raw_gate = torch.sigmoid(self.project_gate(hidden[:, -1]))
        gate = raw_gate.clamp(max=max_mix)
        if min_gate > 0.0 and not bool(raw_gate.ge(min_gate).any()):
            return base_logits, {
                "gate": float(raw_gate.detach().mean()),
                "mix_gate": 0.0,
                "source_positions": [],
                "candidate_ids": [],
                "search_span": 0,
                "skipped_by_exact_need_gate": True,
            }
        query = F.normalize(self.project_queries(hidden[:, -1]), dim=-1)
        used_full_scan_fallback = False
        if anchor_memory is not None:
            if query_anchor_ids is None:
                raise ValueError("query_anchor_ids are required with anchor_memory")
            positions, indexed_mask = (
                indexed_candidates
                if indexed_candidates is not None
                else anchor_memory.candidate_key_positions(query_anchor_ids, max_candidate_chunks)
            )
            if anchor_memory.key_vectors is not None:
                gather_keys = positions.unsqueeze(-1).expand(-1, -1, anchor_memory.key_vectors.size(-1))
                keys = anchor_memory.key_vectors.gather(1, gather_keys).float()
            else:
                gather_hidden = positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
                source_hidden = hidden.gather(1, gather_hidden).float()
                keys = F.normalize(self.project_keys(source_hidden), dim=-1)
            prompt_candidate_ids = anchor_memory.token_ids.long().gather(1, positions + 1)
            candidate_ids = anchor_memory.payload_first_tokens(
                positions, prompt_candidate_ids
            )
            source_mask = indexed_mask
            start = 0
        else:
            effective_window = self.window if search_window is None else max(int(search_window), 1)
            start = max(source_end - effective_window - 1, 0)
            source_hidden = hidden[:, start : source_end - 1].float()
            candidate_ids = input_ids[:, start + 1 : source_end]
            source_mask = (
                attention_mask[:, start : source_end - 1].bool()
                & attention_mask[:, start + 1 : source_end].bool()
            )
            keys = F.normalize(self.project_keys(source_hidden), dim=-1)
        scores = torch.einsum("bd,bnd->bn", query, keys) * self.logit_scale.exp().clamp(max=50.0)
        scores = scores.masked_fill(~source_mask, -1e4)

        # The fuzzy anchor index is a fast proposal mechanism, not a correctness
        # boundary. On the first copy step, compare it with the lossless source
        # store and fall back when the globally best key is missing. Cursor mode
        # makes the remaining copied span O(1) per token.
        if full_scan_fallback and anchor_memory is not None and anchor_memory.key_vectors is not None:
            full_keys = anchor_memory.key_vectors.float()
            full_mask = (
                anchor_memory.valid_mask[:, :-1]
                & anchor_memory.valid_mask[:, 1:]
            )
            if anchor_memory.registered_key_mask is not None:
                full_mask &= anchor_memory.registered_key_mask
            full_scores = torch.einsum("bd,bnd->bn", query, full_keys) * self.logit_scale.exp().clamp(max=50.0)
            full_scores = full_scores.masked_fill(~full_mask, -1e4)
            indexed_best = scores.max(dim=-1).values
            full_best = full_scores.max(dim=-1).values
            if bool((full_best > indexed_best + float(fallback_margin)).any()):
                cap = min(max(int(candidate_cap), 1), full_scores.size(1))
                positions = full_scores.topk(cap, dim=-1).indices
                source_mask = full_mask.gather(1, positions)
                scores = full_scores.gather(1, positions).masked_fill(~source_mask, -1e4)
                prompt_candidate_ids = anchor_memory.token_ids.long().gather(
                    1, positions + 1
                )
                candidate_ids = anchor_memory.payload_first_tokens(
                    positions, prompt_candidate_ids
                )
                used_full_scan_fallback = True
        pointer_weights = torch.softmax(scores, dim=-1)
        pointer_vocab = torch.zeros_like(base_logits)
        pointer_vocab.scatter_add_(1, candidate_ids, pointer_weights)
        if min_gate > 0.0:
            gate = torch.where(raw_gate >= min_gate, gate, torch.zeros_like(gate))
        if not bool(gate.gt(0.0).any()):
            mixed_logits = base_logits
        else:
            base_probs = torch.softmax(base_logits.float(), dim=-1)
            mixed = (1.0 - gate) * base_probs + gate * pointer_vocab
            mixed_logits = mixed.clamp_min(1e-12).log()
        top_count = min(4, scores.size(-1))
        top_scores, top_offsets = scores.topk(top_count, dim=-1)
        top_candidate_ids = candidate_ids.gather(1, top_offsets)
        top_positions = positions.gather(1, top_offsets) if anchor_memory is not None else top_offsets + start
        top_span_ends = None
        span_end_source = "none"
        if (
            anchor_memory is not None
            and anchor_memory.registered_payload_ids is not None
        ):
            span_end_source = "explicit_payload"
        elif anchor_memory is not None and anchor_memory.span_end_positions is not None:
            top_span_ends = anchor_memory.span_end_positions.long().gather(
                1,
                top_positions + 1,
            )
            span_end_source = "explicit"
        elif anchor_memory is not None:
            top_span_ends = anchor_memory.inferred_span_end_positions(
                top_positions,
                min_commit=commit_threshold,
            )
            span_end_source = "learned_commit"
        top_pointer_confidence = pointer_weights.gather(1, top_offsets)[:, 0]
        score_margin = (
            top_scores[:, 0] - top_scores[:, 1]
            if top_count > 1
            else torch.full_like(top_scores[:, 0], float("inf"))
        )
        if anchor_memory is not None:
            top_commit_confidence = anchor_memory.commit_scores.float().gather(
                1, top_positions
            )[:, 0]
        else:
            top_commit_confidence = self.commit_scores(hidden).float()[:, -1]
        source_tokens = (
            anchor_memory.token_ids.long()
            if anchor_memory is not None
            else input_ids[:, :source_end]
        )
        source_valid = (
            anchor_memory.valid_mask
            if anchor_memory is not None
            else attention_mask[:, :source_end].bool()
        )
        candidate_occurrences = []
        cursor_continuation_supported = []
        for batch_index in range(source_tokens.size(0)):
            batch_occurrences = []
            batch_continuations = []
            length = source_tokens.size(1)
            transition_valid = source_valid[batch_index, :-1] & source_valid[batch_index, 1:]
            for candidate_index in range(top_candidate_ids.size(1)):
                candidate_id = top_candidate_ids[batch_index, candidate_index]
                occurrences = transition_valid & source_tokens[batch_index, 1:].eq(candidate_id)
                batch_occurrences.append(int(occurrences.sum().detach().cpu()))
                top_position = int(top_positions[batch_index, candidate_index].detach().cpu())
                has_top_continuation = (
                    top_position + 2 < length
                    and bool(source_valid[batch_index, top_position + 2])
                )
                continuation_occurrences = occurrences[:-1] & source_valid[batch_index, 2:]
                continuation_tokens = source_tokens[batch_index, 2:][continuation_occurrences]
                consistent = (
                    has_top_continuation
                    and continuation_tokens.numel() > 0
                    and bool(
                        continuation_tokens.eq(
                            source_tokens[batch_index, top_position + 2]
                        ).all()
                    )
                )
                batch_continuations.append(consistent)
            candidate_occurrences.append(batch_occurrences)
            cursor_continuation_supported.append(batch_continuations)
        diagnostics = {
            "gate": float(raw_gate.detach().mean()),
            "mix_gate": float(gate.detach().mean()),
            "source_positions": top_positions.detach().cpu().tolist(),
            "span_end_positions": (
                top_span_ends.detach().cpu().tolist()
                if top_span_ends is not None
                else []
            ),
            "span_end_source": span_end_source,
            "candidate_ids": top_candidate_ids.detach().cpu().tolist(),
            "candidate_payload_ids": (
                anchor_memory.payloads_for_key_positions(top_positions)
                if anchor_memory is not None
                else []
            ),
            "pointer_confidence": float(top_pointer_confidence.detach().mean()),
            "score_margin": float(score_margin.detach().mean()),
            "commit_confidence": float(top_commit_confidence.detach().mean()),
            "candidate_occurrences": candidate_occurrences,
            "cursor_continuation_supported": cursor_continuation_supported,
            "search_span": int(source_mask.sum(dim=1).max()),
            "used_full_scan_fallback": used_full_scan_fallback,
        }
        return mixed_logits, diagnostics
