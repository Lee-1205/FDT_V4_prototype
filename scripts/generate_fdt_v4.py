from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.next_tools import apply_ngram_loop_penalty_, apply_repetition_penalty_  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
import evaluate_fdt_v4 as evaluator  # noqa: E402


PRODUCT_MIN_NGRAM_LOOP_PENALTY = 13.0


def resolve_ngram_loop_penalty(configured: float, override: float | None) -> float:
    if override is not None:
        return float(override)
    return max(float(configured), PRODUCT_MIN_NGRAM_LOOP_PENALTY)


def exact_copy_contract_enabled(
    pointer_available: bool,
    mode: str,
    span_end_positions: torch.Tensor | None,
) -> bool:
    return bool(
        pointer_available
        and mode in {"retrieve", "copy"}
        and span_end_positions is not None
    )


def boundary(text: str) -> bool:
    return bool(re.search(r"[.!?;:]\s*$", text) or "\n" in text)


def load_exact_span_end_positions(path: Path, prompt_length: int) -> torch.Tensor:
    """Load an explicit source-token to inclusive span-end contract."""
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("span_end_positions")
    ends = torch.as_tensor(payload, dtype=torch.long)
    if ends.ndim == 1:
        ends = ends.unsqueeze(0)
    if tuple(ends.shape) != (1, prompt_length):
        raise ValueError("exact span map must contain one end position per prompt token")
    positions = torch.arange(prompt_length).view(1, -1)
    if bool((ends < positions).any()) or bool((ends >= prompt_length).any()):
        raise ValueError("exact span map contains an invalid source bound")
    return ends


def apply_non_copy_generation_controls(
    logits: torch.Tensor,
    generated_only: torch.Tensor,
    *,
    copy_active: bool,
    exempt_token_ids: torch.Tensor | None = None,
    repetition_penalty: float,
    ngram_order: int,
    ngram_penalty: float,
    ngram_window: int | None = None,
    ngram_hard_block_after: int = 0,
) -> torch.Tensor:
    """Control free decoding while preserving only verified copy tokens."""
    if copy_active or (repetition_penalty <= 1.0 and ngram_penalty <= 0.0):
        return logits
    controlled = logits.clone()
    if repetition_penalty > 1.0:
        apply_repetition_penalty_(controlled, generated_only, repetition_penalty)
    if ngram_penalty > 0.0:
        apply_ngram_loop_penalty_(
            controlled,
            generated_only,
            ngram_order=ngram_order,
            penalty=ngram_penalty,
            window=ngram_window,
            hard_block_after=ngram_hard_block_after,
        )
    if exempt_token_ids is not None and exempt_token_ids.numel() > 0:
        ids = exempt_token_ids.to(device=logits.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        controlled.scatter_(1, ids, logits.gather(1, ids))
    return controlled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument(
        "--ngram-loop-penalty",
        type=float,
        default=None,
        help="Override the checkpoint default; pass 0 to disable explicitly.",
    )
    parser.add_argument("--min-pointer-gate", type=float, default=0.80)
    parser.add_argument(
        "--exact-span-map",
        type=Path,
        help="Explicit JSON span contract required to activate lossless copy mode.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config, _ = evaluator.load_checkpoint(args.checkpoint.resolve())
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    tokenizer = load_tokenizer(str(args.tokenizer.resolve()))
    ngram_loop_penalty = resolve_ngram_loop_penalty(
        config.generation_ngram_penalty,
        args.ngram_loop_penalty,
    )

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    span_end_positions = (
        load_exact_span_end_positions(args.exact_span_map, len(prompt_ids)).to(device=device)
        if args.exact_span_map is not None
        else None
    )
    max_length = min(config.max_seq_len, len(prompt_ids) + args.max_new_tokens)
    generated = torch.empty((1, max_length), dtype=torch.long, device=device)
    generated[:, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    cursor = len(prompt_ids)
    exact_enabled = exact_copy_contract_enabled(
        model.exact_pointer is not None,
        config.exact_memory_mode,
        span_end_positions,
    )
    exact_state = LexicalPointerDecodeState(
        source_length=len(prompt_ids),
        max_activation_steps=16,
        max_copy_tokens=96,
    ) if exact_enabled else None
    trace: list[dict] = []
    new_ids: list[int] = []

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        prompt = generated[:, :cursor]
        full_recompute = evaluator.requires_full_recompute_generation(config)
        if full_recompute:
            output = model(prompt, attention_mask=torch.ones_like(prompt))
            cache = None
        else:
            output, cache = model.prefill(prompt, torch.ones_like(prompt))
        memory = (
            model.build_exact_memory(
                output["hidden"],
                prompt,
                torch.ones_like(prompt),
                source_length=len(prompt_ids),
                span_end_positions=span_end_positions,
            )
            if exact_enabled
            else None
        )
        for _ in range(min(args.max_new_tokens, config.max_seq_len - len(prompt_ids))):
            current = generated[:, :cursor]
            generated_only = generated[:, len(prompt_ids):cursor]
            logits = output["logits"][:, -1].float()
            diagnostics = {"mode": "base", "mix_gate": 0.0, "gate": 0.0}
            mixed = logits
            if exact_state is not None and memory is not None:
                query_anchors = model.exact_route_indices(output["hidden"])[:, -1]
                mixed, diagnostics = exact_state.prepare_logits(
                    model.exact_pointer,
                    logits,
                    output["hidden"],
                    current,
                    torch.ones_like(current),
                    min_gate=args.min_pointer_gate,
                    anchor_memory=memory,
                    query_anchor_ids=query_anchors,
                    max_candidate_chunks=config.exact_pointer_candidate_chunks,
                    full_scan_fallback=config.exact_memory_full_scan_fallback,
                    fallback_margin=config.exact_memory_fallback_margin,
                    candidate_cap=config.exact_memory_candidate_cap,
                    commit_threshold=config.exact_memory_commit_threshold,
                    hard_copy=config.exact_memory_hard_copy,
                    hard_copy_gate_threshold=config.exact_memory_hard_copy_gate_threshold,
                    hard_copy_pointer_threshold=config.exact_memory_hard_copy_pointer_threshold,
                    hard_copy_margin_threshold=config.exact_memory_hard_copy_margin_threshold,
                )

            copy_active = diagnostics.get("mode") in {"hard_copy", "cursor"}
            exempt_token_ids = None
            if (
                not copy_active
                and diagnostics.get("mode") == "mixed"
                and float(diagnostics.get("mix_gate", 0.0)) >= args.min_pointer_gate
                and diagnostics.get("candidate_ids")
                and exact_state.candidate_commit_eligible(diagnostics)
            ):
                # Exempt only the highest-ranked exact token. A weak or unused
                # pointer path must not disable loop controls globally.
                exempt_token_ids = torch.tensor(
                    [[int(diagnostics["candidate_ids"][0][0])]],
                    device=mixed.device,
                )
            mixed = apply_non_copy_generation_controls(
                mixed,
                generated_only,
                copy_active=copy_active,
                exempt_token_ids=exempt_token_ids,
                repetition_penalty=args.repetition_penalty,
                ngram_order=config.generation_ngram_order,
                ngram_penalty=ngram_loop_penalty,
                ngram_window=config.generation_ngram_window,
                ngram_hard_block_after=config.generation_ngram_hard_block_after,
            )

            selected = int(mixed.argmax(dim=-1))
            piece = tokenizer.decode([selected], skip_special_tokens=True)
            if exact_state is not None:
                exact_state.commit(selected, diagnostics, boundary=boundary(piece))
            diagnostics["selected_id"] = selected
            diagnostics["copy_active"] = copy_active
            diagnostics["loop_control_exempt_token_ids"] = (
                exempt_token_ids.detach().cpu().tolist()
                if exempt_token_ids is not None
                else []
            )
            trace.append(diagnostics)
            new_ids.append(selected)
            next_id = torch.tensor([[selected]], device=device)
            generated[:, cursor].copy_(next_id.squeeze(1))
            cursor += 1
            if selected == tokenizer.eos_token_id:
                break
            if full_recompute:
                current = generated[:, :cursor]
                output = model(current, attention_mask=torch.ones_like(current))
            else:
                output, cache = model.decode_step(next_id, cache)

    print(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "model_type": config.model_type,
        "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
        "generated_tokens": len(new_ids),
        "exact_memory_mode": config.exact_memory_mode,
        "trace": trace,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
