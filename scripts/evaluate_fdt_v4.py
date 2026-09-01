from __future__ import annotations

"""CPU FP32 evaluation slice for FDT v4.

This evaluator deliberately does not silently substitute a synthetic score for
an unavailable dataset, unsupported context length, or missing tokenizer.  The
artifact is intended for an external evaluator (for example Terra) to consume.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import apply_ngram_loop_penalty_
from fdt_rlm.tokenization import load_tokenizer


OFFICIAL_DTYPE = "float32"
EXACT_MODES = ("off", "store", "retrieve", "copy")
EXACT_LENGTHS = (4, 8, 16, 32, 64)
RETRIEVAL_DISTANCES = (64, 512, 1400, 2048, 4096, 8192)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def unsupported(reason: str) -> dict[str, Any]:
    return {"status": "unsupported", "reason": reason}


def proportion(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"count": 0, "total": 0, "rate": None}
    return {"count": int(successes), "total": int(total), "rate": successes / total}


def edit_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, value in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (value != other)))
        previous = current
    return previous[-1]


def first_divergence(generated: list[int], expected: list[int]) -> int | None:
    for index, (actual, gold) in enumerate(zip(generated, expected)):
        if actual != gold:
            return index
    return len(expected) if len(generated) < len(expected) else None


def loop_metrics(tokens: list[int], order: int = 3) -> dict[str, Any]:
    if len(tokens) < order:
        return {"loop_free": True, "trigram_repetition_rate": 0.0, "first_loop": None}
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    first = None
    for end in range(order, len(tokens) + 1):
        gram = tuple(tokens[end - order : end])
        if gram in seen:
            repeated += 1
            if first is None:
                first = end - order
        seen.add(gram)
    opportunities = len(tokens) - order + 1
    return {
        "loop_free": first is None,
        "trigram_repetition_rate": repeated / max(opportunities, 1),
        "first_loop": first,
    }


def python_metrics(source: str, reference: str | None = None) -> dict[str, Any]:
    source = source.strip()
    try:
        tree = ast.parse(source)
        parseable = True
    except SyntaxError:
        tree = None
        parseable = False
    try:
        compile(source, "<fdt-v4-generation>", "exec")
        compilable = True
    except (SyntaxError, ValueError, TypeError):
        compilable = False
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))] if tree else []
    signature = None
    if len(functions) == 1:
        node = functions[0]
        signature = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "defaults": len(node.args.defaults),
        }
    ast_exact = None
    signature_match = None
    if reference is not None:
        try:
            reference_tree = ast.parse(reference.strip())
            ast_exact = bool(tree is not None and ast.dump(tree, include_attributes=False) == ast.dump(reference_tree, include_attributes=False))
            reference_functions = [node for node in ast.walk(reference_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(reference_functions) == 1 and signature is not None:
                ref = reference_functions[0]
                signature_match = signature == {
                    "name": ref.name,
                    "args": [arg.arg for arg in ref.args.args],
                    "defaults": len(ref.args.defaults),
                }
        except SyntaxError:
            pass
    return {
        "parseable": parseable,
        "compilable": compilable,
        "function_count": len(functions),
        "usable_structure": bool(compilable and len(functions) == 1),
        "signature": signature,
        "ast_exact": ast_exact,
        "signature_match": signature_match,
    }


def json_metrics(text: str, schema: dict[str, Any] | None = None, reference: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
        valid = True
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
        valid = False
    schema_valid = None
    if schema is not None and valid:
        expected_type = schema.get("type")
        type_map = {"object": dict, "array": list, "string": str, "number": (int, float), "boolean": bool, "null": type(None)}
        type_ok = expected_type not in type_map or isinstance(value, type_map[expected_type])
        required = schema.get("required", [])
        required_ok = not isinstance(value, dict) or all(key in value for key in required)
        schema_valid = bool(type_ok and required_ok)
    value_exact = None
    if reference is not None:
        try:
            value_exact = valid and value == json.loads(reference.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            value_exact = None
    return {"valid": valid, "schema_valid": schema_valid, "value_exact": value_exact}


def python_unit_metrics(source: str, cases: Any, enabled: bool) -> dict[str, Any]:
    """Run only declarative function-call cases, and only after explicit opt-in."""
    if not cases:
        return unsupported("dataset supplied no declarative unit-test cases")
    if not enabled:
        return unsupported("not run without --allow-python-unit-exec")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        return unsupported("unit_tests must be a list of declarative objects")
    harness = (
        "import json, sys\n"
        "source = sys.stdin.readline()\n"
        "cases = json.loads(sys.stdin.readline())\n"
        "scope = {'__builtins__': {'abs': abs, 'bool': bool, 'dict': dict, 'float': float, "
        "'int': int, 'len': len, 'list': list, 'max': max, 'min': min, 'range': range, "
        "'str': str, 'sum': sum, 'tuple': tuple}}\n"
        "exec(compile(source, '<candidate>', 'exec'), scope, scope)\n"
        "results = []\n"
        "for case in cases:\n"
        "    function = scope.get(case['function'])\n"
        "    value = function(*case.get('args', []), **case.get('kwargs', {}))\n"
        "    results.append(value == case.get('expected'))\n"
        "print(json.dumps(results))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            input=source + "\n" + json.dumps(cases) + "\n",
            text=True,
            capture_output=True,
            timeout=2,
            cwd=tempfile.gettempdir(),
            env={"PYTHONNOUSERSITE": "1", "PATH": os.environ.get("SystemRoot", "")},
            check=False,
        )
        if completed.returncode != 0:
            return {"status": "error", "returncode": completed.returncode, "stderr": completed.stderr[-500:]}
        results = json.loads(completed.stdout)
        return {"status": "ok", "passed": sum(bool(value) for value in results), "total": len(results), "rate": sum(bool(value) for value in results) / max(len(results), 1)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": 2}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def ids_for_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode_tokens(tokenizer: Any | None, ids: Iterable[int]) -> str | None:
    if tokenizer is None:
        return None
    return tokenizer.decode(list(ids), skip_special_tokens=True)


def load_checkpoint(
    checkpoint: Path, *, require_v4: bool = True
) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = ModelConfig(**payload["model_config"])
    if require_v4 and config.model_type != "fdt_v4":
        raise ValueError(f"checkpoint model_type must be fdt_v4, received {config.model_type!r}")
    model = build_model(config).to(device="cpu", dtype=torch.float32).eval()
    if payload.get("checkpoint_format") == "fdt_v4_adapter_overlay_v1":
        parent = Path(payload["parent_checkpoint"]).resolve()
        if sha256_file(parent) != str(payload["parent_checkpoint_sha256"]).upper():
            raise ValueError("adapter overlay parent checkpoint hash mismatch")
        parent_payload = torch.load(
            parent, map_location="cpu", weights_only=True, mmap=True
        )
        incompatible = model.load_state_dict(
            parent_payload["model_state_dict"], strict=False
        )
        expected_adapter = {
            name for name in model.state_dict() if name.startswith("loop_controller.")
        }
        if set(incompatible.missing_keys) != expected_adapter or incompatible.unexpected_keys:
            raise ValueError("adapter overlay parent state is not structurally compatible")
        adapter_state = payload.get("adapter_state_dict")
        if not isinstance(adapter_state, dict) or set(adapter_state) != expected_adapter:
            raise ValueError("adapter overlay state is incomplete")
        incompatible = model.load_state_dict(adapter_state, strict=False)
        if incompatible.unexpected_keys or expected_adapter.intersection(
            incompatible.missing_keys
        ):
            raise ValueError("adapter overlay could not be applied")
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_config": payload["model_config"],
        "checkpoint_keys": sorted(payload.keys()),
    }
    if payload.get("checkpoint_format") == "fdt_v4_adapter_overlay_v1":
        metadata["checkpoint_format"] = payload["checkpoint_format"]
        metadata["parent_checkpoint"] = str(Path(payload["parent_checkpoint"]).resolve())
        metadata["parent_checkpoint_sha256"] = payload["parent_checkpoint_sha256"]
    return model, config, metadata


@torch.inference_mode()
def forward_row(model: torch.nn.Module, ids: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    output = model(tensor, attention_mask=torch.ones_like(tensor))
    return output["logits"], output


@torch.inference_mode()
def retrieve_proposal_diagnostics(
    model: torch.nn.Module,
    output: dict[str, Any],
    memory: Any,
    config: ModelConfig,
) -> dict[str, Any]:
    """Inspect anchor-index proposals without calculating or applying pointer logits."""
    if model.exact_pointer is None:
        return {"mode": "retrieve", "source_positions": [], "candidate_ids": [], "gate": None, "mix_gate": 0.0}
    hidden = output["hidden"]
    query_anchor_ids = model.exact_route_indices(hidden)[:, -1]
    positions, valid = memory.candidate_key_positions(query_anchor_ids, config.exact_pointer_candidate_chunks)
    next_positions = (positions + 1).clamp_max(memory.token_ids.size(1) - 1)
    candidate_ids = memory.token_ids.long().gather(1, next_positions)
    source_positions = [
        [int(position) for position, keep in zip(row_positions.tolist(), row_valid.tolist()) if keep]
        for row_positions, row_valid in zip(positions, valid)
    ]
    candidates = [
        [int(token) for token, keep in zip(row_ids.tolist(), row_valid.tolist()) if keep]
        for row_ids, row_valid in zip(candidate_ids, valid)
    ]
    raw_gate = torch.sigmoid(model.exact_pointer.gate_proj(hidden[:, -1].float()))
    return {
        "mode": "retrieve",
        "gate": float(raw_gate.detach().mean()),
        "mix_gate": 0.0,
        "source_positions": source_positions,
        "candidate_ids": candidates,
        "proposal_count": sum(len(row) for row in source_positions),
        "logits_mixed": False,
    }


@torch.inference_mode()
def requires_full_recompute_generation(config: ModelConfig) -> bool:
    alpha = float(getattr(config, "rope_transition_alpha", 1.0))
    return bool(
        getattr(config, "use_rope", False)
        and getattr(config, "rope_transition_mode", "lerp") == "output_blend"
        and 0.0 < alpha < 1.0
    )


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    config: ModelConfig,
    prompt_ids: list[int],
    max_new_tokens: int,
    exact_mode: str,
    min_gate: float = 0.0,
    ngram_loop_penalty: float = 0.0,
) -> tuple[list[int], list[dict[str, Any]]]:
    if exact_mode not in EXACT_MODES:
        raise ValueError(f"unknown exact ablation mode: {exact_mode}")
    if not prompt_ids:
        raise ValueError("generation prompt must contain at least one token")
    prompt = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated = prompt.clone()
    generated_only = generated[:, :0]
    full_recompute = requires_full_recompute_generation(config)
    if full_recompute:
        output = model(prompt, attention_mask=torch.ones_like(prompt))
        cache = None
    else:
        output, cache = model.prefill(prompt, torch.ones_like(prompt))
    memory = None
    state = None
    exact_available = model.exact_pointer is not None and exact_mode in {"store", "retrieve", "copy"}
    if exact_available:
        memory = model.build_exact_memory(output["hidden"], prompt, torch.ones_like(prompt), source_length=len(prompt_ids))
    if exact_available and exact_mode == "copy":
        state = LexicalPointerDecodeState(source_length=len(prompt_ids), max_activation_steps=max_new_tokens, max_copy_tokens=max_new_tokens)
    selected_ids: list[int] = []
    trace: list[dict[str, Any]] = []
    for _ in range(min(max_new_tokens, config.max_seq_len - len(prompt_ids))):
        logits = output["logits"][:, -1].float()
        diagnostics: dict[str, Any] = {
            "mode": "base",
            "gate": 0.0,
            "mix_gate": 0.0,
            "ablation": exact_mode,
            "exact_memory_built": memory is not None,
            "logits_mixed": False,
            "decode_backend": "full_recompute" if full_recompute else "incremental_cache",
        }
        mixed = logits
        if exact_mode == "retrieve" and memory is not None:
            diagnostics.update(retrieve_proposal_diagnostics(model, output, memory, config))
            diagnostics["ablation"] = exact_mode
            diagnostics["exact_memory_built"] = True
            diagnostics["logits_mixed"] = False
        elif state is not None and memory is not None:
            query = model.exact_route_indices(output["hidden"])[:, -1]
            proposed, pointer_diagnostics = state.prepare_logits(
                model.exact_pointer,
                logits,
                output["hidden"],
                generated,
                torch.ones_like(generated),
                min_gate=min_gate,
                anchor_memory=memory,
                query_anchor_ids=query,
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
            diagnostics.update(pointer_diagnostics)
            diagnostics["ablation"] = exact_mode
            diagnostics["exact_memory_built"] = True
            diagnostics["logits_mixed"] = False
            if exact_mode == "copy":
                mixed = proposed
                diagnostics["logits_mixed"] = True
        copy_active = diagnostics.get("mode") in {"hard_copy", "cursor"}
        if float(ngram_loop_penalty) > 0.0 and not copy_active:
            mixed = mixed.clone()
            apply_ngram_loop_penalty_(
                mixed,
                generated_only,
                ngram_order=int(config.generation_ngram_order),
                penalty=float(ngram_loop_penalty),
                window=int(config.generation_ngram_window),
                hard_block_after=int(config.generation_ngram_hard_block_after),
            )
            diagnostics["ngram_loop_penalty"] = float(ngram_loop_penalty)
            diagnostics["ngram_loop_control_applied"] = True
        else:
            diagnostics["ngram_loop_penalty"] = 0.0
            diagnostics["ngram_loop_control_applied"] = False
        selected = int(mixed.argmax(dim=-1).item())
        # Retrieval measures proposal quality only.  It must not activate the
        # cursor or force a token; only copy commits cursor state.
        if state is not None and exact_mode == "copy":
            state.commit(selected, diagnostics)
        diagnostics["selected_id"] = selected
        selected_ids.append(selected)
        trace.append(diagnostics)
        if selected == config.eos_token_id:
            break
        next_id = torch.tensor([[selected]], dtype=torch.long, device=generated.device)
        generated = torch.cat((generated, next_id), dim=1)
        generated_only = torch.cat((generated_only, next_id), dim=1)
        if full_recompute:
            output = model(generated, attention_mask=torch.ones_like(generated))
        else:
            output, cache = model.decode_step(next_id, cache)
    return selected_ids, trace


def token_code(vocab_size: int, length: int, offset: int) -> list[int]:
    usable = max(vocab_size - 4, 1)
    return [4 + ((offset * 17 + position * 29) % usable) for position in range(length)]


def exact_case_prompt(vocab_size: int, length: int, distance: int) -> tuple[list[int], list[int], int]:
    code = token_code(vocab_size, length, distance + length)
    prefix = [2]
    filler_count = max(distance - length, 0)
    filler = token_code(vocab_size, filler_count, distance + 1000)
    prompt = prefix + code + filler + [3]
    return prompt, code, len(prefix) - 1


def exact_result(generated: list[int], expected: list[int], tokenizer: Any | None) -> dict[str, Any]:
    matches = sum(actual == gold for actual, gold in zip(generated, expected))
    target_text = decode_tokens(tokenizer, expected)
    generated_text = decode_tokens(tokenizer, generated[: len(expected)])
    char_accuracy = None
    if target_text is not None and generated_text is not None:
        denominator = max(len(target_text), 1)
        char_accuracy = sum(a == b for a, b in zip(target_text, generated_text)) / denominator
    return {
        "free_exact": generated[: len(expected)] == expected,
        "token_accuracy": matches / max(len(expected), 1),
        "char_accuracy": char_accuracy,
        "token_edit_distance": edit_distance(generated[: len(expected)], expected),
        "first_divergence": first_divergence(generated, expected),
        "generated_tokens": len(generated),
    }


def evaluate_exact_memory(model: torch.nn.Module, config: ModelConfig, tokenizer: Any | None) -> dict[str, Any]:
    if model.exact_pointer is None:
        return unsupported("checkpoint has no exact episodic memory parameters")
    matrix: dict[str, Any] = {}
    for length in EXACT_LENGTHS:
        cells = {}
        for distance in RETRIEVAL_DISTANCES:
            prompt, target, target_source_position = exact_case_prompt(config.vocab_size, length, distance)
            if len(prompt) + length > config.max_seq_len:
                cells[str(distance)] = unsupported(f"requires {len(prompt) + length} tokens; checkpoint max_seq_len is {config.max_seq_len}")
                continue
            modes: dict[str, Any] = {}
            for mode in EXACT_MODES:
                try:
                    generated, trace = generate(model, config, prompt, length, mode)
                    first = trace[0] if trace else {}
                    candidates = (first.get("source_positions") or [[]])[0]
                    entry = {
                        "candidate_recall": target_source_position in candidates if mode in {"retrieve", "copy"} else None,
                        "used_full_scan_fallback": first.get("used_full_scan_fallback") if mode in {"retrieve", "copy"} else None,
                        "gate": first.get("gate"),
                        "mix_gate": first.get("mix_gate"),
                    }
                    if mode == "copy":
                        entry.update(exact_result(generated, target, tokenizer))
                    else:
                        entry["free_exact"] = None
                    modes[mode] = entry
                except Exception as exc:  # Record an evaluator limitation without fabricating a score.
                    modes[mode] = unsupported(f"{type(exc).__name__}: {exc}")
            cells[str(distance)] = {"status": "ok", "target_source_position": target_source_position, "modes": modes}
        matrix[str(length)] = cells
    return {"status": "ok", "lengths": list(EXACT_LENGTHS), "distances": list(RETRIEVAL_DISTANCES), "matrix": matrix}


def load_dataset(path: Path, tokenizer: Any | None, limit: int) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        raw = path.read_text(encoding="utf-8")
        source = json.loads(raw) if path.suffix.lower() == ".json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
        source = source.get("rows", []) if isinstance(source, dict) else source
        rows = []
        for row in source[:limit]:
            if "input_ids" in row:
                ids = [int(item) for item in row["input_ids"]]
            elif tokenizer is not None and "text" in row:
                ids = ids_for_text(tokenizer, str(row["text"]))
            elif tokenizer is not None and "prompt" in row and "target" in row:
                ids = ids_for_text(tokenizer, str(row["prompt"]) + str(row["target"]))
            else:
                continue
            rows.append({**row, "input_ids": ids})
        return rows
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = payload.get("input_ids")
    masks = payload.get("attention_mask")
    if ids is None:
        raise ValueError("tensor dataset must contain input_ids")
    rows = []
    for index in range(min(int(ids.size(0)), limit)):
        active = ids[index] if masks is None else ids[index][masks[index].bool()]
        rows.append({"input_ids": [int(value) for value in active.tolist()]})
    return rows


def require_nonempty_rows(rows: list[dict[str, Any]], path: Path, purpose: str) -> None:
    if rows:
        return
    raise ValueError(
        f"{purpose} dataset produced zero usable rows: {path}. "
        "JSON prompt/text rows require --tokenizer; tensor rows require input_ids."
    )


def tokenizer_json_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        candidate = path / "tokenizer.json"
    else:
        candidate = path
    if candidate.name != "tokenizer.json" or not candidate.is_file():
        raise ValueError("tokenizer must be a directory containing tokenizer.json")
    return candidate


def tokenizer_metadata(path: Path | None) -> tuple[Any | None, dict[str, Any]]:
    if path is None:
        return None, {"status": "not_supplied"}
    tokenizer_json = tokenizer_json_path(path)
    tokenizer_dir = tokenizer_json.parent
    return load_tokenizer(str(tokenizer_dir)), {
        "status": "ok",
        "directory": str(tokenizer_dir),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
    }


@torch.inference_mode()
def per_row_teacher_forced(model: torch.nn.Module, config: ModelConfig, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed rows for paired comparisons; no synthetic fallback is permitted."""
    scores: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        ids = row["input_ids"][: config.max_seq_len]
        if len(ids) < 2 or any(token < 0 or token >= config.vocab_size for token in ids):
            continue
        tensor = torch.tensor(
            [ids],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        logits = model(tensor, attention_mask=torch.ones_like(tensor))["logits"][:, :-1].float()
        labels = tensor[:, 1:]
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        scores.append({
            "row_index": row_index,
            "tokens": int(labels.numel()),
            "nll": float(losses.mean()),
            "top1": float(logits.argmax(dim=-1).eq(labels).float().mean()),
        })
    return scores


def bootstrap_interval(values: list[float], samples: int = 2000, seed: int = 20260823) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples))
    lower = means[int((samples - 1) * 0.025)]
    upper = means[int((samples - 1) * 0.975)]
    return [lower, upper]


def paired_bootstrap(
    candidate: torch.nn.Module,
    candidate_config: ModelConfig,
    comparator: torch.nn.Module | None,
    comparator_config: ModelConfig | None,
    rows: list[dict[str, Any]],
    samples: int = 2000,
    seed: int = 20260823,
) -> dict[str, Any]:
    if comparator is None or comparator_config is None:
        return unsupported("no fixed comparator checkpoint supplied")
    if candidate_config.vocab_size != comparator_config.vocab_size:
        return unsupported("candidate and comparator vocabularies differ")
    candidate_rows = per_row_teacher_forced(candidate, candidate_config, rows)
    comparator_rows = per_row_teacher_forced(comparator, comparator_config, rows)
    if not candidate_rows or len(candidate_rows) != len(comparator_rows):
        return unsupported("fixed paired inputs were unusable by one or both checkpoints")
    if [entry["row_index"] for entry in candidate_rows] != [entry["row_index"] for entry in comparator_rows]:
        return unsupported("candidate and comparator did not score the same fixed rows")
    nll_deltas = [new["nll"] - old["nll"] for new, old in zip(candidate_rows, comparator_rows)]
    top1_deltas = [new["top1"] - old["top1"] for new, old in zip(candidate_rows, comparator_rows)]
    return {
        "status": "ok",
        "input_rows": len(candidate_rows),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "candidate_minus_comparator": {
            "mean_nll": sum(nll_deltas) / len(nll_deltas),
            "mean_nll_bootstrap_95": bootstrap_interval(nll_deltas, samples, seed),
            "mean_top1": sum(top1_deltas) / len(top1_deltas),
            "mean_top1_bootstrap_95": bootstrap_interval(top1_deltas, samples, seed + 1),
        },
    }


def independent_repetition_protocol(
    model: torch.nn.Module,
    config: ModelConfig,
    rows: list[dict[str, Any]],
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    if not rows:
        return unsupported("no independent repetition rows supplied")
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        prompt = row.get("prompt_ids") or row.get("input_ids")
        if not isinstance(prompt, list) or not prompt:
            continue
        prompt_ids = [int(value) for value in prompt[: config.max_seq_len - 1]]
        if not prompt_ids or any(value < 0 or value >= config.vocab_size for value in prompt_ids):
            continue
        generated, _ = generate(model, config, prompt_ids, min(int(row.get("max_new_tokens", max_new_tokens)), max_new_tokens), "off")
        record = loop_metrics(generated)
        record["row_index"] = row_index
        records.append(record)
    if not records:
        return unsupported("independent repetition rows had no usable prompts")
    rates = [float(record["trigram_repetition_rate"]) for record in records]
    loop_free = [1.0 if bool(record["loop_free"]) else 0.0 for record in records]
    return {
        "status": "ok",
        "protocol": "independent_generated_prefix_trigram_v1",
        "rows": records,
        "summary": {
            "count": len(records),
            "mean_repetition_rate": sum(rates) / len(rates),
            "mean_repetition_bootstrap_95": bootstrap_interval(rates),
            "loop_free": proportion(sum(loop_free), len(loop_free)),
            "loop_free_bootstrap_95": bootstrap_interval(loop_free, seed=20260824),
        },
    }


@torch.inference_mode()
def evaluate_teacher_forced(model: torch.nn.Module, config: ModelConfig, tokenizer: Any | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return unsupported("no usable evaluation rows")
    loss_sum = 0.0
    top1 = 0
    tokens = 0
    byte_count = 0
    routing: list[dict[str, float]] = []
    for row in rows:
        ids = row["input_ids"][: config.max_seq_len]
        if len(ids) < 2:
            continue
        tensor = torch.tensor(
            [ids],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        output = model(tensor, attention_mask=torch.ones_like(tensor))
        logits = output["logits"][:, :-1].float()
        labels = tensor[:, 1:]
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        loss_sum += float(losses.sum())
        top1 += int(logits.argmax(dim=-1).eq(labels).sum())
        tokens += int(labels.numel())
        if tokenizer is not None:
            byte_count += len(tokenizer.decode(labels[0].tolist(), skip_special_tokens=True).encode("utf-8"))
        for stat in output.get("anchor_stats", []):
            membership = stat.membership.float()
            if membership.numel() == 0 or membership.size(-1) < 2 or not bool(torch.isfinite(membership).all()):
                continue
            per_token_entropy = -(membership.clamp_min(1e-12) * membership.clamp_min(1e-12).log()).sum(dim=-1)
            normalized = per_token_entropy / math.log(max(membership.size(-1), 2))
            effective = per_token_entropy.exp()
            load = stat.load_prob.float()
            if load.numel() == 0 or not bool(torch.isfinite(load).all()):
                continue
            routing.append({
                "entropy_normalized": float(normalized.mean()),
                "effective_k": float(effective.mean()),
                "dead_anchor_fraction": float(load.le(1e-8).float().mean()),
            })
    if tokens == 0:
        return unsupported("all rows were shorter than two tokens")
    report: dict[str, Any] = {
        "status": "ok",
        "tokens": tokens,
        "nll": loss_sum / tokens,
        "top1": proportion(top1, tokens),
        "bpb": (loss_sum / (math.log(2.0) * byte_count)) if byte_count else unsupported("tokenizer unavailable or decoded target byte count was zero"),
        "routing": unsupported("checkpoint emitted no anchor statistics") if not routing else {
            "entropy_normalized": sum(value["entropy_normalized"] for value in routing) / len(routing),
            "effective_k": sum(value["effective_k"] for value in routing) / len(routing),
            "dead_anchor_fraction": sum(value["dead_anchor_fraction"] for value in routing) / len(routing),
            "observations": len(routing),
        },
    }
    return report


@torch.inference_mode()
def cache_integrity(model: torch.nn.Module, config: ModelConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if requires_full_recompute_generation(config):
        return unsupported(
            "intermediate output-blend transition intentionally has no single-KV "
            "cache state; cache integrity must be audited at alpha endpoints"
        )
    eligible = next((row["input_ids"] for row in rows if len(row["input_ids"]) >= 3), None)
    if eligible is None:
        return unsupported("requires a dataset row with at least three tokens")
    prompt = eligible[: min(len(eligible) - 1, max(2, config.max_seq_len // 2))]
    full = torch.tensor(
        [prompt],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    output, cache = model.prefill(full, torch.ones_like(full))
    max_error = 0.0
    token_identity = True
    steps = min(8, config.max_seq_len - len(prompt))
    for _ in range(steps):
        next_id = output["logits"][:, -1].float().argmax(dim=-1, keepdim=True)
        incremental, cache = model.decode_step(next_id, cache)
        full = torch.cat((full, next_id), dim=1)
        recomputed = model(full, attention_mask=torch.ones_like(full))["logits"][:, -1:]
        max_error = max(max_error, float((incremental["logits"].float() - recomputed.float()).abs().max()))
        token_identity &= bool(incremental["logits"].argmax(dim=-1).eq(recomputed.argmax(dim=-1)).all())
        output = incremental
    return {
        "status": "ok",
        "steps": steps,
        "token_identity": token_identity,
        "max_logit_error": max_error,
        "tolerance": 3e-4,
        "pass": bool(token_identity and max_error <= 3e-4),
    }


def category_generation(
    model: torch.nn.Module,
    config: ModelConfig,
    tokenizer: Any | None,
    rows: list[dict[str, Any]],
    allow_python_unit_exec: bool,
) -> dict[str, Any]:
    if tokenizer is None:
        return unsupported("a tokenizer is required for free-generation category evaluation")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = row.get("category")
        if category in {"json", "python", "python_code", "natural", "factual", "retrieval"} and row.get("prompt") is not None:
            groups.setdefault(category, []).append(row)
    if not groups:
        return unsupported("dataset has no prompt/target category rows")
    output: dict[str, Any] = {}
    for category, examples in groups.items():
        records = []
        for row in examples[:16]:
            prompt = ids_for_text(tokenizer, str(row["prompt"]))
            if not prompt or len(prompt) >= config.max_seq_len:
                continue
            generated, _ = generate(model, config, prompt, min(128, config.max_seq_len - len(prompt)), "off")
            completion = decode_tokens(tokenizer, generated) or ""
            target = str(row.get("target", ""))
            record: dict[str, Any] = {
                "prompt": str(row["prompt"]),
                "target": target,
                "completion": completion,
                "generated_tokens": len(generated),
                "loop": loop_metrics(generated),
            }
            if category == "json":
                record["json"] = json_metrics(completion, row.get("schema"), target)
            elif category in {"python", "python_code"}:
                record["python"] = python_metrics(completion, target or None)
                record["unit_tests"] = python_unit_metrics(completion, row.get("unit_tests"), allow_python_unit_exec)
            else:
                record["target_prefix_exact"] = bool(target and completion.strip().startswith(target.strip()))
            records.append(record)
        output[category] = {"rows": records, "count": len(records)}
    return {"status": "ok", "categories": output}


def integrity_audit(
    model: torch.nn.Module,
    checkpoint_metadata: dict[str, Any],
    tokenizer_info: dict[str, Any],
    dataset_info: dict[str, Any],
    repetition_info: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "official_cpu_fp32": all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters()),
        "checkpoint_digest_present": bool(checkpoint_metadata.get("checkpoint_sha256")),
        "tokenizer_digest_present_when_supplied": tokenizer_info.get("status") != "ok" or bool(tokenizer_info.get("tokenizer_json_sha256")),
        "dataset_digest_present_when_supplied": dataset_info.get("status") != "ok" or bool(dataset_info.get("sha256")),
        "repetition_digest_present_when_supplied": repetition_info.get("status") != "ok" or bool(repetition_info.get("sha256")),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limitations": [
            "No unsupported metric is converted into a numeric score.",
            "Paired comparison and independent repetition require explicitly supplied fixed inputs.",
        ],
    }


def audit_axes(report: dict[str, Any], config: ModelConfig, parameter_count: int) -> dict[str, Any]:
    """Final contract: every axis is explicit, including unavailable evidence."""
    exact = report["exact_memory"]
    repetition = report["independent_repetition"]
    cache = report["cache_full_recompute"]
    teacher_forced = report["teacher_forced"]
    free_generation = report["free_generation"]
    long_context_cells = []
    if exact.get("status") == "ok":
        for length in exact.get("lengths", []):
            for distance in exact.get("distances", []):
                if distance > 64:
                    cell = exact.get("matrix", {}).get(str(length), {}).get(str(distance), {})
                    if cell.get("status") == "ok":
                        long_context_cells.append((length, distance))
    architecture_status = "PASS" if config.model_type == "fdt_v4" and 400_000_000 <= parameter_count <= 450_000_000 else "PARTIAL"
    if config.model_type != "fdt_v4":
        architecture_status = "FAIL"
    exact_status = "NOT_TESTED" if exact.get("status") == "unsupported" else "PARTIAL"
    generation_status = "NOT_TESTED" if repetition.get("status") == "unsupported" else "PARTIAL"
    quality_status = "NOT_TESTED" if teacher_forced.get("status") == "unsupported" else "PARTIAL"
    if free_generation.get("status") == "ok" and teacher_forced.get("status") == "ok":
        quality_status = "PARTIAL"
    cache_status = "NOT_TESTED" if cache.get("status") == "unsupported" else ("PASS" if cache.get("pass") else "FAIL")
    reproducibility_status = "PASS" if report["integrity_audit"].get("status") == "PASS" else "FAIL"
    return {
        "ARCHITECTURE": {
            "status": architecture_status,
            "model_type": config.model_type,
            "parameter_count": int(parameter_count),
            "expected_single_lineage_range": [400_000_000, 450_000_000],
        },
        "EXACT_MEMORY": {
            "status": exact_status,
            "result": exact,
            "reason": "Exact modes are reported separately; store does not retrieve, retrieve does not mix, copy uses retrieval plus cursor.",
        },
        "GENERATION_STABILITY": {
            "status": generation_status,
            "repetition": repetition,
            "control": report["generation_protocol"]["repetition_control"],
        },
        "LONG_CONTEXT": {
            "status": "PARTIAL" if long_context_cells else "NOT_TESTED",
            "supported_exact_cells_over_64": [{"length": length, "distance": distance} for length, distance in long_context_cells],
            "full_target_requires": "fixed disjoint 2K/4K/8K context cohorts; unsupported cells remain unsupported.",
        },
        "INFERENCE_INTEGRITY": {"status": cache_status, "cache_full_recompute": cache},
        "PERFORMANCE": {
            "status": "NOT_TESTED",
            "reason": "Official CPU FP32 correctness evaluation does not claim throughput or GPU performance.",
        },
        "QUALITY": {
            "status": quality_status,
            "teacher_forced": teacher_forced,
            "free_generation": free_generation,
            "semantic_python_and_json": "Reported only when fixed category rows supply prompts and targets.",
        },
        "REPRODUCIBILITY": {
            "status": reproducibility_status,
            "integrity_audit": report["integrity_audit"],
            "provenance": report["provenance"],
        },
    }


def evaluate(
    checkpoint: Path,
    output: Path,
    tokenizer_path: Path | None = None,
    dataset_path: Path | None = None,
    dataset_limit: int = 32,
    allow_python_unit_exec: bool = False,
    comparator_checkpoint: Path | None = None,
    repetition_dataset_path: Path | None = None,
    bootstrap_samples: int = 2000,
    run_exact_memory: bool = True,
) -> dict[str, Any]:
    started = time.time()
    checkpoint = checkpoint.resolve()
    model, config, checkpoint_metadata = load_checkpoint(checkpoint)
    tokenizer, tokenizer_info = tokenizer_metadata(tokenizer_path)
    rows = []
    dataset_metadata: dict[str, Any] = {"status": "not_supplied"}
    if dataset_path is not None:
        dataset_path = dataset_path.resolve()
        rows = load_dataset(dataset_path, tokenizer, dataset_limit)
        require_nonempty_rows(rows, dataset_path, "evaluation")
        dataset_metadata = {"status": "ok", "path": str(dataset_path), "sha256": sha256_file(dataset_path), "rows_loaded": len(rows), "format": "fixed_tensor" if dataset_path.suffix.lower() not in {".json", ".jsonl"} else "json_rows"}
    comparator_model = None
    comparator_config = None
    comparator_metadata: dict[str, Any] = {"status": "not_supplied"}
    if comparator_checkpoint is not None:
        comparator_model, comparator_config, metadata = load_checkpoint(
            comparator_checkpoint.resolve(), require_v4=False
        )
        comparator_metadata = {"status": "ok", **metadata}
    repetition_rows: list[dict[str, Any]] = []
    repetition_metadata: dict[str, Any] = {"status": "not_supplied"}
    if repetition_dataset_path is not None:
        repetition_dataset_path = repetition_dataset_path.resolve()
        repetition_rows = load_dataset(repetition_dataset_path, tokenizer, 100)
        require_nonempty_rows(repetition_rows, repetition_dataset_path, "repetition")
        repetition_metadata = {
            "status": "ok",
            "path": str(repetition_dataset_path),
            "sha256": sha256_file(repetition_dataset_path),
            "rows_loaded": len(repetition_rows),
            "protocol": "independent_generated_prefix_trigram_v1",
        }
    report = {
        "schema": "fdt_v4_evaluation_v2",
        "official_evaluation": {"quantization": "none", "dtype": OFFICIAL_DTYPE, "device": "cpu", "gpu_launched": False},
        "checkpoint": checkpoint_metadata,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "generation_protocol": {
            "decoder": "greedy",
            "transition_decode_contract": "full_recompute_for_intermediate_output_blend",
            "repetition_control": {
                "repetition_penalty": 1.0,
                "scope": "generated_only_when_enabled",
                "enabled": False,
                "ngram_order": 3,
                "exact_copy_exempt": True,
            },
        },
        "dataset": dataset_metadata,
        "tokenizer": tokenizer_info,
        "comparator": comparator_metadata,
        "teacher_forced": evaluate_teacher_forced(model, config, tokenizer, rows) if rows else unsupported("no dataset supplied"),
        "paired_bootstrap": paired_bootstrap(model, config, comparator_model, comparator_config, rows, bootstrap_samples) if rows else unsupported("no fixed evaluation dataset supplied"),
        "exact_memory": (
            evaluate_exact_memory(model, config, tokenizer)
            if run_exact_memory
            else unsupported("deferred to the separately preserved Exact Memory pilot")
        ),
        "free_generation": category_generation(model, config, tokenizer, rows, allow_python_unit_exec) if rows else unsupported("no dataset supplied"),
        "independent_repetition": independent_repetition_protocol(model, config, repetition_rows) if repetition_rows else unsupported("no independent repetition dataset supplied"),
        "independent_repetition_dataset": repetition_metadata,
        "cache_full_recompute": cache_integrity(model, config, rows) if rows else unsupported("no dataset supplied"),
        "integrity_audit": integrity_audit(model, checkpoint_metadata, tokenizer_info, dataset_metadata, repetition_metadata),
        "provenance": {
            "checkpoint": checkpoint_metadata,
            "tokenizer": tokenizer_info,
            "tensor_dataset": dataset_metadata,
            "comparator": comparator_metadata,
            "independent_repetition_tensor_dataset": repetition_metadata,
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda_available_but_unused": torch.cuda.is_available(),
            },
        },
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    report["audit_axes"] = audit_axes(report, config, sum(parameter.numel() for parameter in model.parameters()))
    atomic_json(output.resolve(), report)
    report["output"] = {"path": str(output.resolve()), "sha256": sha256_file(output.resolve())}
    atomic_text(output.resolve().with_name(output.name + ".sha256"), report["output"]["sha256"] + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="FDT v4 CPU FP32 evaluation slice")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-limit", type=int, default=32)
    parser.add_argument("--allow-python-unit-exec", action="store_true")
    parser.add_argument("--comparator-checkpoint", type=Path)
    parser.add_argument("--repetition-dataset", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-exact-memory", action="store_true", help="Record Exact Memory as deferred instead of rerunning its separate pilot.")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Official evaluation is CPU FP32 only.")
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint,
        args.output,
        args.tokenizer,
        args.dataset,
        args.dataset_limit,
        args.allow_python_unit_exec,
        args.comparator_checkpoint,
        args.repetition_dataset,
        args.bootstrap_samples,
        not args.skip_exact_memory,
    )
    print(json.dumps({"output": report["output"], "official_evaluation": report["official_evaluation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
