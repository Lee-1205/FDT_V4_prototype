# Pure-FDT v4 Sol Cache Integrity Diagnosis

## Scope

- Target commit: `6ce911c995a3e87916fc05f45fb69f85776f9130`
- Audit-only checkpoint SHA-256: `2E9D5764CFE145CC9DEBB53540F84CC33E9CB64124BAA76920F2B5C53CAF785F`
- Primary evidence: `artifacts/final_audit_inputs/20260823_6ce911c/cache_integrity_cuda_fp32.json`
- GPU work during Sol diagnosis: none
- Raw integrity status: `FAIL` (unchanged)
- Tolerance: `3e-4` (unchanged)

## Verdict

`POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT`, severity `major`, `requires_sol=true`.

The evidence does not establish an auditor aliasing bug. It also does not yet prove one localized architecture defect strongly enough to justify a model-code change.

At 8192 tokens, incremental versus full normalized anchor-state error is `0.0138512`; repeated full versus full is `0.0005697`; excess is `0.0132814`. Logits remain within tolerance and greedy tokens agree. This pattern is consistent with a last-token top-k route or membership discontinuity after a small decode-versus-full hidden-state difference. CUDA `scatter_add_` nondeterminism explains raw accumulator mismatch and some repeat drift, but does not by itself explain the measured 8K normalized excess.

## Code Review Findings

- Full prefill and one-token decode use matching anchor recency scales and accumulator update semantics in `src/fdt_rlm/models/fdt_v3.py`.
- `decode_step` intentionally mutates its incremental cache in place. The auditor then constructs two separate full-prefill caches, so no obvious mutated-reference self-comparison was found.
- Full local attention and incremental local attention do not use an identical numerical kernel path. Small hidden-state differences can cross the discontinuous top-k routing boundary and produce contribution-sized anchor-state deltas while leaving the immediate logits close.
- Existing evidence does not contain per-layer last-token route indices, memberships, or anchor-input deltas, so the route-flip mechanism remains the strongest hypothesis rather than a measured fact.

## Required Next Diagnostic

Instrument each active anchor layer for the appended token and compare:

1. incremental-decode versus full-recompute anchor input;
2. top-k anchor indices;
3. membership values and top-k boundary margin;
4. resulting numerator and mass contribution before mutation.

If routes differ in the failing layers, align decode local-attention math with the chronological full-prefill path and rerun the unchanged 2K/4K/8K audit. If routes match, investigate accumulator mutation and CUDA reduction state further. Do not alter 256 anchors, top-k 8, router dimension 256, fuzzy-routing semantics, or the `3e-4` criterion.

## Changes And Tests

- Model or auditor source changes: none.
- Existing pre-fix evidence: preserved.
- Focused CPU tests: 2 passed.
  - `tests/test_fdt_v4_cache_audit.py::test_small_cpu_fp32_incremental_anchor_state_matches_full_recompute`
  - `tests/test_fdt_v3.py::test_v3_incremental_cache_matches_recompute`
- Large-checkpoint or GPU rerun: not performed by instruction.

