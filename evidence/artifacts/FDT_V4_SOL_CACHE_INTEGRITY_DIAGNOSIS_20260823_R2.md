# Pure-FDT v4 Sol Cache Integrity Diagnosis R2

## Correction Scope

This R2 supersedes the decision use of the earlier diagnosis files, which referenced an invalid abbreviated commit and obsolete evidence. Those earlier files remain preserved as incident evidence and were not overwritten.

- Exact target commit: `6ce911ce1a14e4d882f3065ed8685736e4a89b37`
- Audit-only checkpoint: `runs/fdt_v4_audit_warmstart_20260823_0af5577/latest.pt`
- Checkpoint SHA-256: `2E9D5764CFE145CC9DEBB53540F84CC33E9CB64124BAA76920F2B5C53CAF785F`
- Checkpoint stage: `AUDIT_UNTRAINED_WARM_START`
- Primary evidence: `artifacts/final_audit_inputs/20260823_6ce911ce/cache_integrity_cuda_fp32.json`
- Primary evidence SHA-256: `795AB02FFD5D54499AFD659D3A23762B00C0AE288B05EDBA9F805C4873C8CDFA`
- Evaluator SHA-256: `0AE0A62AB611D2422CB391DC7A4B406EE7830907E894F58EC26A49F19657B06E`
- GPU work during this correction: none

## Exact 8192-Token Evidence

- Prefill versus full maximum logit error: `0.000014185905456542969`
- Prefill token agreement: `true`
- Decode versus full maximum logit error: `0.000011920928955078125`
- Decode token agreement: `true`
- Incremental versus full raw anchor-state error: `0.9068603515625`
- Incremental versus full normalized anchor-state error: `0.013851165771484375`
- Repeat full versus full raw anchor-state error: `1.9453125`
- Repeat full versus full normalized anchor-state error: `0.0018981695175170898`
- Normalized excess over repeat baseline: `0.011952996253967285`
- Fixed tolerance: `0.0003`
- Raw context status: `FAIL`
- `INFERENCE_INTEGRITY`: `FAIL`
- `LONG_CONTEXT`: `PARTIAL`

## Sol Verdict

`POSSIBLE_CAUSAL_OR_CACHE_STATE_DRIFT`, severity `major`, `requires_sol=true`.

The corrected evidence does not support reducing the incident to an auditor aliasing or comparison-order bug. The auditor decodes the live incremental cache and then constructs separate full-prefill caches. CUDA `scatter_add_` reduction nondeterminism is directly visible in the repeat full-versus-full baseline, but the corrected normalized incremental excess of `0.011952996253967285` remains far above the unchanged `0.0003` criterion.

The strongest supported mechanism remains a discontinuous last-token top-k route or membership change caused by small numerical differences between incremental decode and full recompute. Full and incremental anchor accumulation use matching recency and update semantics. The unresolved upstream difference is that full local attention and one-token cached attention do not execute an identical numerical path; a small hidden-state difference can cross a top-k routing boundary and produce a contribution-sized internal-state change while immediate logits and greedy tokens still agree.

This mechanism is not yet directly measured because the primary evidence does not record per-layer appended-token anchor inputs, top-k indices, memberships, boundary margins, or isolated pre-mutation contributions. Therefore no speculative model-code change is authorized by this R2 diagnosis.

## Required Next Diagnostic

For every active anchor layer, capture and compare the appended token's:

1. incremental-decode and full-recompute anchor input;
2. top-k anchor indices;
3. membership values and the top-k boundary margin;
4. numerator and mass contribution before cache mutation;
5. pre-decode anchor state preserved as an immutable clone.

If routes differ in the failing layers, align the cached local-attention numerical path with the chronological full-prefill calculation and rerun the unchanged 2K, 4K, and 8K audit. If routes match, continue into accumulator mutation and CUDA reduction diagnosis. Do not change fuzzy-routing semantics, 256 anchors, top-k 8, router dimension 256, or the `0.0003` criterion.

## Changes And Tests

- Existing incident artifacts overwritten: none
- Model source changes: none
- Auditor source changes: none
- GPU work: none
- Focused CPU tests rerun for R2: not needed because no source changed; the prior two focused CPU tests remain supporting small-context evidence, not a substitute for the unresolved 8K device diagnosis

