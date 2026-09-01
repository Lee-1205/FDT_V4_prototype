# FDT v4.1 Resume, Repair, and Promotion Decision

Date: 2026-09-01

## Decision

`PRODUCT_RUNTIME_CANDIDATE_VALIDATED__HOLD_STAGE_B_RAW_GENERATION_GATE`

The product runtime is now loop-safe and Exact Memory is lossless under an
explicit identity contract. The raw model is not promoted: external-control-off
generation remains visibly unusable and Stage A still has a +0.974% paired NLL
regression against V20.

## Selected checkpoint

- Path: `runs/fdt_v4_1_exact_pointer_pilot_20260831_r3/latest.pt`
- SHA-256: `96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE`
- Size: 1,707,955,211 bytes
- Parent Stage A SHA-256:
  `08A9CCF9CAD6C6145BAFD623FEE818A0A03D2BAD0D13524D409D788523FFFBF8`

The checkpoint has the same state-key set as Stage A. Exactly eight
`exact_pointer.*` tensors differ; every base-language tensor is bit-identical.

## Cache and transition

The structural prefix-invariance repair passed 63/64/65, 511/512/513, and
2K/4K/8K/16K fixtures. The release generator no longer uses a single-KV cache
for intermediate `output_blend` alpha. Alpha 0.25 uses the verified full
recompute path.

Stage A remains a controlled transition result, not a complete model result:

- Paired FP32 NLL vs V20: +0.974%
- External-control-off loop-free generation: 6/100

## Exact Memory

The sealed payload audit passed all 60 cells:

- retrieval success: 60/60
- whole-string exact: 60/60
- copy gate activation: 60/60
- full-scan count remained bounded

Artifact SHA-256:
`DB1ACBD2A5010ACBF5E8218C83ECE2A9FA445504A10C2E7845C1B28C839DD506`

An additional natural-generation cross-check found a critical activation bug:
unconditional Exact copy changed 100/100 ordinary outputs and visibly damaged
some text. The release generator now requires an explicit span map before copy
mode can activate. Without that contract, the pointer path does not mix logits.
Verified hard-copy cursor steps remain exempt from loop control.

## Free-run training experiments

### Full-model counterfactual unlikelihood

The 16.15M-token run preserved validation (`2.08980585`) but produced only
8/100 loop-free generations. On 512 real failure states it reduced mean loop
token probability from 0.50531 to 0.45776, yet changed only 6/512 top-1 tokens.
The objective was working numerically but was too weak at the decision boundary.

### Learned loop-controller overlay

A zero-initialized 1.65M-parameter hidden-state controller was trained while all
426.96M parent parameters were frozen. It safety-stopped at 6.70M tokens when
validation reached 2.10220943, beyond the predeclared +5% transition limit. It
reached only 9/100 loop-free on held-out prompts. The overlay is preserved as a
negative result and rejected.

These results rule out blind continuation of either learned intervention.

## Product loop control

The runtime already contained a generated-only selective trigram controller,
but its checkpoint value of 8 was not used by default and was empirically weak.
The release generator now uses a calibrated minimum of 13 unless the user
explicitly overrides it. It applies only to generated tokens, and the third
persistent closure is hard-blocked.

Results on the unchanged 100-prompt suite:

| Contract | Loop-free | Mean trigram repetition |
| --- | ---: | ---: |
| Raw Stage A, controls off | 6/100 | about 0.443 |
| Generated-only penalty 8 | 79/100 | 0.007428 |
| Generated-only penalty 13 | 100/100 | 0.000000 |
| Exact checkpoint + penalty 13 | 100/100 | 0.000000 |

Direct output review confirms that penalty 13 removes exact loops without the
old endless repetition. It does not create knowledge or factual correctness;
some outputs remain generic, weakly grounded, or semantically repetitive.

## Code and tests

The implementation adds:

1. adapter-overlay checkpoint loading and evidence-only controller support;
2. real loop-state counterfactual auditing;
3. top-k loop-candidate and non-loop escape dataset construction;
4. corrected candidate-mass unlikelihood math;
5. release-default calibrated loop control;
6. explicit-contract-only Exact copy activation;
7. full recompute for intermediate output-blend generation.

Focused verification passed 110 tests with zero failures. The two warnings are
third-party SWIG deprecation warnings and do not affect FDT behavior.

## Promotion boundary

The selected checkpoint plus repaired generator is a validated product-runtime
candidate. It is not a validated autonomous raw model and is not a final Diana
release. Stage B and large-scale training remain blocked until a new isolated
objective materially improves external-control-off natural generation while
keeping the unchanged V20 capability and validation gates.

No preserved checkpoint was deleted or overwritten. No D: path was used.
