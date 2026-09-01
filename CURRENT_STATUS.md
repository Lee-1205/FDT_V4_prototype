# Current FDT v4.1 Status

Snapshot date: 2026-09-01 KST

## Decision

`PRODUCT_RUNTIME_CANDIDATE_VALIDATED__HOLD_STAGE_B_RAW_GENERATION_GATE`

The repaired runtime is suitable for controlled testing, but the raw model is
not promoted and large-scale Stage B training is not authorized.

## Selected Checkpoint

- Model: FDT v4.1 Exact Pointer pilot
- Parameters: 426M class
- Local path: `runs/fdt_v4_1_exact_pointer_pilot_20260831_r3/latest.pt`
- SHA-256: `96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE`
- Parent Stage A SHA-256: `08A9CCF9CAD6C6145BAFD623FEE818A0A03D2BAD0D13524D409D788523FFFBF8`
- State difference from Stage A: eight `exact_pointer.*` tensors only; all
  base-language tensors are bit-identical.

## Verified Capabilities

- Alpha-zero FP32 transition starts bit-exact to V20.
- Cache boundary fixtures pass at 63/64/65, 511/512/513, 2K, 4K, 8K, and 16K.
- Exact Memory passes 60/60 retrieval and 60/60 whole-string exact tests under
  an explicit identity/span contract.
- Exact copy is disabled on ordinary prompts unless an explicit span map is
  supplied.
- Generated-only trigram control with calibrated strength 13 passes 100/100
  loop-free runtime generations without modifying verified copy-cursor tokens.
- Focused implementation verification passes 110 tests with zero failures.

## Unresolved Model Limits

- Stage A paired FP32 NLL remains 0.974% worse than V20.
- Raw generation with loop controls disabled is loop-free on only 6/100 prompts.
- The same fixed 100-prompt audit also gives V20 6/100. The paired mean
  repetition difference is +0.01887 with a 95% interval spanning zero, so a
  v4.1 raw-generation regression is not confirmed.
- Direct output review still finds generic, weakly grounded, and semantically
  repetitive natural-language continuations.
- A 16.15M-token full-model unlikelihood intervention reached only 8/100
  loop-free outputs.
- A frozen-base 1.65M-parameter loop controller safety-stopped at 6.70M tokens
  and reached only 9/100 loop-free outputs.

Runtime controls solve the product loop failure transparently; they do not add
knowledge, factual correctness, or raw autonomous generation capability.

## Training Boundary

Do not resume Stage B or a large curriculum from either rejected loop-training
experiment. Same-input evidence clears raw-generation non-regression as a
transition blocker, but not the absolute usability problem. Stage B remains
held because the +0.974% paired NLL result leaves almost no room under the 1%
transition gate and local storage cannot currently fit an optimizer recovery.
Any next stage must preserve V20 loss, top-1, routing, cache, Exact Memory, and
validation gates in a fresh output path.

The earlier step-3000 V4 release remains an immutable historical snapshot. The
current v4.1 compact evidence is under `evidence/v4_1_20260901/`.
