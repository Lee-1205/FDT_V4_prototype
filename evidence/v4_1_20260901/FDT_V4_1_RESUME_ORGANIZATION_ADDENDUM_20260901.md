# FDT v4.1 Resume and Organization Addendum

Date: 2026-09-01

## Revised Decision

`RAW_GENERATION_REGRESSION_NOT_CONFIRMED__STAGE_B_STILL_HELD_FOR_NLL_HEADROOM_AND_STORAGE`

The previous Stage A decision correctly rejected raw-model promotion, but it
could not determine whether the 6/100 penalty-off loop-free result was caused
by the v4.1 transition or inherited from V20. A same-input, same-decoder
counterfactual now resolves that question.

## Same-Input Raw Generation

Both checkpoints were evaluated on the exact fixed 100-row dataset with greedy
decoding, FP32, no Exact Memory, and all repetition controls disabled.

| Checkpoint | Loop-free | Mean trigram repetition |
| --- | ---: | ---: |
| V20-T1 | 6/100 | 0.42674734 |
| v4.1 Stage A | 6/100 | 0.44561284 |

Paired row-level comparison:

- both loop-free: 3
- V20-only loop-free: 3
- v4.1-only loop-free: 3
- v4.1 minus V20 mean repetition: +0.01886550
- paired bootstrap 95% interval: [-0.02264441, +0.05991590]

The interval includes zero. Raw repetition regression from V20 to Stage A is
not confirmed. The severe penalty-off loop behavior is an inherited base-model
limitation, not evidence that the behavior-preserving RoPE transition created
a new loop failure.

## Consequences

1. The two rejected learned loop interventions remain rejected; neither
   materially improved the inherited absolute behavior.
2. The repaired product runtime remains the usable path: generated-only
   trigram control at strength 13 passes 100/100, while explicit-contract Exact
   Memory remains 60/60 lossless.
3. Raw-model promotion is still blocked because 6/100 is visibly unusable even
   when it is not a v4.1 regression.
4. The raw-generation non-regression blocker for a controlled positional
   transition is cleared.
5. Stage B is still not launched: Stage A paired FP32 NLL is already +0.974%
   versus V20, leaving almost no room under the 1% transition gate, and C: has
   only about 3.15 GiB free, less than one optimizer-bearing atomic recovery.

## Archive

- Public repository commit: `abddcb6`
- Release: `v4.1-runtime-candidate-20260901`
- Released checkpoint SHA-256:
  `96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE`
- GitHub server digest exactly matches the local checkpoint.

No checkpoint was deleted or overwritten. Large training remains paused until
storage headroom is restored and a Stage B contract explicitly protects the
V20 NLL gate.
