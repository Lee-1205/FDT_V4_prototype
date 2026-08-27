# Pure FDT v4 Architecture

## Decision

FDT v4 is a causal recurrent language model with three deliberately separate
memory paths:

1. **Local working memory**: RoPE local attention with a fixed 64-token window.
2. **Fuzzy semantic memory**: the proven 256-anchor, top-8 FDT state used for
   meaning, relationships, and generalization.
3. **Exact episodic memory**: an append-only raw-token store with compact learned
   keys, anchor-indexed proposals, a correctness fallback, and a copy cursor.

The previous Anchor Exact v2 is rejected as an architecture parent. Its
realigned pointer reached 4/4 on immediate continuation but 0/4 at distances
64, 512, and 1400. The fuzzy anchor proposal therefore remains an accelerator,
not a correctness boundary.

## Invariants

- Fuzzy routing remains 256 anchors, top-k 8, router width 256, cosine
  temperature 0.25.
- RoPE is applied to local-attention Q/K only. The semantic router does not get
  an independent positional rotation.
- Exact raw storage is lossless and O(N). Claims of O(1) memory apply only to
  the fuzzy semantic state.
- Exact lookup is gated. Anchor-indexed candidates are checked against a
  lossless full-source fallback on the first copy step. The remaining span uses
  an O(1) cursor and does not search again.
- Copy mode is exempt from repetition penalties because repeated source tokens
  may be correct.
- Normal generation defaults to generated-token-only repetition control and a
  targeted n-gram loop detector.
- Exact-pointer training uses prompt-source copy supervision only. Generic LM
  tokens must never be treated as pointer targets.

## Scale Policy

Only one trainable model lineage is created: the 426M-class research model.

- dim 1216, 20 layers, 19 heads
- anchor layers 0, 2, 4, ..., 18
- maximum positional range 16384, with 8K prompt-plus-decode validation required
- natural language and factual knowledge form the majority of effective update
  contribution; retrieval is substantial; code and JSON are auxiliary

Tiny randomly initialized CPU instances are used only for unit tests. They are
never trained, checkpointed, registered, or treated as a model lineage. This
avoids spending storage and training budget on a separate 116M checkpoint.

## Training Objectives

Base language-model CE remains primary. EOS receives a predeclared weight of
2.0. Prompt-source exact copy is a separate auxiliary stream and optimizer
group. Generated-prefix recovery is introduced at low frequency only after a
minimum stable-token threshold. Every category logs supervised tokens, weighted
loss, and gradient contribution.

## Required Ablations

Every accepted checkpoint must support the same-weight ablations:

- FDT only
- FDT plus exact store
- FDT plus exact retrieval
- FDT plus exact retrieval and copy cursor

No capability claim may be attributed to exact memory without this comparison.

## Acceptance Gates

- FP32 cache/full-recompute greedy token identity and max logit error <= 3e-4.
- Exact target candidate recall 100% on controlled 4/8/16/32/64-token cases.
- Cursor continuation 100% after a correct activation, with no repeated search.
- No material natural-language BPB regression from exact-memory activation.
- Paired repetition, first-loop, EOS, factual, retrieval, JSON-value, and Python
  semantic metrics remain explicit and separate.
- 2K/4K/8K long-context integrity is mandatory; 16K is exploratory until it has
  corresponding mixed-length training evidence.
- Runtime claims are measured again on the actual RTX 5060 Ti. GTX 1650 timing
  constants are not generalized.

## Ownership

Routine training supervision is assigned to Luna. Completed atomic checkpoint
manifests are handed to Terra for official unquantized FP32 evaluation. Neither
supervisor may promote, overwrite, or delete a checkpoint. Promotion remains a
separate recorded decision.

Abnormalities use a mandatory escalation chain. Luna first writes an atomic
recovery checkpoint, pauses the affected stage, and hands the complete incident
record to Terra. Terra diagnoses and classifies the required change. A major
architecture, objective, or dataset-contract change is routed to Sol and cannot
be applied automatically. A bounded operational, configuration, or runtime
repair may be applied directly from Terra's recorded recommendation. Luna may
not silently redesign or continue through a failed safety/capability gate.
