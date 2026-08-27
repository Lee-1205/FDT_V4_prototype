# Pure FDT v4 Review Synthesis

## Scope

This decision incorporates the complete `PureFDT-리뷰-20260822` review bundle,
the three supplied ChatGPT discussions, the current source tree, the corrected
R2 evaluation history, and the Anchor Exact v2 incident/evaluation record.

## Evidence

- The 426M-class lineage has seen only about 528.9M tokens, approximately 1.24
  tokens per parameter. Its present capability is therefore an undertraining
  result, not a credible architecture ceiling.
- Overall BPB 1.550 trails GPT-2 124M at 1.453, while the FDT lineage is stronger
  on TinyStories/simple narrative and Python. Natural language and factual
  knowledge remain the primary training priority.
- The previous Anchor Exact v2 pointer is not a valid architecture parent. Its
  corrected smoke achieved 4/4 immediate continuation but 0/4 at distances 64,
  512, and 1400, with 0/4 focused candidate recall.
- Frozen fuzzy anchors remained highly stable across capability stages. The
  exact path must complement, not replace or enlarge, the semantic anchor path.
- Generation repetition and exact copying are different problems. Repetition
  control must inspect generated tokens only and must never suppress a correct
  copy cursor.
- RoPE removes the learned absolute-position table but does not by itself prove
  long-context capability. Mixed 2K/4K/8K training and integrity tests are
  required; 16K remains exploratory.
- Runtime evidence from GTX 1650 cannot be generalized to RTX 5060 Ti. The
  in-place `scatter_add_` change is accepted because it is output-identical and
  removes an observed allocation/kernel bottleneck.

## Architecture Decision

Build one 426M-class FDT v4 model only. No separately trained small model or
checkpoint is permitted.

The model has three distinct paths:

1. RoPE local working memory for recent token order.
2. The existing 256-anchor sparse fuzzy semantic memory for compressed meaning.
3. Lossless exact episodic memory containing raw prompt token IDs and compact
   learned keys.

The fuzzy anchor index proposes exact-memory chunks but is never a correctness
boundary. The first exact lookup compares indexed proposals with a full-source
fallback. After a correct activation, an O(1) cursor continues the source span.

## Training Decision

- Main configuration: dimension 1216, 20 layers, 19 heads, approximately
  424.47M trainable parameters with the exact pointer enabled.
- Base LM cross-entropy remains primary; natural language plus factual knowledge
  are a majority of effective update contribution and retrieval is substantial.
- Exact pointer loss is computed only on explicitly marked prompt-source copy
  examples. Generic next-token labels are never pointer targets.
- EOS weight is 2.0. Generated-prefix recovery may be introduced at low
  frequency only after a stable-token threshold.
- Routing remains frozen initially. Code and JSON remain auxiliary.
- Atomic PAUSED checkpoints contain model, AdamW, cursor, token counter, and RNG
  state. COMPLETE checkpoints are model-only and immutable.

## Evaluation Decision

Terra evaluates the same checkpoint in four modes: FDT-only, exact store,
exact retrieval, and retrieval plus copy cursor. Official claims use original
unquantized FP32 operations. Required evidence includes BPB, factual and
retrieval exactness/rank, exact-copy distance, candidate recall, cursor
continuation, repetition and first-loop position, semantic Python, parsed JSON,
routing, long-context state retention, and cache/full-recompute integrity.

## Delegation

Luna performs routine training supervision and writes an immutable handoff after
checkpoint verification. Terra consumes that handoff for official evaluation.
Neither agent may overwrite, delete, or promote a checkpoint.

For an abnormality, Luna checkpoints and pauses before handing the complete
incident to Terra. Terra diagnoses and emits a machine-readable severity and
change classification. Major architecture, objective, or dataset-contract
changes are handed to Sol and are never auto-applied. Lightweight operational,
configuration, or runtime repairs may be handled directly under Terra's bounded
recommendation.
