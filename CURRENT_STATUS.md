# Current Snapshot Status

Snapshot time: 2026-08-27 KST

## Model

- Exact parameter count: 424,474,072
- Model type: `fdt_v4`
- Maximum configured context: 16K
- Core routing: frozen-shape fuzzy anchor path with 256 anchors and top-8 membership
- Exact Memory: enabled in copy mode, candidate cap 16, cursor continuation enabled
- Repetition control: generated-text-only trigram penalty 8.0, 96-token window, hard block after the second repeat

## Training Evidence

- Immutable release snapshot: optimizer step 3,000
- Tokens seen: 17,362,528
- Fixed validation loss: 4.7819013595581055
- Validation history in the snapshot: 4.8148013, 4.7969342, 4.7819014
- Overfit trigger: false
- Dead-anchor hard gate: 1% over a complete 25-step routing window
- Short-sequence optimized path: activation checkpointing disabled only up to 512 tokens
- Long-context path: activation checkpointing retained at 8K and 16K
- Verified short-path benchmark: 1,267 tokens/s with sampled update maximum absolute difference 0
- Live session throughput around the snapshot: approximately 1,369 tokens/s

## Important Limits

- The model is still early in the planned one-billion-token V4 curriculum.
- A single metric batch is not sufficient to judge Exact Memory proposal recall; use the recorded metric series and strict copy audit.
- Partial routing windows do not establish dead-anchor failure.
- The current curriculum contains 512, 8K, and 16K data. A future 512 -> 2K -> 4K -> 8K -> 16K bridge remains a planned improvement and is not silently claimed as present.

