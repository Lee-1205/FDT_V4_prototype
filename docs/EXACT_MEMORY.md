# FDT v4 Exact Episodic Memory

## Contract

Fuzzy memory represents meaning. Exact memory preserves identity. Exact memory
stores raw token IDs and validity/position metadata; compact learned keys and
fuzzy-anchor postings are proposal indices only.

## Training

Exact-copy rows carry explicit `labels`, with prompt positions set to `-100`.
They also carry `prompt_mask`, `source_boundary`, `copy_source_positions`, and
`copy_target_mask`. Generic all-token LM rows are rejected by the exact loss.
The loss scans the full prompt source in bounded chunks, records proposal
recall and hard negatives, and supervises cursor continuation across local
window and chunk boundaries.

## Decode

The fuzzy anchor index proposes a bounded candidate set. A lossless full-source
fallback is allowed on the first lookup only. After a source transition is
selected, the copy cursor advances without another retrieval. Copy mode is
exempt from generated-token repetition penalties so legitimate repeated values
remain representable.

## Evidence Status

CPU unit tests cover raw-token storage, prompt-source label contracts, global
source positions, chunk boundaries, repeated-token cursor continuation, and
single-use fallback. Main-checkpoint exact-copy accuracy and current-GPU cost
remain unverified until the final audit produces immutable results.
