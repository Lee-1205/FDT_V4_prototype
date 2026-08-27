# FDT v4 Long Context

The main configuration has a 16,384-token RoPE position capacity so an 8K
prompt can be followed by generated tokens. RoPE is applied only to
local-attention Q/K. Fuzzy-anchor routing retains its original
cosine representation and incremental numerator/mass state.

Position capacity is not evidence of long-context capability. The final audit
must separately test 512, 1K, 2K, 4K, and 8K full/cached consistency, retrieval,
state retention, exact identity, repetition, peak memory, prefill, and decode
latency. A 16K result is optional and any OOM must be preserved as evidence.

The intended algorithmic interaction shape is local window plus sparse state
access. It is not a claim that all model FLOPs are constant or that real GPU
latency is context-independent. Current-hardware profiling must measure the
decode-latency slope before such a claim is accepted.
