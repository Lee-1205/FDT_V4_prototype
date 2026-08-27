# FDT v4 runtime/code audit - 2026-08-27

## Scope

- C-only FDT v4 424,474,072-parameter training, Exact Memory, routing, generation/cache, checkpoint/resume, and active runtime paths.
- No Luna supervisor is attached to the active trainer.

## Corrected findings

1. The former Luna supervisor retained a fully deserialized optimizer checkpoint while the child trainer ran. Removing the supervisor restored free commit from roughly 3 GiB to 20-42 GiB.
2. The earlier 1,252 tok/s benchmark did not reproduce the trainer's deterministic CUDA policy. Checkpoint-matched measurement found 166.7 tok/s with that policy versus 1,178.4 tok/s with the fast backend.
3. The fast backend produced zero sampled optimizer-update difference, zero base/exact loss difference, identical sample order/cursors, and zero 8K/16K loss difference. Measured long-context rates were 1,423.1 and 1,414.5 tok/s.
4. Resume throughput was divided by process-session time using lifetime run tokens. It now uses only session-local tokens and emits `session_local_v2`.
5. Per-microbatch finite checks and scalar logging forced repeated GPU/CPU synchronization. Finite loss is now synchronized once per optimizer step; metrics are transferred only when due.
6. Generated-prefix recovery built a full zero-gradient graph before its 2% curriculum ramp. Its deterministic data cursor still advances, but the graph is skipped while its exact weight is zero.
7. Explicit-copy proposal recall was hardcoded to zero. It now mirrors inference: chunk top-anchor summaries plus commit score select candidate chunks, then the mapped source chunk is checked. It is computed on metric steps only.

## Verification

- 146 focused FDT v4 tests passed.
- Trainer SHA-256: `B351DF0B9B6CDB9E20B0858897F5849647E281110B71A96B6CF83708D5E6C547`
- Exact pointer SHA-256: `65D392DD32447279674CEB2D7298E669A99FFA35D9953228A2F323C763608BB8`
- Backend evidence SHA-256: `0A720AFAE65059CBF32B520E923C63FE50A31ED9CDF8EDED5DD8F88D6190D5C8`
- R8 source migration SHA-256: `8705ADCD0FD142FC69F02F3A850BF07E49495B539F2D5673A826DCB452C47D92`
- First R8 recovery: step 1,750, 10,106,296 tokens, atomic, no temporary residue.
- Live R8 throughput: 1,259-1,272 tok/s; finite losses and gradients; stderr empty.
- Fixed validation: 4.88679755 at step 1,500, then 4.86861876 at step 1,750. There is no validated overfit signal.

## Remaining bounded risks

- The current immutable dataset still jumps from 512 to 8K/16K. Add globally disjoint 2K and 4K buckets only in the next fresh tranche; do not mutate this run's pinned data order.
- Logged dead-anchor fraction before a complete 25-step window is diagnostic. The hard gate is evaluated on the full union window.
- Fast CUDA operations are checkpoint-matched and numerically identical on the current hardware/driver, but cross-driver bitwise reproducibility is not promised.
- Strict capability evaluation remains deferred while training is active and v4 has processed only about 1% of its 1B-token target.
- C: free space is below the post-completion 80 GiB target because preserved R6/R7/R8 checkpoints coexist. Do not delete them automatically; reclaim only through the user's checkpoint-management policy after verified completion.
