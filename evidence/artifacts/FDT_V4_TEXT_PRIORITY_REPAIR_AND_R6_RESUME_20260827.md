# FDT v4 text-priority repair and R6 resume

Status: PASS / TRAINING RESUMED

## Corrected training controls

- The prior overfit stop was invalid evidence: each validation check consumed a different single row. R6 evaluates the same fixed eight-row grid on every check and requires consecutive relative regressions before stopping.
- The invalidated validation history and original SAFETY_STOP checkpoint remain preserved. The migrated optimizer-bearing PAUSED checkpoint is immutable and verified at step 1250 / 7,059,624 tokens.
- Exact Memory now follows a deterministic 4 -> 8 -> 16 -> 32 -> 64 target-token curriculum without changing source order, optimizer state, routing, or the 424,474,072-parameter architecture.
- Generated-prefix recovery and abnormal-loop unlikelihood ramp from 2% to 20% training progress. Valid code, number, and exact-copy spans are not assigned generic loop negatives.
- RoPE, W64 local attention, 256/top-8 fuzzy anchors, frozen routing, Exact Memory, Copy Cursor, EOS objective, 8K/16K checkpointing, and FP32 master weights are unchanged.

## Inference path

- Decode reuses the cache container and precomputed ring order instead of rebuilding both every token.
- Generation uses a preallocated token buffer instead of repeated concatenation.
- Generic repetition intervention is OFF by default and remains explicitly available as a diagnostic/interactive option.

## Verification

- Focused tests: 43 passed.
- Migrated recovery SHA-256: `84C61D63CA0E435E3AD3763321A479BD33382D71370687C741F91600E9FAFF6F`.
- R6 resumed through Luna. Initial post-resume events at steps 1260 and 1270 had finite loss/gradients, dead anchors 0%, no stderr, and C: about 79.67 GiB free.

## Pinned implementation

- trainer: `AC2313E59E8A344D43FD55F92C1139E2512502D59A8A92150848CDEF96E3CC45`
- config: `44058EDF7A338B7E0572A097B5F728C7B121ED5D3D86636538DE83954165D8FA`
- model runtime: `B47B2C1795ECB56278D71B786D09065B016B3D730DDD06D2695CA9F79F3A3074`
- generation runtime: `E0E468DCC2C09DFCEF36CCA21929C0F9DD47C09849B4820412044A857835E8AE`
