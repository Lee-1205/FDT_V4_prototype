# FDT 520M Retention-Alignment v4 Pilot Journal

## Rationale

The rejected 519M diversity pilot improved broad loss, Python loss, JSON loss, and mean prefix repetition, but credibly regressed factual and retrieval teacher-forced metrics. The main mismatch was that the prior retrieval pretraining rows were mostly long document continuations while the strict capability rows ask for a short value after a prompt.

This branch restarts from the promoted 517M base and adds exact-novel prompted retrieval documents as ordinary all-token pretraining. It does not use assistant masking, conversational SFT, or any exact strict-evaluation row.

## Dataset

- Dataset: `prepared_data/v3_retention_alignment_v4_3m_pilot_ctx2048`.
- Manifest SHA-256: `F068E80690E34D1D066104BAAA78E86C1A011A2E0E1CA01F0D0454C618ED1F0E`.
- Usable tokens: 2,981,888 across 1,456 sequences and 11,194 unique documents.
- Global exact overlap: 0 against all referenced base, accepted continuation, failed pilot, and rejected 519M datasets.
- Mix: FineWeb-Edu 25%, Wikipedia 15%, CodeXGLUE Python 25%, factual cloze 10%, compact JSON 5%, prompted retrieval 10%, long retrieval 10%.
- Repetition filtering rejected 11 FineWeb-Edu, 22 Wikipedia, and 197 Python candidates.

## Training plan

- Parent: promoted 517M base SHA-256 `962F9780C0A619B90AF861BF536D29B000AC15153EDF8392AAD6A0E15CF60AFD`.
- Target: 519,962,624 cumulative tokens after 182 optimizer steps.
- Constant LR: 4e-7, no warmup.
- Promotion requires every strengthened strict-v3 gate to pass, including paired factual/retrieval confidence intervals and prefix repetition non-regression.

## Completion and promotion

- Completed normally at step 31,736 and 519,962,624 cumulative tokens.
- Final checkpoint SHA-256: `FC758EF3028734E31B763019B8F96DFCC341AD5120D9751BF1211920E9AA1D59`.
- Intermediate step-31,680 SHA-256: `42A4161427DDDE997236AE37F6B69432EEDAE762E9A1DD87C4B5E1E66B03E9AD`; it is deleted after this journal because the final checkpoint supersedes it.
- Strict broad loss changed 3.168314 -> 3.172511 (+0.1325%), within the 0.3% non-regression gate. Broad top-1 changed 43.7282% -> 43.6517% (-0.0765 percentage points).
- Factual teacher-forced loss improved by 0.05521 per row with paired bootstrap 95% CI [-0.06669, -0.04388]; top-1 improved by 1.601 percentage points per row with CI [+0.897, +2.394].
- Retrieval teacher-forced loss improved by 0.52722 per row with CI [-0.55129, -0.50355]; top-1 improved by 11.25 percentage points per row with CI [+9.448, +13.055].
- Ranked retrieval improved 37.5% -> 40.0% overall and 30.0% -> 35.0% at distance 1400.
- Python teacher-forced loss was non-regressive and mean row top-1 was statistically flat. JSON loss and top-1 improved; valid JSON remained 50/50.
- Prefix loop-free stayed 18/50 while mean repetition improved 40.54% -> 39.05% and mean unique-token rate increased 43.42% -> 46.19%.
- Routing remained healthy: dead anchors 0%, normalized entropy 0.53639, effective-K 3.05075.
- Context-1792 measured 3,464.6 prefill tok/s, 63.64 decode tok/s, and 2,336.1 MiB peak allocated VRAM.
- Strengthened paired comparator status: PASS on every gate.
- Decision: promote as the best pretraining base. Keep the frozen 517M checkpoint as rollback evidence. Leave `artifacts/current_best_fdt_bundle.json` unchanged because strict free generation is still not ready.

## Remaining blockers

- Factual exact generation: 0/50.
- Retrieval greedy exact: 0/120 despite better ranking and teacher-forced scores.
- Python parseable/compilable/usable generation: 0/25, although generation repetition dropped materially.
- JSON valid generation is 50/50 but semantic exact is 0/50.
- Conversational SFT remains blocked.
