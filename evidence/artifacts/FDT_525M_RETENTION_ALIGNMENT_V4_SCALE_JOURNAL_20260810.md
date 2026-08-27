# FDT 525M Retention-Alignment v4 Scale Journal

## Rationale

The 522M prompted-code branch was rejected because paired Python and factual losses credibly regressed and strict Python generation remained unusable. This branch restarts from the promoted 520M checkpoint and returns to the exact mixture that passed every strengthened strict-v3 gate at 520M. It adds ordinary all-token base pretraining only; it does not use prompted Python, assistant masking, conversational SFT, or exact evaluation rows.

## Dataset

- Dataset: `prepared_data/v3_retention_alignment_v4_scale_5m_ctx2048`.
- Manifest SHA-256: `DEF3D653639386AF850AAE2773A9285DA2503987D112CB51962CC72FBF5B1D58`.
- Requested tokens: 5,000,000; usable tokens: 4,997,120 across 2,440 optimizer-aligned sequences.
- Unique documents: 19,103.
- Global exact overlap: 0 against all referenced accepted and rejected datasets through the 522M branch.
- Mix: FineWeb-Edu 25%, Wikipedia 15%, plain CodeXGLUE Python 25%, SQuAD factual cloze 10%, compact JSON 5%, prompted retrieval 10%, long retrieval 10%.
- Dataset validation: `PASS`; all shard hashes, tensor shapes, and storage dtypes verified.

## Training plan

- Parent: promoted 520M checkpoint SHA-256 `FC758EF3028734E31B763019B8F96DFCC341AD5120D9751BF1211920E9AA1D59`.
- Target: 524,959,744 cumulative tokens at optimizer step 32,041 after 305 steps.
- Constant LR: 3e-7, no warmup. Same-stage resumes must use the stage checkpoint and must not re-warm.
- Checkpoint and paired validation use the trainer's global-step modulus: steps 31,800 and 31,950, plus atomic final step 32,041.
- Promotion requires every strengthened strict-v3 gate against the 520M base. In particular, factual and retrieval paired confidence intervals, Python non-regression, JSON validity, repetition, and routing must all pass.

## Checkpoint milestones

- Step 31,800 / 521,011,200 tokens: atomic `latest.pt` and `step_31800.pt` hashes matched at `99F58FDE2B0DF95D398B00B09E49B655550817DF04E33B2D9DF43AF2D004BAE7`.
- Fixed validation loss 3.40452275 versus launch 3.40415078 (+0.0109% relative), top-1 40.7694%, entropy-normalized 0.536576, effective K 3.05194, dead anchors 0%. This remains comfortably inside the 0.3% pilot guardrail.
- Step 31,950 / 523,468,800 tokens: validation loss 3.40441627 (+0.00780% versus launch), top-1 40.7621%, entropy-normalized 0.536793, effective K 3.05332, dead anchors 0%. The retained numbered checkpoint SHA-256 is `900C781C23E782DB40591DEABC1B577F7C19DC1497448B382F32654DF19B7605`.

## Training completion

- Completed normally at optimizer step 32,041 and 524,959,744 cumulative tokens after all 305 planned steps.
- Final checkpoint SHA-256: `4F94FEBE191318F621B63A4D4807D6269BE6DE30B1030F90D8C69B1EB536F8AE`.
- No early stop, external stop, nonfinite metric, routing collapse, paging failure, or security stop indicator occurred.

## Strict evaluation against promoted 520M

- Strict result: `artifacts/fdt_426m_strict_base_eval_latest_525m_retention_alignment_v4_scale_standard_20260810.json`.
- Strict artifact SHA-256: `913C9D2344FEC53D8CB540FDE2C566E575872FA12E2D0F4A200313EFC3467281`.
- Paired comparison: `artifacts/fdt_525m_retention_alignment_v4_scale_paired_bootstrap_vs_520m_20260810.json`.
- Comparison SHA-256: `832D951AABEB46375A95667BD9665C84A00219C37A21CA38E0313B651CA6DC96`.
- Strengthened comparator result: **FAIL**.
- Broad loss changed 3.17251101 -> 3.17349592 (+0.03105% relative), within the 0.3% guardrail; broad top-1 was effectively flat at 43.6537%.
- Paired factual loss changed -0.002621 with bootstrap 95% CI [-0.009179, +0.003859], and factual top-1 changed -0.398 percentage points with CI [-1.024, +0.202]. Both were statistically non-regressive.
- Paired retrieval loss improved -0.152157, CI [-0.169978, -0.134362], and paired retrieval top-1 improved +5.376 percentage points, CI [+3.898, +6.940]. However base-continuation rank declined 40.0% -> 38.33% overall while distance-1,400 remained 35%.
- Paired Python loss worsened +0.003141, CI [+0.001363, +0.004942], although it remained inside the aggregate 0.5% non-regression allowance and paired top-1 was statistically flat. Strict Python generation remained 0/25 parseable, compilable, function-bearing, and structurally usable, with 52.48% mean repetition.
- JSON remained 50/50 syntactically valid and 0/50 semantic exact. Paired JSON loss improved and top-1 was statistically flat.
- Prefix mean repetition improved 39.05% -> 38.09%, but loop-free generation declined 18/50 -> 17/50. This failed the explicit prefix-generation gate.
- Routing remained healthy: normalized entropy 0.537042, effective K 3.05490, dead anchors 0%, top-1 membership 0.660236.
- Context-1792: 3,470.80 prefill tok/s, 65.24 decode tok/s, 2,336.06 MiB peak allocated VRAM.

## Decision

**Rejected.** The 5M scale extension improved prompted retrieval probabilities but did not produce a broad or generation-quality gain sufficient to replace 520M. The lower base-continuation retrieval rank, one-sample loop-free regression, statistically worse paired Python loss, and unchanged unusable Python generation outweigh the small improvements. Promotion gates are not relaxed after observing the result.

The promoted 520M retention-alignment v4 checkpoint remains the best pretraining base and `artifacts/current_best_fdt_base.json` is unchanged. The 525M strict evaluation, paired comparison, dataset, config, logs, hashes, and this journal are retained. The rejected 525M checkpoint binaries are removed after this record because the branch is reproducible from the preserved 520M parent and exact-novel dataset.
