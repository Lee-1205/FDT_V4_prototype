# FDT v4.1 Stage A Final Cross-Evaluation

## Decision

`STAGE_A_TRANSITION_PASS_WITH_GENERATION_BLOCK`

Stage A proved that the V20-to-output-blend-RoPE transition can reach alpha 0.25 without the large behavior regression seen in the original v4 warm start. It did not repair free-running generation, and it did not test Exact Memory. Do not launch alpha 0.25 to 0.50 or a large curriculum yet.

## Immutable Inputs

- Candidate: `runs/fdt_v4_1_424m_rope_stage_a_20260830_r2_pinned_validation_resume/latest.pt`
- Candidate SHA-256: `08A9CCF9CAD6C6145BAFD623FEE818A0A03D2BAD0D13524D409D788523FFFBF8`
- V20 comparator SHA-256: `7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179`
- Fixed 52-row dataset SHA-256: `7F3A28AA049904A4D715ACE2307CDF7A80E7D6E2EB65A707EAAD8764AB908E9E`
- Independent 100-row repetition dataset SHA-256: `E7438870589B6CCAB912D2F16F266C5EE654F8F47D3617DF90B24E9F206BAED5`
- Tokenizer SHA-256: `4FADE0C72E8673BA1E4319C8821593BB5E62A8C27792E2412DE3BDC52D232862`
- Official evaluation: original unquantized CPU FP32, greedy decoding, repetition penalty disabled.

## Transition Result

The 19,720,000-token Stage A run completed at alpha 0.25 with exact natural/factual token balance. Final pinned validation remained within both predeclared gates:

- Legacy-control regression: about `+0.802%` (gate `<= +1%`).
- Scheduled-transition regression: about `+3.83%` (gate `<= +5%`).
- Dead anchors: `0%` on completed routing windows.
- Source epochs: natural `0`, factual `0`; no source repeat.

On the independent 52-row paired evaluator:

- Candidate NLL: `3.5754051790`.
- V20 NLL inferred from the paired delta: `3.5409130589`.
- Candidate minus V20 NLL: `+0.0344921201`, about `+0.974%`; bootstrap 95% CI `[+0.024991, +0.044140]`.
- Candidate top-1: `40.9244%`.
- V20 top-1 inferred from the paired delta: `41.5360%`.
- Candidate minus V20 top-1: `-0.6115 percentage points`; bootstrap 95% CI `[-0.9603, -0.2740]`.

This is a small but statistically resolved transition cost, not the original v4-scale collapse. Alpha 0 behavior was separately proven bit-exact against V20 before training.

## Free-Generation Result

The unchanged independent 100-row, penalty-off protocol remained unacceptable:

- Loop-free: `6/100` (95% bootstrap interval `2%` to `12%`).
- Mean trigram repetition rate: `0.4456` (95% bootstrap interval `0.3924` to `0.5002`).
- Original v4 at 200M was `1/100`; Stage A is directionally better but still visibly unusable.

The 52-row qualitative slice further separated the failure modes:

| Category | Loop-free | Exact/usable result |
|---|---:|---:|
| factual | 8/12 | 0/12 target-prefix exact; many wrong short answers or early EOS |
| natural | 0/12 | all 12 entered repetition loops |
| retrieval | 12/12 | 2/12 exact copied values |
| JSON | 8/8 syntactically valid | 0/8 parsed-value exact |
| Python | 0/8 | 0/8 parseable or structurally usable |

Representative raw failures included `the Duchy of the Duchy of the Duchy`, long `the first of the ...` continuations, and repetitive water narratives. Retrieval sometimes copied exactly (`CW114208`, `ND545002`) but usually altered one or more characters. JSON preserved shape and ordinary fields but substituted record identifiers. These are genuine model behaviors under the predeclared decoder, not cache artifacts.

## Scope And Next Gates

- Stage A transition: pass.
- Free-running stability: fail/block.
- Exact Memory: not tested in Stage A; remain isolated until source-top1, first-token, commit, cursor, and whole-string 4/8/16/32/64 gates are run.
- Cache integrity: not defined for an intermediate output blend because legacy and RoPE paths do not share a mathematically valid single KV state. Repair endpoint cache fixtures separately at 63/64/65, 511/512/513, and 2K/4K/8K/16K before later stages.
- Next alpha stage: blocked until cache, Exact, and real model-prefix/selective trigram-unlikelihood pilots pass without regressing V20 capability.

## Evaluator Integrity

Two evaluator incidents were found and preserved rather than hidden:

1. The original cache generator rejected intermediate output blend. The evaluator now uses exact full recomputation for intermediate alpha and leaves endpoint cache auditing separate.
2. A tokenizer omission produced zero decoded JSONL rows. The evaluator now rejects any supplied dataset that decodes to zero rows.

Focused evaluator and v4.1 tests passed `27/27`. The valid result is `fdt_v4_1_stage_a_official_fp32_eval.json`, SHA-256 `FA61659B8C526D5E637848C786C6F971B8AA8498D153E8F05C2B39D2B9E6DF93`.
