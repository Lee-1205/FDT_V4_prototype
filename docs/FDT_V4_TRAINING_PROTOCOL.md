# FDT v4 Main Training Protocol

This protocol defines one trainable architecture: `fdt_v4_main_426m`, configured
at `dim=1216`, `20` layers, `19` heads, local RoPE window `64`, `256` anchors,
top-8 routing, router dimension `256`, and anchor layers `0,2,...,18`. Tiny v4
instances may exist inside CPU tests only; no 116M model is a training target or
saved research artifact.

The main model has a 16,384-token RoPE position capacity so an 8,192-token
prompt still leaves room for decode. Initial curriculum shards may use shorter
sequences; training sequence length is not evidence of 8K generalization, so
2K/4K/8K are audited separately and 16K remains exploratory.

The curriculum gives the majority of effective tokens to natural language and
factual knowledge. The exact-copy objective is isolated to prompt-source
auxiliary batches. Generic all-token pointer loss is forbidden. EOS receives
weight `2.0`. Generated-prefix recovery is optional, low-weight, and cannot begin
before its configured minimum step. Each log record reports effective tokens,
objective loss fractions, and periodic per-objective gradient contribution norms.

Every recovery contains model weights, AdamW state, Python/Torch/CUDA RNG state,
and deterministic source cursors. `latest_recovery.pt` and `latest.pt` are
written through temporary files followed by atomic replacement. A resume is
accepted only when the recovery belongs to the same C: run directory, contains
the full optimizer/cursor state, and identifies the same `fdt_v4` configuration.
No checkpoint is deleted by the trainer or supervisor.

The launch preflight constructs the configured model on the meta device and
rejects anything outside the 426M class. The tokenizer contract is pinned to
the tokenizer directory, `tokenizer.json` SHA-256, vocabulary size `24576`,
pad `0`, and EOS `2`. Training parameters remain FP32 for optimizer/master
semantics; CUDA uses BF16 autocast only. CPU tests do not launch GPU work.

Train shards must explicitly contain `input_ids`, `attention_mask`, and
`labels`. Exact-copy shards additionally require `prompt_mask`,
`source_boundary`, `copy_source_positions`, and `copy_target_mask`, with
prompt labels set to `-100`. The exact objective receives those explicit
labels and is never applied to generic all-token LM batches. The loader streams
one shard at a time and records every manifest/shard hash, tokenizer hash,
configuration hash, precision contract, and environment in the run manifest.
If a V20 FDT v3 parent is selected, only the strict allowlist is converted;
RoPE and exact-memory parameters remain new, anchor transfer is verified before
anchors are frozen, and `v20_conversion_manifest.json` records the decision.

The live log records finite loss, normalized sampled routing entropy when an
anchor metric is available, tokens per second, additional tokens, target
additional tokens, and an ETA-compatible counter. Gradient probes are disabled
by default. Validation snapshots and the configured overfit gate may stop a run
after an atomic recovery; they do not change the model architecture or data
contract.

## Luna, Terra, Sol escalation

Luna supervises routine training only. On any abnormality, gate failure, safety
stop, non-finite value, or unexpected trainer exit, Luna writes/retains the
verified recovery, creates a stop marker, and atomically writes a Terra handoff
manifest. Luna does not redesign, change objectives, continue past a failed gate,
or claim to have evaluated the model. The handoff includes machine-readable
`severity`, `classification`, `trigger`, `required_remedy_classification`, and
`next_route` fields.

Normal handoffs declare `status=READY` and `handoff_type=EVALUATION` and include
the fixed evaluation tensor file/hash and tokenizer hash. Abnormal handoffs
declare `status=ABNORMAL` and `handoff_type=INCIDENT`. An explicit Console resume
must pass `--resume-paused`; without it a PAUSED recovery is handed to Terra and
is not consumed as a trainer input.

Terra owns diagnosis and evaluation. Terra may prescribe a lightweight
operational/config/runtime repair for Luna or direct automation to apply. If the
required remedy is a major architecture, objective, or data-contract change,
Terra routes the issue to Sol. A Terra handoff must not be interpreted as an
evaluation result until Terra writes its own verified evidence.
