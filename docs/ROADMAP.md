# FDT v4 Roadmap

1. Freeze the repaired 424,474,072-parameter implementation in the private
   repository with hashes and a clean commit.
2. Complete CPU architecture, causal, RoPE, fuzzy-state, exact-memory, cursor,
   repetition-scope, checkpoint, and handoff regression tests.
3. Build fresh C-only tensor datasets with explicit labels, pinned tokenizer,
   source provenance, licenses, revisions, overlap audit, and fixed validation.
4. Run bounded current-GPU memory and FP32 correctness/performance audits at
   increasing context lengths. Do not start long training if checkpoint safety
   or 8K integrity fails.
5. Train the single 426M-class lineage under Luna supervision. Terra evaluates
   verified checkpoints. Major architecture/objective/data-contract changes
   require Sol; bounded operational repairs may be applied directly.
6. Keep natural language and factual knowledge primary, retrieval substantial,
   and code/JSON auxiliary. No 116M lineage, conversational SFT, or quantized
   official evaluation.
