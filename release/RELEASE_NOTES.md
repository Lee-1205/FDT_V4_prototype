# Pure FDT V4 Prototype, Step 3,000 Snapshot

Private immutable snapshot of the 424,474,072-parameter Pure FDT V4 research model.

Included in the repository:

- complete Python model/runtime source
- V4 trainer, generator, evaluator, audits, benchmarks, and migration tools
- focused V4 tests
- exact tokenizer
- pinned training configurations
- architecture, Exact Memory, long-context, and training documentation
- current and historical V4 evidence, incidents, and run metadata
- external review and design-reference materials
- SHA-256 file inventory

Included as Release assets:

- unquantized model-only checkpoint
- split optimizer-bearing exact-resume checkpoint
- full V4 curriculum dataset payload
- frozen step-3,000 metadata bundle
- SHA-256 asset manifest

Snapshot metrics:

- tokens seen: 17,362,528
- fixed validation loss: 4.7819013595581055
- overfit gate: not triggered
- dead-anchor hard gate: healthy
- observed optimized short-path throughput: 1,267 to approximately 1,369 tokens/s

This is a research prototype snapshot, not a promoted production release.

