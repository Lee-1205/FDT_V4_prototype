# FDT V4 Prototype

Private research and reproducibility archive for the current Pure FDT V4 prototype.

## Snapshot

- Architecture: Pure FDT V4
- Parameters: 424,474,072
- Vocabulary: 24,576 tokens
- Context ceiling: 16,384 tokens
- Core memory: 256 fuzzy anchors, top-8 cosine routing, 64-token local window
- Exact Memory: proposal-driven copy path with cursor continuation and full-scan fallback
- Generation control: generated-text-scoped trigram penalty and hard blocking
- Training snapshot: step 3,000, 17,362,528 tokens
- Fixed validation loss: 4.7819013596
- Overfit gate: not triggered
- Training precision: BF16 autocast with FP32 optimizer master weights

The active experiment continued after this immutable snapshot. The repository records the exact code, configuration, tests, manifests, and evidence used at the snapshot. Large binary assets are published in the matching private GitHub Release.

## Layout

- `src/`: model and runtime implementation
- `scripts/`: V4 training, generation, evaluation, audit, migration, and benchmark tools
- `tests/`: focused V4 tests
- `configs/`: pinned V4 training configurations
- `docs/`: architecture and protocol notes
- `apps/`: V4 Training Console
- `tokenizers/`: exact tokenizer files
- `datasets/manifests/`: data contracts and source manifests
- `evidence/`: experiment artifacts and non-checkpoint run records
- `release/`: release-asset inventory and verification instructions

## Environment

Windows, Python 3.10+, CUDA-capable PyTorch, and an NVIDIA GPU with enough memory for the selected sequence bucket are expected.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q tests
```

Official evaluation uses the original unquantized model operations. Quantized results are not accepted as official evidence.

## Reproduce Evaluation

After downloading the model asset from the matching Release:

```powershell
python scripts/evaluate_fdt_v4.py --help
python scripts/final_audit_fdt_v4.py --help
```

Use `scripts/generate_fdt_v4.py` for generation and retain the configured repetition controls when comparing outputs.

## Resume Training

The optimizer-bearing recovery checkpoint is split across Release assets because it exceeds GitHub's per-file limit. Reassemble and verify it using `release/Restore-FdtV4Recovery.ps1`, then pass the result to `scripts/train_fdt_v4_curriculum_speed.py --resume` in a fresh output directory. Never overwrite a preserved run.

## Research Status

This is an experimental prototype, not a promoted production model. At the snapshot, loss, routing, Exact Memory proposal recall, checkpoint atomicity, and overfit gates were healthy, but large-scale capability claims still require the established unquantized evaluation suite and sealed holdout criteria.

See [CURRENT_STATUS.md](CURRENT_STATUS.md), [DATASETS.md](DATASETS.md), and [release/RELEASE_ASSETS.md](release/RELEASE_ASSETS.md).

