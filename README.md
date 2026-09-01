# FDT V4 Research Archive

Public source, evaluation, and reproducibility archive for Pure FDT V4 and the
behavior-preserving FDT v4.1 repair program.

## Current Snapshot

- Architecture: Pure FDT, 426M class
- Vocabulary: 24,576 tokens
- Context ceiling: 16,384 tokens
- Core memory: 256 fuzzy anchors, top-8 cosine routing, 64-token local window
- Position transition: behavior-preserving V20 to staged RoPE output blend
- Exact Memory: explicit-contract retrieval and lossless cursor copy
- Product loop control: generated-only trigram control, calibrated strength 13
- Current decision: product runtime candidate; raw model and Stage B held
- Selected checkpoint SHA-256:
  `96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE`

The selected checkpoint passes the sealed Exact Memory matrix and repaired
cache fixtures. It is not a final autonomous language-model release: paired NLL
is still 0.974% worse than V20 and penalty-off raw generation is loop-free on
only 6/100 prompts. V20 scores the same 6/100 on those exact inputs, so this is
an inherited base limitation rather than a confirmed v4.1 transition regression.

## Layout

- `src/`: model and runtime implementation
- `scripts/`: training, generation, evaluation, and diagnostic tools
- `tests/`: focused regression tests
- `configs/`: historical V4 and current v4.1 experiment contracts
- `docs/`: architecture and protocol notes
- `datasets/manifests/`: reproducibility manifests without large tensor shards
- `evidence/`: compact experiment artifacts and decision records
- `release/`: immutable historical step-3000 asset inventory

## Verification

Windows, Python 3.10+, CUDA-capable PyTorch, and an NVIDIA GPU are expected for
full model evaluation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q tests
```

Official results use original unquantized FP32 model operations. Run
`scripts/evaluate_fdt_v4.py` for model evaluation and
`scripts/generate_fdt_v4.py` for the repaired product runtime. Exact copy mode
requires an explicit span map; it must not activate on ordinary prompts.

## Research Boundary

This repository preserves both positive and negative results. Runtime loop
control and explicit Exact Memory are validated, but they do not prove raw
language fluency or factual competence. Large training remains blocked until an
isolated objective improves external-control-off generation without weakening
the unchanged V20 capability gates.

See [CURRENT_STATUS.md](CURRENT_STATUS.md) and the hashed decision evidence in
`evidence/v4_1_20260901/`.
