# Dataset Archive

The exact V4 curriculum payload used by the step-3,000 snapshot is published as a private Release asset. The repository keeps all source manifests and preflight evidence in plain text.

## Curriculum Components

- Natural language: 20,000 rows at 512 tokens
- Factual language: 20,000 rows at 512 tokens
- Exact-copy supervision: 4,096 rows with explicit source positions and copy-target masks
- Generated-prefix recovery: 2,048 rows with loop-negative supervision
- Long context: 64 rows at 8K and 64 rows at 16K
- Fixed validation: 800 rows at 512 tokens

The current payload is about 237 MiB. It is kept outside Git history to prevent permanent binary repository growth.

## Integrity

Canonical manifest and shard hashes are recorded in:

- `datasets/manifests/manifest.json`
- `datasets/manifests/*/manifest.json`
- `evidence/runs/fdt_v4_main_424m_curriculum_speed_20260827_r8_fast_backend/data_preflight.json`
- `evidence/runs/fdt_v4_main_424m_curriculum_speed_20260827_r8_fast_backend/run_manifest.json`

Historical overlap limitations and source provenance remain in the archived artifacts. Do not infer a scan against payloads that were previously pruned after hash journaling.

