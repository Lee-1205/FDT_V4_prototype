# Release Assets

Tag: `v4-prototype-step3000-20260827`

Expected assets:

- `fdt_v4_step3000_model.pt`: model-only checkpoint for inference and evaluation
- `fdt_v4_step3000_recovery.pt.part001`
- `fdt_v4_step3000_recovery.pt.part002`
- `fdt_v4_step3000_recovery.pt.part003`
- `fdt_v4_curriculum_v4.tar.gz`: exact dataset payload
- `fdt_v4_step3000_metadata.tar.gz`: frozen run metadata and logs
- `SHA256SUMS.txt`: complete asset digest list

The recovery parts concatenate byte-for-byte into the original optimizer-bearing checkpoint. Use `Restore-FdtV4Recovery.ps1` and verify the expected SHA-256 from `SHA256SUMS.txt` before resume.

The model-only asset is sufficient for generation and evaluation. Exact training continuation requires the reconstructed recovery checkpoint because it includes AdamW state, source cursors, token counter, and RNG state.

