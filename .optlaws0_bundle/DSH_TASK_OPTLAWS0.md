# DSH TASK — OPTLAWS0 / MG0-S + OC0

This task is diagnostic only.

## Priority

1. Run/finish MG0 exactly as frozen.
2. While MG0 gradients are available, compute MG0-S streaming summaries.
3. Run OC0 renderer-state replay only after MG0-S row IDs are frozen.

Do not change MG0 gates.

## MG0-S required artifacts

For each scene/checkpoint cell at 20K and 25K:

- full per-row Adam-coordinate variance vector V_j
- full concentration summary
- frozen 4096-row low-rank sample IDs
- enough preconditioned gradient data for those rows to independently verify
  rank metrics
- low-rank summary JSON

Required filenames:
- `MG0S_VARIANCE_{scene}_{ckpt}.npz`
- `MG0S_CONCENTRATION_{scene}_{ckpt}.json`
- `MG0S_LOWRANK_SAMPLE_{scene}_{ckpt}.npz`
- `MG0S_LOWRANK_{scene}_{ckpt}.json`

## OC0 required steps

Use the exact frozen row/camera construction in `OC0_PREREG.md`.

Implement diagnostic native replay instrumentation only for the selected
row-camera pairs.

Do NOT change production renderer.

Export exact renderer summaries and reuse MG0 gradient rows.

Required:
- `OC0_SELECTION.json`
- `OC0_RENDERER_FEATURES_{scene}_{ckpt}.parquet`
- `OC0_GRADIENT_ROWS_{scene}_{ckpt}.npz`
- `OC0_INSTRUMENTATION_PARITY.json`
- `OC0_CONTAMINATION.json`

Parity requirement:
instrumented diagnostic replay must match uninstrumented native RGB/loss within
the frozen C0 numerical envelope before renderer features are accepted.

## Forbidden

Do NOT:
- run new full training;
- use formal test views;
- change loss/resolution;
- tune thresholds;
- add learned conflict features;
- alter MG0 hot-set K;
- implement OneView-GS production trainer;
- issue a scientific verdict.

Return raw artifacts to ChatGPT for independent analysis and STOP.
