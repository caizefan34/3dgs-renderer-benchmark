# H1 authoritative B1A vs B2

## Terminal status

Source reconstruction/build, ABI, room/cam0 B2 correctness, and B1A count-regression gates passed. The matched 3-scene × 3-camera performance cohort was **not run**: the required GPU was saturated by unrelated processes. Therefore there are no B1A/B2 forward, backward, or F+B timing tables, no geomeans, and no current-performance classification. The run is incomplete, not a failed renderer comparison.

## Validated room/cam0 smoke

Room 2048×1365, camera 0, 115,278 total Gaussians:

| Mode | Visible | Intersections | Active tiles |
|---|---:|---:|---:|
| B1A AccuTile | 44,844 | 953,085 | 11,008 |
| B2 metadata | 44,908 | 953,144 | not reported by this metadata path |

B1A mean intersections/tile was 86.5811, p95 179, and tiles/Gaussian 21.2533. RGB B1A/B2 max absolute difference was `1.78814e-7`, mean absolute `3.33358e-9`, relative L2 `3.057e-8`; alpha matched exactly. Fixed nonzero-VJP gradients passed: means relative L2 `4.0313e-6`, quats `6.1253e-4`, scales `3.1495e-5`, opacity `1.0102e-6`, SH `4.0725e-7`; cosine ≥`0.99999988`, no zero/nonzero mismatch or NaN/Inf.

Checkpoint health on this smoke fixture: 44,450/44,844 visible rows had nonzero gradients (0.991214); max radius 19,096, p99 radius 155, max opacity 1, alpha saturation fraction 0.814623. This fixture is classified `OK`, not pathological under the requested <1% gradient criterion.

The earlier room/cam0 smoke artifact had recorded the B2 visible/intersection metadata values as if they were B1A. Direct execution of both current and previously validated B1A binaries agrees at 44,844 / 953,085; see freeze regression JSON.

## Build/source/ABI identity

Base `77ab983ffe43420b2131669cb35776b883ca4c3c`; B2 patch SHA256 `74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c`. Exact extension paths and hashes are in `artifacts/h1-b2-freeze/build_manifest.json`. ABI passed with no argument shift. Full records: `source_manifest.json`, `abi_audit.json`, and `build_manifest.json`.

## Why cohort was withheld

At final check the target A100 UUID `GPU-84d2099e-9223-fb5f-f807-20e9932d07e0` reported 38,143 MiB used, 2,299 MiB free, and 100% utilization. Unrelated PIDs 220824 and 295572 were actively consuming 19,318 and 18,810 MiB. Running 20 warmup/100 measurement event timing under that contention would invalidate the requested same-GPU controlled comparison and risk OOM, so neither process was disturbed and no performance result was fabricated.

Artifacts under `artifacts/h1-authoritative-b1a-vs-b2/` contain explicit `NOT_RUN` records for the unexecuted cohort. Classification is `NOT_CLASSIFIED`; no performance conclusion should be drawn.
