# H8-0 — Moment-Space Geometry Adjoint Compile + Arithmetic Gate

**Verdict: `H8_0_COMPILE_FAIL` — STOP.** No production H8-M implementation or benchmark was authorized. No training ran and no renderer source changed.

## Exact algebra and ABI

The frozen SCALAR_ADJOINT source has the required `if (opac * vis <= MAX_ALPHA)` branch. Its baseline branch computes `v_sigma = -opac*vis*v_alpha`, then the two mean, three conic, and opacity gradients. Accumulating `R0,Rx,Ry,Rxx,Rxy,Ryy` only in that same branch and applying the requested reconstruction is an exact linear factorization. `v_opacity = R0` agrees with the source semantics.

Existing buffers suffice: `v_means2d=(Rx,Ry)`, `v_conics_out=(Rxx,Rxy,Ryy)`, and `v_opacities_flat=R0`; no persistent tensor, temporary global tensor, or forward checkpoint is needed. `v_means2d` is Python-returned and must be overwritten with its reconstructed real gradient before backend return. `v_conics_out` is internal to projection.

The blocker is decisive: the projection VJP already reads forward `conics` and both gradient buffers, but does **not** receive `opacities`. Reconstruction needs opacity and it cannot be inferred from moments. Passing it requires an additional global FP32 opacity load per live projected Gaussian. That violates the mandatory “no additional global loads” condition.

## SASS and resources

An isolated `sm_80`, `CDIM=3`, `PX=2` validation-only probe compiled on mx A100. Valid-contribution arithmetic is 14 operations for baseline (12 `FMUL`, 2 `FFMA`) versus 5 `FMUL` for moments: **64.29% less before reconstruction**. The deferred reconstruction is 11 operations per existing flush unit (6 `FMUL`, 5 `FFMA`). No special FP operation appears in these bodies.

| Probe body | FFMA | FMUL | loads | stores | registers | spills | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline contribution | 2 | 12 | 13 | 6 | 18 | 0 | 0 B |
| Moment contribution | 0 | 5 | 7 | 6 | 20 | 0 | 0 B |
| Deferred reconstruction | 5 | 6 | 12 | 6 | 18 | 0 | 0 B |

The isolated moment body is exactly `+2` registers and has no local memory, spills, added shared memory, or synchronization. Frozen full production SCALAR_ADJOINT remains 56 registers, 0 spills, 5120 B dynamic shared memory, and one existing barrier. An integrated production compile was deliberately not performed after the traffic gate failed.

## Real-state replay and multiplicity

Real camera-0 state was captured from room, bicycle, and garden at a 2048-pixel long side, with fixed random upstream VJP seed 4200. FP replay used 16 intersection-count-stratified tiles per scene (135,579 / 138,372 / 96,316 valid contributions in FP64). FP64 is at rounding level; deterministic FP32 is the expected reassociation-level difference. All nine support comparisons are zero.

| Scene | Valid contributions | Flush units | Mean / p50 / p75 / p90 / p95 / p99 lanes | Amortization |
|---|---:|---:|---|---:|
| room | 100,070,612 | 2,741,069 | 36.51 / 33 / 64 / 64 / 64 / 64 | 36.51x |
| bicycle | 103,039,394 | 3,590,345 | 28.70 / 22 / 52 / 64 / 64 / 64 | 28.70x |
| garden | 66,985,141 | 1,612,836 | 41.53 / 48 / 64 / 64 / 64 / 64 | 41.53x |

A reduction unit is the existing `(fine tile, sorted Gaussian, PX=2 warp)` reduction with at most 64 pixel lanes. “Flush units” preserves the current `warp.any(any_valid)` schedule; geometry-nonzero counts differ by at most nine units across these scenes.

| Scene | Effective baseline ops/valid | Effective moment ops/valid | Arithmetic removed |
|---|---:|---:|---:|
| room | 14.000 | 5.301 | 62.13% |
| bicycle | 14.000 | 5.383 | 61.55% |
| garden | 14.000 | 5.265 | 62.39% |

This is an arithmetic ceiling, not a runtime-speedup claim.

## Gate result

Arithmetic, multiplicity, no-spill, and isolated `+2` register conditions are strong. However, `H8_0_STRONG` also requires **zero new global traffic**. The mandatory opacity load fails it. The result is therefore not `H8_0_STRONG`, `MARGINAL`, or `WEAK`; it is the stricter terminal outcome `H8_0_COMPILE_FAIL`.

Evidence is in [artifacts/higs-h8-0](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h8-0).
