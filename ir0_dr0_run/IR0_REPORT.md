# IRGS / DR0 — Opportunity Oracle Report (IR0)

- Working name: **IRGS** (IncrementalRaster-GS); `DeltaRaster-GS` is a descriptive codename only.
- Protocol: `IR0_PREREG.md` (FROZEN), packet `IRGS_DR0_PACKET_20260930`.
- Run date: 2026-09-30
- Execution host: **EPIC-05** remote GPU node (`8.130.30.251:1024`, key `~/.ssh/epic-node-new`), 8x A100-80GB. NOT executed locally (local host lacks the frozen renderer / `.so` / checkpoints).
- Python env: `/root/miniforge3/envs/gsplat/bin/python` (must `import torch` before loading the frozen `.so`).
- Machinery: reused the frozen A0 library `a0_lib.py` (`/root/epic05_moonshot/a0/a0_lib.py`) + formal A0 artifacts; the frozen renderer is driven **read-only** via the same runtime capture patch A0 used. No production CUDA was modified. No densification.

## Verdict (owned by `ir0_analyze.py`)

```
IR0_DROP
selected_fixed_fraction: null
strong_by_fraction: {0.10: false, 0.20: false}
promote_by_fraction: {0.10: false, 0.20: false}
```

The analyzer (`ir0_analyze.py`, sha256 verified against the packet manifest) produced this verdict from `IR0_SUMMARY.json`. No manual override was applied.

## Why it dropped

The **IR0-A immediate-stop gate fired in every cell**. Per `IR0_PREREG.md` §3:

> If, for both f=0.10 and f=0.20, ANY cell has
> `median_8cams(dirty_segment_hit_fraction_B=32) > 0.50` → `IR0_NO_LOCALITY`, STOP before dynamic-repair probes.

Measured `median_8cams(dirty_segment_hit_fraction_B=32)`:

| cell | f=0.10 | f=0.20 |
|---|---|---|
| room@20000 | 0.9846 | 0.9940 |
| room@25000 | 0.9870 | 0.9953 |
| bicycle@20000 | 0.9266 | 0.9624 |
| bicycle@25000 | 0.9361 | 0.9673 |

All eight (cell, f) medians far exceed 0.50. The optimistic **fixed-support** dirty-work lower bound already shows that a top-10%/top-20% active set touches ~93-99.5% of B=32 segment hit slots. The frozen rule STOPs here; **IR0-B (§5), IR0-C (§6) and §7 timing were NOT run** and are reported as `NOT_MEASURED` sentinels. No rescue via new selection heuristics is permitted.

## Source / identity gate (§1)

Identity gate: **PASS** (`IR0_SOURCE_IDENTITY_*.json: pass=true`, all four cells). Live EPIC-05 hashes matched the authoritative values; the frozen source tree was not mutated.

- `core gsplat_cuda.so` = `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e` (prereg §1, exact)
- room@25K checkpoint = `b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d` (prereg §1, exact)

**Prereg §1 C0-extension SHA256 typo (documented).** The prereg inline value for the C0 extension `.so` is 63 hex chars (malformed; a dropped `b7`):

```
prereg inline (63): d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11
authoritative (64):  d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11
```

Per `IR0_PREREG.md` §1 ("recover the COMPLETE expected values from the formal A0/MOONSHOT manifests. Never infer or truncate."), the authoritative value is the A0 manifest hash (`A0_CELL_SUMMARY` identity.env.so_sha256), which equals the live EPIC-05 hash exactly. The probe gate was patched to compare against the authoritative 64-char value and to record the typo; core + room@25K prereg values are exact.

## IR0-A fixed-support dirty-work lower bound (§3)

Median over 8 frozen cameras (`IR0_FIXED_SUPPORT_LOCALITY_*.json`). Optimistic LOWER BOUND: support/order held at `theta_t`; only active-Gaussian hits/segments marked dirty. `dirty_pixel_fraction = 1.0` in all cells (the utility-ranked active set touches every rendered pixel).

| cell | f | dirty_hit | dirty_tile_isect | dirty_pixel | dirty_seg_frac_B32 | dirty_seg_hit_B32 |
|---|---|---|---|---|---|---|
| room@20000 | 0.10 | 0.4949 | 0.3436 | 1.0000 | 0.9801 | 0.9846 |
| room@20000 | 0.20 | 0.6201 | 0.4720 | 1.0000 | 0.9921 | 0.9940 |
| room@25000 | 0.10 | 0.5623 | 0.3707 | 1.0000 | 0.9849 | 0.9870 |
| room@25000 | 0.20 | 0.6806 | 0.5010 | 1.0000 | 0.9938 | 0.9953 |
| bicycle@20000 | 0.10 | 0.3308 | 0.1815 | 1.0000 | 0.9316 | 0.9266 |
| bicycle@20000 | 0.20 | 0.4629 | 0.3029 | 1.0000 | 0.9654 | 0.9624 |
| bicycle@25000 | 0.10 | 0.3922 | 0.2039 | 1.0000 | 0.9407 | 0.9361 |
| bicycle@25000 | 0.20 | 0.5181 | 0.3285 | 1.0000 | 0.9712 | 0.9673 |

Read: the incremental compositing savings that IRGS would need do not exist at the oracle level — the "sparsity" of active Gaussians does not translate into sparse segment work.

## Cache-memory model (§4)

`IR0_CACHE_MEMORY_*.json`. 16 B/segment minimal (RGB premult C=3xf32 + T=1xf32) and conservative 32 B/node. Pooled across the 8-camera cohort.

| cell | nodes_B32 | cache_GiB_16B | cache_GiB_32B |
|---|---|---|---|
| room@20000 | 74,053,680 | 1.103 | 2.207 |
| room@25000 | 68,941,855 | 1.027 | 2.055 |
| bicycle@20000 | 66,858,554 | 0.996 | 1.993 |
| bicycle@25000 | 61,781,141 | 0.921 | 1.841 |

`cache_GiB_32B(B=32) <= 16` in every cell (2.1 / 2.0 / 2.0 / 1.8), so the hard memory gate would pass — but it is moot: promotion already fails on the IR0-A locality gate. This is a model estimate only, not an implementation footprint.

## Active-set oracle / utility capture (§2)

`IR0_ACTIVE_UTILITY_*.json`. First-order positive optimizer utility proxy `u_i = max(0, -<g_i, delta_theta_i>)`, top-f visible selection (ties by Gaussian ID). Oracle ranking only — not a deployable selector.

| cell | utility_capture f=0.10 | f=0.20 |
|---|---|---|
| room@20000 | 0.8225 | 0.9160 |
| room@25000 | 0.8301 | 0.9237 |
| bicycle@20000 | 0.7258 | 0.8785 |
| bicycle@25000 | 0.7510 | 0.8960 |

Optimizer-utility concentration is decent (top-20% active captures ~88-92% of positive utility), which is exactly why the result is a DROP rather than a trivial no-signal: utility is concentrated, but that concentration does NOT yield sparse segment work.

## Not measured (frozen stop)

- IR0-B exact sparse-update repair census (§5): `NOT_MEASURED` — gate fired.
- IR0-C one-step utility retention (§6): `NOT_MEASURED` — gate fired; sentinels `R_loss=0`, `S_oracle_upper=0`, `non_descent_count=8` (cannot satisfy promote).
- §7 phase timing / `S_oracle_upper`: `NOT_MEASURED` — gate fired.

`IR0_DYNAMIC_REPAIR.json`, `IR0_ONE_STEP_UTILITY.json`, `IR0_PHASE_TIMING.json` are therefore absent (permitted: the frozen rule forbids running those stages after the IR0-A hard stop).

## Interpretation boundary

`IR0_DROP` falsifies the DR0 premise: under the current exact support, the optimistic fixed-support incremental-work lower bound is already too large. This does NOT prove IRGS is impossible in principle — a different (non-utility) active-set selection or a fundamentally different compositing structure could change the picture — but per the frozen protocol no such rescue is admissible at IR0. Per `00_START_HERE.md`, only `IR0_STRONG_PROMOTE` / `IR0_PROMOTE` may proceed to IR1; `IR0_DROP` ends DR0.

## Artifacts

Local bundle: `ir0_dr0_run/`
- `cells/<cell>/IR0_{SOURCE_IDENTITY,ENVIRONMENT,ACTIVE_UTILITY,FIXED_SUPPORT_LOCALITY,CACHE_MEMORY,GATE,GATE_SUMMARY}_<cell>.json`
- `cells/*.log` (per-cell driver logs)
- `out/IR0_SUMMARY.json`, `out/IR0_VERDICT.json`
- `packet/` (frozen protocol docs + analyzer)
- `SHA256_MANIFEST.json` (this bundle)
- `IR0_REPORT.md` (this file)