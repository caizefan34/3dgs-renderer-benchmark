# BES0 Real-Scene Opportunity Screen — FALSIFICATION

**Date:** 2026-10-05
**Author:** DSH (autonomous, MX A100-80 idle window)
**Status:** COMPLETE — decisive negative result
**Question:** Does BES0's "bounded-event" routing yield a >=3x training-speedup
opportunity on real 3DGS workloads? (3x is the claim, not a measured value.)

## TL;DR
**No.** On three real MipNp360 scenes (room, bicycle, garden; 12 dev cameras
each = 36 cameras, ~958M Gaussian occurrences at the protocol tile size), the
per-occurrence bounded-alpha routing that defines BES0 routes **exactly 1 of
957,816,964 occurrences to the cheap "R" path** (rho_R = 1.04e-9). The
per-pixel early-termination opportunity (where 3DGS speedup actually comes
from) is **0.8-1.9%** over a complete renderer. Neither mechanism approaches
3x. The per-occurrence formulation is **structurally vacuous** on real 3DGS.

## Method
Faithful reimplementation of BES0's `build_bounds` R0 enumeration against the
frozen `scene_adapter.py` (gsplat-1.5.3-convention projection, verified in
BES0 E0: means2d matches gsplat to 6.1e-5 px; E0 math identities PASS). Two
opportunity structures are measured per dev camera:

1. **Per-occurrence R/D routing (BES0 literal mechanism).** For each
   (gaussian, tile) occurrence over the tile's `ts x ts` pixels:
   `hi = max alpha` (tile pixel nearest the peak), `lo = min alpha` (farthest
   tile corner), `is_R = (lo > 0) & (hi/lo <= gamma)`. `rho_R = n_R / n_occ`;
   implied upper-bound speedup `= 1/(1-rho_R)`. Tile-size sweep `ts in
   {16,8,4,2}`; gamma sweep `{2,5,10,30,100}`.
2. **Per-pixel early-termination.** Strided pixel sample (~2500 px/cam),
   front-to-back cumulative transmittance `T`, count occurrences after
   `T < eps` (`eps = 1/255`). `skipped_fraction -> speedup = 1/(1-sf)`.

This is an **opportunity screen (analytic upper bound)**, not a CUDA timing.
The realized speedup is <= these bounds.

Guard: identical fail-closed MX policy (`mx_guard`): host `bms-39468022-001`,
8x A100-40GB SM80, allowed GPUs `[1,2,3,4,6,7]`, idle window <=2% util /
<=512 MiB / 0 procs for 20s. Run on GPU 7. No co-resident job was touched.

## Results

### Per-occurrence routing (rho_R) — the BES0 mechanism
| scene | visible gaussians | n_occ (ts=16) | n_R (ts=16) | rho_R (ts=16) | rho_R (ts=8) | rho_R (ts=4) | rho_R (ts=2) |
|-------|------------------:|--------------:|------------:|--------------:|-------------:|-------------:|-------------:|
| room   | 9,501,967  | 168,700,377 | 0 | 0.0        | 0.0        | 0.0        | 1.09e-7 |
| bicycle| 34,154,646 | 540,816,678 | 1 | 1.85e-9    | 0.0        | 2.73e-9    | 1.04e-7 |
| garden | 21,359,326 | 248,299,909 | 0 | 0.0        | 3.59e-9    | 1.04e-8    | 9.94e-8 |
| **total** | **65,015,939** | **957,816,964** | **1** | **1.04e-9** | | | |

- Gamma sweep at ts=16: rho_R = 0.0 for **all** gamma in {2,5,10,30,100}.
- Tile-size sweep: rho_R stays <=1.1e-7 even at ts=2px (1 routable occurrence
  per ~10M). The protocol tile size (gsplat 16px) gives rho_R = 1.04e-9.

### Per-pixel early-termination (the real 3DGS speedup source)
| scene | skipped/total pairs | skipped_fraction | speedup bound |
|-------|--------------------:|-----------------:|--------------:|
| room   | 53,700 / 6,380,492  | 0.0084 | 1.008x |
| bicycle| 67,704 / 8,029,038  | 0.0084 | 1.008x |
| garden | 59,856 / 3,089,326  | 0.0194 | 1.020x |

- Camera-dependent: most views sf ~ 0; a few reach 0.08 (room cam 226),
  0.11 (garden cam 152), 0.30 (garden cam 51). Aggregate ~1%.

## Root cause (why per-occurrence routing is vacuous)
Real 3DGS Gaussians are **anisotropic thin splats** with a **sub-pixel minor
axis**. Diagnostic (room cam 1): conic = [9.93, -0.87, 0.076] -> sigma_short
~ 0.32px, sigma_long ~ 115px; screen radius median 93px, p99 15540px; active
opacity median 0.037. Within any tile >= 2px, the sub-pixel minor axis makes
alpha underflow to 0 at the tile's far corners, so `lo = 0` for essentially
every occurrence -> `is_R = False` universally. The gamma-regularity condition
`hi/lo <= gamma` is therefore violated everywhere; it is not a gamma or
threshold artifact (rho_R = 0 for gamma up to 100 and tile down to 2px).

The per-GAUSSIAN global bound (min alpha over the whole bbox) is even more
vacuous: `lo = 0` for 100% of gaussians, because the bbox always includes the
diagonal far corner.

## Verdict
**FALSIFIED.** BES0's >=3x training-speedup claim is not supported on real
3DGS scenes:
- The per-occurrence bounded-alpha routing (BES0's defining mechanism) is
  structurally vacuous: rho_R ~ 1e-9, because real Gaussians have sub-pixel
  minor axes that collapse the per-tile alpha lower bound to 0.
- The per-pixel transmittance-bounded formulation (the *correct* bounded-event
  for 3DGS) is exactly gsplat's existing early-stop; the additional headroom
  over a complete renderer is ~1-2% (camera-dependent, <=30% in a few views).

Combined with BES0 E0 (math PASS) and E1 (adapter correct after fixture fix),
the bounded-event *per-occurrence* research direction is a dead end for 3DGS
training speedup. The honest BES0 verdict remains
`INCONCLUSIVE_CUDA_NOT_IMPLEMENTED` (no production CUDA renderer), and this
screen adds: even with a perfect renderer, the routing opportunity is ~0.

## Implications for the research program
- Per-occurrence alpha-bound routing: **abandon** (structurally empty).
- Per-pixel early-stop: already in gsplat; headroom ~1-2% aggregate. A
  quality-aware threshold could capture the camera-dependent tail (up to 30%
  in saturated views) — the only positive, actionable lead.
- Strongest positive remains the HiGS line (B1A: 1.104x geomean, +0.110 dB
  dPSNR, 11 scenes x 3 seeds, committed).

## Reproduce
- Probe: `bes_opp_probe/probe_opportunity.py` (v4, tile+gamma sweep).
- Launcher: `bes_opp_probe/run_opp.sh`, chain: `bes_opp_probe/chain_opp.sh`.
- Raw per-camera JSON: `bes_opp_probe/out/opportunity_{room,bicycle,garden}.json`.
- Chain log: `bes_opp_probe/out/chain.log`.
- Run on MX: `bash chain_opp.sh 7` (GPU 7), results in `/dev/shm/bes_opp/out/`.

## Caveats
- Opportunity screen (analytic upper bound), not measured CUDA speedup.
- Adapter projection is gsplat-1.5.3-convention verified; the gsplat CUDA
  *rasterizer* itself is currently unloadable on MX (gsplat 1.5.3 JIT vs
  torch 2.7.1 `_jit_compile` signature mismatch). This does not affect the
  screen, which uses the adapter's pure-torch projection + numpy bounds.
- `/dev/shm` scratch; checkpoints on `/mnt/storage_pool` (hashed, see
  `config/scene_manifest.json`).
