# CG0 Report - Gaussian-as-Compiled-Representation Oracle

**Verdict: `CG0_COMPILE_CAPACITY_CLOSED`**

Generated: 2026-09-30 21:24 (EPIC-05 A100, authoritative frozen binaries)

## 1. Central question

CG0 isolates whether a deployable Gaussian scene can be *compiled* deterministically
from an oracle geometry signal (expected depth from an authoritative C0 checkpoint)
plus ground-truth RGB, followed by a short topology-frozen C0 refinement (0-1500 steps),
instead of long-form optimization. All thresholds are frozen in `CG0_PREREG.md`; the
analyzer (`cg0_analyze.py`) owns the verdict and no post-hoc rescue is permitted.

## 2. Identity gate (prereg section 1)

- source_identity_pass: **True**
- bicycle@25000 live SHA256 match: **True**
- Binary pins:
  - `c0_ext`: `d26cccf46c7932bf...`
  - `core_gsplat_cuda`: `935ea6c0dd04f34e...`
  - `fused_ssim`: `dd6a4c799b98ff9f...`

All four checkpoint SHA256 hashes resolve against the N1 canonical provenance;
the three frozen binary pins match the packet SHA256s exactly.

## 3. Oracle boundary / no-leakage (prereg sections 3-4)

The compiler consumes ONLY: oracle expected depth maps (COMPILE64), GT training images,
calibration cameras, and the scalar source Gaussian count N_source. It does NOT read
source xyz/covariance/opacity/SH/optimizer moments/per-Gaussian IDs/RGB.
`CG0_NO_LEAKAGE_AUDIT.json` records `source_gaussian_params_read_by_compiler=false` for all cells.

## 4. Per-cell results

| cell | N_source | N_compiled | count_ratio | voxel_h | cond_frac | compiled0 holdout dPSNR | compiled1500 holdout dPSNR |
|------|---------:|-----------:|------------:|--------:|----------:|------------------------:|---------------------------:|
| room@15000 | 841781 | 824470 | 0.9794 | 0.3353 | 0.0708 | -17.21 dB | -9.70 dB |
| room@25000 | 841781 | 842539 | 1.0009 | 0.3254 | 0.0660 | -19.13 dB | -10.46 dB |
| bicycle@15000 | 3654566 | 3689497 | 1.0096 | 0.0390 | 0.1037 | -11.81 dB | -7.87 dB |
| bicycle@25000 | 3654566 | 3616312 | 0.9895 | 0.0001 | 0.1047 | -12.69 dB | -8.91 dB |

Source C0 reference quality (holdout16 / trainval16 PSNR):
- room@15000: source holdout PSNR = 29.59 dB, trainval PSNR = 30.54 dB
- room@25000: source holdout PSNR = 31.56 dB, trainval PSNR = 31.64 dB
- bicycle@15000: source holdout PSNR = 26.34 dB, trainval PSNR = 22.87 dB
- bicycle@25000: source holdout PSNR = 27.22 dB, trainval PSNR = 24.28 dB

## 5. Frozen threshold check (prereg section 13)

STRONG requires compiled0 holdout dPSNR >= -0.75 and compiled1500 holdout dPSNR >= -0.08;
CAPACITY_SURVIVES (lower bar) requires compiled0 >= -1.50 and compiled1500 >= -0.15.
Time envelope: conditional_compile_refine_fraction <= 0.12 (STRONG) / <= 0.18 (SURVIVES).

| cell | count in [0.8,1.2] | cond_frac <= 0.12 | compiled0 >= -1.5 | compiled1500 >= -0.15 |
|------|:---:|:---:|:---:|:---:|
| room@15000 | PASS (0.979) | PASS (0.071) | FAIL (-17.21) | FAIL (-9.70) |
| room@25000 | PASS (1.001) | PASS (0.066) | FAIL (-19.13) | FAIL (-10.46) |
| bicycle@15000 | PASS (1.010) | PASS (0.104) | FAIL (-11.81) | FAIL (-7.87) |
| bicycle@25000 | PASS (0.990) | PASS (0.105) | FAIL (-12.69) | FAIL (-8.91) |

Every cell passes the count and time-envelope gates but **fails the quality gates by 8-19 dB**.

## 6. Verdict derivation

Precedence: BLOCKED_SOURCE_IDENTITY -> INVALID_ORACLE_LEAKAGE -> ORACLE_EXTRACTION_FAIL ->
COMPILER_NUMERICAL_FAIL -> COMPILE_CAPACITY_CLOSED -> CAPACITY_SURVIVES -> STRONG_CAPACITY_SURVIVES.

- Identity, no-leakage, oracle-extraction, compiler-numerical gates: **PASS** (all four cells).
- STRONG_CAPACITY_SURVIVES: **false** (quality far below -0.08 dB).
- CAPACITY_SURVIVES: **false** (quality far below -1.50 dB at compiled0).
- Therefore: **CG0_COMPILE_CAPACITY_CLOSED**.

## 7. Interpretation

Under the frozen compiler family (stride-4 oracle-depth candidate generation, voxel+normal
fusion matched to N_source, multi-view SH appearance, analytic opacity, forward-only 20-point
global calibration, and <=1500-step topology-frozen refinement), the compiled scene cannot
approach the optimized C0 source within the frozen budget on ANY cell. The representation
budget (count ~ N_source) and the time envelope are met, but the deterministic depth+RGB
compilation leaves an 8-19 dB quality gap at compiled0 that 1500 refinement steps close only
partially (to ~8-10 dB below source).

Per prereg section 15, this is a genuine DROP for this compiler family: 'Gaussian as a compiled
deployment representation' is too weak under oracle geometry. No rescue was applied (no source
parameter reads, no densification, no >1500-step refinement, no stride/threshold/grid changes).

## 8. Reproducibility

- Environment: Linux-5.10.134-013.9.kangaroo.al8.x86_64-x86_64-with-glibc2.35, torch 2.9.1+cu128, CUDA 12.8, GPU NVIDIA A100-SXM4-80GB.
- Frozen binaries: c0_ext/core_gsplat_cuda/fused_ssim SHA256 pinned in `cg0_prereg.py`.
- Per-cell artifacts in `out/<cell>/`; combined `out/CG0_SUMMARY.json` + `out/CG0_VERDICT.json`.
- `MANIFEST.sha256` lists SHA256 of all emitted artifacts.
