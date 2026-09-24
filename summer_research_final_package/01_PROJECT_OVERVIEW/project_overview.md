# 01 · Project Overview

## Title

**Benchmarking and Optimizing 3D Gaussian Splatting (3DGS) Renderers: A Candidate-Driven Study of Tile Geometry, Sparse Computation, and Training Performance**

## Abstract (state of evidence)

This summer research project built a reproducible, cohort-strict evidence base for
optimizing 3D Gaussian Splatting (3DGS) rendering and training. It established a
validated reference implementation ("Reference V1") of the canonical 3DGS renderer
using gsplat 1.5.3 as the backend, then systematically evaluated a large portfolio
of candidate optimizations across three independent but shared-infrastructure
tracks: (1) a **renderer survey** across five renderer families; (2) **forward-pass
and sorting candidates** (C1 depth compression, segmented sort, C17 tile-local
queues, C18 incremental sort, C19 occupancy/cache studies, C43 adaptive tile size);
(3) **backward-pass / sparse computation candidates** (C25 sparse backward, C51
selective backward, C49 attribute decoupling) — and (4) a separate **storage
compression** track (SPZ, FCGS). The final, formally pre-registered conclusion was
that a fused render + pruning candidate (`gsplat_30k_fused_prune10_rclip05`) passes
all quality gates while delivering a measured **1.164x mean wall-clock speedup**
(95% CI lower bound 1.034) at 30k steps on 11 scenes × 3 seeds, with quality
non-inferiority (PSNR CI lower bound −0.022 dB ≥ −0.10 dB gate). A large body of
candidates was dropped or falsified; those negative results are preserved as first-class
evidence in this package.

## Three research tracks

1. **Survey track** — `paper/survey-claims.json`
   Which open 3DGS renderers are reproducible and comparable? Result: a
   source-pinned registry of 10 families and a 5-renderer A100 matrix;
   systematic-review claims remain blocked pending a frozen search log.

2. **Differentiable HiGS track** — `paper/higs-claims.json`
   Can hierarchical 3DGS support a correct native backward pass and reduce
   end-to-end training cost without sacrificing converged quality?
   Result: native CUDA backward implemented and verified; trainability
   established; mean peak-memory −21.6%; but full-convergence training
   speedup remained blocked until the fused `gsplat_30k_fused_prune10_rclip05`
   candidate passed the pre-registered confirmatory gate (see `03_G`.

3. **Storage compression track** — `paper/compression-claims.json`
   Which same-checkpoint format minimizes storage under a declared
   near-lossless gate? Result: SPZ 8/8 scenes passes all gates at
   5.57x–6.07x compression with <0.02 dB absolute PSNR change.

## Core methodology

- **Rules** (see `05_Protocol/`): any claim must live in exactly one evidence
  class; hardware cohorts are immutable containers for comparison; no absolute
  cross-cohort latency claims; all quality gates are pre-registered; all
  numbers trace to JSON/CSV artifacts in `results/`.
- **Definition of "speedup"**: denominator must be stated (kernel / forward /
  end-to-end / wall-clock). The confirmatory result is *wall-clock* end-to-end
  training time per step.
- **Seed policy**: Reference V1 Room = 42; HiGS protocol = seeds 0,1,2;
  confirmatory = seeds 3,4,5.
- **Evaluation**: official train/test splits; 3 held-out cameras at every
  300 steps for the confirmatory protocol; eval snapshots at
  [500,1k,2k,5k,10k,15k,20k,25k,30k].

## Where the evidence lives (key files)

| What | Path |
|---|---|
| README / project intro | `README.md` |
| Research program (3 tracks) | `docs/research-program.md` |
| HiGS paper plan | `docs/higs-paper-plan.md` |
| Confirmatory protocol | `paper/confirmatory-protocol.md` |
| Baseline lock report | `reports/reference-v1-baseline-lock.md` |
| C42 analysis | `reports/c42_discrepancy_analysis.md`, `reports/c42-adaptive-fix.md`, `reports/c42-*-*.md` |
| C51 investigation | `reports/phase_c51_sparse_backward.md`, `reports/phase-c51-*.md` |
| C49 lifecycle | `reports/phase_c49_gaussian_lifecycle_research.md`, `reports/phase-r2*.md` |
| Tile geometry | `reports/epic05/hardware-aware-tile-study-2026-08-19.md`, `reports/epic05/tile-mechanism-training-study-2026-08-19.md` |
| Raw measurements | `results/` (mirrored into `17_Raw_Data/`) |
| Source of truth for claims | `paper/claims*.json`, `paper/*-claims.json`, `paper/higs/tables/*.json` |

## How this package is organized

See `MANIFEST.md` for a complete file-by-file index. The curated documents are
split into sections `01`–`16` matching the report's likely structure; the
writer should prefer the curated summaries for narrative and the raw files for
exact numbers, and should always follow the `05_Protocol/` rules when making
claims.
