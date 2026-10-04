# BES0 Complementary — Certified Quality-Speed Headroom of Per-Pixel Early-Stop

**Date:** 2026-10-05  **Author:** DSH (autonomous, MX A100 idle window)
**Companion to:** `reports/bes0_realscene_opportunity.md` (per-occurrence falsification)

## Why this
The per-occurrence falsification left one candidate: per-pixel transmittance
early-stop with a *quality-aware* threshold. This measures its certified
headroom on the same 3 scenes x 12 dev cameras.

## Method
For each strided pixel, gather covering gaussians (== scene_adapter), sort
front-to-back, and for a sweep of transmittance floors eps compute the
fraction of (pixel,gaussian) occurrences skippable once cumulative
transmittance T < eps. Certified bound: truncating a pixel at T < eps changes
its radiance by at most eps in [0,1], i.e. an error ceiling eps (PSNR ceiling
~-20 log10 eps dB). Opportunity screen (analytic), not a CUDA timing.

## Results — skipped fraction sf(eps)
| scene | eps=1/255 (48dB) | 0.01 (40dB) | 0.05 (26dB) | 0.10 (20dB) | 0.15 (16.5dB) |
|-------|-----------------:|------------:|------------:|------------:|--------------:|
| room   | 0.0084 | 0.0084 | 0.0087 | 0.0093 | 0.0106 |
| bicycle| 0.0084 | 0.0095 | 0.0145 | 0.0233 | 0.0324 |
| garden | 0.0194 | 0.0197 | 0.0207 | 0.0221 | 0.0241 |

## Finding
The early-stop headroom is **~1-2% and strongly quality-insensitive**: relaxing
the transmittance floor from 1/255 (48 dB certified ceiling) all the way to
0.15 (16.5 dB — a catastrophic quality floor) raises the skippable fraction
only from ~1% to ~1-3.2%. Quality-awareness does **not** unlock headroom,
because per-pixel transmittance rarely saturates on these scenes: the cost is
dominated by thin, near-zero-alpha gaussians (long-axis bbox overlap) that
neither reduce T nor can be skipped by an early-stop test.

## Implication
Combined with the per-occurrence result (rho_R ~ 1e-9), **bounded-event
render-cost reduction is a dead end on real 3DGS scenes in every formulation**:
- per-occurrence bounded-alpha routing: structurally vacuous (sub-pixel minor
  axes -> per-tile alpha lower bound = 0);
- per-pixel transmittance early-stop (= gsplat's existing mechanism): ~1-2%
  quality-insensitive headroom.

The only certified lever is gsplat's fixed-threshold early-stop itself; there
is no meaningful additional headroom to capture by re-formulating "bounded
events".

## Reproduce
- `bes_opp_probe/headroom_sweep.py`, chain `bes_opp_probe/chain_headroom.sh`.
- Raw: `bes_opp_probe/out/headroom_{room,bicycle,garden}.json`,
  `bes_opp_probe/out/headroom.log`.
- Run: `bash chain_headroom.sh 7` (GPU 7).
