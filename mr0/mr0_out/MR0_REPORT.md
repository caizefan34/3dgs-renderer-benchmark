# MacroRep-GS MR0 - Direct Macro-Primitive Capacity Upper Bound

Generated: 2026-09-30T13:30:35Z

## Verdict

**MR0_DROP**  (selected_K=None, strong_K=[], promote_K=[])

## Question

Can many fine C0 Gaussians be replaced by far fewer directly-rendered,
view-independent macro Gaussians (one mean/covariance/opacity/C0 SH) while
preserving multi-view radiance/transmittance and collective first-order
response, removing BOTH raster work and N-scaling optimizer work?
This is an optimistic capacity oracle, not a training method.

## Protocol summary

- Cells: room/bicycle at 10K and 15K iterations (mipnerf 360).
- 512 intersection-weighted deterministic greedy-NN fine groups per cell,
  K in {4,8,16} nested; construction cams are TRAIN cams disjoint from eval.
- Fit ONE macro Gaussian per group per K on 8 per-group highest-visibility
  TRAIN cams; objective = RGB contribution err + alpha/transmittance err +
  12-perturbation first-order collective response err (forward FD).
- Greedy full-scene replacement at 25/50/75% fine-Gaussian coverage; eval on
  8 train-eval + 8 true-holdout cams; PSNR/SSIM delta, intersection-count
  reduction, primitive/variable reduction, response capture, HM0-predicted
  iteration speed.

## Proxies and disclosures

- n_groups: 512
- K_values: ['4', '8', '16']
- coverage_levels: [0.25, 0.5, 0.75]
- n_fit_steps: 6
- n_fit_cameras: 8
- fit_cam_selection: per-group highest precomputed visibility TRAIN cams (prereg sec.4)
- n_vis_cams_construction: 32
- n_isect_reduction_sample_groups: 128
- n_isect_reduction_cams: 4
- intersection_reduction_units: tile-gaussian intersection pairs (a0 capture patch); group and macro both tile-based
- response_perturbations: 12
- response_estimator: forward finite differences in fit; exact autograd for the target
- note: no local GPU; remote A100 only; no full training; optimization allowed to be expensive per prereg
- Intersection-count reduction is measured on a 128-highest-weight group
  sample over 4 construction cams; both group and macro counts are
  tile-gaussian intersection pairs (a0 capture patch).

## Results at 75% coverage

| cell | K | train dPSNR | holdout dPSNR med | holdout dPSNR worst | SSIM d | resp cos med | resp cos p10 | isect red | var red | speedup | nan/inf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| room@10000 | 4 | -0.029 | -0.032 | -0.177 | -0.0010 | 0.770 | 0.308 | 0.959 | 0.750 | 2.05x | 0 |
| room@10000 | 8 | -0.092 | -0.106 | -0.435 | -0.0025 | 0.825 | 0.590 | 0.949 | 0.875 | 2.47x | 0 |
| room@10000 | 16 | -0.247 | -0.356 | -1.293 | -0.0070 | 0.867 | 0.669 | 0.964 | 0.938 | 2.76x | 0 |
| room@15000 | 4 | -0.029 | -0.085 | -0.241 | -0.0016 | 0.746 | 0.187 | 0.940 | 0.750 | 2.04x | 0 |
| room@15000 | 8 | -0.118 | -0.157 | -0.513 | -0.0029 | 0.819 | 0.402 | 0.936 | 0.875 | 2.46x | 0 |
| room@15000 | 16 | -0.353 | -0.461 | -1.307 | -0.0059 | 0.847 | 0.611 | 0.933 | 0.938 | 2.74x | 0 |
| bicycle@10000 | 4 | -0.006 | -0.019 | -0.040 | -0.0000 | 0.490 | -0.109 | 0.978 | 0.750 | 2.14x | 0 |
| bicycle@10000 | 8 | -0.007 | -0.029 | -0.062 | -0.0002 | 0.658 | 0.000 | 0.978 | 0.875 | 2.63x | 0 |
| bicycle@10000 | 16 | -0.015 | -0.067 | -0.107 | -0.0005 | 0.713 | 0.015 | 0.979 | 0.938 | 2.97x | 0 |
| bicycle@15000 | 4 | -0.002 | -0.012 | -0.060 | +0.0000 | 0.435 | -0.041 | 0.973 | 0.750 | 2.15x | 0 |
| bicycle@15000 | 8 | -0.006 | -0.030 | -0.113 | -0.0001 | 0.649 | 0.073 | 0.977 | 0.875 | 2.66x | 0 |
| bicycle@15000 | 16 | -0.013 | -0.052 | -0.219 | -0.0004 | 0.735 | 0.023 | 0.981 | 0.938 | 3.02x | 0 |

## Gate (prereg sec.6, single K across BOTH scenes at BOTH 10K/15K)

MR0_STRONG_PROMOTE requires, at 75% coverage: train/holdout PSNR med >= -0.08 dB,
holdout worst >= -0.20 dB, SSIM med >= -0.002, response cos med >= 0.90, p10 >= 0.80,
intersection reduction >= 0.65, variable reduction >= 0.65, predicted speed >= 3.0x, no NaN/Inf.
MR0_PROMOTE relaxes these to -0.15/-0.35/-0.004/0.82/0.68/0.55/0.55/3.0.
A single K must pass on BOTH scenes at BOTH iterations.

Predicted iteration speed at K=16, 75% coverage: room@10000=2.76x, room@15000=2.74x, bicycle@10000=2.97x, bicycle@15000=3.02x

room fails the 3.0x speedup gate at every K<=16 (room@10000: K4=2.05x, K8=2.47x, K16=2.76x), so no single K passes both scenes -> MR0_DROP.

## Interpretation

MR0_DROP: a directly rendered macro Gaussian does not give enough local
capacity (at the tested K<=16) to remove both raster and N-scaling optimizer
work while holding quality. This does not rule out macro/anchor representations
in general; it rules out this direct single-macro-per-group upper bound at
these capacities. PROMOTE would only have justified a new training-representation study.

