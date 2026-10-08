# SplatMG SG0 — Future-Displacement Coarse-Space Oracle

Frozen prereg: SG0_PREREG (2026-09-30). Oracle upper-bound study; no CUDA authoring, no 30K, no threshold tuning.

## Environment
- host: dsw-933622-64cc845c7d-lqswc  gpu: NVIDIA A100-SXM4-80GB  torch: 2.9.1+cu128
- renderer arm: C0 (frozen gsplat/HIGS)
- loss: 0.8*L1 + 0.2*(1-SSIM), full resolution, no downsampling

## Verdict
**SG0_BASELINE_UNSTABLE**  (selected fixed config: None)

## Per-cell, per-config metrics

| cell | config | dof_ratio | K4 med/P10 P_hold | K8 med/P10 P_hold | K8 P_des | K8 improve | K8 non-descent |
|---|---|---|---|---|---|---|---|
| bicycle@20000 | L32/SIM13 | 0.0359 | 0.709/0.481 | NA/NA | 0.136 | 8/8 | 8/8 |
| bicycle@20000 | L32/AFF22 | 0.0607 | 1.109/0.710 | NA/NA | 0.175 | 8/8 | 8/8 |
| bicycle@20000 | L128/SIM13 | 0.0090 | 0.605/0.385 | NA/NA | 0.074 | 8/8 | 8/8 |
| bicycle@20000 | L128/AFF22 | 0.0152 | 0.814/0.498 | NA/NA | 0.090 | 8/8 | 8/8 |
| bicycle@25000 | L32/SIM13 | 0.0359 | NA/NA | NA/NA | 0.127 | 8/8 | 8/8 |
| bicycle@25000 | L32/AFF22 | 0.0607 | NA/NA | NA/NA | 0.148 | 8/8 | 8/8 |
| bicycle@25000 | L128/SIM13 | 0.0090 | NA/NA | NA/NA | 0.069 | 8/8 | 8/8 |
| bicycle@25000 | L128/AFF22 | 0.0152 | NA/NA | NA/NA | 0.077 | 8/8 | 8/8 |
| room@20000 | L32/SIM13 | 0.0389 | 0.643/0.566 | 1.021/0.941 | 0.190 | 8/8 | 4/8 |
| room@20000 | L32/AFF22 | 0.0659 | 0.642/0.567 | 1.017/0.939 | 0.208 | 8/8 | 4/8 |
| room@20000 | L128/SIM13 | 0.0097 | 0.513/0.450 | 0.832/0.767 | 0.119 | 8/8 | 4/8 |
| room@20000 | L128/AFF22 | 0.0165 | 0.521/0.457 | 0.839/0.774 | 0.127 | 8/8 | 4/8 |
| room@25000 | L32/SIM13 | 0.0389 | 0.779/0.418 | NA/NA | 0.161 | 8/8 | 8/8 |
| room@25000 | L32/AFF22 | 0.0659 | 0.795/0.426 | NA/NA | 0.170 | 8/8 | 8/8 |
| room@25000 | L128/SIM13 | 0.0097 | 0.634/0.337 | NA/NA | 0.094 | 8/8 | 8/8 |
| room@25000 | L128/AFF22 | 0.0165 | 0.646/0.341 | NA/NA | 0.098 | 8/8 | 8/8 |

## Hierarchy (from theta_0 means only)
- bicycle@20000: L32 n=131072 sizes[27..28]; L128 n=32768 sizes[111..112]
- bicycle@25000: L32 n=131072 sizes[27..28]; L128 n=32768 sizes[111..112]
- room@20000: L32 n=32768 sizes[25..26]; L128 n=8192 sizes[102..103]
- room@25000: L32 n=32768 sizes[25..26]; L128 n=8192 sizes[102..103]

## Controls
- bicycle@20000: identity max_err=3.02e-07 pass=True; random-cluster K8 P_hold median=NA (diagnostic)
- bicycle@25000: identity max_err=3.02e-07 pass=True; random-cluster K8 P_hold median=NA (diagnostic)
- room@20000: identity max_err=2.98e-07 pass=True; random-cluster K8 P_hold median=0.165 (diagnostic)
- room@25000: identity max_err=2.98e-07 pass=True; random-cluster K8 P_hold median=NA (diagnostic)

## Interpretation
More than 2/8 K=8 rollouts are BASE_NON_DESCENT in a cell: the frozen baseline is not a reliable descent, so the oracle study is stopped per prereg.
