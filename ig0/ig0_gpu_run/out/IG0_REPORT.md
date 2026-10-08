# InfoGeo-GS IG0 GPU Report (EPIC-05 A100)

## STATUS
```
IG0_DROP
```
selected_fixed_variant = None ; strong=[] ; promote=[]
ig0_analyze.py owns verdict; manual override = false.

## EXECUTION
```
gpu_execution_host: dsw-933622-64cc845c7d-lqswc
GPU: NVIDIA A100-SXM4-80GB
CUDA_VISIBLE_DEVICES: None
torch: 2.9.1+cu128 ; CUDA: 12.8 ; python: 3.10.21
remote/non-local confirmation: Executed on EPIC-05 A100 (8.130.30.251:1024) via SSH; local RTX GPU NOT used.
```

## IDENTITY
```
C0 extension sha256: d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11
core gsplat_cuda.so sha256: 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
fused_ssim_cuda.so sha256: dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb
prereg_literal_valid=False authoritative_live_hash_match=True
trainer: publication_trainer_p12_n1snap.py (arm=c0, seed=42, loss=fused)
checkpoints (iter/arm/N/has_optimizer_state):
  room@10000: iter=10000 arm=c0 N=710652 opt=True
  room@15000: iter=15000 arm=c0 N=841781 opt=True
  bicycle@10000: iter=10000 arm=c0 N=2960907 opt=True
  bicycle@15000: iter=15000 arm=c0 N=3654566 opt=True
```

## CORRECTNESS
CPU reference (EPIC-05, fp64): Fisher PSD / closed-form-vs-differential / quaternion Jacobian / FR-GLOBAL & FR-LOCAL radius parity / NaN — 6/6 PASS.
GPU radius parity (real native Adam displacement, per probe): worst FR-GLOBAL rel = 3.41e-16; worst FR-LOCAL median = 1.12e-16; worst FR-LOCAL p99 = 3.13e-16 (tolerances 1e-5 / 1e-5 / 1e-4). NaN/Inf = 0 across all 32 probes.

## PRIMARY RESULTS (4 cells x 2 variants)
| cell | variant | valid | R_src_median | R_src_p10 | peer_win | holdout_win | holdout_PSNR_med | holdout_PSNR_worst | correct |
|---|---|---|---|---|---|---|---|---|---|
| room@10000 | FR-GLOBAL | 8/8 | 0.632 | 0.581 | 0.000 | 0.000 | -0.047 | -0.062 | True |
| room@10000 | FR-LOCAL | 8/8 | 0.847 | 0.815 | 0.000 | 0.000 | -0.039 | -0.058 | True |
| room@15000 | FR-GLOBAL | 8/8 | 0.659 | 0.607 | 0.125 | 0.125 | -0.025 | -0.038 | True |
| room@15000 | FR-LOCAL | 8/8 | 0.865 | 0.819 | 0.125 | 0.125 | -0.020 | -0.036 | True |
| bicycle@10000 | FR-GLOBAL | 8/8 | 0.413 | 0.349 | 0.125 | 0.125 | -0.009 | -0.012 | True |
| bicycle@10000 | FR-LOCAL | 8/8 | 0.838 | 0.787 | 0.125 | 0.250 | -0.007 | -0.009 | True |
| bicycle@15000 | FR-GLOBAL | 8/8 | 0.494 | 0.362 | 0.125 | 0.750 | 0.001 | -0.003 | True |
| bicycle@15000 | FR-LOCAL | 8/8 | 0.810 | 0.773 | 0.125 | 0.750 | 0.001 | -0.002 | True |

PROMOTE thresholds (per cell): valid>=6, R_src_median>=1.15, R_src_p10>=0.80, peer_win>=0.55, holdout_win>=0.55, holdout_PSNR_med>=-0.02, worst>=-0.10.
STRONG thresholds: valid>=7, R_src_median>=1.30, R_src_p10>=1.00, peer_win>=0.65, holdout_win>=0.60, PSNR_med>=0.00, worst>=-0.05.
No cell meets even PROMOTE for either variant (R_src_median 0.41-0.87 < 1.15; peer_win <= 0.125 < 0.55; room holdout_win = 0.0).

## INTERPRETATION
OBSERVED: With geometry direction from the intrinsic Fisher-Rao metric, magnitude-matched to native Adam's Fisher trust radius, the one-step source loss descent is 0.41x-0.87x of Adam's (R_src_median), and cross-view (peer) and true-holdout mean losses are not lower than Adam's (win rate 0.0-0.75, mostly <=0.125).
EVIDENCE: 32 source probes (4 cells x 8), valid source descent 8/8 per cell, correctness PASS, NaN/Inf=0; frozen analyzer verdict IG0_DROP.
CONCLUSION: The Fisher-Rao matched-radius geometry direction does not improve over native Adam for standard photometric 3DGS geometry updates; it is worse on source and does not help cross-view/holdout. InfoGeo-GS IG0 CLOSED (no rescue per prereg section 12/19).

## ARTIFACTS
remote: /root/ig0_run/out/ ; local mirror: C:\Users\36570\3dgs-renderer-benchmark\ig0\ig0_gpu_run\ ; manifest: MANIFEST.sha256 (self-entry excluded).
knowledge note: D:\学习\3DGS渲染器对比\笔记 - InfoGeo-GS IG0 Fisher-Rao几何方向Oracle 2026-09-30.md
