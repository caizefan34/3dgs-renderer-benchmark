# FastGS scalar-adjoint component gate (frozen before timing)

Date 2026-09-27. This is an exact-backward implementation screen, **not** the original 3DGS method or full-goal validation. The full user target still requires complete training wall, full effective resolution/original loss each step, multi-scene/multi-seed formal unseen-view PSNR/SSIM/LPIPS noninferiority, and same-protocol strong baselines. Engineering speed alone is insufficient.

## Hypothesis and code scope

At fixed raster forward state, `q = dot(ar, dL_dpixel) - T_final*dot(bg,dL_dpixel)` replaces three `ar` lane-shuffled floats and one `T_final` lane-shuffled float in the native bucket raster backward with one scalar. Each accepted interaction first adds `alpha*T*dot(c,dL_dpixel)` to `q`, then uses `dL_dalpha=T*dot(c,dL_dpixel)+q/(1-alpha)`. The native forward, gates, loss, full-resolution input, all six parameter groups, color gradients, absolute screen gradient and topology policy are unchanged. This is algebraic VJP state compression; it does not by itself establish algorithmic originality.

Only the isolated `fastgs_scalar_adjoint_trial/cuda_rasterizer/backward.cu` copy is edited. Source FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`; author original backward.cu SHA256 `851041054cc0dedd7a2b049fae6e10f2a486d1df5ac2978b4df85c1f552bc90e`, isolated modified file SHA256 `a449abb19b6bd2452cfee1bb68c31e55582b5bfe58ed4d998493b4f9d5c3e64e`. The remote isolated wheel SHA256 is `32290dbbb50fb508a53d958ff9ed4e68b292f75925f3402fb0277c746ec36998`; it imports from its own target path, with the baseline installed package unchanged.

## Gate A: correctness

At four fixed room@4100 training views, full effective width 1600 and original `0.8 L1 + 0.2(1-fusedSSIM)`, require exact forward RGB parity (max abs ≤1e-4), and for each of six parameter groups and screen-space gradient require relative L2 ≤1e-3 **or** max abs ≤1e-7. No held-out/test views are used. Passing this stage only authorizes the bounded timing stage. A later broader stage would need room/garden early/late states, boundary and early-stop coverage before training use.

## Gate B: material whole-block opportunity

Use the same four frozen room training views, native full image/loss/backward, 16 warmup steps, then five 128-step blocks per binary. Hold model and camera cycle fixed; clear grads each step. Run three alternating process pairs on idle A100 GPU3 in order baseline→candidate, candidate→baseline, baseline→candidate. Compare medians of five block wall times within each process, and CUDA-event block times as a second signal. Promote to two-scene/both-stage and complete-training work only if the paired candidate/native wall ratio median is ≤0.95 and all three pairs are faster; otherwise stop this scalar-adjoint version. This gate measures frozen-state full backward blocks, **not** complete training speed or image quality. No threshold adjustment after seeing results.
