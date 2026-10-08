# FastGS accepted-mask + 32x64 transpose CPU gate

Date: 2026-09-27

## Decision

**Synthetic 32x64 FP32 CPU gate PASS for one fixed 70-Gaussian case.** It uses 64 distinct pixels, per-pixel accepted masks and early-stop states, a 32/32/6 bucket layout, explicit `[bucket, gaussian-lane, pixel-lane, w/h]` arrays, and primitive-major pairwise FP32 reduction across pixel lanes. Reconstructed color, opacity, mean, conic, and per-contribution absgrad match an independent literal-order per-pixel FastGS reference within the stated tolerance. This supports continuing to a native CUDA correctness/resource/timing gate, but it is not a general numerical proof or GPU parity. Static resources do not establish occupancy, forward mask traffic, barriers, atomics, or end-to-end cost. No production or complete-training speedup, quality, or originality claim follows.

## CPU semantic exercise

Run `python cpu_gate.py`. The script first runs five targeted single-pixel edge cases, then runs an actual software 32x64 microtile model with 64 distinct pixels and 70 ordered Gaussians. Each pixel computes its own FP32 forward state, `n_contrib`, background output, and three 32-bit mask words. The 70-entry sequence has two full buckets and a six-entry tail. The fixture includes early termination, 17 background-only pixels, and an accepted saturated-alpha hit. The stop-triggering sample is excluded from the mask, consistent with the source forward loop.

The per-pixel reference computes native FastGS VJP contributions in source order and reduces them by Gaussian. The candidate separately reads each pixel's accepted mask, visits pixels under each Gaussian, uses cached `w=T*alpha` and `h=G*dL_dalpha` plus RGB adjoint and Gaussian/pixel attributes, then reduces primitive-major across all 64 pixels with a pairwise tree, changing the reduction order. It does not recompute `dL_dalpha` or inspect suffix/T/alpha/G in stage two. It takes `abs` for each mean contribution before the primitive reduction. There are 950 accepted hits and mask popcount is also 950. One pixel reaches early stop; 17 have only background; one accepted hit is saturated. Maximum absolute errors are `9.54e-7` mean (including absgrad), `5.96e-8` conic, `5.96e-8` RGB color, and `1.42e-7` opacity; maximum relative error to each output group's reference peak is below `9.1e-8`. The fixed synthetic run passes the script's `2e-5` absolute tolerance.

The saturated-alpha case confirms the source-specific rule in `PerGaussianRenderCUDA`: alpha is clamped to .99, but the backward uses the unclamped `opacity*G` chain for opacity, mean, and conic derivatives. This is a custom VJP, not the ordinary derivative of the clamp. Background contribution and `T_final` enter `dL_dalpha`. The early-stop sample never enters the accepted mask; the partial case exercises the third mask word with only three valid list entries. `absgrad` takes absolute values per contribution before primitive aggregation.

## Static resource review

Source checked: `public_baselines/fastgs/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu` and `backward.cu`; tile constants are 16x16, RGB, bucket width 32.

- Existing per-bucket forward state is one FP32 transmittance plus three FP32 accumulated-color values per pixel: 16 bytes/pixel. One 32-bit accepted mask adds 4 bytes/pixel, a nominal 25% capacity increase for these boundary records.
- Two FP32 values over a 32-Gaussian x 64-pixel microblock require 32 x 64 x 2 x 4 = 16,384 bytes (16 KiB) of shared memory per block.
- The current per-Gaussian kernel declares about 772 bytes of shared floats (0.754 KiB), though the proposed kernel mapping may change this allocation and occupancy profile.
- The source performs atomics for mean2D x/y and absgrad x/y, conic x/y/w, opacity, and RGB colors. A transpose that changes aggregation ownership must show where all these outputs are reduced without increasing contended writes.

Static bytes do not establish actual allocation, occupancy, memory traffic, or speed. The multi-pixel oracle uses NumPy scalar FP32 operations on a deterministic synthetic scene, not a compiled native kernel or real training view. It does not cover projection/covariance/SH preprocessing parity, and the stable CPU reduction order does not model nondeterministic GPU atomics. The result does not imply complete training speedup, training benefit, quality improvement, or originality. Next gate: compiled native CUDA VJP parity plus measured forward mask writes, occupancy, synchronization/atomic cost, and complete-step timing.

Artifacts: `cpu_gate.py`, `cpu_gate_result.json`.
