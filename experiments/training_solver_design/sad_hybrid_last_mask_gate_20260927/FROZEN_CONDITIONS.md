# Frozen discriminator: SAD-GS hybrid SH last-camera mask

Date: 2026-09-27. This is a source-guided, CPU-only logic discriminator. It does not edit or import the author package and does not launch CUDA or training.

## Pinned source

- Author clone: `/root/codex_sadgs_baseline_20260927`
- Git HEAD: `029814fb5ed773e4a794cec5115951a64f610ab8`
- Git worktree was clean at inspection.
- `train.py` SHA256: `c7f51d40f2dac224b42c2750f3c5bd249cce26df853facc498f987dc2c12047b`
- `scene/gaussian_model.py` SHA256: `1d224980637b2056572eea9053e50ddf48a696d157a05abb63269aa62d8994cc`
- `submodules/diff-gaussian-rasterization_structgs/diff_gaussian_rasterization_structgs/__init__.py` SHA256: `bd6d0d177d5ec2dcde56cce94fcb967850f75a3486d5f6f938e342cfc2a42657`
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/adam.cu` SHA256: `e200a015002328f81462eccec0db826bf94d424ce2359135a0432eba147e771b`

## Source path being tested

1. `train.py:235-263` loops over `opt.batch_size`, calls `loss.backward()` for each camera, and does not divide each loss by batch size (`loss = rgb_loss`, with division commented out).
2. `train.py:269-280` overwrites `last_radii` on every camera and, after the loop, assigns `radii = last_radii`.
3. For `optimizer_type == "hybrid"`, `scene/gaussian_model.py:322-324` puts ordinary parameters in dense Adam but puts only `_features_rest` in `SparseGaussianAdam`.
4. `train.py:429-434` constructs `visible = radii > 0` from the final camera, runs sparse SH Adam with that mask, then calls `shoptimizer.zero_grad(set_to_none=True)`.
5. `SparseGaussianAdam.step` (`__init__.py:253-276`) passes its visibility tensor to `_C.adamUpdate`. `cuda_rasterizer/adam.cu:21-33` updates a Gaussian's SH coefficient elements only inside `if (tiles_touched[g_idx])`.

## Minimal frozen condition

Use `batch_size=2` and two Gaussians, each with a small vector of SH coefficients. Gaussian 0 is visible in camera 1 and receives nonzero camera-1 SH gradient; it is invisible in camera 2 and receives zero camera-2 SH gradient. Gaussian 1 is visible in both cameras as a positive control. Accumulate both cameras' gradients, use camera 2's visibility as `last_radii > 0`, execute the source-equivalent sparse per-Gaussian update predicate, then model `zero_grad(set_to_none=True)`. Compare with a counterfactual union-of-camera visibility mask. No dataset frequencies or runtime speed are inferred.
