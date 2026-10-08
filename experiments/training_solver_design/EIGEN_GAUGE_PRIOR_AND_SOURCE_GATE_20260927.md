# Eigenvector coordinate selection: prior-art and source gate

2026-09-27. **Rejected before experiments.** A mathematical optimization opportunity exists at strictly isotropic primitives, but the inspected trainers guarantee it only at initialization, not as a persistent source of dominant training cost. No CPU/GPU experiment or training arm was run.

## Mathematical opportunity and limits

For `Sigma = sigma^2 I`, changing the rotation leaves the covariance and current rendered image unchanged, and the covariance derivative with respect to rotation vanishes. Scale changes in a fixed rotation frame initially span only its diagonal covariance directions. Choosing the eigenvectors of the symmetric original-loss covariance gradient as the rotation frame makes its full gradient direction representable by subsequent scale changes.

This is a statement about reachable directions, not a claim that native Adam becomes a full covariance-gradient update: Adam's coordinatewise normalization changes the actual step. Nor does a better first-order training-loss direction establish PSNR/SSIM/LPIPS noninferiority. Extending the operation to merely near-isotropic primitives changes the represented scene and requires a different argument. Nonzero optimizer history also requires treatment beyond a free coordinate change.

## Frozen source gate

FastGS author commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, verified against the local `public_baselines/fastgs` checkout:

- [scene/gaussian_model.py:178](https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/scene/gaussian_model.py#L178) repeats one scale in three axes at initialization.
- Lines 443–444 divide all parent scales by the same split factor and copy rotation; lines 462–463 clone parent scales/rotation. These preserve parent axis ratios rather than guaranteeing new isotropic primitives.
- Lines 395–402 append zero first/second moments but reuse the stored optimizer state, including its group-level step. Newborn rows do not have fully fresh optimizer history.

SAD author commit `029814fb5ed773e4a794cec5115951a64f610ab8`; actual raw-file line numbers were counted directly, not taken from web-tool display indices:

- [scene/gaussian_model.py:272–274](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/scene/gaussian_model.py#L272) uses isotropic initialization.
- [train.py:82–150](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/train.py#L82) optionally adds `6 * 16^2 = 1536` isotropic bounding-box-face rows before the training loop. This is another initialization component, not recurring births.
- `scene/gaussian_model.py:722–723` divides each scale by its axis-specific `ks**scale_power` and copies rotation for structural splitting. Lines 880–881 implement ordinary uniform split scaling; lines 903–904 and 941–942 clone parent scales/rotation. None guarantees recurring strict isotropy.
- Lines 578–582 append zero Adam/AMSGrad moments while retaining other optimizer state.

Source inspection does not determine how many later primitives happen to be exactly isotropic, or for how long. Therefore the supported conclusion is **only initialization is guaranteed**, not that later opportunities are impossible or have been empirically counted.

## Bounded prior-art check

Searches covered covariance parameterization, isotropic rotation-gradient degeneracy, gradient-aligned eigenvectors, and Gaussian gauge transformations. [GeoSplat](https://arxiv.org/html/2509.05075v1) already addresses geometry-guided covariance initialization and constrained optimization; [ReorgGS](https://arxiv.org/html/2605.08739v1) addresses distribution reorganization. Searches also surfaced Cholesky-based Gaussian parameterizations. No direct match to this exact render-preserving eigenframe selection was established in the bounded check. **A search miss is not an originality proof.** Generic covariance reparameterization or direct positive-definite matrix optimization cannot be claimed as new.

## Decision

The operation does not reduce current accepted Gaussian–pixel interactions and adds gradient collection/eigendecomposition work. Its only proposed speed path is a better initialization reducing later iterations; no evidence connects that path to SAD's late broad-scale residual or material total work. The source gate supplies no guaranteed recurring opportunity, while local one-step improvement would not decide complete-training benefit. Do not launch a local descent probe, CUDA implementation or full training arm for this candidate on the present evidence.
