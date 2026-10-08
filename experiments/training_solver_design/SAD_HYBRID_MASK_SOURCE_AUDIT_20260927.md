# SAD-GS hybrid batch visibility mask: source audit

2026-09-27. **Confirmed source-level conditional gradient loss; scene prevalence, quality impact and speed impact are unmeasured.** No training, rendering or CPU experiment was run for this audit. This is an author-code correctness/protocol finding, not an original 3DGS mechanism.

## Frozen source and exact locations

Author commit: `029814fb5ed773e4a794cec5115951a64f610ab8`. Line numbers below were counted from the raw file bytes with PowerShell `Invoke-WebRequest` and newline splitting. Web-tool rendered raw-source line indices can omit blank lines and are not the actual GitHub file line numbers.

- [train.py:235](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/train.py#L235): camera batch loop. Lines 262–263 backpropagate each camera loss without dividing by batch size. Line 270 overwrites `last_radii` on each camera; line 279 retains only that final camera's radii.
- [train.py:429–434](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/train.py#L429): hybrid mode updates the ordinary optimizer, then calls `shoptimizer.step(radii > 0, radii.shape[0])` and clears its gradients.
- [scene/gaussian_model.py:313](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/scene/gaussian_model.py#L313) and [322–324](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/scene/gaussian_model.py#L322): the sparse SH optimizer owns only `_features_rest`; position, DC, opacity, scale and rotation belong to ordinary dense Adam in hybrid mode.
- [SparseGaussianAdam wrapper:254–276](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/submodules/diff-gaussian-rasterization_structgs/diff_gaussian_rasterization_structgs/__init__.py#L254): forwards the supplied visibility mask to `_C.adamUpdate` with the accumulated parameter gradient and Adam moments.
- [cuda_rasterizer/adam.cu:26](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/adam.cu#L26): `if (tiles_touched[g_idx])` encloses the parameter, first-moment and second-moment updates. A false mask leaves all three unchanged, irrespective of a nonzero input gradient.

## Minimal discriminating condition

Consider a normal fixed-topology step with B cameras. Let `v_j[i] = (radii_j[i] > 0)` and let `g_j[i]` be the high-order SH gradient from camera j. The accumulated gradient supplied to sparse Adam is `g[i] = sum_j g_j[i]`, but its update mask is only `v_B`.

For B=2, the minimal witness is:

```text
v_1[i] = true
v_2[i] = false
g_1[i] != 0
g_2[i] = 0
```

Then `g[i] = g_1[i] != 0`, yet parameter/moment updates for row i are skipped and the subsequent `zero_grad(set_to_none=True)` discards that gradient. No claim is made that every first-camera-only row has a nonzero gradient: frustum/radius visibility is weaker than actual accepted contribution.

An exact pre-step discriminator, using the real accumulated gradient, is:

```python
g = gaussians._features_rest.grad
visible_last = last_radii > 0
discarded_rows = (~visible_last) & g.reshape(g.shape[0], -1).ne(0).any(dim=1)
discarded_gradient = g * (~visible_last).reshape(-1, 1, 1)
```

The Boolean support candidate is `(OR_j v_j) & ~v_B`; the actual nonzero-gradient discriminator above is stricter. A future diagnostic may report row count and the squared-gradient-norm fraction `||discarded_gradient||^2 / ||g||^2` when the denominator is nonzero. Neither statistic is a fraction of lost optimizer displacement, lost training time or quality degradation. No empirical value has been measured in this audit.

## Applicability and limits

- The frozen SAD room original-loss 3K and 5K commands both use `batch_size=2` and `optimizer_type=hybrid`; this code path therefore applies. Use ordinary post-topology steps for a minimal check, since densification/pruning can replace parameter rows or gradients. The final training iteration skips optimization separately and is not a mask-loss witness.
- Only higher-order SH has this mask behavior in hybrid mode. The geometry/DC gradients from both cameras still reach dense Adam. It is incorrect to say that the entire first camera's backward pass is wasted.
- For B=1 there is no earlier-camera visibility omitted by the final-camera mask. The `sparse_adam` branch uses the same final radii but is not the active frozen room configuration.
- The summed rather than averaged batch loss is a separate source observation; it is not needed for the mask-loss proof and is not diagnosed here as the cause of image quality differences.
- A batch visibility union is a straightforward correctness candidate, not a novel research method. It admits more sparse optimizer rows and does not by itself reduce raster accepted interactions or complete training cost. Any convergence or three-metric improvement remains to be demonstrated.
- This finding does not establish that the mask explains SAD's broad residual, the isolated held-out veil, or its PSNR deficit against FastGS-Big. No new full training arm follows from the source audit alone. Luna's independent bounded CPU check is separate pending evidence.
