# gsplat HiGS trainability analysis

> **STATUS: SUPERSEDED (2026-07-31).** This analysis describes the inference-only state.
> HiGS is now trainable end-to-end with a native CUDA backward (`backward_backend="higs_native"`),
> implemented and benchmarked on EPIC-05 — see [implementation report](higs-trainability-implementation.md)
> and README. gsplat recomputation remains only as an explicit fallback.

Pinned source: gsplat `77ab983ffe43420b2131669cb35776b883ca4c3c`.

## Why the current implementation is inference-only

The limitation is explicit in the implementation rather than an assumption
based on benchmark behavior:

1. [`check_inference_grad_mode`](https://github.com/nerfstudio-project/gsplat/blob/77ab983ffe43420b2131669cb35776b883ca4c3c/gsplat/experimental/render/_common.py#L24-L31)
   rejects ordinary grad-enabled execution and requires `inference_mode()` or
   `no_grad()`.
2. [`GaussianInferenceScene.from_gaussian_tensors`](https://github.com/nerfstudio-project/gsplat/blob/77ab983ffe43420b2131669cb35776b883ca4c3c/gsplat/scene/components/gaussian_inference_scene.py#L495-L512)
   detaches all grad-tracked Gaussian inputs before packing and warns that the
   packed tensors will not participate in autograd.
3. The experimental CUDA registration explicitly installs an Autograd
   [fallthrough](https://github.com/nerfstudio-project/gsplat/blob/77ab983ffe43420b2131669cb35776b883ca4c3c/gsplat/experimental/render/kernels/cuda/ext.cpp#L73-L80)
   because `gaussian_render_inference_only` has no backward kernel.
4. The stateful renderer checks that the number of Gaussians has not changed
   and requires recreation after scene mutation. Densification, pruning, and
   cloning therefore invalidate packed indices, hierarchy offsets, visibility
   masks, sort storage, and persistent workspaces.
5. The packed representation changes numerical contracts: quaternion, scale,
   and opacity data are stored in FP16-oriented layouts, while optional SH
   compression is a lossy scene-time transform. Training needs gradients with
   respect to the original parameterization, not only the packed values.

HiGS gains speed precisely by moving work out of the per-frame differentiable
path and reusing state. Those choices conflict with the parameter and topology
changes made by normal 3DGS training.

## Can it be made trainable?

Yes, but there are three materially different designs.

### 1. Backward recomputation proxy

Use HiGS for the forward value, then recompute a standard differentiable gsplat
render during backward. This is the shortest correctness prototype and needs
no HiGS-native backward kernel. It does not promise faster training: the normal
renderer cost is merely deferred to backward, and forward/gradient numerical
paths are not identical.

### 2. Frozen-topology late-stage HiGS

Stop densification and pruning, retain stable Gaussian identities, and add a
backward kernel for projection, SH evaluation, alpha compositing, and original
Gaussian parameters. Save or recompute compact visible-set and order state.
This is the lowest-risk version that could produce a real iteration-time gain.

### 3. Fully dynamic trainable HiGS

Add versioned packed buffers and asynchronously rebuild the hierarchy after
topology changes. A training step must bind its forward and backward to the
same hierarchy version; old buffers cannot be reclaimed until backward ends.
This can support full training but adds rebuild cost, extra memory, and complex
stream synchronization.

## Recommended implementation order

1. Verify forward parity and gradient parity for the recomputation proxy on a
   small fixed scene.
2. Implement frozen-topology native backward with FP32 master parameters and
   packed FP16 forward buffers.
3. Measure iteration time, peak VRAM, gradient cosine similarity, and final
   PSNR/SSIM/LPIPS against standard gsplat on EPIC-05.
4. Only after the fixed-topology version wins, add lazy/asynchronous rebuilds
   and test densification/pruning schedules.

The repository therefore treats “HiGS can train” as technically feasible but
not yet measured. Disabling densification alone is insufficient: the current
code still detaches inputs and supplies no backward implementation.
