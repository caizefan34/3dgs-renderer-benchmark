# R21 P4 opacity-aware projection candidate

Status: SOURCE_AUDIT_COMPLETE / IMPLEMENTED; full MX P4/P5 NOT_EXECUTED by Codex.
Date: 2026-10-09. Branch: `feature/r21-higs-densification-parity`.

## Provenance and actual formula

The frozen source was retrieved read-only over `ssh mx` from
`/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx`.
`evidence/MX_SOURCE_IDENTITY.json` identifies every retrieved file and binary.
The main binary is exactly
`40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`.
The actual frozen projection file SHA256 is
`ff00a36c40813fbf7e2aa39c1a8c84378334b2e7282a09a4e3114aeae3e04ce0`.
No formula below is inferred from another public gsplat version.

Frozen `ProjectionEWA3DGSFused.cu:175–215` and `Common.h:97–99` implement:

1. Transform mean/covariance; reject depth outside near/far; project covariance;
   add `eps2d` to the 2D covariance and reject nonpositive determinant.
2. With opacity supplied, read `opacities[bid*N+gid]`. Multiply by compensation
   **only if compensations are requested**. Classic R20 uses no compensation.
3. If opacity `< 1/255`, write both radii zero and return.
4. `extend = min(3.33, sqrt(2 * __logf(opacity / (1/255))))`.
5. `rx = ceilf(extend * sqrtf(Sigma2d[0][0]))`, and similarly `ry`.
6. Reject if both radii `<= radius_clip`, or the resulting rectangle lies
   outside the image. At opacity exactly `1/255`, extend is zero, so the
   default radius_clip=0 rejects both zero radii. No extra Python clamp is added.

These are componentwise pixel radii, not a norm or the old largest-eigenvalue
circle. Opacity-aware bounds affect nondifferentiable support/visibility; the
continuous means/conics/depth formula is unchanged for retained Gaussians.

Frozen `_wrapper.py:659` accepts `opacities: Optional[Tensor]` of shape
`[...,N]`, already activated in [0,1]. It forwards that tensor to the registered
`projection_ewa_3dgs_fused` op. R20 trainer calls `model.get_opacity`; actual
`baseline/reference_v1/gaussian_model.py` applies sigmoid to raw logits and
returns `[N]`. This patch does **not** sigmoid again. Master `[N]` becomes
`[1,N]`; compact `[Nvisible]` becomes `[1,Nvisible]`; cameras are `[1,C,4,4]`
and `[1,C,3,3]`. Projection broadcasts opacity across cameras in CUDA through
`bid*N+gid`; raster opacity is separately expanded to `[1,C,Nvisible]`.
Training inputs stay CUDA FP32 and contiguous. Projection radii are int32.
The kernel dispatch also permits floating scalar types, but R21 validates the
existing FP32 training protocol, not mixed precision.

Frozen `rendering.py:597` calls the native `rasterization_3dgs` op.
`Rendering.cpp:1041` passes activated opacity to EWA projection;
`Rendering.cpp:1103` broadcasts projected opacity across cameras. The separate
`rasterization_autograd` implementation in rendering.py is not the frozen
native std_ll training entry point and is not used to establish the formula.

## Actual paths and minimal changes

Only `source/gsplat/experimental/render/functional/gaussian_inference.py` is
overlaid. `R21_OPACITY_FIX.diff` is relative to the exact R20 wrapper whose
SHA256 is `06ec8ed45f2ee38f21bab5e30b66c6b3c7298c85d0254e96cc00ed508879c1ae`;
the same SHA was read on MX's existing R20 candidate.

| Path (R20 line numbers) | Audit and change |
|---|---|
| `_cull_gaussians_batched`, 696/720 | Full master FP32 projection now receives `[1,N]` activated opacity. Union uses `any(camera, all(axis, radii>0))`; ascending master indices are unchanged. |
| `_cull_visible_cached`, 743/774 | Carries opacity into projection. Cache hits require identical live tensor objects, mutation versions and scalar projection settings. Changed opacity, geometry, camera or bounds invalidates it, including reset and count changes. |
| `_HigsAutogradFunction.forward`, 1780/1790 | Both handle and direct paths pass master opacity. The existing common gather gathers opacity with the same master IDs as means/quats/scales/colors. |
| `_native_forward_capture`, 2030 | Compact projection receives gathered `[1,Nvisible]` opacity. Existing AccuTile receives the same opacity expanded over cameras. |
| `_cull_gaussians`, 668 | Unused private single-view helper is also updated to require opacity. No repository call site uses it; its existing shape limitations are outside the training path. |
| Native culling disabled | Full render projection still receives opacity; no master cull is required. |
| `gsplat_recompute` forward/backward, 1928/2465 | Already passes compact activated opacity into native `gsplat.rendering.rasterization`; master cull is now aligned too. Existing R20 refusal of absgrad=True fallback is retained. |
| Frozen/dynamic trainable APIs | Both call the same custom Function. Dynamic count edits still require mark_dirty; existing version/pending-backward guards remain intact. |
| FP16 direct inference / optional preview | `_cull_gaussians_higs` and inference renderer are separate nontraining paths. They do not supply the trainable Function's radii, gradients or statistics; preview cannot override the production result. |

Internal culling helpers now require the opacity keyword to prevent accidental
legacy calls silently losing the contract. Public API and backward signatures
are unchanged. Cache references are weak: no retained autograd graph, no
additional Gaussian-sized tensor, no device-to-host equality check. Inference
mode tensors have no mutation version and therefore always refresh. Direct
`.data` mutations bypass PyTorch version tracking and are not supported cache
updates; normal optimizer/in-place operations and fresh sigmoid outputs are.
New camera views conservatively miss the cache. The former interval>1
approximation no longer reuses changed inputs. Default interval=1 is unchanged.

## AccuTile / SNUGBOX

The actual version is pinned by source SHA, not a guessed release number:
`IntersectTile.cu` SHA256
`b2468d28a329e7e5d7693f76eae1ba93c927865b50c1ce106c591e42843c8c0f`.
Lines 293–379 enter AccuTile when both conics and opacity are provided. It
computes `t=min(3.33^2,2*__logf(opacity/(1/255)))`, uses the full conic including
its cross term to obtain SNUGBOX extents, and executes the same tile routine
in count and emit passes. Positive radii are checked first. Both frozen native
rendering and HiGS capture supply conics and opacity; the AABB-only fallback is
not their normal classic training path. R21 changes neither isect count/emit,
tile masks, sorting, blending nor their thresholds. Tile/pixel parity still
requires the supplied MX numerical test; mathematical support alignment alone
is insufficient evidence of no lost pixel contribution.

## Real evidence and remaining gates

`evidence/KNOWN_MISMATCH_FIXTURES.json` was extracted using CPU-only reads of
DSH's actual iteration-1 states, with equality checks for all raw Gaussian
parameters and stable IDs between backends. It retains original row/stable IDs,
camera, raw parameters, both radii/visibility records, and source SHA256.
Each scene includes 32 mismatches and eight visible controls; it is a regression
subset, not a replacement for full-scene parity.

Bicycle: std 21,930 vs HiGS 23,131 visible (1,201 difference); removing std
opacity makes the sets equal. Room: std 70,217 vs HiGS 74,760 (4,543 difference);
removing std opacity leaves one mismatch. The ablation JSON's boolean
`answer.opacity_reduces_std_visibility=false` contradicts its raw counts;
we do not use that field as evidence. Room's residual one-point difference
must be investigated if it remains after the patch. The user-reported
iteration-600 split counts 331/269 and 68/49 and branch-E restoration are gate
targets; this candidate does not itself prove historical recovery.

## Numerical and performance risks

No CUDA/C++ or R20 True AbsGrad source is changed; signed/raw backward code is
AST-identical. Bounds now use the frozen implementation and its float log,
sqrt/ceil and image/depth inequalities. Boundary rounding can change visibility
discretely; original FP32 atomic accumulation remains nondeterministic.
Opacity pointer reads and log/sqrt are added to the two existing projections.
There is no additional O(N) global intermediate or backward accumulator.
Tighter visibility can reduce gathering/raster work; cache refreshes can
increase projection work for interval>1. Actual performance must be remeasured;
R20's reported 6.3% advantage is not carried forward. Thresholds, normalization,
percent_dense, training state and original/frozen binaries are untouched.
