# R21 source audit and data flow

Status: LOCAL_SOURCE_AUDIT_COMPLETE; exact MX runtime source export and causal
evidence PENDING. Date: 2026-10-09. Branch:
`feature/r21-higs-densification-parity`. Base commit:
`387a7b31f35fa0b9ab1d160fa12181e1a60c77c2`.

This is a diagnostic-only change. R18 binaries, R20 validated candidate,
reference GaussianModel and all original checkpoints remain untouched.
The new R21 objective reports MX R20 True AbsGrad parity and 5K failures.
Those numbers are user-provided evidence; this checkout does not contain their
raw MX return package. No R21 GPU experiment has been run by this deliverable.

## Actual source map

Paths below are relative to the repository root.

| Component | Actual inspected source | Finding |
|---|---|---|
| Training recipe/render adapters | `experiments/r20_true_absgrad/R20_TRAINER.py:165` | Same raw activations/loss, eps2d=.1, black background, C=1; std packed=False; HiGS dynamic topology and culling enabled |
| Training phase order | same file:333–395 | LR/SH, camera, render, loss/backward, stats, topology, opacity reset, Adam, zero_grad |
| Native projection call | `HIGS_SOURCE_HANDOFF_ACTUAL/source/gsplat/cuda/csrc/Rendering.cpp:1041` | Dense EWA projection explicitly receives `opacities` |
| Projection Python contract | `HIGS_SOURCE_HANDOFF_ACTUAL/source/gsplat/cuda/_wrapper.py:659` | opacities optional; documented tighter bounds when provided; int32 per-axis radii in pixels |
| HiGS culling | `experiments/r20_true_absgrad/source/gsplat/experimental/render/functional/gaussian_inference.py:695` | `fully_fused_projection` over all master rows; omits opacities; union positive radii over cameras; ascending master row gather |
| HiGS render projection | same file:2030 | Projects gathered FP32 raw-activated subset; also omits opacities; captures exact forward projection for backward |
| HiGS master mapping | same file:1941,2380 | Scatter radii, signed and absolute gradients back by master-visible row; all invisible zero |
| True AbsGrad | `.../kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu:270,541` | Per-pixel component fabs before PX/warp sum; independent FP32 atomic accumulation; original signed VJP retained |
| Native AbsGrad | `HIGS_SOURCE_HANDOFF_ACTUAL/source/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu:233` | Same componentwise per-pixel definition; FP32 accumulation order can differ |
| Stats | `baseline/reference_v1/gaussian_model.py:261` | AbsGrad preferred, camera slice 0, W/2 and H/2 scaling, norm, visibility-gated accumulator and denom |
| Clone/split | same file:462,490 | Grad threshold >= .0008, activated scale partition .01×extent; no radii input to either selection |
| Pruning | same file:568 | Clone, split, remove split parents, then opacity/screen/world pruning |
| Optimizer migration | same file:297,322,395 | Persistent Adam; append zero moment rows; survivor moments indexed with surviving parameters |
| Native IDs | same file:64,422,457 | Monotone local IDs; survivors retained, children assigned fresh IDs; IDs alone do not establish cross-run parent lineage |
| Camera protocol | `src/benchmark_framework/cameras.py:80`; `scripts/epic05/phase7/dataset.py:44,117` | 3DGS JSON camera-to-world inverse, +Z forward, K resized to 1920×1080 preserving FoV; matching GT image names |

Source hashes are in `R21_SOURCE_IDENTITY.json`. At runtime the collector copies
the actual renderer sources into `source_evidence/`, records source and binary
hashes, and retains R20's frozen-main/candidate-identity checks.

## Evidence limits on projection

The frozen export has `Rendering.cpp`, `_wrapper.py` and real HiGS CUDA/wrapper
sources. It lacks `gsplat/rendering.py`,
`gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`, and common projection headers.
Consequently the exact frozen CUDA radius cutoff/rounding formula and Python
dispatch into the exported C++ path still require the runtime export.

The local public_baselines checkout contains an opacity-aware radius formula,
but its identity is not the frozen MX identity. It is NOT substituted as proof
of the frozen implementation. The native C++ call supplying opacity versus
HiGS omitting it is a real source-level contract difference and a testable
H1/H2 lead, not a confirmed explanation of iteration 600.

Python projection calls are observed without recomputation. A native C++ render
can bypass that Python observer; its actual projection outputs are captured
from renderer metadata. An empty `projection_calls` list in std is not proof
that projection did not run. Missing required metadata stops capture explicitly.

## Mathematical flow and tensor contract

N is the current master Gaussian count, V the gathered visible subset,
C=1 for this actual trainer, K=(active_sh_degree+1)^2. All renderer floating
inputs and statistics here are FP32 unless a snapshot records otherwise.

1. Raw parameters: xyz `[N,3]` (world units), scaling `[N,3]` (log world scale),
   quaternion `[N,4]` (raw), opacity `[N]` (logit), SH `[N,16,3]` (coefficient).
   Activations are exp, normalize and sigmoid; SH takes the active leading K.
   Raw gradients have identical shapes and include activation chain rules.
2. World-to-camera uses float32 `[4,4]` viewmatrix and `[3,3]` K. The ordinary
   pinhole projection has pixel means `(fx*x/z+cx, fy*y/z+cy)`; covariance is
   propagated through camera/projection Jacobians, with eps2d=.1 added on screen.
   Actual clipping/clamped Jacobian details must be read from exported MX CUDA.
3. Projection contract: means `[C,N,2]` pixels, depth `[C,N]` camera z,
   conics `[C,N,3]` inverse screen covariance (pixel^-2), radii `[C,N,2]` int32
   screen bounding half-extents. Positive radii mark valid projection. Invalid
   means/conics/depth slots may be unwritten; canonical comparisons zero only
   those invalid slots and retain the raw capture as evidence.
4. HiGS culls/gathers master rows to V and projects again; it scatters radius
   and gradient outputs back to master N. The proxy `means2d` is a gradient
   carrier, NOT actual projected coordinates. The collector takes actual means
   from forward projection telemetry, never from proxy values.
5. For loss L and per-pixel screen derivative g_ip, native absolute gradient is
   `a_i=(sum_p abs(g_ip.x), sum_p abs(g_ip.y))`, shape `[C,N,2]`, FP32,
   units loss/pixel. Signed sum is separately propagated to raw parameters.
   There is no added loss averaging or extra camera factor in R21.
6. Trainer consumes `v_i = any(radii_i > 0)` and
   `h_i = norm((W/2*a_ix, H/2*a_iy))`. Multiplication converts derivatives to
   normalized screen coordinates; the existing model comment reverses that
   verbal direction. R21 preserves the actual arithmetic.
7. For every step before densify_until_iter: `A_i += v_i*h_i`,
   `D_i += v_i`, `M_i=max(M_i,max(radii_i))` on visible rows. A/D are `[N,1]`,
   M `[N]`. A visible Gaussian with zero absolute gradient still increments D.
   Therefore equal per-step absolute gradients alone do not prove equal grow
   scores if visibility histories differ.
8. At iteration t divisible by 100, with 500<t<15000:
   `g_i=A_i/D_i`; NaN is replaced with zero (other nonfinite values are not
   silently repaired). Clone iff `norm(g_i)>=.0008` and
   `max(exp(scale_i))<=.01*extent`; split iff `g_i>=.0008` and world scale is
   greater. The initial row gradients are cached before clone resets stats;
   zero padding excludes fresh clones from the split decision.
9. Clone copies parents exactly. Split emits two child blocks using
   `R(q)*normal(0,s)+xyz`, child scale `s/1.6`; then removes split parents.
   Postfix resets A,D,M for all surviving/appended rows. Even an empty clone
   invokes postfix. The final prune mask is evaluated on the post-split rows.
   Opacity threshold .005 always applies; screen/world checks apply only when
   t>3000. In this actual model M was reset by clone before screen pruning,
   which is a shared reference behavior to record, not an R21 unilateral fix.
10. Adam follows topology. Replacing Parameters during postfix/prune drops the
    old current-step `.grad`; survivor moments are retained and new moments
    zeroed. This existing ordering is preserved and measured. Subsequent steps
    accumulate gradients for the new Parameters normally.

The image is clamped to [0,1]; loss is .8×mean L1 + .2×separable DSSIM. Thus
screen derivatives already contain the actual image loss scaling. Neither the
threshold nor image resolution, loss, camera sequence, culling, dtype or SH
progression is changed by instrumentation.

## Identity, synchronization and lifetime

Raw parameters/stats are master-row aligned. Native integer IDs are preserved
for local state reconstruction. Diagnostic stable IDs use initial IDs, then
SHA256-derived negative int64 IDs keyed by `(parent stable ID, iteration,
clone/split, child ordinal)`. Split ordinal follows `.repeat(N,...)` block
ordering. All IDs are checked for uniqueness, including birth collisions.
Comparison and substitutions align by stable IDs and reject unequal sets for
interventions. After topology divergence, unmatched descendants are reported,
not paired by coincident native integer IDs.

All observations detach before CPU copy/hash; no gradient-producing autograd
operation or hook is added. Copies synchronize streams and hashing causes
substantial host overhead. CPU snapshots own their storage. The original
trainer event/synchronize boundaries remain, but diagnostic timing is invalid
for performance claims. CUDA atomic ordering can remain nondeterministic;
repeat A/A and B/B runs and distinguish bit differences from significant error.

The original capture omits optimizer/RNG/IDs; restore calls training_setup
after loading A/D, resetting them, and regenerates IDs. R21 uses a separate
full-state contract and directly restores Parameters, Adam groups/moments,
statistics, counters, active SH and all RNG states without invoking that path.

## Requirement findings

- Radii describe screen extents on both interfaces, but exact opacity-aware
  support/visibility equivalence is unproved and runtime comparisons are required.
- `radii > 0` affects statistics every step; C=1 camera slicing is matched.
- `max_radii2D` does not enter clone/split, and screen pruning is disabled at 600.
- A/D equality is not guaranteed by isolated fixed-state AbsGrad parity.
- Scale partition crossings and accumulated raw-parameter drift are independent
  candidate mechanisms; both are reported with threshold margins.
- Optimizer operations appear common to both arms; actual pre/post moment and
  lineage equality must be tested. No H1–H6 root cause is marked confirmed.
