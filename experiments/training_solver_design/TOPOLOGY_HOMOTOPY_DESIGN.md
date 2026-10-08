# Full-resolution topology homotopy: research hypothesis, not a validated method

**2026-09-28 status correction:** `FASTGS_VECTOR_GATE_RESULTS.md` has already optimized all 9,345 continuous family gates for 12 Adam updates on ten full-resolution training views **with geometry/appearance frozen**. It reduced training loss but regressed all three 39-view room development metrics against leaving the event unsplit (PSNR −0.002344 dB, SSIM −0.000165, LPIPS +0.000470). Thus "learnable continuous gates" alone are screened and do not justify native fusion. What remains untested is joint nonzero-gate child geometry/appearance adaptation during probation, including a frozen exploration rule, endpoint settlement, all construction/renderer costs and long-horizon quality. The prior result supplies no positive reason to fund that larger implementation now; the conditional steps below are not an active experiment queue.

2026-09-27. The desired end state is shorter **complete** training wall time and
higher quality on untouched novel views than current strong 3DGS, with every
optimization step using the same effective full-resolution image and the real
photometric objective. The parallel FastGS camera loader is a measured systems
component, not the original reconstruction contribution sought here. This
document specifies a separate candidate that must pass correctness, cost,
quality, and originality gates before it can be promoted.

## Mechanism

FastGS commits a discrete split immediately, replacing one parent splat with
native child splats. A split can create a large one-step image change even when
the extra degrees of freedom might help after training. Prior local finite-edit
rankings changed over longer horizons, so immediate error alone is not a
reliable accept/reject rule. Make the topology transition a temporary
trainable scalar, not a permanent grouped rendering primitive.

For a selected parent `p`, initialize the author's native child geometry,
opacity and SH parameters. Let `a_p(x)` and `a_j(x)` be the normal clamped
per-pixel alpha responses and let `g` be a family activation in `[0,1]`.
During a short probation interval, use physically depth-sorted splats with
optical-depth gates:

```text
A_p(x,g) = 1 - (1-a_p(x))^(1-g)
A_j(x,g) = 1 - (1-a_j(x))^g,  j=1,2
```

At `g=0`, the children are transparent and the old scene is rendered; at
`g=1`, the parent is transparent and the native children are rendered. These
endpoint statements hold even when unrelated splats lie between the parent
and children in physical depth order. Special-case both endpoints in the GPU
implementation so their float results match the author renderer exactly.
Intermediate images are a new model, not an exact Gaussian split or a convex
blend of finished images. A clone keeps its parent at full alpha and gates
only the added clone. An inexpensive linear alpha gate is a possible systems
control; it also has exact endpoints, but its intermediate gradient differs
and must be treated as a separate ablation.

The activation starts at zero and may increase only after full-resolution
training views support it. A candidate is accepted by removing its parent
after `g` reaches one; a candidate is rejected by removing the children after
probation. The first prototype should use the author's child initialization
and optimizer-state rules, with one gate per split family, to isolate the
transition effect. An eventual policy may price the expected future
Gaussian-tile work of an accepted family, but adding such a penalty before the
basic transition succeeds would confound the result. Keep camera selection,
full-resolution loss and final evaluation split explicit; official test views
must never update the gate.

## A non-optional raster correctness problem

At `g=0`, a standard opacity cutoff or tile cull omits transparent
children, making their gate gradient vanish. The proposed *smooth fixed-support
extension* has a one-sided derivative containing `-log(1-a_j)` from each child;
the native hard-cut rasterizer instead has a zero derivative in the dead zone.
The parent's smooth derivative is
`(1-a_p)log(1-a_p)`. Omitting child terms can reverse the decision. The
`topology_homotopy_oracle.py` FP64 compositing check covered 10,000 random
five-layer orderings: maximum endpoint image errors were 1.11e-16 at birth
and 2.22e-16 at release. In a constructed one-pixel case with alpha cutoff
`1/255`, the smooth-extension loss derivative with respect to `g` was `-0.04367347`
(activation helps), while culling zero-alpha children yielded `+0.01386294`
(false rejection). This is a scalar oracle, not production CUDA parity.
The rerun also retained children when their *ungated* alpha exceeded the
cutoff: it recovered `-0.04367347` and matched the full scalar gradient.
This is only a sufficient support rule in the one-pixel example. It does not
validate tile assignment, transmittance early termination, or the CUDA VJP.

Source inspection of author FastGS makes the problem sharper: tile bounding
uses `t = mult*2*log(opacity*255)` in `auxiliary.h`, before the forward pixel
cutoff or backward cutoff. A zero-gated child cannot simply be inserted as an
ordinary Gaussian and expected to receive a gate gradient. At the real
room@4100 native split event, 9,345 selected parents had median center-pixel
gate onset `.00257` after the native pixel alpha cap `.99`; 860 still fail the
center cutoff at `g=.01`. Every
selected child has a strictly positive onset, so exact zero-start learning
requires a shadow support/VJP path or a different transient operator. These
are one-event, center-bound observations, not a measured renderer overhead or
quality gain; see `FASTGS_GATE_DEADZONE_RESULTS.md`.

A GPU implementation must keep candidate-child support in a gate-gradient
traversal even at zero alpha, while preserving the original image and all
ordinary parameter gradients at `g=0`. It must honor FastGS's alpha clamp,
tile assignment, physical depth sort, per-splat cutoff and transmittance early
termination. At `g=1`, removing the zero-alpha parent must reproduce the
native post-split renderer. Verify RGB, alpha and VJP against the author path
at both endpoints on real 1600x1066 cameras before training. A special
zero-gate backward pass may be cheaper than evaluating logarithms at every
regular interaction; this is an implementation question, not a free saving.

The event-only shadow VJP has a compact pixel formula. For an ordered active
layer `k`, incoming transmittance `T_k` and the suffix color `S_{k+1}` behind
it, `dI/dg=sum_k T_k(c_k-S_{k+1})(d alpha_k/dg)`. A 2,500-case FP64
fixed-support RGB oracle with native-style exclusive early stop and arbitrary
interleaving matched independent autograd VJP to `3.55e-15`; see
`topology_shadow_vjp_oracle.json`. This proves the formula only under the
reference semantics. A FastGS implementation still needs an ungated-opacity
tile list and correct visibility/early-stop support, and must measure all
extra tile/pixel traffic against later saved work.

Children have no parameter gradient exactly at `g=0`, so the first gate
update depends on their **initialized** finite-edit direction. Earlier
project experiments found immediate split-score rankings unstable over 64
and 512 updates. A gate that rejects every initially harmful edit could miss
later useful capacity. Probation length, internal multi-view evidence and
rollback must be measured, not chosen from the official test set. Repeated
densification, opacity reset and pruning need transactional settlement of any
active families; keep the first probation shorter than FastGS's 100-step
event interval to simplify a pilot, then audit all event collisions.

## Why this might move both objectives, and why it might fail

The **hypothesis** is that delaying commitment allows original full-resolution
losses to reject harmful splits, while retaining useful child freedom without
a sudden representational jump. Rejected children do not impose all future
raster, SH or optimizer cost, which could shorten training wall and reduce
overfitted floaters. The full-resolution image is still rendered every step.
None of those downstream benefits is measured. The temporary parent, shadow
child traversal, power-alpha arithmetic, gate optimizer, rollback and later
changed point count all cost time. If the gate rejects useful long-horizon
splits, or saved future pixel work does not repay probation, the method fails.
The existing FastGS loader trial's 248.50 s room result becomes a stronger
systems comparator; a topology gain may not be attributed to data loading.

## Prior-art boundary from source/abstract checks

- [Revising Densification](https://arxiv.org/abs/2404.06109) already derives
  `1-sqrt(1-a)` for opacity correction during cloning. The square-root and
  optical-depth identity are not original here.
- [Improving Densification](https://arxiv.org/abs/2508.12313) already has
  Long-Axis Split, Recovery-Aware Pruning and Growth Control; its “Multi-step
  Update” means parameter updates every `N>1` iterations after densification,
  not this gate. Similarity of goals still requires full-text/source review.
- [CLoD-GS](https://arxiv.org/abs/2510.09997) has learnable opacity decay for
  continuous level of detail, and
  [SplitAvatar](https://arxiv.org/abs/2605.25751) uses soft masks to control
  avatar density. “Learnable opacity gate” in general is prior art.
- FastGS already uses multi-view error scoring; Taming 3DGS budgets density;
  3DGS-MCMC relocates Gaussians. This candidate cannot claim those ideas.
- A previous project prototype made paired newborn splats jointly conserve
  transmittance but retained parent depth order after movement; real views
  showed severe interleaving. This proposal instead uses actual physical
  sorting throughout and is only endpoint-preserving, not motion-equivalent.
  It has not been implemented or compared with that prototype.
- A separate first-pass arXiv search found direct prior art for spatially
  varying color/opacity inside Gaussians:
  [Textured-GS](https://arxiv.org/abs/2407.09733),
  [SVGS](https://arxiv.org/abs/2411.18966),
  [Textured Gaussians](https://arxiv.org/abs/2411.18625), and
  [LiteTex-GS](https://arxiv.org/abs/2609.23380). Adding local texture is not
  a safe novelty pivot.

These checks do **not** establish priority for the specific homotopy scheme.
They are an early rejection screen, not a systematic literature review.

## Minimum falsification sequence

1. On a pinned FastGS checkpoint with real optimizer state, choose fixed
   split parents and cameras before seeing results. Implement only the gate
   reference on a small real image region. Check both endpoint RGB/alpha and
   parent/child/gate derivatives under cutoff, early exit and arbitrary depth
   interleaving. Stop if gate-gradient support is not correct or affordable.
2. Measure temporary touched tile/pixel pairs, per-interaction arithmetic,
   gate/rollback wall time and expected future cost for candidate families.
   Use the complete 30K wall boundary; a scalar oracle is no speed evidence.
3. Compare native split, unchanged replay, fixed ramp, learned gate and
   gate-without-cost-prior on identical child initialization and camera order.
   Include short and 512-step horizons, internal training-view holdouts,
   several seeds, and both room and an untouched scene. If long-horizon gains
   are below replay variation or the gate rejects beneficial edits, stop.
4. Only after those gates, integrate repeated events and run full 30K on
   untouched scenes against author FastGS-Big, Faster-GS, and available newer
   methods. Count complete wall, all event and loader costs, points, PSNR,
   SSIM, LPIPS and per-view failures. Demonstrating equal quality or a faster
   local branch alone is insufficient for the user's joint objective.

Current status (2026-09-27): one selected native-rule family has an isolated
real-camera CUDA fixed-support gate diagnostic. Its full 1600×1066 endpoint
RGB matches native pre/post renders within 1.79e-7; three selected pixels'
gate VJPs match an independent FP64 reference within 6.09e-15. A separate
full-image pass costs ~1.95 ms, before constructing support or doing any
training. See `FASTGS_SHADOW_GATE_CUDA_RESULTS.md`. The whole-event family
gradient, true overhead, long-horizon training benefit, and originality are
still unverified. Do not infer a joint speed/quality gain from endpoint parity.

Later cost/quality audit: `FASTGS_EVENT_GATE_RESULTS.md` extends this to all
9,345 event families and ten training views. The separate all-family VJP pass
costs 3.7–4.4 ms on one 1600×1066 camera; 64.6% of tile pixel slots are
affected. An event-only score is cheaper than per-step probation but highly
view-dependent. A conservative 1,831-family veto showed only tiny room
64/512-step development-view gains with no proven complete-wall saving;
the scoring time was omitted from branch runs. Persistent trainable gates
remain untested, and this one-shot veto is a different policy, not a validated
version of the homotopy proposal.

A second child-seed replication of the one-shot veto kept all three 512-step
room metric means weakly favorable, but the deltas shrank and their descriptive
paired-view bootstrap intervals all crossed zero. Its process wall moved in
the opposite direction, with score cost still excluded. The one-shot veto is
deprioritized as a strong candidate; this does not validate or falsify a
future fused trainable gate, whose full probation/rollback cost remains open.
