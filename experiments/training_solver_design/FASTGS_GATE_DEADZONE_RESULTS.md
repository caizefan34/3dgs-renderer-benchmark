# Native FastGS split gate has a zero-start dead zone

2026-09-27. This falsifies the simple implementation of the proposed topology
homotopy: multiplying native child opacity by a zero-start gate and relying on
the author rasterizer's autograd cannot learn the gate. It does **not** show
that the proposed method improves reconstruction quality or full training time.

The FastGS forward preprocess calls `duplicateToTilesTouched` using the passed
opacity. `auxiliary.h:334-351` derives its compact tile box from
`t = mult * 2 * log(opacity * 255)`. The forward pixel kernel then skips alpha
below `1/255`, and backward repeats that skip while restricting work to the
forward `last_contributor`. At `g=0`, gated child opacity is zero and cannot
produce the smooth activation derivative in the current kernels. This failure
starts at tile construction, before the ordinary pixel alpha cutoff.

If a native child's ungated center alpha is `a=min(0.99,opacity)`, the first gate value at which
`1-(1-a)^g` reaches the author pixel cutoff is

```text
g_min(a) = log(1-1/255) / log(1-a).
```

This is a **lower bound** on the onset at an off-center pixel, whose ungated
alpha is no larger. If `a < 1/255`, the child never reaches that cutoff even
at `g=1`; the author renderer itself would not use it at its center. The
formula only describes the pixel cutoff and does not model compact-box
rounding, clipping, sorting or early termination.

We loaded the unchanged author's `room_big_4099.pth` checkpoint and ran one
1600x1066 training step through the native room@4100 densification event.
The diagnostic wrapper called the author's score once, reproduced the actual
split mask before `densify_and_split_fastgs`, and observed `get_opacity`.
FastGS copies that parent opacity directly to each native split child.
Source commit, checkpoint SHA256 and exact counts are in
`fastgs_gate_deadzone_results.json`. The wrapper suppressed only final model
save; the first attempt stopped before training because PyTorch defaulted to
safe-only checkpoint loading. The successful isolated process loaded this
known project checkpoint with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

| Population | Count | Median center `g_min` | `g_min > .01` | `g_min > .05` | `g_min > .1` |
|---|---:|---:|---:|---:|---:|
| All current Gaussians | 651,547 | .00685 among cutoff-reachable | 324,521 | 177,322 | 139,694 |
| Native selected split parents | 9,345 | .00257 | 860 (9.20%) | 40 (0.43%) | 3 (0.032%) |

Of the all-Gaussian population, 93,267 (14.31%) are below the native center
cutoff even at full opacity. None of the 9,345 selected split parents is in
that group. The selected set is more opaque than the general population, so
the all-Gaussian percentages must not be used as split-candidate statistics.
The separate 4100 PLY audit found the same count and nearly identical center
quantiles; it is a cross-check, not another independent event. Repeating the
native event with the pixel cap `0.99` explicitly applied yielded identical
selected counts and reported quantiles.

The earlier scalar oracle's negative zero-gate derivative is the derivative
of a **smooth, fixed-support extension** of alpha composition. The native
hard-cut FastGS **child** contribution is locally zero with respect to `g` at
birth. The parent still fades, so the native total derivative contains the
parent term but misses the proposed child's smooth derivative. A
shadow candidate-support pass would deliberately use a different derivative
while keeping both endpoint images and the photometric loss. It needs an
explicit definition of support, tile placement, early-stop handling and VJP;
it is not an automatic property of the author rasterizer. Nonzero gate
initialization is a cheaper ablation, but sacrifices exact zero-birth and
cannot recover off-center dead zones uniformly.

For a candidate layer `k` in a *fixed* depth-ordered traversal, let `T_k`
be incoming transmittance and `S_{k+1}` the RGB obtained by compositing all
later accepted layers over the background with unit incoming transmittance.
Then its pixel-color sensitivity is

```text
dI/dg = sum_k T_k (c_k - S_{k+1}) (d alpha_k/dg).
```

At birth, `d alpha_parent/dg=(1-a_parent)log(1-a_parent)` and
`d alpha_child/dg=-log(1-a_child)`. The event-only shadow traversal can in
principle compute this VJP using the ordinary full-resolution image loss's
pixel adjoint. A new 2,500-case FP64 RGB reference uses ungated-alpha support,
the author's exclusive `1e-4` early-stop rule and arbitrary depth ordering:
271 cases exited early; 1,682 placed at least one child after its parent.
Both endpoint RGB errors were zero and the largest independent analytic-vs-
autograd shadow VJP error was `3.55e-15`. This validates only the scalar
fixed-support compositor, not the FastGS GPU kernel, all floating-point
branches, or its work. Source/output: `topology_shadow_vjp_oracle.py` and
`topology_shadow_vjp_oracle.json`.

Next gate: before modifying a full trainer, implement a small real-camera
region with ungated child support, the author's physical depth order and
cutoff/early-stop semantics. Check endpoint images and per-candidate gate VJP
against a differentiable reference, then measure candidate tile/pixel work.
If that cost cannot be repaid by rejected families, stop this hypothesis.
