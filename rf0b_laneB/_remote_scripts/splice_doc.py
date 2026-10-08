# -*- coding: utf-8 -*-
import io
p = "RF0B_POLICY_DIFF.md"
t = io.open(p, encoding="utf-8").read()

t = t.replace("photometric_loss_v * flagged_v", "photometric_loss_v * footprint_count_v")

h4 = t.find("## 4. Every unavoidable semantic mismatch")
h5 = t.find("## 5. Exact source line / function mapping")
assert h4 >= 0 and h5 >= 0 and h5 > h4, (h4, h5)

PM = u"\u00b1"   # +/-
SQ = u"\u00b2"   # superscript 2
PI = u"\u03c0"   # pi
APP = u"\u2248"  # approx
LE = u"\u2264"   # <=
ARR = u"\u2192"  # right arrow

new_s4 = (
"## 4. Every unavoidable semantic mismatch\n"
"\n"
"These are forced by holding the C0 renderer/loss fixed while porting a policy\n"
"that was designed against the original 3DGS rasterizer + graphdeco camera. Each\n"
"is documented; none are tunable post-hoc.\n"
"\n"
"1. **`accum_metric_counts` screen-footprint reconstruction (renderer-side count).**\n"
"   FastGS obtains `accum_metric_counts` from a modified CUDA rasterizer that,\n"
"   with `get_flag=True` + a binary `metric_map`, returns a per-Gaussian count of\n"
"   high-error **covered** pixels. The frozen C0 renderer does NOT return this\n"
"   count and exposes `means2d` only as an adjoint buffer (zero in forward;\n"
"   verified signed nonzero = 0). We therefore reconstruct the equivalent\n"
"   per-Gaussian count with **minimum bookkeeping**: project world means to pixel\n"
"   coords via the frozen camera's `world_view_transform` + `K`\n"
"   (`_project_to_pixel`), take the per-Gaussian screen radius `r` from the C0\n"
"   render's `radii`, and count high-error pixels inside the footprint with an\n"
"   **integral-image (summed-area table) box-sum** over the binary metric map\n"
"   (`compute_fastgs_scores`). This reproduces a covered-pixel count (can exceed\n"
"   1 per view, exactly like FastGS), so `importance_score > 5` is a\n"
"   magnitude-threshold, not a view-majority vote. Residual approximation: we\n"
"   sum the axis-aligned bounding box `[px" + PM + "r, py" + PM + "r]` (area `(2r)" + SQ + ")`)\n"
"   rather than the exact elliptical footprint (`" + APP + PI + "r" + SQ + "`), inflating counts\n"
"   by " + LE + " ~27% uniformly; the box is used because the C0 render exposes no\n"
"   per-Gaussian footprint mask. See item #3 for the count-unit consequence.\n"
"\n"
"2. **Signed" + ARR + "absgrad for the clone qualifier (+ pixel" + ARR + "NDC grad scaling).**\n"
"   FastGS `add_densification_stats` reads `viewspace_point_tensor.grad[:, :2]`\n"
"   (signed, pixel-space) for `xyz_gradient_accum` (clone qualifier,\n"
"   `grad_thresh=0.0002`) and `[:, 2:]` (abs) for `xyz_gradient_accum_abs` (split\n"
"   qualifier, `grad_abs_thresh=0.0012`). Under C0 H8_MR the signed `.grad` is\n"
"   structurally zero (verified), so BOTH accumulators read `means2d.absgrad`.\n"
"   Consequence: the clone qualifier threshold (0.0002) is applied to an absgrad\n"
"   signal. This is forced " + u"\u2014" + " the C0 renderer does not expose a contracted\n"
"   signed screen gradient. The split qualifier is unaffected (already abs).\n"
"   The C0 `means2d` gradient is pixel-space; we apply a fixed `width/2`,\n"
"   `height/2` factor before the norm to normalize to the NDC convention that\n"
"   FastGS's `grad_thresh`/`grad_abs_thresh` are calibrated against. This is\n"
"   deterministic bookkeeping (not tuned) and preserves the gradient direction\n"
"   and cross-Gaussian ordering.\n"
"\n"
"3. **`importance_score` count unit (box vs ellipse).**\n"
"   FastGS `importance_score = floor(sum_v accum_metric_counts_v / n_views)` where\n"
"   `accum_metric_counts_v` is a true covered-pixel count. Our reconstruction\n"
"   (item #1) is also a covered-pixel count but over the footprint bounding box\n"
"   rather than the ellipse, so the per-Gaussian magnitude is inflated by a\n"
"   near-constant factor (`" + APP + " (2r)" + SQ + " / " + PI + "r" + SQ + " " + APP + " 1.27`). Because the\n"
"   factor is approximately scale-invariant across Gaussians of similar radius,\n"
"   the `importance_score > 5` threshold retains its relative ordering; the\n"
"   absolute cutoff is, however, not unit-identical to FastGS. This is a\n"
"   documented approximation forced by the frozen renderer (no footprint mask\n"
"   exposed).\n"
"\n"
"4. **`photometric_loss` weight.**\n"
"   FastGS `compute_photometric_loss` = `(1-0.2)*L1 + 0.2*(1-fused_ssim)` per\n"
"   view. We use the identical functional form with the frozen SepSSIM (the\n"
"   `--loss sep` default). Only the SSIM backend differs from FastGS's fused_ssim;\n"
"   the loss *form* and weight (0.2) match.\n"
"\n"
"5. **`sampling_cameras` RNG.**\n"
"   FastGS uses the global `random` module (process-seeded). We use a dedicated\n"
"   `random.Random(seed*1000003 + iteration)` for reproducibility and to avoid\n"
"   perturbing the frozen camera-sequence RNG. Distribution (10 distinct train\n"
"   cams, without replacement) is identical.\n"
"\n"
"6. **Score-camera render cost (algorithmic, not semantic).**\n"
"   FastGS renders 10 extra views per densification event and per final-prune\n"
"   event. This is intrinsic to FastGS's multi-view-consistent policy and is\n"
"   retained (it is part of the algorithm's cost, not a backend substitution). It\n"
"   is included in wall-clock timing.\n"
"\n"
"---\n"
"\n"
)

t = t[:h4] + new_s4 + t[h5:]
io.open(p, "w", encoding="utf-8").write(t)
print("SPLICED OK; length", len(t))