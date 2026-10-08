import io
p = r"C:\Users\36570\3dgs-renderer-benchmark\rf0b_laneB\RF0B_POLICY_DIFF.md"
t = open(p, encoding="utf-8").read()

# --- Fix 1: section 3 row, flagged_v -> footprint_count_v ---
old_row = "`importance_score`) + min-max normalized `sum_v photometric_loss_v * flagged_v` (`pruning_score`)."
new_row = "`importance_score`) + min-max normalized `sum_v photometric_loss_v * footprint_count_v` (`pruning_score`)."
assert old_row in t, "row not found"
t = t.replace(old_row, new_row)

# --- Fix 2: section 4 item #1 ---
old1 = """1. **`means2d` projection proxy (importance/pruning incidence).**
   FastGS obtains `accum_metric_counts` from a modified CUDA rasterizer that,
   with `get_flag=True` + a binary `metric_map`, returns a per-Gaussian count of
   high-error pixels covered. The frozen C0 renderer exposes `means2d` only as an
   adjoint buffer that is **zero in forward** (verified: signed nonzero = 0), and
   does NOT return `accum_metric_counts`. We therefore reconstruct a per-Gaussian
   screen footprint by **manual projection** of world means through the frozen
   camera's `world_view_transform` + `K`, and flag a Gaussian when its projected
   center lands on a high-error pixel (`metric_map[iy,ix] > 0.5`). This is a
   center-point proxy, not a covered-pixel count. Threshold semantics are
   preserved (`importance_score > 5` of 10 views; `pruning_score` min-max
   normalized)."""
new1 = """1. **`accum_metric_counts` screen-footprint reconstruction (renderer-side count).**
   FastGS obtains `accum_metric_counts` from a modified CUDA rasterizer that,
   with `get_flag=True` + a binary `metric_map`, returns a per-Gaussian count of
   high-error **covered** pixels. The frozen C0 renderer does NOT return this
   count and exposes `means2d` only as an adjoint buffer (zero in forward;
   verified signed nonzero = 0). We therefore reconstruct the equivalent
   per-Gaussian count with **minimum bookkeeping**: project world means to pixel
   coords via the frozen camera's `world_view_transform` + `K`
   (`_project_to_pixel`), take the per-Gaussian screen radius `r` from the C0
   render's `radii`, and count high-error pixels inside the footprint with an
   **integral-image (summed-area table) box-sum** over the binary metric map
   (`compute_fastgs_scores`). This reproduces a covered-pixel count (can exceed
   1 per view, exactly like FastGS), so `importance_score > 5` is a
   magnitude-threshold, not a view-majority vote. Residual approximation: we
   sum the axis-aligned bounding box `[px\\u00b1r, py\\u00b1r]` (area `(2r)\\u00b2`)
   rather than the exact elliptical footprint (`\\u2248\\u03c0r\\u00b2`), inflating counts
   by \\u2272 ~27% uniformly; the box is used because the C0 render exposes no
   per-Gaussian footprint mask. See item #3 for the count-unit consequence."""
assert old1 in t, "item1 not found"
t = t.replace(old1, new1)

# --- Fix 3: append pixel->NDC grad-scaling note to item #2 ---
old2tail = "   screen gradient. The split qualifier is unaffected (already abs)."
new2tail = ("   screen gradient. The split qualifier is unaffected (already abs).\n"
            "   The C0 `means2d` gradient is pixel-space; we apply a fixed\n"
            "   `width/2`, `height/2` factor before the norm to normalize to the\n"
            "   NDC convention that FastGS's `grad_thresh`/`grad_abs_thresh` are\n"
            "   calibrated against. This is deterministic bookkeeping (not tuned)\n"
            "   and preserves the gradient direction and cross-Gaussian ordering.")
assert old2tail in t, "item2 tail not found"
t = t.replace(old2tail, new2tail)

# --- Fix 4: section 4 item #3 ---
old3 = """3. **`importance_score` count semantics.**
   FastGS `importance_score = floor(sum_v accum_metric_counts_v / n_views)` where
   `accum_metric_counts_v` is a covered-pixel count (can be >1 per view). Our
   proxy is 0/1 per view (center hit or not), so `importance_score \\u2208 [0,10]` and
   `>5` means "flagged as high-error from a majority of views". The threshold
   value (5) and the majority-of-views intent are preserved; the per-view
   magnitude unit differs (see #1)."""
new3 = """3. **`importance_score` count unit (box vs ellipse).**
   FastGS `importance_score = floor(sum_v accum_metric_counts_v / n_views)` where
   `accum_metric_counts_v` is a true covered-pixel count. Our reconstruction
   (item #1) is also a covered-pixel count but over the footprint bounding box
   rather than the ellipse, so the per-Gaussian magnitude is inflated by a near-
   constant factor (`\\u2248 (2r)\\u00b2 / \\u03c0r\\u00b2 \\u2248 1.27`). Because the factor is
   approximately scale-invariant across Gaussians of similar radius, the
   `importance_score > 5` threshold retains its relative ordering; the absolute
   cutoff is, however, not unit-identical to FastGS. This is a documented
   approximation forced by the frozen renderer (no footprint mask exposed)."""
assert old3 in t, "item3 not found"
t = t.replace(old3, new3)

open(p, "w", encoding="utf-8").write(t)
print("PATCHED OK; new length", len(t))