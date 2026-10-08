"""RF0B Lane B: FastGS density-control policy ported onto the frozen C0 model.

This module subclasses the FROZEN C0 GaussianModel
(baseline/reference_v1/gaussian_model.py, sha-pinned in RF0B_SOURCE_LOCK.json)
and adds ONLY the FastGS multi-view-consistent densification/pruning policy
(fastgs/FastGS @ 44e02a5). It does NOT touch the renderer, loss, optimizer,
LR groups, iteration budget, tile size, or holdout split.

Optimizer-state migration (_prune_optimizer / cat_tensors_to_optimizer /
replace_tensor_to_optimizer / densification_postfix / prune_points) is
INHERITED UNCHANGED from the frozen C0 model. Only the POLICY decisions
(which Gaussians clone/split/prune, and the multi-view score bookkeeping)
are overridden.

Source mapping (fastgs/FastGS @ 44e02a5 -> here):
  scene/gaussian_model.py:528 add_densification_stats   -> add_densification_stats (override)
  utils/fast_utils.py:62    compute_gaussian_score_fastgs -> module fn compute_fastgs_scores
  utils/fast_utils.py:15    sampling_cameras            -> module fn sample_score_cameras
  scene/gaussian_model.py:468 densify_and_prune_fastgs  -> densify_and_prune_fastgs
  scene/gaussian_model.py:431 densify_and_split_fastgs  -> densify_and_split_fastgs
  scene/gaussian_model.py:455 densify_and_clone_fastgs  -> densify_and_clone_fastgs
  scene/gaussian_model.py:533 final_prune_fastgs        -> final_prune_fastgs
  scene/gaussian_model.py:279 reset_opacity             -> reset_opacity (inherited, 0.01 cap)
"""
import torch
import torch.nn.functional as F
from gaussian_model import GaussianModel


def sample_score_cameras(train_pool, rng, n=10):
    """FastGS utils/fast_utils.py:15 sampling_cameras (10 distinct random cams).

    FastGS pops from a fresh copy of the train stack (without replacement); we
    mirror that with rng.sample when the pool is large enough, falling back to
    with-replacement only if the pool is smaller than n.
    """
    m = len(train_pool)
    if m >= n:
        positions = rng.sample(range(m), n)
    else:
        positions = [rng.randint(0, m - 1) for _ in range(n)]
    return [train_pool[i] for i in positions]


def _ndc_error_map(render, gt, loss_thresh):
    """FastGS utils/fast_utils.py get_loss + metric_map, on a single view.

    render,gt: [3,H,W] in [0,1]. Returns binary metric_map [H,W] (1 = high error).
    """
    # C0 renderer yields HWC [H,W,3]; reduce over the LAST (channel) axis.
    l1 = torch.abs(render - gt).mean(-1)                          # [H,W]
    l1 = l1 - l1.min()
    denom = (l1.max() - l1.min())
    l1_norm = l1 / denom if float(denom) > 1e-12 else torch.zeros_like(l1)
    return (l1_norm > loss_thresh).float()                         # [H,W]


def _project_to_pixel(model, cam, W, H):
    """Minimum bookkeeping: project world means to pixel coords via the frozen
    camera's world_view_transform + K. Returns (px [N], py [N], valid [N] bool).

    The C0 renderer's densification_info['means2d'] is an adjoint buffer that is
    zero in forward; FastGS's accum_metric_counts is a renderer-side count. We
    reconstruct the equivalent per-Gaussian screen footprint by manual
    projection (documented semantic mismatch #1 in RF0B_POLICY_DIFF.md).
    """
    X = model.get_xyz                                  # [N,3] world
    ones = torch.ones_like(X[:, :1])
    Xh = torch.cat([X, ones], dim=1)                   # [N,4]
    wvt = cam.world_view_transform.to(X.device)        # [4,4] world->view
    Xv = (wvt @ Xh.T).T[:, :3]                         # [N,3] camera coords (z fwd)
    K = cam.K.to(X.device)                             # [3,3]
    P = (K @ Xv.T).T                                   # [N,3]  (K is 3x3: drop homog)
    z = P[:, 2].clamp_min(1e-8)
    px = P[:, 0] / z
    py = P[:, 1] / z
    valid = (Xv[:, 2] > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, valid


def compute_fastgs_scores(model, cam_pool, dataset, render_fn, phot_fn,
                          loss_thresh, n_views, rng, sh_degree):
    """FastGS utils/fast_utils.py:62 compute_gaussian_score_fastgs.

    For each of n_views sampled training cameras: render (frozen C0 renderer),
    build the binary high-error map, and accumulate per-Gaussian counts of
    high-error pixels inside the Gaussian's SCREEN FOOTPRINT (reconstructed from
    the render's per-Gaussian radii via an integral image of the metric map).
    This reconstructs FastGS's renderer-side ``accum_metric_counts`` (which the
    frozen C0 renderer does not expose) closely enough that the policy's
    ``importance_score > 5`` threshold is meaningful. Returns
      importance_score [N] int  = floor( mean_v footprint_count_v )
                                  (FastGS floors the mean count; threshold >5).
      pruning_score   [N] float  = min-max normalized sum_v photometric_loss_v *
                                  footprint_count_v  (FastGS pruning_score).
    """
    n = model.get_xyz.shape[0]
    counts = torch.zeros(n, device="cuda")
    score_accum = torch.zeros(n, device="cuda")
    pool = list(range(len(cam_pool)))
    chosen = sample_score_cameras(pool, rng, n_views)
    for ci in chosen:
        cam, gt = dataset.get_item(cam_pool[ci])
        with torch.no_grad():
            img, radii, _ = render_fn(model, cam, sh_degree)
            img = img.clamp(0, 1)
        gt = gt.to("cuda").clamp(0, 1)
        H, W = cam.image_height, cam.image_width
        metric = _ndc_error_map(img, gt, loss_thresh)          # [H,W] in [0,1]
        # Integral image (summed-area table) of the binary high-error map.
        binary = (metric > 0.5).float()
        integral = torch.zeros((H + 1, W + 1), device="cuda")
        integral[1:, 1:] = binary.cumsum(0).cumsum(1)
        px, py, valid = _project_to_pixel(model, cam, W, H)
        # Per-Gaussian screen footprint radius (radii is [*, N, 2] or [N]).
        r = radii.float()
        if r.dim() == 3:
            r = r[0].max(dim=-1).values                       # [N]
        elif r.dim() == 2:
            r = r.max(dim=-1).values                           # [N]
        r = r.clamp(min=1.0)
        phot = phot_fn(img, gt)                                # FastGS loss form (scalar)
        # Count high-error pixels within each footprint bounding box via the
        # integral image (4 lookups per Gaussian). Invalid -> count 0.
        x0 = (px - r).floor().long().clamp(0, W - 1)
        x1 = (px + r).ceil().long().clamp(0, W - 1)
        y0 = (py - r).floor().long().clamp(0, H - 1)
        y1 = (py + r).ceil().long().clamp(0, H - 1)
        box = (integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1]
               - integral[y1 + 1, x0] + integral[y0, x0])
        footprint = torch.where(valid, box, torch.zeros_like(box))
        counts += footprint.float()
        score_accum += phot * footprint.float()
    # FastGS: importance_score = floor(mean_v count_v); pruning = minmax(sum_v loss*count).
    importance = torch.div(counts, n_views, rounding_mode="floor")
    mn, mx = score_accum.min(), score_accum.max()
    pruning = (score_accum - mn) / (mx - mn) if float(mx - mn) > 1e-12 \
        else torch.zeros_like(score_accum)
    return importance.long(), pruning.float()


# ================================================================ FastGS policy subclass
# FastGS args (arguments/__init__.py:84-103) mirrored as constants.
FGS_GRAD_THRESH = 0.0002      # clone qualifier (signed in FastGS; absgrad on C0)
FGS_GRAD_ABS_THRESH = 0.0012  # split qualifier (abs)
FGS_DENSE = 0.001             # clone/split size boundary (args.dense)
FGS_MIN_OPACITY = 0.005       # prune opacity floor
FGS_FINAL_MIN_OPACITY = 0.1   # final_prune opacity floor
FGS_LOSS_THRESH = 0.1         # metric_map high-error threshold
FGS_METRIC_MASK_MIN = 5       # importance_score > 5
FGS_PRUNE_SCORE_THRESH = 0.9  # final_prune pruning_score > 0.9
FGS_REMOVE_BUDGET_FRAC = 0.5  # budget prune fraction
FGS_OPACITY_CAP = 0.8         # post-densify opacity cap (vs 0.01 in reset_opacity)
FGS_N_SCORE_VIEWS = 10     # n_score_views (train.py compute_gaussian_score_fastgs camlist size)


class FastGSGaussianModel(GaussianModel):
    """Frozen C0 model + FastGS density-control policy only.

    Inherits ALL optimizer-state migration and topology arithmetic from the
    frozen C0 GaussianModel (densification_postfix, prune_points,
    _prune_optimizer, cat_tensors_to_optimizer, replace_tensor_to_optimizer,
    densify_and_clone, densify_and_split). Overrides only:
      - training_setup: add xyz_gradient_accum_abs
      - add_densification_stats: populate dual accumulators
      - densification_postfix / prune_points: also migrate xyz_gradient_accum_abs
      - densify_and_prune_fastgs / _clone_fastgs / _split_fastgs / final_prune_fastgs
    """

    def training_setup(self, config):
        super().training_setup(config)
        self.xyz_gradient_accum_abs = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")

    def add_densification_stats(self, means2d, visibility_filter,
                                width=None, height=None):
        """Populate dual accumulators from the C0 absgrad signal.

        FastGS L528 reads viewspace_point_tensor.grad[:, :2] (signed) for
        xyz_gradient_accum and [:, 2:] (abs) for xyz_gradient_accum_abs. The
        C0 HIGS renderer exposes means2d.absgrad (abs) and a structurally-zero
        signed .grad under H8_MR (verified: signed nonzero == 0). Both
        accumulators therefore read absgrad -> forced mismatch #2.
        """
        grad = getattr(means2d, "absgrad", None)
        if grad is None:
            grad = means2d.grad
        if grad is None:
            return
        if grad.dim() == 3:
            grad = grad[0]
        if width is not None and height is not None:
            grad = grad.clone()
            grad[:, 0] *= width / 2.0
            grad[:, 1] *= height / 2.0
        g2 = grad[visibility_filter, :2]
        gnorm = torch.norm(g2, dim=-1, keepdim=True)
        self.xyz_gradient_accum[visibility_filter] += gnorm
        self.xyz_gradient_accum_abs[visibility_filter] += gnorm
        self.denom[visibility_filter] += 1

    def densification_postfix(self, new_xyz, new_shs, new_opacities,
                              new_scaling, new_rotation, new_tmp_radii=None):
        super().densification_postfix(new_xyz, new_shs, new_opacities,
                                      new_scaling, new_rotation, new_tmp_radii)
        self.xyz_gradient_accum_abs = torch.zeros(
            (self._xyz.shape[0], 1), device="cuda")

    def prune_points(self, mask):
        super().prune_points(mask)
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[~mask]

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        new_xyz = self._xyz[selected_pts_mask]
        new_shs = self._shs[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask] if self.tmp_radii is not None else None
        n_cloned = selected_pts_mask.sum().item()
        self.densification_postfix(new_xyz, new_shs, new_opacities,
                                    new_scaling, new_rotation, new_tmp_radii)
        return n_cloned

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask
        n_split = int(selected_pts_mask.sum())
        if n_split == 0:
            return 0
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self._build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_shs = self._shs[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N) if self.tmp_radii is not None else None
        self.densification_postfix(new_xyz, new_shs, new_opacity,
                                    new_scaling, new_rotation, new_tmp_radii)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * n_split, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return n_split

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent,
                                 radii, importance_score, pruning_score, args):
        """FastGS scene/gaussian_model.py:468 densify_and_prune_fastgs."""
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.norm(grad_vars, dim=-1) >= FGS_GRAD_THRESH
        grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= FGS_GRAD_ABS_THRESH
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= FGS_DENSE * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > FGS_DENSE * extent
        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        metric_mask = importance_score > FGS_METRIC_MASK_MIN
        n_cloned = self.densify_and_clone_fastgs(metric_mask, all_clones)
        n_split = self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        pruned_opacity = int(prune_mask.sum())
        pruned_screen = 0
        pruned_world = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            pruned_screen = int((big_points_vs & ~prune_mask).sum())
            pruned_world = int((big_points_ws & ~(prune_mask | big_points_vs)).sum())
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score
        to_remove = int(torch.sum(prune_mask))
        remove_budget = int(FGS_REMOVE_BUDGET_FRAC * to_remove)
        total_pruned = 0
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32, device="cuda")
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            total_pruned = int(final_prune.sum())
            self.prune_points(final_prune)

        opacities_new = torch.logit(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * FGS_OPACITY_CAP))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self.tmp_radii = None
        torch.cuda.empty_cache()
        return {"cloned": n_cloned, "split": n_split, "pruned_total": total_pruned,
                "pruned_opacity": pruned_opacity, "pruned_screen": pruned_screen,
                "pruned_world": pruned_world, "budget_remove": remove_budget}

    def final_prune_fastgs(self, min_opacity, pruning_score=None):
        """FastGS scene/gaussian_model.py:533 final_prune_fastgs."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        scores_mask = pruning_score > FGS_PRUNE_SCORE_THRESH
        final_prune = torch.logical_or(prune_mask, scores_mask)
        n = int(final_prune.sum())
        if n:
            self.prune_points(final_prune)
        return n
