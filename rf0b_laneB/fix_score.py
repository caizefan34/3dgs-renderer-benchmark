p = "rf0b_fastgs_model.py"
s = open(p, encoding="utf-8").read()

old = '''def compute_fastgs_scores(model, cam_pool, dataset, render_fn, phot_fn,
                          loss_thresh, n_views, rng, sh_degree):
    """FastGS utils/fast_utils.py:62 compute_gaussian_score_fastgs.

    For each of n_views sampled training cameras: render (frozen C0 renderer),
    build the binary high-error map, and flag each Gaussian whose projected
    center falls on a high-error pixel (accum_metric_counts proxy). Returns
      importance_score [N] int  = #views flagging the Gaussian (FastGS floors
                                  mean count; here count is 0/1 per view so the
                                  mean* n_views == count, threshold semantics
                                  preserved: >5 of 10 views).
      pruning_score   [N] float  = min-max normalized sum_v photometric_loss_v *
                                  flagged_v  (FastGS pruning_score).
    """
    n = model.get_xyz.shape[0]
    counts = torch.zeros(n, device="cuda")
    score_accum = torch.zeros(n, device="cuda")
    pool = list(range(len(cam_pool)))
    chosen = sample_score_cameras(pool, rng, n_views)
    for ci in chosen:
        cam, gt = dataset.get_item(cam_pool[ci])
        with torch.no_grad():
            img, _, _ = render_fn(model, cam, sh_degree)
            img = img.clamp(0, 1)
        gt = gt.to("cuda").clamp(0, 1)
        metric = _ndc_error_map(img, gt, loss_thresh)          # [H,W]
        px, py, valid = _project_to_pixel(model, cam, cam.image_width, cam.image_height)
        phot = phot_fn(img, gt)                                # FastGS loss form
        ix = px.round().long().clamp(0, cam.image_width - 1)
        iy = py.round().long().clamp(0, cam.image_height - 1)
        hit = valid & (metric[iy, ix] > 0.5)
        counts += hit.float()
        score_accum += phot * hit.float()
    importance = counts                                       # #views flagging (0..n_views)
    mn, mx = score_accum.min(), score_accum.max()
    pruning = (score_accum - mn) / (mx - mn) if float(mx - mn) > 1e-12 \\
        else torch.zeros_like(score_accum)
    return importance.long(), pruning.float()'''

new = '''def compute_fastgs_scores(model, cam_pool, dataset, render_fn, phot_fn,
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
    pruning = (score_accum - mn) / (mx - mn) if float(mx - mn) > 1e-12 \\
        else torch.zeros_like(score_accum)
    return importance.long(), pruning.float()'''

assert old in s, "compute_fastgs_scores block not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("FOOTPRINT SCORING FIX APPLIED")