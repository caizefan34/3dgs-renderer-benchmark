import sys
p = "rf0b_fastgs_model.py"
s = open(p, encoding="utf-8").read()
orig = s

old1 = '''def sample_score_cameras(train_pool, rng, n=10):
    """FastGS utils/fast_utils.py:15 sampling_cameras (10 random cams, with replacement)."""
    idx = [rng.randint(0, len(train_pool) - 1) for _ in range(n)]
    return [train_pool[i] for i in idx]'''
new1 = '''def sample_score_cameras(train_pool, rng, n=10):
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
    return [train_pool[i] for i in positions]'''
assert old1 in s, "old1 not found"
s = s.replace(old1, new1)

old2 = "def compute_fastgs_scores(model, cam_pool, cam_pool_cams, dataset, render_fn,\n                          loss_thresh, n_views, rng, sh_degree):"
new2 = "def compute_fastgs_scores(model, cam_pool, dataset, render_fn, phot_fn,\n                          loss_thresh, n_views, rng, sh_degree):"
assert old2 in s, "old2 not found"
s = s.replace(old2, new2)

old3 = "        phot = float(torch.abs(img - gt).mean())"
new3 = "        phot = phot_fn(img, gt)                                # FastGS loss form"
assert old3 in s, "old3 not found"
s = s.replace(old3, new3)

old4 = "    def add_densification_stats_fastgs(self, means2d, visibility_filter,\n                                       width=None, height=None):"
new4 = "    def add_densification_stats(self, means2d, visibility_filter,\n                                width=None, height=None):"
assert old4 in s, "old4 not found"
s = s.replace(old4, new4)

old5 = '''        final_prune = prune_mask
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32, device="cuda")
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
        total_pruned = int(final_prune.sum())
        if total_pruned:
            self.prune_points(final_prune)'''
new5 = '''        total_pruned = 0
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32, device="cuda")
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            total_pruned = int(final_prune.sum())
            self.prune_points(final_prune)'''
assert old5 in s, "old5 not found"
s = s.replace(old5, new5)

old6 = "  scene/gaussian_model.py:528 add_densification_stats   -> add_densification_stats_fastgs"
new6 = "  scene/gaussian_model.py:528 add_densification_stats   -> add_densification_stats (override)"
assert old6 in s, "old6 not found"
s = s.replace(old6, new6)
old7 = "  scene/gaussian_model.py:279 reset_opacity             -> reset_opacity_fastgs (0.8 cap)"
new7 = "  scene/gaussian_model.py:279 reset_opacity             -> reset_opacity (inherited, 0.01 cap)"
assert old7 in s, "old7 not found"
s = s.replace(old7, new7)

assert s != orig
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("MODEL PATCH APPLIED OK; bytes:", len(s))