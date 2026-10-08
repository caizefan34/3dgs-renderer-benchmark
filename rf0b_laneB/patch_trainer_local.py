import sys
p = "rf0b_trainer.py"
s = open(p, encoding="utf-8").read()
orig = s

# --- Edit A: import block already imports compute_fastgs_scores, FGS_LOSS_THRESH,
#     FGS_N_SCORE_VIEWS. Add sample_score_cameras? Not needed (model handles it).
#     Add phot_fn import? We define phot_fn locally. No change needed here.

# --- Edit B: absgrad guard must include c0fgs (line ~624)
oldB = '        if args.arm in ("c0", "a0", "a1", "a2") and iteration <= config.densify_until_iter:\n            # Loud guard: under H8_MR the signed .grad fallback is moment-space'
newB = '        if args.arm in ("c0", "c0fgs", "a0", "a1", "a2") and iteration <= config.densify_until_iter:\n            # Loud guard: under H8_MR the signed .grad fallback is moment-space'
assert oldB in s, "B not found"
s = s.replace(oldB, newB)

# --- Edit C: add_densification_stats dispatch already routes non-b0 to the 4-arg call.
#     c0fgs uses the override (same 4-arg signature). No change needed. BUT the
#     pre-event grad-norm record block reads model.xyz_gradient_accum/denom which
#     for c0fgs is populated by the override. OK.

# --- Edit D: replace densify_and_prune call with c0fgs branch
oldD = '''            current_radii = radii.float().max(dim=-1).values
            ev = model.densify_and_prune(
                max_grad=thresh_eff,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii)
            if args.arm in ("c0", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed'''
newD = '''            current_radii = radii.float().max(dim=-1).values
            if args.arm == "c0fgs":
                # FastGS multi-view-consistent densification (policy only).
                # Score bookkeeping: render n_score_views random TRAIN cameras,
                # build per-Gaussian high-error incidence (importance_score) and
                # min-max normalized photometric*incidence (pruning_score).
                train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
                score_rng = random.Random(config.seed * 1000003 + iteration)
                importance_score, pruning_score = compute_fastgs_scores(
                    model, train_pool, dataset, render_with_meta, phot_fn,
                    FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                    model.active_sh_degree)
                ev = model.densify_and_prune_fastgs(
                    size_threshold, config.min_opacity, scene_extent,
                    current_radii, importance_score, pruning_score, None)
                _HIGS_DYNAMIC_SCENE.mark_dirty()
            else:
                ev = model.densify_and_prune(
                    max_grad=thresh_eff,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=size_threshold,
                    radii=current_radii)
                if args.arm in ("c0", "a0", "a1", "a2"):
                    _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed'''
assert oldD in s, "D not found"
s = s.replace(oldD, newD)

# --- Edit E: final_prune_fastgs block before pev[4].record() (after opacity reset)
oldE = '''        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            total_resets += 1
            if args.arm in ("c0", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value

        pev[4].record()'''
newE = '''        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            total_resets += 1
            if args.arm in ("c0", "c0fgs", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value

        if args.arm == "c0fgs" and iteration % 3000 == 0 and iteration > 15000 and iteration < 30000:
            # FastGS final_prune_fastgs (train.py:151-158): every 3k after 15k
            # before 30k, prune low-opacity + low multi-view-consistency Gaussians.
            train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
            score_rng = random.Random(config.seed * 1000003 + iteration + 777)
            _, pruning_score = compute_fastgs_scores(
                model, train_pool, dataset, render_with_meta, phot_fn,
                FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                model.active_sh_degree)
            n_fp = model.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
            total_final_prunes += n_fp
            if n_fp:
                _HIGS_DYNAMIC_SCENE.mark_dirty()

        pev[4].record()'''
assert oldE in s, "E not found"
s = s.replace(oldE, newE)

# --- Edit F: counter init
oldF = "    total_clones = total_splits = total_prunes = total_resets = 0\n"
newF = "    total_clones = total_splits = total_prunes = total_resets = 0\n    total_final_prunes = 0\n"
assert oldF in s, "F not found"
s = s.replace(oldF, newF)

# --- Edit G: results dict + training_results.json
oldG1 = '''        "total_clones": total_clones, "total_splits": total_splits,
        "total_prunes": total_prunes, "total_opacity_resets": total_resets,
        "peak_vram_gb": peak_vram,'''
newG1 = '''        "total_clones": total_clones, "total_splits": total_splits,
        "total_prunes": total_prunes, "total_opacity_resets": total_resets,
        "total_final_prunes": total_final_prunes,
        "peak_vram_gb": peak_vram,'''
assert oldG1 in s, "G1 not found"
s = s.replace(oldG1, newG1)

oldG2 = '''            "total_clones": total_clones, "total_splits": total_splits,
            "total_prunes": total_prunes, "total_opacity_resets": total_resets,
        }, f, indent=2)'''
newG2 = '''            "total_clones": total_clones, "total_splits": total_splits,
            "total_prunes": total_prunes, "total_opacity_resets": total_resets,
            "total_final_prunes": total_final_prunes,
        }, f, indent=2)'''
assert oldG2 in s, "G2 not found"
s = s.replace(oldG2, newG2)

# --- Edit H: renderer dict must have c0fgs key (else KeyError at results build)
oldH = '''            "c0": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2},
            "a0":'''
newH = '''            "c0": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2},
            "c0fgs": {"eps2d": eps2d_eff, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2,
                    "policy": "fastgs_density_control_only"},
            "a0":'''
assert oldH in s, "H not found"
s = s.replace(oldH, newH)

# --- Edit I: define phot_fn right after ssim_fn = SepSSIM (line ~461)
oldI = "    ssim_fn = SepSSIM(device=\"cuda\")\n"
newI = '''    ssim_fn = SepSSIM(device="cuda")

    # FastGS photometric-loss form for score weighting (train.py:compute_photometric_loss).
    # L = (1-0.2)*L1 + 0.2*(1-SSIM); used ONLY as the per-view weight in the FastGS
    # pruning_score. The training loss above is unchanged.
    def phot_fn(pred_hwc, target_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = target_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim_val = ssim_fn(b, g)
        return (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * (1.0 - ssim_val)
'''
assert oldI in s, "I not found"
s = s.replace(oldI, newI, 1)

assert s != orig
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("TRAINER PATCH APPLIED OK; bytes:", len(s))