src = open("rf0b_trainer.py").read()
orig = src

# --- A) absgrad guard: include c0fgs ---
src = src.replace(
    'if args.arm in ("c0", "a0", "a1", "a2") and iteration <= config.densify_until_iter:',
    'if args.arm in ("c0", "c0fgs", "a0", "a1", "a2") and iteration <= config.densify_until_iter:')

# --- B) mark_dirty after densify: include c0fgs ---
src = src.replace(
    '            if args.arm in ("c0", "a0", "a1", "a2"):\n                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed',
    '            if args.arm in ("c0", "c0fgs", "a0", "a1", "a2"):\n                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed')

# --- C) mark_dirty after opacity reset: include c0fgs ---
src = src.replace(
    '            if args.arm in ("c0", "a0", "a1", "a2"):\n                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value',
    '            if args.arm in ("c0", "c0fgs", "a0", "a1", "a2"):\n                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value')

# --- D) densify branch: insert c0fgs FastGS policy path ---
old_densify = '''            current_radii = radii.float().max(dim=-1).values
            ev = model.densify_and_prune(
                max_grad=thresh_eff,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii)
            if args.arm in ("c0", "a0", "a1", "a2"):'''
new_densify = '''            current_radii = radii.float().max(dim=-1).values
            if args.arm == "c0fgs":
                # RF0B Lane B: FastGS multi-view-consistent densification/pruning
                # policy (fastgs/FastGS @ 44e02a5). Renderer/loss/optimizer/horizon
                # UNCHANGED; only the density-control policy differs from frozen C0.
                with torch.no_grad():
                    score_rng = random.Random(config.seed * 1000003 + iteration)
                    train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
                    importance_score, pruning_score = compute_fastgs_scores(
                        model, train_pool, dataset, render_with_meta,
                        FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                        model.active_sh_degree)
                ev = model.densify_and_prune_fastgs(
                    max_screen_size=size_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    radii=current_radii,
                    importance_score=importance_score,
                    pruning_score=pruning_score,
                    args=None)
            else:
                ev = model.densify_and_prune(
                    max_grad=thresh_eff,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=size_threshold,
                    radii=current_radii)
            if args.arm in ("c0", "c0fgs", "a0", "a1", "a2"):'''
assert old_densify in src, "densify block not found"
src = src.replace(old_densify, new_densify)

# --- E) final_prune_fastgs: add after opacity-reset block, before optimizer step ---
old_opt = '''        pev[4].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        pev[5].record()'''
new_opt = '''        # RF0B Lane B: FastGS final-stage pruning (fastgs train.py:153-158).
        # Every 3k iters AFTER 15k and BEFORE 30k: prune opacity<0.1 OR
        # pruning_score>0.9 (multi-view inconsistency). Matches FastGS exactly.
        if (args.arm == "c0fgs" and iteration % 3000 == 0
                and iteration > 15_000 and iteration < 30_000):
            with torch.no_grad():
                score_rng = random.Random(config.seed * 1000003 + iteration * 7 + 1)
                train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
                _, pruning_score_fp = compute_fastgs_scores(
                    model, train_pool, dataset, render_with_meta,
                    FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                    model.active_sh_degree)
            n_fp = model.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score_fp)
            total_final_prunes += n_fp
            if args.arm in ("c0", "c0fgs", "a0", "a1", "a2"):
                _HIGS_DYNAMIC_SCENE.mark_dirty()

        pev[4].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        pev[5].record()'''
assert old_opt in src, "optimizer block not found"
src = src.replace(old_opt, new_opt)

# --- F) init total_final_prunes counter next to total_prunes ---
src = src.replace(
    "        t_start = time.perf_counter()\n        total_clones = total_splits = total_prunes = total_resets = 0",
    "        t_start = time.perf_counter()\n        total_clones = total_splits = total_prunes = total_resets = 0\n        total_final_prunes = 0")

# --- G) record total_final_prunes into results dict ---
src = src.replace(
    '"total_clones": total_clones, "total_splits": total_splits,\n        "total_prunes": total_prunes, "total_opacity_resets": total_resets,',
    '"total_clones": total_clones, "total_splits": total_splits,\n        "total_prunes": total_prunes, "total_opacity_resets": total_resets,\n        "total_final_prunes": total_final_prunes,')

assert src != orig
open("rf0b_trainer.py","w").write(src)
print("patched rf0b_trainer.py", len(src), "bytes")
for tag in ["c0fgs", "compute_fastgs_scores", "densify_and_prune_fastgs", "final_prune_fastgs", "total_final_prunes", "mark_dirty"]:
    print(f"  {tag}: {src.count(tag)}")