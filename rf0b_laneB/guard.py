p = "rf0b_trainer.py"
s = open(p, encoding="utf-8").read()

# 1) Add a context manager right after phot_fn definition.
oldI = '''    def phot_fn(pred_hwc, target_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = target_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim_val = ssim_fn(b, g)
        return (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * (1.0 - ssim_val)
'''
newI = oldI + '''
    # FastGS scoring renders extra views purely to COMPUTE scores; they must not
    # perturb the HIGS renderer's forward/backward lifecycle. Each probe render
    # calls _begin_forward() (sets _pending_backward=True) but, being no-grad,
    # never calls _end_backward(). We snapshot/restore the flag so the subsequent
    # mark_dirty() (after the real backward) sees the correct state. This is pure
    # bookkeeping — no renderer or policy change.
    from contextlib import contextmanager
    @contextmanager
    def _fastgs_scoring_guard():
        was_pending = _HIGS_DYNAMIC_SCENE._pending_backward
        try:
            yield
        finally:
            _HIGS_DYNAMIC_SCENE._pending_backward = was_pending
'''
assert oldI in s, "phot_fn block not found"
s = s.replace(oldI, newI)

# 2) Wrap the densify-branch scoring call
oldD = '''                train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
                score_rng = random.Random(config.seed * 1000003 + iteration)
                importance_score, pruning_score = compute_fastgs_scores(
                    model, train_pool, dataset, render_with_meta, phot_fn,
                    FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                    model.active_sh_degree)
                ev = model.densify_and_prune_fastgs('''
newD = '''                train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
                score_rng = random.Random(config.seed * 1000003 + iteration)
                with _fastgs_scoring_guard():
                    importance_score, pruning_score = compute_fastgs_scores(
                        model, train_pool, dataset, render_with_meta, phot_fn,
                        FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                        model.active_sh_degree)
                ev = model.densify_and_prune_fastgs('''
assert oldD in s, "D call site not found"
s = s.replace(oldD, newD)

# 3) Wrap the final-prune scoring call
oldE = '''            train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
            score_rng = random.Random(config.seed * 1000003 + iteration + 777)
            _, pruning_score = compute_fastgs_scores(
                model, train_pool, dataset, render_with_meta, phot_fn,
                FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                model.active_sh_degree)
            n_fp = model.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)'''
newE = '''            train_pool = [i for i in range(n_cameras) if i % HOLDOUT_STEP != 0]
            score_rng = random.Random(config.seed * 1000003 + iteration + 777)
            with _fastgs_scoring_guard():
                _, pruning_score = compute_fastgs_scores(
                    model, train_pool, dataset, render_with_meta, phot_fn,
                    FGS_LOSS_THRESH, FGS_N_SCORE_VIEWS, score_rng,
                    model.active_sh_degree)
            n_fp = model.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)'''
assert oldE in s, "E call site not found"
s = s.replace(oldE, newE)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("SCORING GUARD APPLIED")