#!/usr/bin/env python3
"""Generate D0 traced sources from the frozen P12 files (metadata-only hooks).

Frozen inputs (never modified):
  baseline/reference_v1/gaussian_model.py   -> d0_gaussian_model.py
  p12_work/publication_trainer_p12.py       -> d0_trainer.py

Every patch is an exact-text insertion anchored on text that must occur
exactly once; any anchor drift aborts generation. Outputs:
  instrumented/d0_gaussian_model.py
  instrumented/d0_trainer.py
  instrumented/D0_SOURCE_DIFF.diff   (unified diff vs frozen, local sources)
"""
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_MODEL = os.path.join(REPO, "baseline", "reference_v1", "gaussian_model.py")
SRC_TRAINER = os.path.join(REPO, "p12_work", "publication_trainer_p12.py")


def apply_patches(text: str, patches, label: str) -> str:
    for i, (old, new) in enumerate(patches):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{label}: patch {i} anchor occurs {n} times "
                             f"(must be 1):\n{old[:200]}")
        text = text.replace(old, new, 1)
    return text


# ============================================================== model
MODEL_PATCHES = [
    # M1: tracer handle
    (
        """        # Explicit Gaussian identity tracking (R0.2 Part H)
        # Does not affect rendering; purely metadata for lineage tracking.
        self._gaussian_ids = torch.empty(0, dtype=torch.long, device="cuda")
        self._next_gaussian_id = 0
""",
        """        # Explicit Gaussian identity tracking (R0.2 Part H)
        # Does not affect rendering; purely metadata for lineage tracking.
        self._gaussian_ids = torch.empty(0, dtype=torch.long, device="cuda")
        self._next_gaussian_id = 0

        # D0 metadata-only tracer handle (None in every non-D0 run).
        self.d0 = None
        self.d0_iter = 0
""",
    ),
    # M2: init registration
    (
        """        print(f"Number of points at initialisation : {N}")
""",
        """        print(f"Number of points at initialisation : {N}")
        if self.d0 is not None:
            self.d0.register_init(int(N))
""",
    ),
    # M3: clone births
    (
        """        n_cloned = selected_pts_mask.sum().item()
        self.densification_postfix(new_xyz, new_shs, new_opacities,
                                    new_scaling, new_rotation, new_tmp_radii)
        return n_cloned
""",
        """        d0_feat = None
        d0_parent_uids = None
        if self.d0 is not None:
            rows = selected_pts_mask.nonzero(as_tuple=True)[0]
            d0_feat = self.d0.capture_birth_features(rows, 1, self.d0_iter)
            d0_parent_uids = self._gaussian_ids[rows].detach()

        n_cloned = selected_pts_mask.sum().item()
        self.densification_postfix(new_xyz, new_shs, new_opacities,
                                    new_scaling, new_rotation, new_tmp_radii)
        if self.d0 is not None and int(n_cloned) > 0:
            self.d0.register_births(
                self._gaussian_ids[-int(n_cloned):].detach(),
                d0_parent_uids, 1, self.d0_iter, d0_feat)
        return n_cloned
""",
    ),
    # M4: split births + split-parent deaths
    (
        """        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N) if self.tmp_radii is not None else None

        self.densification_postfix(new_xyz, new_shs, new_opacity,
                                    new_scaling, new_rotation, new_tmp_radii)

        # Remove split parents (official L432-433)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * n_split, device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter)
""",
        """        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N) if self.tmp_radii is not None else None

        d0_feat = None
        d0_parent_uids = None
        if self.d0 is not None:
            rows = selected_pts_mask.nonzero(as_tuple=True)[0]
            # children are appended as N consecutive blocks over the parents;
            # child block b, slot j <-> parent j  (feat [F, m] -> [F, N*m])
            d0_feat = self.d0.capture_birth_features(
                rows, 2, self.d0_iter).repeat(1, N)
            d0_parent_uids = self._gaussian_ids[rows].repeat(N).detach()

        self.densification_postfix(new_xyz, new_shs, new_opacity,
                                    new_scaling, new_rotation, new_tmp_radii)
        if self.d0 is not None and int(n_split) > 0:
            self.d0.register_births(
                self._gaussian_ids[-N * int(n_split):].detach(),
                d0_parent_uids, 2, self.d0_iter, d0_feat)

        # Remove split parents (official L432-433)
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * n_split, device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter, d0_reason=4)
""",
    ),
    # M5: prune with reason
    (
        """    def prune_points(self, mask: torch.Tensor):
        \"\"\"Remove Gaussians where mask=True. Preserve survivor optimizer state.

        Official L349-364.
        \"\"\"
        valid_points_mask = ~mask
""",
        """    def prune_points(self, mask: torch.Tensor, d0_reason=0):
        \"\"\"Remove Gaussians where mask=True. Preserve survivor optimizer state.

        Official L349-364.  d0_reason is metadata-only (int or per-row tensor).
        \"\"\"
        if self.d0 is not None:
            died_uids = self._gaussian_ids[mask].detach()
            if d0_reason is None:
                # densify_and_prune passes None only while max_screen_size is
                # inactive (iteration <= opacity_reset_interval): the prune
                # mask then contains opacity prunes only (reason 1).
                reasons = 1
            else:
                reasons = (d0_reason[mask] if torch.is_tensor(d0_reason)
                           else int(d0_reason))
            self.d0.register_deaths(died_uids, reasons, self.d0_iter)
        valid_points_mask = ~mask
""",
    ),
    # M6: prune reason codes in densify_and_prune
    (
        """        pruned_screen = 0
        pruned_world = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            pruned_screen = (big_points_vs & ~prune_mask).sum().item()
            pruned_world = (big_points_ws & ~(prune_mask | big_points_vs)).sum().item()
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        total_pruned = prune_mask.sum().item()
        if total_pruned > 0:
            self.prune_points(prune_mask)
""",
        """        pruned_screen = 0
        pruned_world = 0
        d0_reason_codes = None
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            pruned_screen = (big_points_vs & ~prune_mask).sum().item()
            pruned_world = (big_points_ws & ~(prune_mask | big_points_vs)).sum().item()
            if self.d0 is not None:
                # priority opacity > screen > world, matching the reported
                # pruned_opacity / pruned_screen / pruned_world counts above
                d0_reason_codes = torch.zeros_like(prune_mask, dtype=torch.uint8)
                d0_reason_codes[big_points_ws] = 3
                d0_reason_codes[big_points_vs] = 2
                d0_reason_codes[prune_mask] = 1
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        total_pruned = prune_mask.sum().item()
        if total_pruned > 0:
            self.prune_points(prune_mask, d0_reason=d0_reason_codes)
""",
    ),
    # M7: opacity reset event
    (
        """        opacities_new = torch.logit(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )  # 1D
""",
        """        if self.d0 is not None:
            self.d0.register_opacity_reset(self.d0_iter)
        opacities_new = torch.logit(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )  # 1D
""",
    ),
]


# ============================================================ trainer
TRAINER_PATCHES = [
    # T1: CLI flag
    (
        """    args = ap.parse_args()
""",
        """    ap.add_argument("--d0-trace", action="store_true",
                    help="D0 waste-census metadata tracing (D0_INSTRUMENTATION_SPEC.md)")
    args = ap.parse_args()
""",
    ),
    # T2: traced C0 worktree
    (
        """C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
""",
        """# D0: traced runs import the metadata-patched worktree copy; the frozen
# worktree remains the binary source for every other run.
C0_WT = ("/root/epic05_d0/c0_worktree_d0" if "--d0-trace" in sys.argv
         else "/root/epic05_frozen/higs_c0_final30k_worktree")
""",
    ),
    # T3: traced model + tracer import
    (
        """    from gaussian_model import GaussianModel
""",
        """    if args.d0_trace:
        from d0_gaussian_model import GaussianModel
        from d0_trace import D0Tracer
    else:
        from gaussian_model import GaussianModel
""",
    ),
    # T4: b0 return
    (
        """            return frame.clamp(0, 1), radii.reshape(1, -1, 1), means2d
""",
        """            return frame.clamp(0, 1), radii.reshape(1, -1, 1), means2d, None
""",
    ),
    # T5: b1 return
    (
        """            return r[0].clamp(0, 1), meta["radii"], means2d
""",
        """            return r[0].clamp(0, 1), meta["radii"], means2d, None
""",
    ),
    # T6: c0 return + capture dict
    (
        """            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            return frame.clamp(0, 1), info["radii"], means2d
""",
        """            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            d0cap = None
            if args.d0_trace:
                _md = out.get("metadata", {})
                d0cap = {
                    "visible_ids": info["visible_gaussian_ids"],
                    "isect_row_space": "visible",
                    "tiles_per_gauss": _md.get("d0_tiles_per_gauss"),
                    "depths_vis": _md.get("d0_depths_vis"),
                    "means2d_px_visible": _md.get("d0_means2d_px"),
                    "radii_full": info["radii"][0],
                    "means2d_proxy": means2d,
                    "n_isects": int(_md.get("n_isects") or 0),
                    "n_isects_full": int(_md.get("n_isects_full") or 0),
                    "f9_enabled": bool(_md.get("d0_f9_enabled")),
                    "width": float(info["width"]),
                    "height": float(info["height"]),
                    "cam_center": cam.camera_center.detach().clone(),
                }
            return frame.clamp(0, 1), info["radii"], means2d, d0cap
""",
    ),
    # T7: tracer construction (before create_from_pcd so register_init fires)
    (
        """    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
""",
        """    model = GaussianModel(max_sh_degree=config.sh_degree)
    d0 = None
    if args.d0_trace:
        # offset from the scene name only (never from the training RNG)
        _d0_off = 1 + int(hashlib.sha256(
            ("D0:" + args.scene).encode()).hexdigest(), 16) % 20
        d0 = D0Tracer(model, args.scene, args.outdir, n_cameras,
                      config.densify_grad_threshold, _d0_off)
        model.d0 = d0
        print(f"  [D0] trace enabled: sample_offset={_d0_off}", flush=True)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
""",
    ),
    # T8: evaluate unpack
    (
        """                img, _, _ = render_with_meta(model, cam, model.active_sh_degree)
""",
        """                img, _, _, _ = render_with_meta(model, cam, model.active_sh_degree)
""",
    ),
    # T9: train-loop render unpack
    (
        """        image, radii_full, means2d = render_with_meta(model, cam, model.active_sh_degree)
""",
        """        image, radii_full, means2d, d0cap = render_with_meta(model, cam, model.active_sh_degree)
""",
    ),
    # T10: per-iteration metadata hook (after backward, before densify)
    (
        """        radii = radii_full[0]
        visibility_filter = (radii > 0).any(dim=-1)
        n_visible = int(visibility_filter.sum())
""",
        """        radii = radii_full[0]
        visibility_filter = (radii > 0).any(dim=-1)
        n_visible = int(visibility_filter.sum())

        if args.d0_trace:
            model.d0_iter = iteration
            d0.note_render(iteration, cam_idx, d0cap, image, gt_image)
""",
    ),
    # T11: densify event id
    (
        """            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            n_before = int(model._xyz.shape[0])
""",
        """            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            n_before = int(model._xyz.shape[0])
            if args.d0_trace:
                d0.begin_event(iteration)
""",
    ),
    # T12: trajectory row after phase timing
    (
        """        iter_times.append((time.perf_counter() - t0) * 1000)
""",
        """        iter_times.append((time.perf_counter() - t0) * 1000)
        if args.d0_trace:
            d0.note_iteration(iteration, float(loss.item()), phase_ms,
                              iter_times[-1])
""",
    ),
    # T13: checkpoint states + full-camera evals
    (
        """        if iteration in eval_iterations:
            evaluate(eval_indices, "eval", iteration)
""",
        """        if iteration in eval_iterations:
            evaluate(eval_indices, "eval", iteration)

        if args.d0_trace and iteration in (5000, 10000, 15000, 20000, 25000,
                                           30000):
            evaluate(list(range(n_cameras)), "d0_ckpt", iteration)
            torch.save(model.capture(),
                       os.path.join(args.outdir, f"d0_ckpt_{iteration}.pt"))
""",
    ),
    # T14: finalize before artifact writing
    (
        """    # ---- harness-format artifacts (frozen FINAL-30K schema) ----
""",
        """    if args.d0_trace:
        d0_meta = d0.finalize({
            "arm": args.arm, "scene": args.scene, "seed": config.seed,
            "iterations": config.iterations, "n_cameras": n_cameras,
            "outdir": args.outdir, "loss": args.loss,
            "grad_threshold": config.densify_grad_threshold,
            "timing_grade": args.timing_grade,
        })
        results["d0_trace"] = d0_meta
        print(f"  [D0] trace finalized: uids={d0_meta['n_uids']:,} "
              f"births={d0_meta['n_births_post_init']:,} "
              f"samples={d0_meta['n_samples']}", flush=True)

    # ---- harness-format artifacts (frozen FINAL-30K schema) ----
""",
    ),
    # T15: Stage-5 final-state audit (16 deterministic cameras, metadata only,
    # after all training/timing work; feeds D0_FINAL_CONTRIBUTION_AUDIT)
    (
        """    if args.final_eval == "all":
        final_eval = evaluate(list(range(n_cameras)), "final", config.iterations)
    else:
        final_eval = evaluate(eval_indices, "final", config.iterations)
""",
        """    if args.final_eval == "all":
        final_eval = evaluate(list(range(n_cameras)), "final", config.iterations)
    else:
        final_eval = evaluate(eval_indices, "final", config.iterations)

    if args.d0_trace:
        # D0 Stage 5 (descriptive audit, metadata only): render the 16
        # deterministic audit cameras on the final state, after all training
        # and timing work; no recorded training quantity is affected.
        d0.run_final_audit(dataset, render_with_meta, model, config.iterations)
""",
    ),
]


def main():
    with open(SRC_MODEL, encoding="utf-8") as f:
        model_text = f.read()
    with open(SRC_TRAINER, encoding="utf-8") as f:
        trainer_text = f.read()

    new_model = apply_patches(model_text, MODEL_PATCHES, "gaussian_model")
    new_trainer = apply_patches(trainer_text, TRAINER_PATCHES, "trainer")

    out_model = os.path.join(HERE, "d0_gaussian_model.py")
    out_trainer = os.path.join(HERE, "d0_trainer.py")
    with open(out_model, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_model)
    with open(out_trainer, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_trainer)

    diff = []
    for name, a, b in (("gaussian_model.py", model_text, new_model),
                       ("publication_trainer_p12.py", trainer_text, new_trainer)):
        diff.extend(difflib.unified_diff(
            a.splitlines(keepends=True), b.splitlines(keepends=True),
            fromfile=f"frozen/{name}", tofile=f"d0/{name}", n=3))
    diff_path = os.path.join(HERE, "D0_SOURCE_DIFF.diff")
    with open(diff_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(diff))

    print(f"wrote {out_model}")
    print(f"wrote {out_trainer}")
    print(f"wrote {diff_path} ({len(diff)} diff lines)")
    for p in (out_model, out_trainer):
        import ast
        with open(p, encoding="utf-8") as f:
            ast.parse(f.read())
        print(f"  ast-parse OK: {os.path.basename(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
