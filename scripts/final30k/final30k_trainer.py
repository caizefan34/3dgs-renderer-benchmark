#!/usr/bin/env python3
"""FINAL-30K unified trainer: reference_v1 frozen recipe, two renderer arms.

  --arm b1a : gsplat true-accutile rasterization (absgrad=True, accutile=True, eps2d=0.1)
              via the matched-env B1A extension (gsplat_cuda_final30k, sys.modules-injected)
  --arm c0  : C0_V3_FINAL30K dynamic forward (F9 + SCALAR_ADJOINT + H8_MR, PX=2,
              eps2d=0.3, HIGS_BWD_ABSGRAD=1) via the 9baf8655 extension

Everything else -- dataset, GaussianModel, optimizer, densification, camera
sequence, loss, evaluation -- is ONE shared code path copied from
b1a_13scene_train.py (reference_v1), so training semantics are matched by
construction. Used for BOTH the gate E 2K smoke and the 26 benchmark runs.

Recording: N_GS + loss every 100 iters; every densification event (iteration,
N before/after, clone/split/prune counts, selected count, per-gaussian absgrad
norm distribution stats, NaN/Inf counts); eval rows at eval_iterations; final
full-scene eval. Timing is recorded but is NOT publication-grade unless the
run is on a clean GPU (labels carried in the results JSON).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))

REFERENCE_V1_DIR = os.path.join(HERE, "baseline", "reference_v1")
PHASE7_DIR = os.path.join(HERE, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(HERE, "src")

ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
B1A_SO = "/mnt/storage_pool/liaoyuanjun/gsplat_accutile_final30k_cache/gsplat_cuda_final30k/gsplat_cuda_final30k.so"
C0_WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
C0_SO = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_ext_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def bootstrap_b1a():
    mod = load_ext_module("gsplat_cuda_final30k", B1A_SO)
    sys.modules["gsplat.csrc"] = mod
    # sys.path priority (lowest first): accutile tree, phase7, src, reference_v1
    sys.path.insert(0, ACCUTILE_TREE)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    import gsplat
    import inspect
    sig = inspect.signature(gsplat.rasterization)
    assert "accutile" in sig.parameters, "accutile param missing -- wrong tree"
    return {"arm": "b1a", "so_sha256": sha256_file(B1A_SO), "so_path": B1A_SO,
            "gsplat_file": gsplat.__file__}


def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    # shared recipe paths (after worktree; no name conflicts)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: F401
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE  # noqa: F401
    return {"arm": "c0", "so_sha256": sha256_file(C0_SO), "so_path": C0_SO,
            "core_sha256": sha256_file(CORE_SO),
            "env": {k: os.environ[k] for k in
                    ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD")}}


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["b1a", "c0"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--final-eval", choices=["all", "subset"], default="all")
    ap.add_argument("--checkpoints", action="store_true", help="save 30K checkpoint (benchmark mode)")
    ap.add_argument("--timing-grade", default="FUNCTIONAL_ONLY",
                    choices=["FUNCTIONAL_ONLY", "PUBLICATION"],
                    help="label carried into the results JSON; PUBLICATION requires a clean GPU")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if args.arm == "b1a":
        ident = bootstrap_b1a()
    else:
        ident = bootstrap_c0()

    # shared recipe imports (after bootstrap)
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset
    if args.arm == "b1a":
        from gsplat import rasterization
    else:
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE

    # ---------------- SepSSIM / PSNR (verbatim from b1a_13scene_train.py) ----
    class SepSSIM:
        def __init__(self, device="cuda", window_size=11, sigma=1.5):
            coords = torch.arange(window_size, dtype=torch.float32, device=device)
            coords -= window_size // 2
            k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            k1d = k1d / k1d.sum()
            self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
            self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
            self.padding = window_size // 2

        def __call__(self, pred, target):
            if pred.ndim == 3:
                pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
                target = target.unsqueeze(0).permute(0, 3, 1, 2)
            stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
            b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
            b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
            mu_p, mu_t = b[:, 0:3], b[:, 3:6]
            bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
            mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
            sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
            ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                       ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
            return 1.0 - ssim_map.mean()

    def compute_psnr(pred, gt):
        mse = F.mse_loss(pred, gt)
        return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0

    def compute_scene_extent(dataset, n_samples=50):
        centers = []
        n = min(n_samples, len(dataset))
        for i in range(n):
            cam = dataset.get_camera(i)
            centers.append(cam.camera_center.cpu().numpy())
        centers = np.array(centers)
        extent = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                extent = max(extent, d)
        return max(extent, 0.1)

    # ---------------- arm-specific render ---------------------------------
    def render_with_meta(model, cam, sh_degree):
        if args.arm == "b1a":
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                absgrad=True, accutile=True)
            means2d = meta["means2d"]
            if means2d.requires_grad:
                means2d.retain_grad()
            return r[0].clamp(0, 1), meta["radii"], means2d
        else:
            out = rasterize_gaussian_higs_dynamic(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                sh_degree=sh_degree,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB")  # eps2d default 0.3 (frozen C0)
            info = out["densification_info"]
            means2d = info["means2d"]
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            return frame.clamp(0, 1), info["radii"], means2d

    # ---------------- setup ------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = HERE
    config.iterations = args.iterations

    print(f"\n{'=' * 60}\n  FINAL-30K TRAINER  arm={args.arm} scene={args.scene} "
          f"iters={args.iterations}\n  gpu={args.gpu} timing_grade={args.timing_grade}\n{'=' * 60}")

    dataset = GTDataset(scene=args.scene, repo_root=HERE, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)
    scene_extent = compute_scene_extent(dataset)

    sfm_path = os.path.join(HERE, "data", "datasets", "mipnerf360", args.scene,
                            "sparse", "0", "points3D.bin")
    points3d = read_points3D_binary(sfm_path)
    pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)

    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = model._xyz.shape[0]
    model.training_setup({
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    print(f"  cameras={n_cameras} extent={scene_extent:.4f} initial_N={initial_N}")

    lpips_fn = None
    if args.lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
            lpips_fn.eval()
        except Exception as e:
            print(f"  WARNING: LPIPS unavailable: {e}")

    ssim_fn = SepSSIM(device="cuda")

    # frozen camera sequence (identical both arms: same rng, same seed)
    viewpoint_stack = list(range(n_cameras))
    camera_sequence = []
    rng = random.Random(config.seed)
    for iteration in range(1, config.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        camera_sequence.append(viewpoint_stack.pop(rand_idx))
    camera_sequence = np.array(camera_sequence, dtype=np.int32)
    np.save(os.path.join(args.outdir, "camera_sequence.npy"), camera_sequence)

    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))
    eval_iterations = set(i for i in config.eval_iterations if i <= args.iterations)

    # ---------------- recording structures ---------------------------------
    per_iter = []           # every 100 iters: {iter, n_gs, loss, n_visible, nan_loss}
    dens_events = []        # every densification event
    eval_rows = []
    iter_times = []
    peak_vram = 0.0

    def dist_stats(t):
        v = t.detach().float().reshape(-1)
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return {"n": 0}
        return {"n": int(v.numel()), "mean": float(v.mean()), "median": float(v.median()),
                "p90": float(torch.quantile(v, 0.90)), "p95": float(torch.quantile(v, 0.95)),
                "p99": float(torch.quantile(v, 0.99)), "max": float(v.max()), "min": float(v.min())}

    def evaluate(cam_indices, tag, step):
        psnrs, ssims, l1s, lpips_vals = [], [], [], []
        for ci in cam_indices:
            cam, gt = dataset.get_item(ci)
            with torch.no_grad():
                img, _, _ = render_with_meta(model, cam, model.active_sh_degree)
            psnrs.append(compute_psnr(img, gt))
            ssims.append(1.0 - ssim_fn(img, gt).item())
            l1s.append(float(F.l1_loss(img, gt).item()))
            if lpips_fn is not None:
                pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                lpips_vals.append(float(lpips_fn(pred_lp, gt_lp).item()))
        row = {"step": step, "tag": tag,
               "wall_time_s": time.perf_counter() - t_start,
               "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
               "l1": float(np.mean(l1s)),
               "n_eval_cameras": len(cam_indices),
               "n_gaussians": int(model._xyz.shape[0])}
        if lpips_vals:
            row["lpips"] = float(np.mean(lpips_vals))
        eval_rows.append(row)
        lp_str = f" LPIPS={row['lpips']:.4f}" if "lpips" in row else ""
        print(f"  [{args.arm}/{args.scene} eval@{step}] PSNR={row['psnr']:.2f} "
              f"SSIM={row['ssim']:.4f}{lp_str} GS={row['n_gaussians']:,}", flush=True)
        return row

    # ---------------- training loop ----------------------------------------
    t_start = time.perf_counter()
    total_clones = total_splits = total_prunes = total_resets = 0

    # phase timing (CUDA events, reused; one sync/iter as before)
    pev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    phase_ms = {k: [] for k in ("forward", "loss", "backward", "densify", "optimizer")}

    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)

        t0 = time.perf_counter()
        pev[0].record()
        image, radii_full, means2d = render_with_meta(model, cam, model.active_sh_degree)
        pev[1].record()

        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        pev[2].record()

        loss.backward()
        pev[3].record()

        if args.arm == "c0" and iteration <= config.densify_until_iter:
            # Loud guard: under H8_MR the signed .grad fallback is moment-space
            # (NOT the contracted reference gradient). Densification MUST consume
            # the attached absgrad. Fail loudly rather than silently degrade.
            assert hasattr(means2d, "absgrad"), \
                "C0 arm: means2d.absgrad missing (HIGS_BWD_ABSGRAD not honored)"

        radii = radii_full[0]
        visibility_filter = (radii > 0).any(dim=-1)
        n_visible = int(visibility_filter.sum())

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)

        if (iteration > config.densify_from_iter and
                iteration < config.densify_until_iter and
                iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            n_before = int(model._xyz.shape[0])

            # pre-event record: the per-gaussian absgrad norm distribution
            with torch.no_grad():
                grads = model.xyz_gradient_accum / model.denom
                nan_count = int(torch.isnan(grads).sum())
                inf_count = int(torch.isinf(grads).sum())
                grads = torch.nan_to_num(grads, nan=0.0)
                gnorm = grads.reshape(-1)
                selected = gnorm >= config.densify_grad_threshold
                sel_stats = dist_stats(gnorm[selected]) if int(selected.sum()) else {"n": 0}

            current_radii = radii.float().max(dim=-1).values
            ev = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii)
            if args.arm == "c0":
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # topology changed

            n_after = int(model._xyz.shape[0])
            total_clones += ev["cloned"]
            total_splits += ev["split"]
            total_prunes += ev["pruned_total"]
            dens_events.append({
                "iter": iteration, "n_before": n_before, "n_after": n_after,
                "cloned": ev["cloned"], "split": ev["split"],
                "pruned_total": ev["pruned_total"], "pruned_opacity": ev["pruned_opacity"],
                "selected": int(selected.sum()),
                "grad_norm_stats": dist_stats(gnorm),
                "selected_grad_norm_stats": sel_stats,
                "grad_nan_count": nan_count, "grad_inf_count": inf_count,
                "loss": float(loss.item()),
            })

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            total_resets += 1
            if args.arm == "c0":
                _HIGS_DYNAMIC_SCENE.mark_dirty()  # opacity is a packed-buffer value

        pev[4].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        pev[5].record()

        torch.cuda.synchronize()
        phase_ms["forward"].append(pev[0].elapsed_time(pev[1]))
        phase_ms["loss"].append(pev[1].elapsed_time(pev[2]))
        phase_ms["backward"].append(pev[2].elapsed_time(pev[3]))
        phase_ms["densify"].append(pev[3].elapsed_time(pev[4]))
        phase_ms["optimizer"].append(pev[4].elapsed_time(pev[5]))
        iter_times.append((time.perf_counter() - t0) * 1000)
        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_vram = max(peak_vram, vram)

        if iteration % 100 == 0 or iteration == 1:
            lv = float(loss.item())
            per_iter.append({"iter": iteration, "n_gs": int(model._xyz.shape[0]),
                             "loss": lv, "n_visible": n_visible,
                             "loss_nan": not math.isfinite(lv)})

        if iteration in eval_iterations:
            evaluate(eval_indices, "eval", iteration)

        if iteration % 500 == 0:
            print(f"  [{args.arm}/{args.scene} iter {iteration}] loss={float(loss.item()):.4f} "
                  f"GS={model._xyz.shape[0]:,} vis={n_visible:,}", flush=True)

    total_time = time.perf_counter() - t_start

    # ---- phase timing aggregation (means over all iters) ----
    phase_summary = {}
    for k, v in phase_ms.items():
        if v:
            arr = np.array(v)
            phase_summary[k] = {
                "mean_ms": float(np.mean(arr)), "median_ms": float(np.median(arr)),
                "min_ms": float(np.min(arr)), "max_ms": float(np.max(arr)),
            }

    if args.final_eval == "all":
        final_eval = evaluate(list(range(n_cameras)), "final", config.iterations)
    else:
        final_eval = evaluate(eval_indices, "final", config.iterations)

    iter_arr = np.array(iter_times)
    results = {
        "arm": args.arm,
        "scene": args.scene,
        "iterations": config.iterations,
        "seed": config.seed,
        "config": config.to_dict(),
        "renderer": {
            "b1a": {"eps2d": 0.1, "absgrad": True, "accutile": True, "tile_size": 16, "packed": False},
            "c0": {"eps2d": 0.3, "env": ident.get("env"), "backward_mode": "higs_native",
                    "f9": True, "scalar_adjoint": True, "h8_mr": True, "px_runtime": 2},
        }[args.arm],
        "binary_identity": ident,
        "timing_grade": args.timing_grade,
        "timing_note": ("wall/iter timing is NOT publication-grade unless timing_grade=PUBLICATION "
                        "and the run was on a CLEAN_PUBLICATION_GPU"),
        "timing": {
            "total_wall_s": total_time, "total_wall_min": total_time / 60,
            "mean_iter_ms": float(np.mean(iter_arr)), "median_iter_ms": float(np.median(iter_arr)),
            "n_iters": len(iter_arr),
            "phase_ms": phase_summary,
        },
        "initial_N": initial_N,
        "final_N": int(model._xyz.shape[0]),
        "n_cameras": n_cameras,
        "scene_extent": float(scene_extent),
        "total_clones": total_clones, "total_splits": total_splits,
        "total_prunes": total_prunes, "total_opacity_resets": total_resets,
        "peak_vram_gb": peak_vram,
        "per_iter": per_iter,
        "densification_events": dens_events,
        "eval_rows": eval_rows,
        "final_eval": final_eval,
        "torch": torch.__version__,
    }
    if args.checkpoints:
        ckpt = model.capture()
        ckpt["optimizer_state_dict"] = model.optimizer.state_dict()
        ckpt["iteration"] = config.iterations
        ckpt["scene"] = args.scene
        ckpt["arm"] = args.arm
        torch.save(ckpt, os.path.join(args.outdir, f"ckpt_{config.iterations}.pt"))

    # ---- harness-format artifacts (frozen FINAL-30K schema) ----
    all_finite = all(math.isfinite(r["loss"]) for r in per_iter) and \
        all(math.isfinite(r["psnr"]) and math.isfinite(r["ssim"]) for r in eval_rows)
    final_status = {
        "status": "SUCCESS" if all_finite else "NUMERIC_FAILURE",
        "arm": args.arm, "scene": args.scene,
        "iterations": config.iterations, "seed": config.seed,
        "all_losses_finite": all(math.isfinite(r["loss"]) for r in per_iter),
        "contaminated": False,  # scheduler annotates contamination post-hoc
        "timing_grade": args.timing_grade,
    }
    timing_file = {
        "total_wall_s": total_time,
        "iterations_per_s": config.iterations / total_time if total_time > 0 else None,
        "mean_iter_ms": float(np.mean(iter_arr)),
        "median_iter_ms": float(np.median(iter_arr)),
        "forward_ms": phase_summary.get("forward", {}).get("mean_ms"),
        "backward_ms": phase_summary.get("backward", {}).get("mean_ms"),
        "nested_fb_ms": (phase_summary.get("forward", {}).get("mean_ms", 0.0)
                         + phase_summary.get("backward", {}).get("mean_ms", 0.0)) if phase_summary else None,
        "optimizer_ms": phase_summary.get("optimizer", {}).get("mean_ms"),
        "loss_ms": phase_summary.get("loss", {}).get("mean_ms"),
        "phase_ms": phase_summary,
    }
    quality_file = dict(final_status)
    quality_file.update(final_eval)

    with open(os.path.join(args.outdir, "final_status.json"), "w") as f:
        json.dump(final_status, f, indent=2)
    with open(os.path.join(args.outdir, "timing.json"), "w") as f:
        json.dump(timing_file, f, indent=2)
    with open(os.path.join(args.outdir, "quality.json"), "w") as f:
        json.dump(quality_file, f, indent=2)
    with open(os.path.join(args.outdir, "training_results.json"), "w") as f:
        json.dump({
            "initial_N": initial_N, "final_N": int(model._xyz.shape[0]),
            "total_clones": total_clones, "total_splits": total_splits,
            "total_prunes": total_prunes, "total_opacity_resets": total_resets,
        }, f, indent=2)
    with open(os.path.join(args.outdir, "memory.json"), "w") as f:
        json.dump({"peak_vram_gb": peak_vram}, f, indent=2)

    import csv as _csv
    with open(os.path.join(args.outdir, "training_curve.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step", "wall_time", "psnr", "ssim", "lpips", "l1", "n_gaussians", "tag"])
        for r in sorted(eval_rows, key=lambda x: x["step"]):
            w.writerow([r["step"], r.get("wall_time_s", ""), r["psnr"], r["ssim"],
                        r.get("lpips", "null"), r.get("l1", ""), r.get("n_gaussians", ""),
                        r["tag"]])

    out_path = os.path.join(args.outdir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  DONE {args.arm}/{args.scene}: final_N={results['final_N']:,} "
          f"PSNR={final_eval['psnr']:.2f} wall={total_time / 60:.1f}min")
    print(f"  WROTE {out_path} + final_status.json, timing.json, quality.json, "
          f"training_results.json, training_curve.csv")


if __name__ == "__main__":
    main()
