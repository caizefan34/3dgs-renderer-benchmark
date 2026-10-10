#!/usr/bin/env python3
"""R21 30K paired trainer. Copied from R20_TRAINER.py with two R21 additions:
  1. Held-out split (MipNeRF-360 style, every Nth camera held out) for an
     INDEPENDENT held-out evaluation alongside train-view metrics.
  2. Checkpointing at configurable iterations (5K/10K/15K/20K/25K/30K).
  Reference model, loss, optimizer, densification, opacity-reset, SH progression
  and resolution are UNCHANGED. Both backends share the identical split (seed
  + n_cameras) so the comparison is paired.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
REFV1 = REPO / "baseline" / "reference_v1"
PHASE7 = REPO / "scripts" / "epic05" / "phase7"
SRCDIR = REPO / "src"
PROJECT = Path(os.environ.get("HIGS_PROJECT",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"))
sys.path.insert(0, str(SRCDIR))
sys.path.insert(0, str(PHASE7))
sys.path.insert(0, str(REFV1))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

EXPECTED = {
    "std_csrc_sha256": "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50",
    "higs_csrc_sha256": "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3",
}


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def attach_absgrad_hook(means2d_proxy):
    """Explicit signed-mode adapter: abs(sum(pixel_gradient)), not native absgrad."""
    def hook(grad):
        ag = grad.detach().abs()
        means2d_proxy.absgrad = ag
        return grad
    means2d_proxy.register_hook(hook)
    return means2d_proxy


def sfm_from_ply(ply_path, sh_degree=3):
    """Build an SfM-init pcd_data dict from a 3DGS-format sparse PLY.

    Extracts only xyz + DC color (f_dc_0/1/2 -> SH DC coefficient), passing
    opacity/scales/rotations as None so GaussianModel.create_from_pcd applies
    the standard official SfM init (inverse_sigmoid(0.1), KNN scales, identity
    rotation). This keeps initialization identical to the COLMAP-binary path
    while using the real SfM point positions."""
    C0 = 0.28209479177387814
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        n = 0
        props = []
        for ln in header:
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property"):
                props.append(ln.split()[2])
        data = f.read()
    dt = np.dtype([(p, "<f4") for p in props])
    arr = np.frombuffer(data, dtype=dt, count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    # DC SH coefficient from f_dc: color = C0 * f_dc + 0.5  ->  f_dc = (color-0.5)/C0
    fdc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float64)
    shs = np.zeros((n, (sh_degree + 1) ** 2, 3), dtype=np.float32)
    shs[:, 0, :] = (fdc / C0).astype(np.float32)  # store as SH DC; create_from_pcd handles rest
    return {"xyz": torch.from_numpy(xyz), "shs": torch.from_numpy(shs),
            "opacity": None, "scales": None, "rotations": None, "num_points": n}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["std_ll", "higs"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="if >0, eval every N iters (smoke); else use config.eval_iterations")
    ap.add_argument("--densify-mode", choices=["absgrad", "signed"], default="absgrad",
                    help="absgrad=reference recipe (std_ll native absgrad; HiGS uses "
                         "true native absgrad). signed=both arms use |means2d.grad| for "
                         "SYMMETRIC densification (isolates renderer effect from the "
                         "absgrad-signal asymmetry). R20 primary comparison uses absgrad.")
    ap.add_argument("--heldout-mod", type=int, default=8,
                    help="MipNeRF-360 style split: cameras with index mod N == 0 are "
                         "held out for INDEPENDENT eval; the rest are trained on. "
                         "Both backends use the identical split.")
    ap.add_argument("--checkpoint-iters", type=int, nargs="*",
                    default=[5000, 10000, 15000, 20000, 25000, 30000],
                    help="iterations at which to save a full model checkpoint")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # verify the frozen binary for this backend
    so_rel = "gsplat/csrc.so" if args.backend == "std_ll" else \
        "gsplat/experimental/render/kernels/csrc.so"
    so_sha = sha256_file(PROJECT / so_rel)
    exp = EXPECTED["std_csrc_sha256" if args.backend == "std_ll" else "higs_csrc_sha256"]
    binary_ok = (so_sha == exp)
    if sha256_file(PROJECT / 'gsplat/csrc.so') != EXPECTED['std_csrc_sha256']:
        raise RuntimeError('Frozen main extension SHA256 mismatch')
    if args.backend == 'std_ll' and not binary_ok:
        raise RuntimeError('Frozen std_ll SHA256 mismatch')
    if args.backend == 'higs':
        from gsplat.experimental.render.kernels import _backend
        if getattr(_backend._C, 'higs_true_absgrad_version', 0) != 1:
            raise RuntimeError('R20 experimental extension required')
        so_sha = sha256_file(Path(_backend._C.__file__))
        if so_sha == EXPECTED['higs_csrc_sha256']:
            raise RuntimeError('Refusing frozen HiGS binary as R20 candidate')
        identity = json.loads((PROJECT / 'R20_BUILD_IDENTITY.json').read_text())
        if so_sha != identity['binary_sha256']:
            raise RuntimeError('R20 candidate build identity mismatch')
        binary_ok = True

    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset

    os.makedirs(args.outdir, exist_ok=True)
    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO)
    config.iterations = args.iterations
    config.seed = args.seed

    # backend render fn ---------------------------------------------------
    use_absgrad = (args.densify_mode == "absgrad")
    if args.backend == "std_ll":
        from gsplat import rasterization
        def render(model, cam, sh_degree):
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                absgrad=use_absgrad)
            means2d = meta["means2d"]
            if means2d.requires_grad:
                means2d.retain_grad()
            return r[0].clamp(0, 1), meta["radii"][0], means2d
        # in signed mode, expose |signed grad| as .absgrad so densification
        # consumes the identical signal as the HiGS arm
        if not use_absgrad:
            _orig_render = render
            def render(model, cam, sh_degree, _r=_orig_render):
                img, radii, m2d = _r(model, cam, sh_degree)
                if m2d.requires_grad:
                    def hook(g, _t=m2d):
                        _t.absgrad = g.detach().abs()
                        return g
                    m2d.register_hook(hook)
                return img, radii, m2d
    else:
        from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
        def render(model, cam, sh_degree):
            out = rasterize_gaussian_higs_frozen(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                sh_degree=sh_degree, enable_culling=True, use_higs_culling=True,
                freeze_topology=False, render_mode="RGB", eps2d=0.1, absgrad=use_absgrad)
            info = out["densification_info"]
            means2d = info["means2d"]
            # The explicit signed arm keeps its historical hook. True absgrad
            # is supplied by the CUDA wrapper and must never be overwritten.
            if means2d.requires_grad:
                means2d.retain_grad()
                if not use_absgrad:
                    attach_absgrad_hook(means2d)
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            return frame.clamp(0, 1), info["radii"], means2d

    # dataset + model -----------------------------------------------------
    dataset = GTDataset(scene=args.scene, repo_root=REPO, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)

    # scene extent
    centers = []
    for i in range(min(50, n_cameras)):
        centers.append(dataset.get_camera(i).camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            extent = max(extent, float(np.linalg.norm(centers[i] - centers[j])))
    scene_extent = max(extent, 0.1)

    # SfM init: prefer a real COLMAP points3D.bin; fall back to the scene's
    # sparse PLY (xyz + f_dc color), which is the same SfM point set. Both
    # backends use the IDENTICAL init so the comparison is paired.
    sfm_bin = REPO / "data" / "datasets" / "mipnerf360" / args.scene / "sparse" / "0" / "points3D.bin"
    sparse_ply = REPO / "data" / "datasets" / "mipnerf360" / args.scene / "point_cloud_sparse.ply"

    def _is_real_binary(p):
        try:
            with open(p, "rb") as f:
                return f.read(7) != b"version"
        except Exception:
            return False

    init_src = None
    if sfm_bin.is_file() and _is_real_binary(sfm_bin):
        points3d = read_points3D_binary(str(sfm_bin))
        pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
        init_src = f"colmap_bin:{sfm_bin}"
    elif sparse_ply.is_file():
        pcd_data = sfm_from_ply(str(sparse_ply), sh_degree=config.sh_degree)
        init_src = f"sparse_ply:{sparse_ply}"
    else:
        raise FileNotFoundError(f"No SfM init for {args.scene}: {sfm_bin} / {sparse_ply}")
    print(f"  SfM init source: {init_src}")

    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = int(model._xyz.shape[0])
    model.training_setup({
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr, "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr, "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })

    ssim_fn = SepSSIM(device="cuda")
    lpips_fn = None
    if args.lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to("cuda").eval()
        except Exception as e:
            print(f"  WARNING LPIPS unavailable: {e}")

    # frozen camera sequence (identical both backends: same seed)
    # R21: train on train-split cameras only; hold out every Nth for eval.
    train_indices = [i for i in range(n_cameras) if i % args.heldout_mod != 0]
    test_indices = [i for i in range(n_cameras) if i % args.heldout_mod == 0]
    if not train_indices:
        raise RuntimeError(f"heldout-mod {args.heldout_mod} leaves no train cameras")
    print(f"  split: train={len(train_indices)} heldout={len(test_indices)} (mod {args.heldout_mod})", flush=True)
    rng = random.Random(config.seed)
    stack = list(train_indices)
    camera_sequence = []
    for _ in range(config.iterations):
        if not stack:
            stack = list(train_indices)
        camera_sequence.append(stack.pop(rng.randint(0, len(stack) - 1)))
    camera_sequence = np.array(camera_sequence, dtype=np.int32)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "camera_sequence.npy"), camera_sequence)
    np.save(os.path.join(args.outdir, "split_indices.npy"),
            {"train": np.array(train_indices, dtype=np.int32),
             "test": np.array(test_indices, dtype=np.int32)})

    if args.eval_every > 0:
        eval_iterations = set(range(args.eval_every, args.iterations + 1, args.eval_every))
    else:
        eval_iterations = set(i for i in config.eval_iterations if i <= args.iterations)
    # R21: train-view eval = 10 sampled TRAIN cameras; held-out eval = ALL held-out cameras
    rng_eval = random.Random(config.seed + 777)
    train_eval_pool = list(train_indices)
    rng_eval.shuffle(train_eval_pool)
    train_eval_indices = train_eval_pool[:max(1, len(train_eval_pool) // 10)]
    test_eval_indices = list(test_indices)

    def _eval_view(indices):
        psnrs, ssims, l1s, lp = [], [], [], []
        with torch.no_grad():
            for ci in indices:
                cam, gt = dataset.get_item(ci)
                img, _, _ = render(model, cam, model.active_sh_degree)
                psnrs.append(compute_psnr(img, gt))
                ssims.append(1.0 - ssim_fn(img, gt).item())
                l1s.append(float(F.l1_loss(img, gt).item()))
                if lpips_fn is not None:
                    pl = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                    gl = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
                    lp.append(float(lpips_fn(pl, gl).item()))
        row = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
               "l1": float(np.mean(l1s)), "n_eval_cameras": len(indices)}
        if lp:
            row["lpips"] = float(np.mean(lp))
        return row

    def evaluate(step):
        tr = _eval_view(train_eval_indices)
        te = _eval_view(test_eval_indices) if test_eval_indices else {"psnr": None, "ssim": None, "lpips": None, "n_eval_cameras": 0}
        row = {"step": step, "wall_time_s": None,
               "train_psnr": tr["psnr"], "train_ssim": tr["ssim"], "train_lpips": tr.get("lpips"),
               "test_psnr": te["psnr"], "test_ssim": te["ssim"], "test_lpips": te.get("lpips"),
               "n_gaussians": int(model._xyz.shape[0]),
               "n_train_eval": tr["n_eval_cameras"], "n_test_eval": te["n_eval_cameras"]}
        # backward-compatible top-level keys = held-out (the rigorous number)
        row["psnr"] = te["psnr"] if te["psnr"] is not None else tr["psnr"]
        row["ssim"] = te["ssim"] if te["ssim"] is not None else tr["ssim"]
        row["lpips"] = te.get("lpips") if te.get("lpips") is not None else tr.get("lpips")
        return row

    # training loop -------------------------------------------------------
    t_start = time.perf_counter()
    pev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    phase = {k: [] for k in ("forward", "loss", "backward", "densify", "optimizer")}
    per_iter, eval_rows, dens_events = [], [], []
    total_clones = total_splits = total_prunes = total_resets = 0
    peak_vram = 0.0
    first_topology_change = None

    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)

        pev[0].record()
        image, radii_full, means2d = render(model, cam, model.active_sh_degree)
        pev[1].record()
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        pev[2].record()
        loss.backward()
        pev[3].record()

        radii = radii_full
        # normalize radii to [N, 2] (drop any leading camera/batch dims)
        while radii.dim() > 2:
            radii = radii[0]
        if radii.dim() == 1:
            radii = radii.unsqueeze(-1)
        visibility_filter = (radii > 0).any(dim=-1)
        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)

        if iteration > config.densify_from_iter and iteration < config.densify_until_iter \
                and iteration % config.densification_interval == 0:
            n_before = int(model._xyz.shape[0])
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            ev = model.densify_and_prune(
                max_grad=config.densify_grad_threshold, min_opacity=config.min_opacity,
                extent=scene_extent, max_screen_size=size_threshold,
                radii=radii.float().max(dim=-1).values)
            n_after = int(model._xyz.shape[0])
            if n_after != n_before and first_topology_change is None:
                first_topology_change = {"iter": iteration, "n_before": n_before,
                                         "n_after": n_after, "cloned": ev["cloned"],
                                         "split": ev["split"], "pruned_total": ev["pruned_total"]}
            total_clones += ev["cloned"]; total_splits += ev["split"]
            total_prunes += ev["pruned_total"]
            dens_events.append({"iter": iteration, "n_before": n_before, "n_after": n_after,
                                "cloned": ev["cloned"], "split": ev["split"],
                                "pruned_total": ev["pruned_total"],
                                "pruned_opacity": ev["pruned_opacity"],
                                "loss": float(loss.item())})

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            total_resets += 1

        pev[4].record()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        pev[5].record()

        torch.cuda.synchronize()
        phase["forward"].append(pev[0].elapsed_time(pev[1]))
        phase["loss"].append(pev[1].elapsed_time(pev[2]))
        phase["backward"].append(pev[2].elapsed_time(pev[3]))
        phase["densify"].append(pev[3].elapsed_time(pev[4]))
        phase["optimizer"].append(pev[4].elapsed_time(pev[5]))
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024 ** 3)

        if iteration % 100 == 0 or iteration == 1:
            lv = float(loss.item())
            per_iter.append({"iter": iteration, "n_gs": int(model._xyz.shape[0]),
                             "loss": lv, "loss_nan": not math.isfinite(lv)})
        if iteration in eval_iterations:
            row = evaluate(iteration)
            row["wall_time_s"] = time.perf_counter() - t_start
            eval_rows.append(row)
            ckpt_iters = set(args.checkpoint_iters)
            if iteration in ckpt_iters:
                state = model.capture()
                state["iteration"] = iteration
                state["active_sh_degree"] = model.active_sh_degree
                torch.save(state, os.path.join(args.outdir, f"checkpoint_{iteration:06d}.pt"))
                row["checkpoint_saved"] = True
            print(f"  [{args.backend}/{args.scene} eval@{iteration}] testPSNR={row['test_psnr']:.2f} "
                  f"trainPSNR={row['train_psnr']:.2f} SSIM={row['ssim']:.4f} GS={row['n_gaussians']:,}", flush=True)

    total_time = time.perf_counter() - t_start
    phase_summary = {}
    for k, v in phase.items():
        if v:
            a = np.array(v)
            phase_summary[k] = {"mean_ms": float(np.mean(a)), "median_ms": float(np.median(a)),
                                "min_ms": float(np.min(a)), "max_ms": float(np.max(a))}
    final_eval = evaluate(config.iterations)
    final_eval["wall_time_s"] = total_time
    eval_rows.append(final_eval)
    # final checkpoint at iterations end (if not already saved)
    ckpt_set = set(args.checkpoint_iters)
    if config.iterations not in ckpt_set:
        state = model.capture(); state["iteration"] = config.iterations
        state["active_sh_degree"] = model.active_sh_degree
        torch.save(state, os.path.join(args.outdir, f"checkpoint_{config.iterations:06d}.pt"))
    saved_ckpts = sorted(f for f in os.listdir(args.outdir) if f.startswith("checkpoint_") and f.endswith(".pt"))

    out = {
        "backend": args.backend, "scene": args.scene, "iterations": config.iterations,
        "seed": args.seed, "binary_sha256": so_sha, "binary_matches_expected": so_sha == exp,
        "r20_binary_identity_valid": binary_ok, "scene_extent": scene_extent,
        "heldout_mod": args.heldout_mod,
        "n_train_cameras": len(train_indices), "n_test_cameras": len(test_indices),
        "initial_N": initial_N, "final_N": int(model._xyz.shape[0]),
        "total_time_s": total_time,
        "phase_summary_ms": phase_summary,
        "peak_vram_gb": peak_vram,
        "totals": {"clones": total_clones, "splits": total_splits,
                   "prunes": total_prunes, "opacity_resets": total_resets},
        "first_topology_change": first_topology_change,
        "checkpoints": saved_ckpts,
        "final_eval": final_eval,
        "eval_rows": eval_rows,
        "per_iter": per_iter,
        "n_dens_events": len(dens_events),
    }
    with open(os.path.join(args.outdir, "train_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.outdir, "eval.csv"), "w", newline="") as f:
        f.write("step,train_psnr,train_ssim,train_lpips,test_psnr,test_ssim,test_lpips,psnr,ssim,lpips,n_gaussians,wall_time_s\n")
        for r in eval_rows:
            def g(k):
                v = r.get(k)
                return f"{v:.6f}" if isinstance(v, (int, float)) else ""
            f.write(f"{r['step']},{g('train_psnr')},{g('train_ssim')},{g('train_lpips')},"
                    f"{g('test_psnr')},{g('test_ssim')},{g('test_lpips')},"
                    f"{g('psnr')},{g('ssim')},{g('lpips')},{r['n_gaussians']},{r['wall_time_s']:.4f}\n")
    print(f"  DONE {args.backend}/{args.scene}: N={initial_N:,}->{out['final_N']:,} "
          f"time={total_time:.1f}s testPSNR={final_eval['test_psnr']:.2f} trainPSNR={final_eval['train_psnr']:.2f}", flush=True)


if __name__ == "__main__":
    main()
