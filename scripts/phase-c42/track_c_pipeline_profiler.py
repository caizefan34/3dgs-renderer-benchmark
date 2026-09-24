#!/usr/bin/env python3
"""
Track C: Full renderer pipeline profiler.

Profiles each stage of the 3DGS rendering pipeline:
1. Projection + Intersection (rasterization internal)
2. Sorting (isect_tiles sort=True vs sort=False difference)
3. Rasterization forward
4. Rasterization backward
5. Backward pipeline detail: Gaussian contribution distribution, tail analysis

Output: results/a100/phase-c42/track_c_pipeline_profile.json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization, isect_tiles, isect_offset_encode, fully_fused_projection
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
N_TIMING_ITERS = 50
SCENES = ["room", "bicycle", "garden"]

def cuda_timer(func, n_iters=50, warmup=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters

def profile_scene(scene, repo_root):
    print(f"\n{'='*60}")
    print(f"Profiling scene: {scene}")
    print(f"{'='*60}")

    dataset = GTDataset(scene=scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(scene, repo_root, device=DEVICE)
    n_gauss = sfm_data["xyz"].shape[0]
    print(f"  Gaussians: {n_gauss:,}")

    model = GaussianModel(num_points=n_gauss, sh_degree=3, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((n_gauss, 1), 0.1, device=DEVICE)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"))
    model.set_sh_degree(3)
    model.eval()

    results = {"scene": scene, "n_gaussians": n_gauss, "cameras": []}
    cam_indices = list(range(0, len(dataset), max(1, len(dataset) // 6)))[:6]

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        W, H = cam.image_width, cam.image_height
        print(f"\n  Camera {ci}: {W}x{H}")
        cam_result = {"camera_id": ci, "image_size": [W, H]}

        data = model.forward()
        viewmats = cam.viewmatrix.unsqueeze(0)
        Ks = cam.K.unsqueeze(0)
        tile_size = 16
        tile_width = (W + tile_size - 1) // tile_size
        tile_height = (H + tile_size - 1) // tile_size

        # ---- Stage 1+2: Projection + rasterization forward ----
        def do_raster_fwd():
            return rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                tile_size=tile_size, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")

        t_raster_fwd = cuda_timer(do_raster_fwd)
        renders, alphas, last_ids = do_raster_fwd()
        # Detach for analysis (avoid graph issues)
        renders_det = renders.detach()
        alphas_det = alphas.detach()
        print(f"    Rasterize fwd: {t_raster_fwd:.3f} ms")
        cam_result["rasterize_fwd_ms"] = t_raster_fwd

        # ---- Stage 3: Intersection timing (sort vs no-sort) ----
        # fully_fused_projection returns (radii, means2d, depths, conics, compensations)
        radii_proj, means2d, depths, conics, _ = fully_fused_projection(
            means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
            viewmats=viewmats, Ks=Ks, width=W, height=H, eps2d=0.1)

        n_visible = (radii_proj.sum(-1) > 0).sum().item()
        print(f"    Visible Gaussians: {n_visible}/{n_gauss}")
        cam_result["n_visible"] = n_visible

        def do_isect_nosort():
            return isect_tiles(means2d, radii_proj, depths, tile_size, tile_width, tile_height,
                             sort=False, packed=False)

        t_isect_nosort = cuda_timer(do_isect_nosort)
        tpg, iids_nosort, fids_nosort = do_isect_nosort()
        n_isects = fids_nosort.shape[0]
        print(f"    Isect(nosort): {t_isect_nosort:.3f} ms (n_isects: {n_isects:,})")
        cam_result["isect_nosort_ms"] = t_isect_nosort
        cam_result["n_isects"] = n_isects

        def do_isect_sort():
            return isect_tiles(means2d, radii_proj, depths, tile_size, tile_width, tile_height,
                             sort=True, packed=False)

        t_isect_sort = cuda_timer(do_isect_sort)
        tpg2, iids_sorted, fids_sorted = do_isect_sort()
        t_sort = t_isect_sort - t_isect_nosort
        print(f"    Isect(sort):   {t_isect_sort:.3f} ms (sort: {t_sort:.3f} ms)")
        cam_result["isect_sort_ms"] = t_isect_sort
        cam_result["sort_ms"] = t_sort

        # ---- Stage 4: Offset encoding ----
        def do_offset():
            return isect_offset_encode(iids_sorted, 1, tile_width, tile_height)
        t_offset = cuda_timer(do_offset)
        tile_offsets = do_offset()
        print(f"    Offset:        {t_offset:.3f} ms")
        cam_result["offset_ms"] = t_offset

        # ---- Stage 5: Rasterization backward ----
        v_renders = torch.randn_like(renders_det)
        v_alphas = torch.randn_like(alphas_det)

        def do_full_fwd_bwd():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            # Fresh forward to build new autograd graph
            data_fresh = model.forward()
            r, a, l = rasterization(
                means=data_fresh["xyz"], quats=data_fresh["rotations"], scales=data_fresh["scales"],
                opacities=data_fresh["opacity"], colors=data_fresh["shs"],
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                tile_size=tile_size, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            loss = (r * v_renders).sum() + (a * v_alphas).sum()
            loss.backward()

        t_full_fwd_bwd = cuda_timer(do_full_fwd_bwd, n_iters=20, warmup=5)
        t_bwd_only = t_full_fwd_bwd - t_raster_fwd
        print(f"    Full(fwd+bwd): {t_full_fwd_bwd:.3f} ms (bwd only: {t_bwd_only:.3f} ms)")
        cam_result["full_fwd_bwd_ms"] = t_full_fwd_bwd
        cam_result["rasterize_bwd_ms"] = t_bwd_only

        # ---- Full training iteration (fwd + loss + bwd) ----
        def do_train_iter():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            data_fresh = model.forward()
            r, a, l = rasterization(
                means=data_fresh["xyz"], quats=data_fresh["rotations"], scales=data_fresh["scales"],
                opacities=data_fresh["opacity"], colors=data_fresh["shs"],
                viewmats=viewmats, Ks=Ks, width=W, height=H,
                tile_size=tile_size, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            loss = F.l1_loss(r, gt.unsqueeze(0))
            loss.backward()

        t_train_iter = cuda_timer(do_train_iter, n_iters=20, warmup=5)
        print(f"    Train iter:    {t_train_iter:.3f} ms")
        cam_result["train_iter_ms"] = t_train_iter

        # ---- Backward analysis: Gaussian intersection distribution ----
        gauss_isect_counts = torch.bincount(fids_sorted, minlength=n_gauss)
        contrib_stats = {
            "mean": float(gauss_isect_counts.float().mean()),
            "median": float(gauss_isect_counts.float().median()),
            "p10": float(gauss_isect_counts.float().quantile(0.10)),
            "p25": float(gauss_isect_counts.float().quantile(0.25)),
            "p75": float(gauss_isect_counts.float().quantile(0.75)),
            "p90": float(gauss_isect_counts.float().quantile(0.90)),
            "p95": float(gauss_isect_counts.float().quantile(0.95)),
            "p99": float(gauss_isect_counts.float().quantile(0.99)),
            "max": int(gauss_isect_counts.max()),
            "zeros": int((gauss_isect_counts == 0).sum()),
        }

        sorted_counts = gauss_isect_counts.sort()[0]
        total_isects = sorted_counts.sum().item()
        cumulative = sorted_counts.cumsum(0).float() / total_isects
        bottom_1pct = int((cumulative < 0.01).sum())
        bottom_5pct = int((cumulative < 0.05).sum())
        bottom_10pct = int((cumulative < 0.10).sum())

        top_counts = sorted_counts.flip(0)
        top_cumulative = top_counts.cumsum(0).float() / total_isects
        top_1pct_isects = float(top_cumulative[min(int(n_gauss * 0.01), n_gauss - 1)])
        top_5pct_isects = float(top_cumulative[min(int(n_gauss * 0.05), n_gauss - 1)])
        top_10pct_isects = float(top_cumulative[min(int(n_gauss * 0.10), n_gauss - 1)])

        print(f"\n    Gaussian intersection distribution:")
        print(f"      mean={contrib_stats['mean']:.1f} median={contrib_stats['median']:.1f} max={contrib_stats['max']}")
        print(f"      zeros={contrib_stats['zeros']} ({100*contrib_stats['zeros']/n_gauss:.1f}%)")
        print(f"      Bottom 1% of isects from {bottom_1pct} Gaussians ({100*bottom_1pct/n_gauss:.1f}% of all)")
        print(f"      Bottom 5% of isects from {bottom_5pct} Gaussians ({100*bottom_5pct/n_gauss:.1f}% of all)")
        print(f"      Top 1% Gaussians -> {100*top_1pct_isects:.1f}% of isects")
        print(f"      Top 5% Gaussians -> {100*top_5pct_isects:.1f}% of isects")

        cam_result["gauss_isect_distribution"] = contrib_stats
        cam_result["tail_analysis"] = {
            "bottom_1pct_isects_from_n_gauss": bottom_1pct,
            "bottom_5pct_isects_from_n_gauss": bottom_5pct,
            "bottom_10pct_isects_from_n_gauss": bottom_10pct,
            "bottom_1pct_gauss_fraction": bottom_1pct / n_gauss,
            "bottom_5pct_gauss_fraction": bottom_5pct / n_gauss,
            "top_1pct_isect_fraction": top_1pct_isects,
            "top_5pct_isect_fraction": top_5pct_isects,
            "top_10pct_isect_fraction": top_10pct_isects,
        }

        # ---- Per-pixel coverage analysis ----
        alpha_vals = alphas_det[0]
        alpha_stats = {
            "mean": float(alpha_vals.mean()),
            "median": float(alpha_vals.median()),
            "p90": float(alpha_vals.quantile(0.90)),
            "p99": float(alpha_vals.quantile(0.99)),
            "pixels_done_early": float((alpha_vals > 0.99).float().mean()),
        }
        print(f"\n    Alpha (render coverage): mean={alpha_stats['mean']:.3f} done_early={alpha_stats['pixels_done_early']:.3f}")
        cam_result["alpha_stats"] = alpha_stats

        # ---- Per-tile intersection distribution ----
        tile_ids_extracted = (iids_sorted >> 32).int() & ((1 << 13) - 1)
        tile_isect_counts = torch.bincount(tile_ids_extracted, minlength=tile_width * tile_height)
        tile_stats = {
            "mean": float(tile_isect_counts.float().mean()),
            "p50": float(tile_isect_counts.float().quantile(0.50)),
            "p90": float(tile_isect_counts.float().quantile(0.90)),
            "p95": float(tile_isect_counts.float().quantile(0.95)),
            "p99": float(tile_isect_counts.float().quantile(0.99)),
            "max": int(tile_isect_counts.max()),
        }
        print(f"    Tile isect dist: mean={tile_stats['mean']:.0f} p99={tile_stats['p99']:.0f} max={tile_stats['max']}")
        cam_result["tile_isect_dist"] = tile_stats

        # ---- Backward atomicAdd contention estimate ----
        radii_data = radii_proj[0]  # [N, 2]
        visible_mask = (radii_data.sum(-1) > 0)
        radii_area = (radii_data[:, 0].float() * radii_data[:, 1].float())
        tiles_per_g = tpg[0].float()
        pixels_per_tile_per_g = radii_area / (tiles_per_g.clamp(min=1) * tile_size * tile_size)
        pixels_per_tile_per_g = pixels_per_tile_per_g.clamp(max=tile_size * tile_size)

        per_gauss_writes = (pixels_per_tile_per_g * tiles_per_g)
        per_gauss_writes_vis = per_gauss_writes[visible_mask]
        total_bwd_writes = float(per_gauss_writes_vis.sum())
        n_pixels = W * H
        bw_fw_ratio = total_bwd_writes / n_pixels if n_pixels > 0 else 0

        write_stats = {
            "total_backward_writes": total_bwd_writes,
            "forward_writes": n_pixels,
            "bw_to_fwd_ratio": bw_fw_ratio,
            "per_gauss_writes_mean": float(per_gauss_writes_vis.mean()),
            "per_gauss_writes_p50": float(per_gauss_writes_vis.quantile(0.50)),
            "per_gauss_writes_p90": float(per_gauss_writes_vis.quantile(0.90)),
            "per_gauss_writes_p99": float(per_gauss_writes_vis.quantile(0.99)),
            "per_gauss_writes_max": float(per_gauss_writes_vis.max()),
        }
        print(f"\n    Backward atomic write estimate:")
        print(f"      Total bwd writes: {total_bwd_writes:,.0f} vs fwd writes: {n_pixels:,} (ratio: {bw_fw_ratio:.1f}x)")
        print(f"      Per-Gauss writes: mean={write_stats['per_gauss_writes_mean']:.1f} p99={write_stats['per_gauss_writes_p99']:.1f}")
        cam_result["backward_write_analysis"] = write_stats

        # ---- Pipeline breakdown ----
        # Estimate: projection = raster_fwd - isect_sort - rasterize_only
        # raster_fwd includes projection + isect + sort + rasterize
        # We can estimate: projection + rasterize_only = raster_fwd - isect_sort - offset
        t_proj_raster = t_raster_fwd - t_isect_sort - t_offset
        total = t_raster_fwd + t_bwd_only
        cam_result["pipeline_breakdown"] = {
            "rasterize_fwd_ms": t_raster_fwd,
            "isect_nosort_ms": t_isect_nosort,
            "sort_ms": t_sort,
            "offset_ms": t_offset,
            "proj_plus_raster_ms": t_proj_raster,
            "rasterize_bwd_ms": t_bwd_only,
            "total_ms": total,
            "isect_sort_pct": t_isect_sort / total * 100,
            "proj_raster_pct": t_proj_raster / total * 100,
            "rasterize_fwd_pct": t_raster_fwd / total * 100,
            "rasterize_bwd_pct": t_bwd_only / total * 100,
        }
        print(f"\n    Pipeline breakdown (fwd+bwd):")
        print(f"      Isect+sort:   {t_isect_sort:6.3f} ms ({100*t_isect_sort/total:.1f}%)")
        print(f"      Offset:       {t_offset:6.3f} ms ({100*t_offset/total:.1f}%)")
        print(f"      Proj+Raster:  {t_proj_raster:6.3f} ms ({100*t_proj_raster/total:.1f}%)")
        print(f"      Backward:     {t_bwd_only:6.3f} ms ({100*t_bwd_only/total:.1f}%)")
        print(f"      Total:        {total:6.3f} ms")

        results["cameras"].append(cam_result)

    return results

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    torch.manual_seed(42)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    import gsplat
    print(f"gsplat: {gsplat.__version__}")

    all_results = {"gpu": gpu_name, "scenes": {}}
    for scene in SCENES:
        try:
            results = profile_scene(scene, repo_root)
            all_results["scenes"][scene] = results
        except Exception as e:
            print(f"  ERROR profiling {scene}: {e}")
            import traceback; traceback.print_exc()
            all_results["scenes"][scene] = {"error": str(e)}

    save_path = repo_root / "results" / "a100" / "phase-c42" / "track_c_pipeline_profile.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Pipeline breakdown (avg across cameras)")
    print(f"{'='*60}")
    for scene, res in all_results["scenes"].items():
        if "error" in res:
            print(f"  {scene}: ERROR"); continue
        cams = res["cameras"]
        if not cams: continue
        avg = {k: np.mean([c["pipeline_breakdown"].get(k, 0) for c in cams]) for k in
               ["isect_nosort_ms", "sort_ms", "offset_ms", "proj_plus_raster_ms", "rasterize_bwd_ms", "total_ms"]}
        print(f"  {scene} ({res['n_gaussians']:,} Gaussians):")
        print(f"    Isect:     {avg['isect_nosort_ms']:6.2f} ms ({100*avg['isect_nosort_ms']/avg['total_ms']:.1f}%)")
        print(f"    Sort:      {avg['sort_ms']:6.2f} ms ({100*avg['sort_ms']/avg['total_ms']:.1f}%)")
        print(f"    Offset:    {avg['offset_ms']:6.2f} ms ({100*avg['offset_ms']/avg['total_ms']:.1f}%)")
        print(f"    Proj+Rast: {avg['proj_plus_raster_ms']:6.2f} ms ({100*avg['proj_plus_raster_ms']/avg['total_ms']:.1f}%)")
        print(f"    Backward:  {avg['rasterize_bwd_ms']:6.2f} ms ({100*avg['rasterize_bwd_ms']/avg['total_ms']:.1f}%)")
        print(f"    Total:     {avg['total_ms']:6.2f} ms")

if __name__ == "__main__":
    main()
