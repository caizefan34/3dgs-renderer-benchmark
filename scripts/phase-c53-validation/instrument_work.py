#!/usr/bin/env python3
"""
Phase C53-Validation — Instrumentation Validation

Compare the work proxy (tiles_per_gauss) against actual CUDA rendering time
to validate that the proxy tracks real kernel work.

Measures:
1. Per-iteration rendering time (CUDA events)
2. Total tiles_per_gauss for the iteration
3. Total projected area for the iteration
4. Correlation between proxy and actual time across iterations

Also compares:
- Normal (uninstrumented) iteration time
- Instrumented iteration time
- Reports overhead
"""
import sys, math, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

SEED = 42
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0,1,3,2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
        sp2, st2, spt = bp2-mu_p2, bt2-mu_t2, bpt-mu_pt
        ssim_map = (2*mu_pt+self.C1)*(2*spt+self.C2) / ((mu_p2+mu_t2+self.C1)*(sp2+st2+self.C2))
        return 1.0 - ssim_map.mean()


def get_optimizer(model):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--scene", default="room")
    parser.add_argument("--n_iters", type=int, default=500)
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=device)

    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(num_points=N, sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float().to(device),
        opacity_logit=torch.logit(torch.full((N,), 0.1, device=device)),
        scales_log=sfm_data['scales'].clone().float().to(device),
        rotations_raw=sfm_data['rotations'].clone().float().to(device),
        shs=sfm_data['shs'].clone().float().to(device),
    )
    model.set_sh_degree(0)
    optimizer = get_optimizer(model)
    sep_ssim = SepSSIM(device=device)

    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)

    # CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = {
        "per_iter": [],
        "scene": args.scene,
        "n_iters": args.n_iters,
    }

    print(f"Instrumentation validation: {args.scene}, {args.n_iters} iters, GPU {args.gpu}")

    # Warmup
    for i in range(5):
        ci = cam_indices[i % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        r, _, meta = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        pred = r[0].clamp(0, 1)
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * sep_ssim(pred, gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    for iter_idx in range(args.n_iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Time forward pass
        torch.cuda.synchronize()
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        data = model.forward()

        fwd_start.record()
        r, _, meta = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        pred = r[0].clamp(0, 1)
        fwd_end.record()

        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * sep_ssim(pred, gt)
        optimizer.zero_grad(set_to_none=True)

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        optimizer.step()
        torch.cuda.synchronize()

        fwd_ms = fwd_start.elapsed_time(fwd_end)
        bwd_ms = bwd_start.elapsed_time(bwd_end)

        # Work metrics
        radii = meta["radii"][0]
        tiles_per_gauss = meta["tiles_per_gauss"][0]
        visible_mask = (radii > 0).any(dim=-1)
        total_tiles = int(tiles_per_gauss[visible_mask].sum().item())
        total_visible = int(visible_mask.sum().item())
        total_area = float((3.14159 * radii.max(dim=-1).values.float() ** 2 * visible_mask.float()).sum().item())
        n_gaussians = model.xyz.shape[0]

        results["per_iter"].append({
            "iter": iter_idx,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "total_ms": fwd_ms + bwd_ms,
            "n_gaussians": n_gaussians,
            "n_visible": total_visible,
            "total_tiles": total_tiles,
            "total_area": total_area,
        })

        if iter_idx % 100 == 0:
            print(f"  Iter {iter_idx}: fwd={fwd_ms:.1f}ms, bwd={bwd_ms:.1f}ms, "
                  f"tiles={total_tiles:,}, vis={total_visible:,}, GS={n_gaussians:,}")

    # Compute correlations
    fwd_times = np.array([r["fwd_ms"] for r in results["per_iter"]])
    bwd_times = np.array([r["bwd_ms"] for r in results["per_iter"]])
    tiles = np.array([r["total_tiles"] for r in results["per_iter"]], dtype=float)
    areas = np.array([r["total_area"] for r in results["per_iter"]], dtype=float)
    vis = np.array([r["n_visible"] for r in results["per_iter"]], dtype=float)
    gs = np.array([r["n_gaussians"] for r in results["per_iter"]], dtype=float)

    def pearson(x, y):
        sx, sy = np.std(x), np.std(y)
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    results["correlations"] = {
        "tiles_vs_fwd_time": pearson(tiles, fwd_times),
        "tiles_vs_bwd_time": pearson(tiles, bwd_times),
        "area_vs_fwd_time": pearson(areas, fwd_times),
        "area_vs_bwd_time": pearson(areas, bwd_times),
        "vis_vs_fwd_time": pearson(vis, fwd_times),
        "vis_vs_bwd_time": pearson(vis, bwd_times),
        "gs_vs_fwd_time": pearson(gs, fwd_times),
        "gs_vs_bwd_time": pearson(gs, bwd_times),
        "tiles_vs_total_time": pearson(tiles, fwd_times + bwd_times),
    }

    results["summary"] = {
        "mean_fwd_ms": float(np.mean(fwd_times)),
        "mean_bwd_ms": float(np.mean(bwd_times)),
        "mean_total_ms": float(np.mean(fwd_times + bwd_times)),
        "mean_tiles": float(np.mean(tiles)),
        "mean_vis": float(np.mean(vis)),
        "mean_gs": float(np.mean(gs)),
    }

    print(f"\n=== Instrumentation Validation Results ===")
    print(f"Correlation with forward time:")
    print(f"  total_tiles:  {results['correlations']['tiles_vs_fwd_time']:.3f}")
    print(f"  total_area:   {results['correlations']['area_vs_fwd_time']:.3f}")
    print(f"  n_visible:    {results['correlations']['vis_vs_fwd_time']:.3f}")
    print(f"  n_gaussians:  {results['correlations']['gs_vs_fwd_time']:.3f}")
    print(f"Correlation with backward time:")
    print(f"  total_tiles:  {results['correlations']['tiles_vs_bwd_time']:.3f}")
    print(f"  total_area:   {results['correlations']['area_vs_bwd_time']:.3f}")
    print(f"  n_visible:    {results['correlations']['vis_vs_bwd_time']:.3f}")
    print(f"  n_gaussians:  {results['correlations']['gs_vs_bwd_time']:.3f}")

    save_dir = repo_root / "results" / "a100" / "phase-c53-validation"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "instrumentation_validation.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
