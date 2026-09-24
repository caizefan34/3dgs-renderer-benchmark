#!/usr/bin/env python3
"""C30-F: True Cross-Iteration Predictability screening.

For the SAME training trajectory, measure correlation of workload
stats between consecutive iterations (not between different cameras).
"""
from __future__ import annotations
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=500)
    p.add_argument("--scene", default="room")
    args = p.parse_args()
    device = "cuda"; torch.manual_seed(42); np.random.seed(42)
    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"Initial Gs: {model.xyz.shape[0]:,}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])

    records = []
    for step in range(args.steps):
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)
        new_degree = min(3, step // 500)
        if new_degree != model.sh_degree:
            model.set_sh_degree(new_degree)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
        data = model.forward()
        rendered, ret, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=16, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        rendered = rendered[0].clamp(0, 1)

        rendered_pixels = (rendered.sum(dim=-1) > 0).sum().item()
        total_pixels = rendered.shape[0] * rendered.shape[1]
        nz_count = int(info.get("n_gaussians", -1))

        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()

        grad_norms = {}
        for name in ["xyz", "rotations", "scales", "opacity", "shs"]:
            p = getattr(model, name)
            grad_norms[name] = p.grad.norm().item() if p.grad is not None else 0.0

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); torch.cuda.synchronize()

        if step >= 200 and step % 100 == 0:
            model.densification(grad_threshold=2e-4)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
        if step >= 200 and step % 100 == 0:
            model.prune(opacity_threshold=0.005)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        records.append({
            "step": step, "loss": loss.item(), "psnr": psnr,
            "n_gaussians": model.xyz.shape[0],
            "active_pixel_ratio": rendered_pixels / max(total_pixels, 1),
            "isect_count": nz_count if nz_count >= 0 else None,
            "grad_norms": {k: round(v, 8) for k, v in grad_norms.items()},
            "camera_idx": cam_idx,
        })
        if step % 100 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}", flush=True)

    # Compute lag-1 correlations across CONSECUTIVE training steps (same trajectory)
    def _corr(vals):
        if np.std(vals) < 1e-10: return 0.0
        return float(np.corrcoef(vals[:-1], vals[1:])[0, 1])

    ngs = np.array([r["n_gaussians"] for r in records])
    acts = np.array([r["active_pixel_ratio"] for r in records])
    isects = np.array([float(r["isect_count"]) if r["isect_count"] else 0 for r in records])

    corr = {
        "n_gaussians_lag1": _corr(ngs),
        "active_pixel_lag1": _corr(acts),
        "isect_count_lag1": _corr(isects) if np.std(isects) > 1e-10 else 0.0,
    }

    # Also compute lag-k for k=2,3
    for lag in [2, 3]:
        if len(ngs) > lag:
            corr[f"n_gaussians_lag{lag}"] = float(np.corrcoef(ngs[:-lag], ngs[lag:])[0,1]) if np.std(ngs) > 1e-10 else 0.0
            corr[f"active_pixel_lag{lag}"] = float(np.corrcoef(acts[:-lag], acts[lag:])[0,1]) if np.std(acts) > 1e-10 else 0.0

    summary = {
        "schema_version": 2, "phase": "C30-F",
        "n_steps": args.steps, "scene": args.scene,
        "cross_iteration_corr": corr,
        "records": records,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
