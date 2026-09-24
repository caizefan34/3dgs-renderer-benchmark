#!/usr/bin/env python3
"""C30-E: Loss Evaluation Frequency screening.

Hypothesis: Some loss components (SSIM) may not need full evaluation every step.
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
from scripts.epic05.phase7.loss import combined_loss, d_ssim_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

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
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=16, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        rendered = rendered[0].clamp(0, 1)

        # Time individual loss components
        t0 = time.perf_counter()
        l1 = F.l1_loss(rendered, gt_image)
        l1_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dsim = d_ssim_loss(rendered, gt_image)
        dsim_ms = (time.perf_counter() - t0) * 1000

        loss = (1.0 - 0.2) * l1 + 0.2 * dsim

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

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
            "l1": l1.item(), "d_ssim": dsim.item(),
            "l1_cost_ms": round(l1_ms, 4), "d_ssim_cost_ms": round(dsim_ms, 4),
            "bwd_ms": round(bwd_ms, 3),
        })
        if step % 100 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} "
                  f"L1={l1.item():.4f}({l1_ms:.3f}ms) D-SSIM={dsim.item():.4f}({dsim_ms:.3f}ms)", flush=True)

    l1s = np.array([r["l1"] for r in records])
    dsims = np.array([r["d_ssim"] for r in records])
    def _acf(x, lag=1):
        if np.std(x) < 1e-10: return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if len(x) > lag else 0.0

    summary = {
        "schema_version": 2, "phase": "C30-E",
        "n_steps": args.steps, "scene": args.scene,
        "component_timing_ms": {
            "mean_l1_ms": float(np.mean([r["l1_cost_ms"] for r in records])),
            "mean_dssim_ms": float(np.mean([r["d_ssim_cost_ms"] for r in records])),
            "mean_bwd_ms": float(np.mean([r["bwd_ms"] for r in records])),
        },
        "temporal_corr": {
            "l1_lag1": _acf(l1s, 1), "l1_lag2": _acf(l1s, 2),
            "dssim_lag1": _acf(dsims, 1), "dssim_lag2": _acf(dsims, 2),
            "l1_dssim_cross": float(np.corrcoef(l1s, dsims)[0,1]) if np.std(l1s)>1e-10 and np.std(dsims)>1e-10 else 0.0,
        },
        "records": records,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
