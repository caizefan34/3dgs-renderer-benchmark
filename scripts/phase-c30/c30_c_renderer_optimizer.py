#!/usr/bin/env python3
"""C30-C: Renderer/Optimizer Coupled Scheduling screening.

Hypothesis: Renderer workload stats (isect count, active pixels) can predict
optimizer usefulness, enabling joint scheduling.
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
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--scene", default="room")
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(42); np.random.seed(42)
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

        t0 = time.perf_counter()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=16, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000
        rendered = rendered[0].clamp(0, 1)

        # Collect renderer workload stats
        rendered_pixels = (rendered.sum(dim=-1) > 0).sum().item()
        total_pixels = rendered.shape[0] * rendered.shape[1]
        isect_count = info.get("n_gaussians", -1)
        tile_count = info.get("n_tiles", -1)

        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        # Gradient stats per group
        grad_norms = {}
        for name in ["xyz", "rotations", "scales", "opacity", "shs"]:
            p = getattr(model, name)
            grad_norms[name] = p.grad.norm().item() if p.grad is not None else 0.0

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        # Densification
        denf_count = {"cloned": 0, "split": 0, "removed": 0}
        if step >= 200 and step % 100 == 0:
            denf_count = model.densification(grad_threshold=2e-4)
            if denf_count["cloned"] + denf_count["split"] > 0:
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ])
        prune_count = 0
        if step >= 200 and step % 100 == 0:
            prune_count = model.prune(opacity_threshold=0.005)
            if prune_count > 0:
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
            "fwd_ms": round(fwd_ms, 3), "bwd_ms": round(bwd_ms, 3), "opt_ms": round(opt_ms, 3),
            "rendered_pixels": rendered_pixels, "total_pixels": total_pixels,
            "active_pixel_ratio": rendered_pixels / max(total_pixels, 1),
            "isect_count": int(isect_count) if isect_count >= 0 else None,
            "tile_count": int(tile_count) if tile_count >= 0 else None,
            "grad_norms": {k: round(v, 8) for k, v in grad_norms.items()},
            "sh_degree": model.sh_degree, "camera_idx": cam_idx,
            "densified": denf_count["cloned"] + denf_count["split"],
            "pruned": prune_count,
        })

        if step % 100 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,} "
                  f"fwd={fwd_ms:.2f} bwd={bwd_ms:.2f} opt={opt_ms:.2f}", flush=True)

    # Compute correlations between workload and gradient stats (pure numpy)
    def _corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10: return 0.0
        xm, ym = x - x.mean(), y - y.mean()
        r = (xm * ym).sum() / np.sqrt((xm**2).sum() * (ym**2).sum())
        return float(np.clip(r, -1.0, 1.0))

    fwd_times = np.array([r["fwd_ms"] for r in records])
    bwd_times = np.array([r["bwd_ms"] for r in records])
    opt_times = np.array([r["opt_ms"] for r in records])
    active_ratios = np.array([r["active_pixel_ratio"] for r in records])
    isects = np.array([r["isect_count"] if r["isect_count"] else 0 for r in records])
    ng = np.array([r["n_gaussians"] for r in records])

    corr = {
        "fwd_vs_active_pixels": _corr(fwd_times, active_ratios),
        "fwd_vs_isect": _corr(fwd_times, isects) if np.std(isects) > 0 else 0,
        "fwd_vs_n_gaussians": _corr(fwd_times, ng),
        "bwd_vs_active_pixels": _corr(bwd_times, active_ratios),
        "bwd_vs_isect": _corr(bwd_times, isects) if np.std(isects) > 0 else 0,
        "bwd_vs_n_gaussians": _corr(bwd_times, ng),
        "opt_vs_n_gaussians": _corr(opt_times, ng),
        "active_pixels_vs_n_gaussians": _corr(active_ratios, ng),
    }

    summary = {
        "schema_version": 2, "phase": "C30-C",
        "n_steps": args.steps, "scene": args.scene,
        "correlations": corr,
        "records": records,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
