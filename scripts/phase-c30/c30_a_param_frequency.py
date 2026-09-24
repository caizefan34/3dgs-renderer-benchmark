#!/usr/bin/env python3
"""C30-A: Parameter Update Frequency screening.

Hypothesis: Different parameter groups may not need updates at every iteration.
This script MEASURES only — does NOT alter update frequency.
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
    p.add_argument("--resolution", default="1080p")
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-dssim", type=float, default=0.2)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset
    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution=args.resolution, device=device)

    # Model
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"Initial Gs: {model.xyz.shape[0]:,}  spatial_lr_scale={spatial_lr_scale:.2f}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])

    param_groups = ["xyz", "rotations", "scales", "opacity", "shs"]
    param_attr = {"xyz": model.xyz, "rotations": model.rotations, "scales": model.scales,
                   "opacity": model.opacity, "shs": model.shs}
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

        # Capture parameter state BEFORE update
        with torch.no_grad():
            param_before = {}
            for g in param_groups:
                param_before[g] = param_attr[g].detach().clone()

        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        rendered = rendered[0].clamp(0, 1)
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=args.lambda_dssim)
        loss = loss_dict["loss"]

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        # Gradient norms per group (before clipping)
        grad_stats = {}
        for g in param_groups:
            grad = param_attr[g].grad
            if grad is not None:
                gn = grad.norm().item()
            else:
                gn = 0.0
            grad_stats[g] = gn

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        # Parameter update norms
        update_stats = {}
        with torch.no_grad():
            for g in param_groups:
                delta = (param_attr[g] - param_before[g]).norm().item()
                rel = delta / max(param_before[g].norm().item(), 1e-10)
                update_stats[g] = {"delta_norm": delta, "rel_change": rel}

        # Densification (same schedule as original)
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
            "step": step,
            "loss": loss.item(),
            "psnr": psnr,
            "n_gaussians": model.xyz.shape[0],
            "grad_norms": {g: round(v, 8) for g, v in grad_stats.items()},
            "update_norms": {g: round(update_stats[g]["delta_norm"], 8) for g in param_groups},
            "rel_changes": {g: round(update_stats[g]["rel_change"], 8) for g in param_groups},
            "bwd_ms": round(bwd_ms, 3),
            "opt_ms": round(opt_ms, 3),
            "sh_degree": model.sh_degree,
            "densified": denf_count["cloned"] + denf_count["split"],
            "pruned": prune_count,
        })

        if step % 100 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}", flush=True)

    # Aggregate: compute mean gradient/update across early/mid/late
    def _agg(recs, key):
        return {g: float(np.mean([r[key][g] for r in recs])) for g in param_groups}
    n = len(records)
    third = max(n // 3, 1)
    early, mid, late = records[:third], records[third:2*third], records[2*third:]
    summary = {
        "schema_version": 2, "phase": "C30-A",
        "n_steps": args.steps, "scene": args.scene,
        "param_group_stats": {g: {
            "early": {"mean_grad_norm": _agg(early, "grad_norms")[g],
                      "mean_update_norm": _agg(early, "update_norms")[g],
                      "mean_rel_change": _agg(early, "rel_changes")[g]},
            "mid": {"mean_grad_norm": _agg(mid, "grad_norms")[g],
                    "mean_update_norm": _agg(mid, "update_norms")[g],
                    "mean_rel_change": _agg(mid, "rel_changes")[g]},
            "late": {"mean_grad_norm": _agg(late, "grad_norms")[g],
                     "mean_update_norm": _agg(late, "update_norms")[g],
                     "mean_rel_change": _agg(late, "rel_changes")[g]},
        } for g in param_groups},
        "spread": max(_agg(records, "grad_norms").values()) / max(min(v for v in _agg(records, "grad_norms").values() if v > 0), 1e-10),
        "records": records,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
