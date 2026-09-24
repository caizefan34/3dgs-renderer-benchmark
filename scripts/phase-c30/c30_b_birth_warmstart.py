#!/usr/bin/env python3
"""C30-B: Gaussian Birth Warm-Start screening.

Hypothesis: Newly created Gaussians via densification may have a predictable
transient optimization regime distinct from established Gaussians.
"""
from __future__ import annotations
import argparse, gc, json, math, sys, time, copy
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
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
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
    print(f"Initial Gs: {model.xyz.shape[0]:,}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])

    # Track birth events and follow new Gaussians
    birth_events = []  # each: {step, n_before, n_after, new_indices, post_steps: [...]}

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

        n_before = model.xyz.shape[0]
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
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()

        # Record gradient norms for ALL Gaussians before densification
        old_grad_norms = {}
        for name in ["xyz", "rotations", "scales", "opacity", "shs"]:
            p = getattr(model, name)
            if p.grad is not None:
                old_grad_norms[name] = p.grad.norm(dim=-1).detach().clone() if p.grad.dim() > 1 else p.grad.norm().unsqueeze(0)

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()

        # Densification
        this_birth = None
        if step >= 200 and step % 100 == 0:
            n_before_denf = model.xyz.shape[0]
            denf_count = model.densification(grad_threshold=2e-4)
            n_after_denf = model.xyz.shape[0]

            if n_after_denf > n_before_denf:
                n_new = n_after_denf - n_before_denf
                # Track new Gaussians: they are at the end
                this_birth = {"step": step, "n_before": n_before_denf, "n_after": n_after_denf,
                              "n_new": n_new, "new_indices": list(range(n_before_denf, n_after_denf)),
                              "post_tracking": []}
                print(f"  Birth event at step {step}: +{n_new} Gs (total: {n_after_denf:,})", flush=True)
                if denf_count["cloned"] + denf_count["split"] > 0:
                    optimizer = torch.optim.Adam([
                        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                        {"params": [model.rotations], "lr": 1e-3},
                        {"params": [model.scales], "lr": 5e-3},
                        {"params": [model.opacity], "lr": 5e-2},
                        {"params": [model.shs], "lr": 2.5e-3},
                    ])

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

        # Track recent births in following steps
        for ev in reversed(birth_events[-5:]):
            if ev is this_birth or len(ev["post_tracking"]) >= 50:
                continue
            # Only track if indices are still valid
            max_idx = model.xyz.shape[0]
            valid_new = [i for i in ev["new_indices"] if i < max_idx]
            if not valid_new:
                continue
            with torch.no_grad():
                new_xyz = model.xyz[valid_new].norm(dim=-1).mean().item()
                new_opacity = torch.sigmoid(model.opacity[valid_new]).mean().item()
                new_scale = torch.exp(model.scales[valid_new]).mean().item()
                ev["post_tracking"].append({
                    "step": step,
                    "n_surviving_new": len(valid_new),
                    "mean_xyz_norm": new_xyz,
                    "mean_opacity": new_opacity,
                    "mean_scale": new_scale,
                })

        if this_birth:
            birth_events.append(this_birth)

        if step % 100 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}", flush=True)

    summary = {
        "schema_version": 2, "phase": "C30-B",
        "n_steps": args.steps, "scene": args.scene,
        "n_birth_events": len(birth_events),
        "birth_events": birth_events,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
