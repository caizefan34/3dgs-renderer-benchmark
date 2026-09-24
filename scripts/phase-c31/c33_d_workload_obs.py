#!/usr/bin/env python3
"""
C33-D: Workload Predictability Observation.

Records per-iteration workload metrics for 2+ full camera cycles (622+ iterations)
of Phase-7 3DGS training on room scene. No training modification — only logging.

Output: JSON with per-iteration workload vectors + camera metadata.
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

TILE_SIZE = 16

def make_opt(model, sls):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * sls},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15, betas=(0.9, 0.999))

def create(device="cuda:0", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dataset = GTDataset(scene="room", repo_root=ROOT, resolution="1080p", device=device)
    sfm = load_initial_checkpoint("room", ROOT, device=device)
    n_cam = len(dataset)
    model = GaussianModel(num_points=sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm.get("scales"), rotations_raw=sfm.get("rotations"), shs=sfm.get("shs"))
    sls = float(sfm["xyz"].norm(dim=-1).max().item())
    opt = make_opt(model, sls)
    return dataset, model, opt, n_cam, sls

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase-c31/c33_d_workload_data.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_iters", type=int, default=933)  # 3 full cycles of 311
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    print(f"=== C33-D: Workload Predictability ===")
    print(f"Device: {torch.cuda.get_device_properties(args.gpu).name}")
    print(f"Iters: {args.n_iters} (≈{args.n_iters/311:.1f} camera cycles)")

    dataset, model, opt, n_cam, sls = create(device=device, seed=42)

    records = []
    ev_fwd_start = torch.cuda.Event(enable_timing=True)
    ev_fwd_end = torch.cuda.Event(enable_timing=True)
    ev_bwd_start = torch.cuda.Event(enable_timing=True)
    ev_bwd_end = torch.cuda.Event(enable_timing=True)
    ev_total_start = torch.cuda.Event(enable_timing=True)
    ev_total_end = torch.cuda.Event(enable_timing=True)

    for step in range(args.n_iters):
        ci = step % n_cam
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Record topology state
        n_gaussians = model.xyz.shape[0]
        denf_phase = "before_start" if step < 500 else ("active" if step < 15000 else "after_end")

        # Forward + record metadata
        data = model.forward()
        ev_fwd_start.record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=True, sh_degree=0,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        ev_fwd_end.record()
        rendered = rendered[0].clamp(0, 1)

        # Extract workload features from renderer metadata
        radii = info.get("radii", torch.zeros(0, device=device))
        tiles_per_gauss = info.get("tiles_per_gauss", torch.zeros(0, device=device))
        isect_ids = info.get("isect_ids", torch.zeros(0, device=device))
        means2d = info.get("means2d", torch.zeros(0, 2, device=device))

        visible_mask = radii > 0
        n_visible = int(visible_mask.sum().item()) if visible_mask.numel() > 0 else 0
        n_intersections = int(tiles_per_gauss.sum().item()) if tiles_per_gauss.numel() > 0 else 0
        n_active_tiles = int(torch.unique(isect_ids).numel()) if isect_ids.numel() > 0 else 0

        radii_vals = radii[visible_mask] if visible_mask.numel() > 0 else radii
        r_mean = float(radii_vals.float().mean().item()) if radii_vals.numel() > 0 else 0
        r_std = float(radii_vals.float().std().item()) if radii_vals.numel() > 0 else 0
        r_min = float(radii_vals.float().min().item()) if radii_vals.numel() > 0 else 0
        r_max = float(radii_vals.float().max().item()) if radii_vals.numel() > 0 else 0
        r_p50 = float(radii_vals.float().median().item()) if radii_vals.numel() > 0 else 0

        # Tile dimensions
        tile_width = info.get("tile_width", 0)
        tile_height = info.get("tile_height", 0)
        n_total_tiles = tile_width * tile_height

        # Loss + backward (CUDA timed)
        ld = combined_loss(rendered, gt, lambda_dssim=0.2)
        loss = ld["loss"]
        ev_bwd_start.record()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ev_bwd_end.record()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # Synchronize CUDA events
        ev_fwd_end.synchronize()
        fwd_ms = ev_fwd_start.elapsed_time(ev_fwd_end)
        ev_bwd_end.synchronize()
        bwd_ms = ev_bwd_start.elapsed_time(ev_bwd_end)
        total_render_ms = fwd_ms + bwd_ms

        # Densification + pruning (same schedule as Phase-7)
        denf_cloned = denf_split = pruned = 0
        if step >= 500 and step < 15000 and step % 100 == 0:
            denf = model.densification(grad_threshold=2e-4, clone_max_screen_size=20, split_max_screen_size=20)
            denf_cloned = denf["cloned"]; denf_split = denf["split"]
            pruned = model.prune_and_reset(opacity_threshold=0.005, reset_interval=3000, current_step=step)
            if denf_cloned + denf_split + pruned > 0:
                # Rebuild optimizer fresh (discards momentum — acceptable for observation run)
                opt = make_opt(model, sls)

        record = {
            "step": step,
            "camera_id": ci,
            "n_gaussians": n_gaussians,
            "n_visible": n_visible,
            "n_intersections": n_intersections,
            "n_active_tiles": n_active_tiles,
            "n_total_tiles": n_total_tiles,
            "n_radii_features": int(radii_vals.numel()),
            "radii_mean": r_mean,
            "radii_std": r_std,
            "radii_min": r_min,
            "radii_max": r_max,
            "radii_p50": r_p50,
            "fwd_ms": round(fwd_ms, 3),
            "bwd_ms": round(bwd_ms, 3),
            "total_render_ms": round(total_render_ms, 3),
            "denf_phase": denf_phase,
            "denf_cloned": denf_cloned,
            "denf_split": denf_split,
            "pruned": pruned,
        }
        records.append(record)

        if step % 100 == 0 or step == args.n_iters - 1:
            print(f"  [{step:5d}/{args.n_iters}] cam={ci:3d} G={n_gaussians:7d} vis={n_visible:6d} "
                  f"isect={n_intersections:8d} tiles={n_active_tiles:5d}/{n_total_tiles:5d} "
                  f"fwd={fwd_ms:.2f}ms bwd={bwd_ms:.2f}ms"
                  f"{' [DENF]' if denf_cloned+denf_split+pruned > 0 else ''}",
                  flush=True)

    # ── Save ──
    output = {
        "config": {
            "scene": "room",
            "tile_size": TILE_SIZE,
            "packed": True,
            "seed": 42,
            "n_cameras": n_cam,
            "n_iters": args.n_iters,
        },
        "records": records,
        "camera_map": {str(i): {
            "image_name": dataset._valid_image_paths[i].stem if hasattr(dataset, '_valid_image_paths') and dataset._valid_image_paths else f"cam_{i}"
        } for i in range(min(n_cam, 311))},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=1, default=str)
    print(f"\nSaved: {args.out}")
    print(f"  Records: {len(records)}")
    print(f"  Camera cycles: {len(records)/n_cam:.2f}")

if __name__ == "__main__":
    main()
