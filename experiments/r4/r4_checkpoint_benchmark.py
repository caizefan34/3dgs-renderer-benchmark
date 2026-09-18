#!/usr/bin/env python3
"""
R4 Phase 6: Multi-Scene Checkpoint Skip Statistics

Measures skip mask statistics (skip fraction, gradient error) on existing
30K checkpoints for room, bicycle, garden.

For each scene + camera:
  - MODE0: baseline forward + backward
  - Compute skip_mask at various budgets
  - MODE2: skip-enabled backward
  - Report: skip_fraction, gradient_error (L2), timing

Usage:
  PYTHONNOUSERSITE=1 python experiments/r4/r4_checkpoint_benchmark.py \
    --output /mnt/storage_pool/liaoyuanjun/r4_checkpoint_bench
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
sys.path.insert(0, BUILD_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cuda")
    return {
        "xyz": ckpt["xyz"].cuda().float(),
        "rotations": ckpt["rotation"].cuda().float(),
        "scales": ckpt["scaling"].cuda().float(),
        "opacity": ckpt["opacity"].cuda().float(),
        "shs": ckpt["shs"].cuda().float(),
        "active_sh_degree": int(ckpt.get("active_sh_degree", 3)),
    }


def load_cameras(path):
    with open(path) as f:
        return json.load(f)


def get_camera(cameras, idx, images_dir, target_w=1920):
    from PIL import Image
    cam = cameras[idx]
    w, h = cam["width"], cam["height"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = w / 2.0, h / 2.0
    position = np.array(cam["position"], dtype=np.float32)
    rot = np.array(cam["rotation"], dtype=np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rot
    viewmat[:3, 3] = position
    viewmat = torch.from_numpy(viewmat).cuda().float()
    th = (int(h * target_w / w) // 16) * 16
    tw = (target_w // 16) * 16
    sx, sy = tw / w, th / h
    fx *= sx; fy *= sy; cx *= sx; cy *= sy
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    img_path = os.path.join(images_dir, cam["img_name"] + ".JPG")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, cam["img_name"] + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, cam["img_name"] + ".png")
    img_pil = Image.open(img_path)
    img_pil = img_pil.resize((tw, th), Image.LANCZOS)
    img = np.array(img_pil, dtype=np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().float()
    return viewmat, K, tw, th, gt


_CURRENT_SKIP_MASK = None
_EXT_SKIP = None


def patched_backward(ctx, v_render_colors, v_render_alphas):
    (means2d, conics, colors, opacities, backgrounds, masks,
     isect_offsets, flatten_ids, render_alphas, last_ids) = ctx.saved_tensors
    skip_mask = _CURRENT_SKIP_MASK
    if skip_mask is None:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
                means2d, conics, colors, opacities, backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids, render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(), ctx.absgrad)
    else:
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _EXT_SKIP.rasterize_to_pixels_bwd_with_skip(
                means2d, conics, colors, opacities, backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids, render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                skip_mask, ctx.absgrad)
    if ctx.absgrad and v_means2d_abs is not None:
        means2d.absgrad = v_means2d_abs
    v_backgrounds = None
    if ctx.needs_input_grad[4]:
        v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(-3, -2))
    return (v_means2d, v_conics, v_colors, v_opacities,
            v_backgrounds, None, None, None, None, None, None, None)


def run_backward(params, viewmat, K, w, h, gt, skip_mask=None):
    global _CURRENT_SKIP_MASK
    from gsplat import rasterization
    from gsplat.cuda._wrapper import _RasterizeToPixels
    p = {}
    for k, v in params.items():
        p[k] = v.clone().detach().requires_grad_(True) if k != "active_sh_degree" else v
    _CURRENT_SKIP_MASK = skip_mask
    original_backward = _RasterizeToPixels.backward
    if skip_mask is not None:
        _RasterizeToPixels.backward = staticmethod(patched_backward)
    try:
        r, a, meta = rasterization(
            means=p["xyz"], quats=p["rotations"], scales=p["scales"],
            opacities=p["opacity"], colors=p["shs"],
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=w, height=h, tile_size=16, packed=False,
            sh_degree=p["active_sh_degree"],
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True)
        image = r[0].clamp(0, 1)
        loss = F.l1_loss(image, gt)
        loss.backward()
        grads = {
            "v_xyz": p["xyz"].grad.clone(),
            "v_opacity": p["opacity"].grad.clone(),
            "v_shs": p["shs"].grad.clone(),
        }
        means2d = meta["means2d"]
        if means2d.grad is not None:
            grads["v_means2d"] = means2d.grad.clone()
        return loss.item(), grads, meta
    finally:
        _RasterizeToPixels.backward = original_backward
        _CURRENT_SKIP_MASK = None


def benchmark_scene(scene_name, checkpoint_path, cameras_json, images_dir, budgets, n_cameras=3):
    from compute_skip_mask import compute_skip_mask
    print(f"\n=== {scene_name} ===")
    print(f"  Checkpoint: {checkpoint_path}")
    params = load_checkpoint(checkpoint_path)
    print(f"  N Gaussians: {params['xyz'].shape[0]}")
    cameras = load_cameras(cameras_json)
    print(f"  N cameras: {len(cameras)}")

    results = {"scene": scene_name, "n_gaussians": params["xyz"].shape[0], "cameras": {}}

    for ci in range(min(n_cameras, len(cameras))):
        print(f"\n  Camera {ci}:")
        torch.cuda.empty_cache()  # Clear memory between cameras
        viewmat, K, w, h, gt = get_camera(cameras, ci, images_dir)

        # MODE0: baseline
        t0 = time.time()
        loss0, g0, meta = run_backward(params, viewmat, K, w, h, gt, skip_mask=None)
        t0_elapsed = time.time() - t0
        print(f"    MODE0: loss={loss0:.6f}, time={t0_elapsed:.3f}s")

        n_isects = meta["flatten_ids"].shape[0]
        print(f"    n_isects: {n_isects}")

        cam_results = {
            "loss": loss0,
            "n_isects": n_isects,
            "mode0_time_s": t0_elapsed,
            "budgets": {}
        }

        means2d = meta["means2d"].detach()
        conics = meta["conics"].detach()
        opacities = meta["opacities"].detach()
        isect_offsets = meta["isect_offsets"].detach()
        flatten_ids = meta["flatten_ids"].detach()
        shs = params["shs"].detach()
        if shs.dim() == 3:
            colors_dc = shs[:, 0, :].unsqueeze(0)
        elif shs.dim() == 4:
            colors_dc = shs[0, :, 0, :].unsqueeze(0)
        else:
            colors_dc = torch.ones(1, shs.shape[0], 3, device="cuda")

        for budget in budgets:
            print(f"    Budget {budget:.1%}:")
            skip_mask = compute_skip_mask(
                means2d, conics, colors_dc, opacities,
                isect_offsets, flatten_ids, 16, w, h, budget=budget)

            n_skipped = skip_mask.sum().item()
            skip_frac = n_skipped / n_isects

            # MODE2: skip-enabled backward
            t2 = time.time()
            loss2, g2, meta2 = run_backward(params, viewmat, K, w, h, gt, skip_mask=skip_mask)
            t2_elapsed = time.time() - t2

            # Gradient error
            grad_errors = {}
            for key in g0:
                if g0[key] is None or g2.get(key) is None:
                    continue
                d = (g0[key] - g2[key]).abs()
                grad_errors[key] = {
                    "max_abs": d.max().item(),
                    "l2": d.pow(2).mean().sqrt().item(),
                }

            cam_results["budgets"][f"{budget:.4f}"] = {
                "n_skipped": n_skipped,
                "skip_fraction": skip_frac,
                "mode2_time_s": t2_elapsed,
                "loss": loss2,
                "grad_errors": grad_errors,
            }
            print(f"      skipped: {n_skipped}/{n_isects} ({skip_frac:.2%}), "
                  f"time={t2_elapsed:.3f}s, loss={loss2:.6f}")
            for key in grad_errors:
                ge = grad_errors[key]
                print(f"        {key}: max_abs={ge['max_abs']:.2e} l2={ge['l2']:.2e}")

        results["cameras"][f"cam{ci}"] = cam_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cameras", type=int, default=3)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    global _EXT_SKIP
    try:
        import ext_skip
        _EXT_SKIP = ext_skip
        print(f"Loaded ext_skip: {ext_skip.__file__}")
    except ImportError as e:
        print(f"FATAL: Cannot load ext_skip: {e}")
        sys.exit(1)

    repo_root = Path.cwd()

    # Scene configurations
    scenes = [
        {
            "name": "room",
            "checkpoint": str(repo_root / "results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
            "cameras_json": str(repo_root / "data/official/mipnerf360/room/cameras.json"),
            "images_dir": str(repo_root / "data/datasets/mipnerf360/room/images"),
        },
        {
            "name": "bicycle",
            "checkpoint": str(repo_root / "results/reference_v1/s22/bicycle/checkpoints/baseline_iter_30000.pt"),
            "cameras_json": str(repo_root / "data/official/mipnerf360/bicycle/cameras.json"),
            "images_dir": str(repo_root / "data/datasets/mipnerf360/bicycle/images"),
        },
        {
            "name": "garden",
            "checkpoint": str(repo_root / "results/reference_v1/s22/garden/checkpoints/baseline_iter_30000.pt"),
            "cameras_json": str(repo_root / "data/official/mipnerf360/garden/cameras.json"),
            "images_dir": str(repo_root / "data/datasets/mipnerf360/garden/images"),
        },
    ]

    all_results = {"scenes": {}, "budgets": args.budgets}

    for scene in scenes:
        if not os.path.exists(scene["checkpoint"]):
            print(f"Skipping {scene['name']}: checkpoint not found at {scene['checkpoint']}")
            continue
        if not os.path.exists(scene["cameras_json"]):
            print(f"Skipping {scene['name']}: cameras not found at {scene['cameras_json']}")
            continue

        results = benchmark_scene(
            scene["name"], scene["checkpoint"],
            scene["cameras_json"], scene["images_dir"],
            args.budgets, n_cameras=args.n_cameras,
        )
        all_results["scenes"][scene["name"]] = results

    out_path = os.path.join(args.output, "checkpoint_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== Results saved to {out_path} ===")


if __name__ == "__main__":
    main()
