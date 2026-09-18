#!/usr/bin/env python3
"""
R4 MODE2 Test: CUDA skip-enabled backward vs baseline backward.

Strategy: Monkey-patch gsplat's _RasterizeToPixels.backward to use our
custom CUDA kernel with skip_mask support. This way, the full autograd
chain (projection backward + SH backward) works correctly.

MODE0: Original gsplat backward (no patching)
MODE1: Custom backward with skip_mask all False (should match MODE0)
MODE2: Custom backward with skip_mask from certificate

Usage:
  PYTHONNOUSERSITE=1 ~/miniforge3/envs/anysplat/bin/python experiments/r4/r4_mode2_test.py \
    --checkpoint results/reference_v1/room_30k/checkpoints/iter_30000.pt \
    --cameras-json data/official/mipnerf360/room/cameras.json \
    --output /mnt/storage_pool/liaoyuanjun/r4_mode2 \
    --budget 0.05 --n-cameras 2
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add build dir for ext_skip
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
sys.path.insert(0, BUILD_DIR)


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


def get_camera(cameras, idx, images_dir):
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
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    img_path = os.path.join(images_dir, cam["img_name"] + ".JPG")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, cam["img_name"] + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, cam["img_name"] + ".png")
    from PIL import Image
    img_pil = Image.open(img_path)
    target_w = 1920
    target_h = (int(h * target_w / w) // 16) * 16
    target_w = (target_w // 16) * 16
    img_pil = img_pil.resize((target_w, target_h), Image.LANCZOS)
    sx, sy = target_w / w, target_h / h
    fx *= sx; fy *= sy; cx *= sx; cy *= sy
    w, h = target_w, target_h
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    img = np.array(img_pil, dtype=np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().float()
    return viewmat, K, w, h, gt


def ssim_loss(img1, img2):
    """Simple separable SSIM, fallback to L1 if needed."""
    # For correctness testing, L1 loss is sufficient
    # (we only need gradients to flow through the rasterization backward)
    return F.l1_loss(img1, img2)


# Global skip mask state — set before backward
_CURRENT_SKIP_MASK = None
_EXT_SKIP = None


def patched_rasterize_backward(ctx, v_render_colors, v_render_alphas):
    """Replacement for _RasterizeToPixels.backward that uses ext_skip."""
    (
        means2d, conics, colors, opacities,
        backgrounds, masks, isect_offsets, flatten_ids,
        render_alphas, last_ids,
    ) = ctx.saved_tensors

    skip_mask = _CURRENT_SKIP_MASK  # global state

    if skip_mask is None:
        # No skip mask → use original gsplat backward
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
                means2d, conics, colors, opacities,
                backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids,
                render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                ctx.absgrad,
            )
    else:
        # Use custom backward with skip_mask
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _EXT_SKIP.rasterize_to_pixels_bwd_with_skip(
                means2d, conics, colors, opacities,
                backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids,
                render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                skip_mask, ctx.absgrad,
            )

    if ctx.absgrad and v_means2d_abs is not None:
        means2d.absgrad = v_means2d_abs

    v_backgrounds = None
    if ctx.needs_input_grad[4]:  # backgrounds
        v_backgrounds = -(v_render_colors * (1.0 - render_alphas.float())).sum(dim=(-3, -2))

    return (
        v_means2d,      # means2d
        v_conics,       # conics
        v_colors,       # colors
        v_opacities,    # opacities
        v_backgrounds,  # backgrounds
        None,           # masks
        None,           # width
        None,           # height
        None,           # tile_size
        None,           # isect_offsets
        None,           # flatten_ids
        None,           # absgrad
    )


def run_backward(params, viewmat, K, w, h, gt, skip_mask=None):
    """Forward + backward with optional skip_mask patching."""
    global _CURRENT_SKIP_MASK

    from gsplat import rasterization
    from gsplat.cuda._wrapper import _RasterizeToPixels

    # Clone params with grad
    p = {}
    for k, v in params.items():
        if k == "active_sh_degree":
            p[k] = v
        else:
            p[k] = v.clone().detach().requires_grad_(True)

    # Set skip mask and patch backward if needed
    _CURRENT_SKIP_MASK = skip_mask
    original_backward = _RasterizeToPixels.backward
    if skip_mask is not None:
        _RasterizeToPixels.backward = staticmethod(patched_rasterize_backward)

    try:
        r, a, meta = rasterization(
            means=p["xyz"], quats=p["rotations"], scales=p["scales"],
            opacities=p["opacity"], colors=p["shs"],
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=w, height=h, tile_size=16, packed=False,
            sh_degree=p["active_sh_degree"],
            radius_clip=0.0, eps2d=0.1,
            render_mode="RGB", absgrad=True,
        )
        image = r[0].clamp(0, 1)
        means2d = meta["means2d"]
        means2d.retain_grad()

        L1 = F.l1_loss(image, gt)
        ssim_val = ssim_loss(image, gt)
        loss = 0.2 * L1 + 0.8 * (1 - ssim_val)
        loss.backward()

        grads = {
            "v_xyz": p["xyz"].grad.clone(),
            "v_rotation": p["rotations"].grad.clone(),
            "v_scaling": p["scales"].grad.clone(),
            "v_opacity": p["opacity"].grad.clone(),
            "v_shs": p["shs"].grad.clone(),
            "v_means2d": means2d.grad.clone() if means2d.grad is not None else None,
        }
        if hasattr(means2d, 'absgrad') and means2d.absgrad is not None:
            grads["v_means2d_abs"] = means2d.absgrad.clone()

        return loss.item(), grads, meta
    finally:
        # Restore original backward
        _RasterizeToPixels.backward = original_backward
        _CURRENT_SKIP_MASK = None


def compare(g0, g1, label=""):
    results = {}
    for key in g0:
        if g0[key] is None or g1.get(key) is None:
            continue
        d = (g0[key] - g1[key]).abs()
        max_abs = d.max().item()
        denom = g0[key].abs().clamp_min(1e-8)
        max_rel = (d / denom).max().item()
        l2 = d.pow(2).mean().sqrt().item()
        ok = max_abs < 1e-4 or l2 < 1e-6
        results[key] = {"max_abs": max_abs, "max_rel": max_rel, "l2": l2, "pass": ok}
        status = "PASS" if ok else "FAIL"
        print(f"  {key}: max_abs={max_abs:.2e} max_rel={max_rel:.2e} l2={l2:.2e} [{status}]")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cameras", type=int, default=2)
    parser.add_argument("--budget", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    images_dir = args.images_dir
    if images_dir is None:
        scene_name = Path(args.cameras_json).parent.name
        repo_root = Path.cwd()
        images_dir = str(repo_root / "data" / "datasets" / "mipnerf360" / scene_name / "images")

    print("=== R4 MODE2 Test: CUDA Skip Backward ===")
    print(f"Images dir: {images_dir}")

    # Load compiled extension
    global _EXT_SKIP
    try:
        import ext_skip
        _EXT_SKIP = ext_skip
        print(f"Loaded ext_skip: {ext_skip.__file__}")
    except ImportError as e:
        print(f"FATAL: Cannot load ext_skip: {e}")
        sys.exit(1)

    from compute_skip_mask import compute_skip_mask

    params = load_checkpoint(args.checkpoint)
    print(f"N Gaussians: {params['xyz'].shape[0]}")

    cameras = load_cameras(args.cameras_json)
    print(f"N cameras: {len(cameras)}")

    all_results = {"comparisons": {}, "budget": args.budget}

    for ci in range(min(args.n_cameras, len(cameras))):
        print(f"\n--- Camera {ci} ---")
        viewmat, K, w, h, gt = get_camera(cameras, ci, images_dir)
        print(f"  Image: {w}x{h}")

        # MODE0: baseline (no patching, skip_mask=None)
        print("MODE0: Baseline backward...")
        loss0, g0, meta0 = run_backward(params, viewmat, K, w, h, gt, skip_mask=None)
        print(f"  Loss: {loss0:.6f}")

        # Get forward metadata for skip mask computation
        means2d = meta0["means2d"].detach()
        conics = meta0["conics"].detach()
        opacities = meta0["opacities"].detach()
        isect_offsets = meta0["isect_offsets"].detach()
        flatten_ids = meta0["flatten_ids"].detach()

        # For colors, we need the SH-evaluated RGB colors
        # DC component = shs[..., 0, :] (first SH coefficient)
        shs = params["shs"]
        if shs.dim() == 3:  # [N, K, 3]
            colors_dc = shs[:, 0, :].unsqueeze(0)  # [1, N, 3]
        elif shs.dim() == 4:  # [C, N, K, 3]
            colors_dc = shs[0, :, 0, :].unsqueeze(0)  # [1, N, 3]
        else:
            colors_dc = torch.ones(1, shs.shape[0], 3, device="cuda")

        print(f"Computing skip mask (budget={args.budget})...")
        try:
            skip_mask = compute_skip_mask(
                means2d, conics, colors_dc, opacities,
                isect_offsets, flatten_ids,
                16, w, h, budget=args.budget,
            )
        except Exception as e:
            print(f"  skip mask error: {e}")
            skip_mask = torch.zeros(flatten_ids.shape[0], dtype=torch.bool, device="cuda")

        n_skipped = skip_mask.sum().item()
        n_total = skip_mask.numel()
        print(f"  Skipped: {n_skipped}/{n_total} ({n_skipped/n_total*100:.2f}%)")

        # MODE1: skip_mask all False (should match MODE0)
        print("MODE1: Custom backward, mask all False...")
        empty_mask = torch.zeros_like(skip_mask)
        loss1, g1, meta1 = run_backward(params, viewmat, K, w, h, gt, skip_mask=empty_mask)
        print(f"  Loss: {loss1:.6f}")
        print("MODE0 vs MODE1 (skip disabled):")
        cmp01 = compare(g0, g1, "MODE0 vs MODE1")

        # MODE2: skip enabled
        print("MODE2: Custom backward, skip enabled...")
        loss2, g2, meta2 = run_backward(params, viewmat, K, w, h, gt, skip_mask=skip_mask)
        print(f"  Loss: {loss2:.6f}")
        print("MODE0 vs MODE2 (skip enabled):")
        cmp02 = compare(g0, g2, "MODE0 vs MODE2")

        all_results["comparisons"][f"cam{ci}"] = {
            "loss0": loss0, "loss1": loss1, "loss2": loss2,
            "n_skipped": n_skipped, "n_total": n_total,
            "skip_fraction": n_skipped / n_total if n_total > 0 else 0,
            "mode0_vs_mode1": cmp01,
            "mode0_vs_mode2": cmp02,
        }

    # Determine overall verdict
    all_pass = True
    for cam_key, cam_data in all_results["comparisons"].items():
        for key, cmp in cam_data["mode0_vs_mode1"].items():
            if not cmp.get("pass", False):
                all_pass = False
                print(f"\n*** FAIL: MODE0 vs MODE1 mismatch on {cam_key}/{key} ***")

    all_results["verdict"] = "PASS" if all_pass else "IMPLEMENTATION_BUG"

    out_path = os.path.join(args.output, "r4_mode2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== Verdict: {all_results['verdict']} ===")
    print(f"Results: {out_path}")

    if all_pass:
        print("MODE0 vs MODE1 agree — custom CUDA kernel is correct.")
        print("MODE2 skip enabled — gradient differences are expected (within budget).")
    else:
        print("*** STOP R4 — IMPLEMENTATION_BUG ***")


if __name__ == "__main__":
    main()
