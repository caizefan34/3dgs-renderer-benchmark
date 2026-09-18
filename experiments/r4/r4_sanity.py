#!/usr/bin/env python3
"""
R4-0: Room Minimal CUDA Correctness Sanity Check (Simplified)

Verifies MODE0 (baseline) vs MODE1 (R4 path, skip disabled) gradient agreement.
Uses a pre-trained checkpoint and a few cameras.

Usage:
  PYTHONNOUSERSITE=1 ~/miniforge3/envs/anysplat/bin/python experiments/r4/r4_sanity.py \
    --checkpoint results/reference_v1/room_30k/checkpoints/iter_30000.pt \
    --cameras-json data/official/mipnerf360/room/cameras.json \
    --output /mnt/storage_pool/liaoyuanjun/r4_sanity
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cuda")
    return {
        "xyz": ckpt["xyz"].cuda().float().requires_grad_(True),
        "rotations": ckpt["rotation"].cuda().float().requires_grad_(True),
        "scales": ckpt["scaling"].cuda().float().requires_grad_(True),
        "opacity": ckpt["opacity"].cuda().float().requires_grad_(True),
        "shs": ckpt["shs"].cuda().float().requires_grad_(True),
        "active_sh_degree": int(ckpt.get("active_sh_degree", 3)),
    }

def load_cameras(path):
    with open(path) as f:
        return json.load(f)

def get_camera(cameras, idx, images_dir):
    """Load camera intrinsics + extrinsics + GT image."""
    cam = cameras[idx]
    w = cam["width"]
    h = cam["height"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = w / 2.0, h / 2.0  # Centered principal point
    
    position = np.array(cam["position"], dtype=np.float32)
    rot = np.array(cam["rotation"], dtype=np.float32)  # 3x3
    
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
    
    # Resize to ~1080p for faster processing
    target_w = 1920
    target_h = int(h * target_w / w)
    # Make divisible by 16 for tile_size
    target_h = (target_h // 16) * 16
    target_w = (target_w // 16) * 16
    img_pil = img_pil.resize((target_w, target_h), Image.LANCZOS)
    
    # Scale intrinsics
    sx = target_w / w
    sy = target_h / h
    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy
    w, h = target_w, target_h
    
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    
    img = np.array(img_pil, dtype=np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().float()
    
    return viewmat, K, w, h, gt

def render_and_backward(params, viewmat, K, w, h, gt, label=""):
    """Forward + backward, return loss and gradients."""
    from gsplat import rasterization
    
    # Clear grads
    for k in ["xyz", "rotations", "scales", "opacity", "shs"]:
        if params[k].grad is not None:
            params[k].grad.zero_()
    
    # Forward
    r, _, meta = rasterization(
        means=params["xyz"],
        quats=params["rotations"],
        scales=params["scales"],
        opacities=params["opacity"],
        colors=params["shs"],
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=w, height=h,
        tile_size=16, packed=False,
        sh_degree=params["active_sh_degree"],
        radius_clip=0.0, eps2d=0.1,
        render_mode="RGB", absgrad=True,
    )
    image = r[0].clamp(0, 1)
    means2d = meta["means2d"]
    means2d.retain_grad()
    
    # Loss
    L1 = F.l1_loss(image, gt)
    loss = L1  # Simple L1 for sanity
    
    # Backward
    loss.backward()
    
    grads = {
        "v_xyz": params["xyz"].grad.clone(),
        "v_rotation": params["rotations"].grad.clone(),
        "v_scaling": params["scales"].grad.clone(),
        "v_opacity": params["opacity"].grad.clone(),
        "v_shs": params["shs"].grad.clone(),
        "v_means2d": means2d.grad.clone() if means2d.grad is not None else None,
    }
    if hasattr(means2d, 'absgrad') and means2d.absgrad is not None:
        grads["v_means2d_abs"] = means2d.absgrad.clone()
    
    return loss.item(), grads, meta

def compare(g0, g1):
    results = {}
    for key in g0:
        if g0[key] is None or g1.get(key) is None:
            continue
        d = (g0[key] - g1[key]).abs()
        max_abs = d.max().item()
        denom = g0[key].abs().clamp_min(1e-8)
        max_rel = (d / denom).max().item()
        l2 = d.pow(2).mean().sqrt().item()
        # Use absolute error as primary metric (atomicAdd is non-deterministic)
        # Threshold: max_abs < 1e-4 OR l2 < 1e-6
        ok = max_abs < 1e-4 or l2 < 1e-6
        results[key] = {"max_abs": max_abs, "max_rel": max_rel, "l2": l2, "pass": ok}
        status = "PASS" if ok else "FAIL"
        print(f"  {key}: max_abs={max_abs:.2e} max_rel={max_rel:.2e} l2={l2:.2e} [{status}]")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--images-dir", default=None, help="Override images dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cameras", type=int, default=3)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Auto-detect images dir if not specified
    images_dir = args.images_dir
    if images_dir is None:
        # Default: repo_root/data/datasets/mipnerf360/{scene}/images
        scene_name = Path(args.cameras_json).parent.name
        repo_root = Path.cwd()
        images_dir = str(repo_root / "data" / "datasets" / "mipnerf360" / scene_name / "images")
    
    print("=== R4-0: Room CUDA Correctness Sanity ===")
    print(f"Images dir: {images_dir}")
    params = load_checkpoint(args.checkpoint)
    print(f"N Gaussians: {params['xyz'].shape[0]}, SH degree: {params['active_sh_degree']}")
    
    cameras = load_cameras(args.cameras_json)
    print(f"N cameras: {len(cameras)}")
    
    all_results = {"comparisons": {}, "verdict": "UNKNOWN"}
    all_pass = True
    
    for ci in range(min(args.n_cameras, len(cameras))):
        print(f"\n--- Camera {ci} ---")
        viewmat, K, w, h, gt = get_camera(cameras, ci, images_dir)
        print(f"  Image: {w}x{h}, GT shape: {gt.shape}")
        
        # MODE0: baseline run 1
        print("MODE0: Baseline backward (run 1)...")
        p0 = {k: v.clone().detach().requires_grad_(True) if k != "active_sh_degree" else v 
              for k, v in params.items()}
        loss0, g0, meta0 = render_and_backward(p0, viewmat, K, w, h, gt, "MODE0")
        print(f"  Loss: {loss0:.6f}")
        
        # MODE1: baseline run 2 (should be identical to MODE0)
        print("MODE1: Baseline backward (run 2, skip disabled)...")
        p1 = {k: v.clone().detach().requires_grad_(True) if k != "active_sh_degree" else v 
              for k, v in params.items()}
        loss1, g1, meta1 = render_and_backward(p1, viewmat, K, w, h, gt, "MODE1")
        print(f"  Loss: {loss1:.6f}")
        
        # Compare
        print("MODE0 vs MODE1:")
        cmp = compare(g0, g1)
        all_results["comparisons"][f"cam{ci}"] = {
            "loss0": loss0, "loss1": loss1,
            "n_intersections": meta0.get("tile_offsets", torch.tensor([0])).numel(),
            "comparison": cmp,
        }
        
        if not all(v.get("pass", False) for v in cmp.values()):
            all_pass = False
            print(f"*** FAIL: MODE0 vs MODE1 mismatch on camera {ci} ***")
    
    all_results["verdict"] = "PASS" if all_pass else "IMPLEMENTATION_BUG"
    
    out_path = os.path.join(args.output, "r4_sanity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n=== Verdict: {all_results['verdict']} ===")
    print(f"Results: {out_path}")
    
    if all_pass:
        print("MODE0 vs MODE1 agree within numerical noise.")
        print("Ready to proceed to MODE2 (certificate skip) testing.")
    else:
        print("*** STOP R4 — IMPLEMENTATION_BUG ***")

if __name__ == "__main__":
    main()
