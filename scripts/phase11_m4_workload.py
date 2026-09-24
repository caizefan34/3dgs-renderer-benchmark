#!/usr/bin/env python3
"""
Workload analysis for M4 radius_clip — measure actual throughput impact.
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch, numpy as np
from benchmark_framework.scene import load_ply

def load_cameras(json_path, target_h=1080):
    with open(json_path) as f:
        data = json.load(f)
    viewmats, Ks = [], []
    for cd in data:
        fx, fy = cd["fx"], cd["fy"]
        w_orig, h_orig = cd["width"], cd["height"]
        scale = target_h / h_orig
        w = int(round(w_orig * scale))
        h = target_h
        R = np.array(cd["rotation"]).reshape(3, 3)
        t = np.array(cd["position"])
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R.T
        viewmat[:3, 3] = -R.T @ t
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx * scale
        K[1, 1] = fy * scale
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        viewmats.append(torch.from_numpy(viewmat))
        Ks.append(torch.from_numpy(K))
    return torch.stack(viewmats).float().cuda(), torch.stack(Ks).float().cuda(), w, h

def render_with_meta(means, quats, scales, opacities, shs, viewmats, Ks, width, height,
                     sh_degree, radius_clip=0.0, eps2d=0.3):
    from gsplat import rasterization
    rendered, _, meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=shs, viewmats=viewmats, Ks=Ks,
        width=width, height=height,
        sh_degree=sh_degree, radius_clip=radius_clip, eps2d=eps2d,
        packed=True, tile_size=16, render_mode="RGB",
    )
    return rendered, meta

def main():
    data_dir = ROOT / "data"
    output_dir = ROOT / "results" / "epic05" / "phase11"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load scene
    ply_path = data_dir / "official" / "mipnerf360" / "room" / "point_cloud.ply"
    scene = load_ply(str(ply_path), device="cuda")
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    shs = scene["shs"].contiguous()
    sh_degree = scene.get("sh_degree", 3)
    print(f"Loaded {means.shape[0]} Gaussians")

    cam_json = data_dir / "official" / "mipnerf360" / "room" / "cameras.json"
    viewmats, Ks, W, H = load_cameras(str(cam_json), target_h=1080)
    C = len(viewmats)
    print(f"Rendering at {W}x{H}, {C} cameras")

    clip_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for clip_val in clip_values:
        print(f"\n=== radius_clip={clip_val} ===")
        # Batch render with timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Warmup
        render_with_meta(means, quats, scales, opacities, shs, viewmats[:1], Ks[:1], W, H, sh_degree, radius_clip=clip_val)
        torch.cuda.synchronize()

        # Timed batch (16 cameras)
        n_batch = min(16, C)
        torch.cuda.synchronize()
        times = []
        for i in range(n_batch):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            _, meta = render_with_meta(means, quats, scales, opacities, shs, viewmats[i:i+1], Ks[i:i+1], W, H, sh_degree, radius_clip=clip_val)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000)

        key = f"rclip_{clip_val}"
        results[key] = {
            "radius_clip": clip_val,
            "fwd_ms_mean": round(np.mean(times), 4),
            "fwd_ms_median": round(np.median(times), 4),
            "fwd_ms_p95": round(np.percentile(times, 95), 4),
            "fwd_ms_min": round(np.min(times), 4),
            "fwd_ms_max": round(np.max(times), 4),
        }
        print(f"  Forward: mean={np.mean(times):.2f}ms, median={np.median(times):.2f}ms")

        # Try to get meta info
        _, meta = render_with_meta(means, quats, scales, opacities, shs, viewmats[0:1], Ks[0:1], W, H, sh_degree, radius_clip=clip_val)
        for mk in sorted(meta.keys()):
            if isinstance(meta[mk], torch.Tensor):
                print(f"    meta.{mk}: shape={list(meta[mk].shape)}, dtype={meta[mk].dtype}")
            elif isinstance(meta[mk], (int, float)):
                print(f"    meta.{mk}: {meta[mk]}")
            else:
                print(f"    meta.{mk}: {type(meta[mk]).__name__}")

        meta_info = {}
        for mk in meta:
            v = meta[mk]
            if isinstance(v, torch.Tensor):
                meta_info[mk] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            elif isinstance(v, (int, float)):
                meta_info[mk] = v
        results[key]["meta"] = meta_info

    with open(output_dir / "m4_workload_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone. Results saved to {output_dir / 'm4_workload_analysis.json'}")

if __name__ == "__main__":
    main()
