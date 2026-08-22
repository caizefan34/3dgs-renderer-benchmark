#!/usr/bin/env python3
"""
Phase 8C — Backward kernel timing with CUDA events.
Measures the FULL backward pass with CUDA event timing on 2 checkpoints.
Gives us direct bwd timing on this same machine to normalize workload data.
"""

import json, math, sys, os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from gsplat import rasterization

DEVICE = "cuda"
DTYPE = torch.float32
CKPT_DIR = REPO_ROOT / "results" / "epic05" / "phase7"
OUT_DIR = REPO_ROOT / "results" / "epic05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_camera(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=DEVICE, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
                      device=DEVICE, dtype=DTYPE).unsqueeze(0)
    return viewmat, K, W, H

def load_ckpt(path):
    cp = torch.load(path, map_location=DEVICE, weights_only=False)
    ms = cp["model_state"]
    opac = ms["opacity"].detach().clone()
    if opac.dim() == 2 and opac.shape[1] == 1:
        opac = opac.squeeze(1)
    return {"xyz": ms["xyz"].detach().clone(), "rotations": ms["rotations"].detach().clone(),
            "scales": ms["scales"].detach().clone(), "opacity": opac,
            "shs": ms["shs"].detach().clone(), "sh_degree": ms["sh_degree"]}

def measure(params, viewmat, K, W, H, ts, label):
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rot = params["rotations"].detach().clone().requires_grad_(True)
    scl = params["scales"].detach().clone().requires_grad_(True)
    opa = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)
    target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(3):
        r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                 viewmats=viewmat, Ks=K, width=W, height=H,
                                 tile_size=ts, packed=True, sh_degree=params["sh_degree"])
        loss = ((r - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [xyz, rot, scl, opa, shs]:
            if p.grad is not None: p.grad = None

    fwd_times = []
    bwd_times = []
    for _ in range(5):
        for p in [xyz, rot, scl, opa, shs]:
            if p.grad is not None: p.grad = None

        ev_s.record()
        r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                 viewmats=viewmat, Ks=K, width=W, height=H,
                                 tile_size=ts, packed=True, sh_degree=params["sh_degree"])
        ev_e.record()
        torch.cuda.synchronize()
        fwd_times.append(ev_s.elapsed_time(ev_e))

        ev_s.record()
        loss = ((r - target) ** 2).mean()
        loss.backward()
        ev_e.record()
        torch.cuda.synchronize()
        bwd_times.append(ev_s.elapsed_time(ev_e))

    fwd_arr = np.array(fwd_times)
    bwd_arr = np.array(bwd_times)
    return {
        "fwd_median_ms": float(np.median(fwd_arr)),
        "fwd_mean_ms": float(np.mean(fwd_arr)),
        "fwd_std_ms": float(np.std(fwd_arr)),
        "fwd_samples": [float(x) for x in fwd_arr],
        "bwd_median_ms": float(np.median(bwd_arr)),
        "bwd_mean_ms": float(np.mean(bwd_arr)),
        "bwd_std_ms": float(np.std(bwd_arr)),
        "bwd_samples": [float(x) for x in bwd_arr],
    }

def main():
    print("="*60)
    print("Phase 8C — Backward kernel timing")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*60)

    results = {}
    viewmat, K, W, H = make_camera()

    for ckpt_name in ["room_iter5000", "room_iter30000"]:
        ckpt_path = CKPT_DIR / {
            "room_iter5000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
            "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
        }[ckpt_name]

        if not ckpt_path.exists():
            print(f"SKIP {ckpt_name}: not found")
            continue

        params = load_ckpt(str(ckpt_path))
        N = params["xyz"].shape[0]
        print(f"\n{ckpt_name}: N={N:,}")

        entry = {"num_gaussians": N, "tile_sizes": {}}
        for ts in [16, 32]:
            label = f"tile{ts}"
            print(f"  [{label}] measuring...")
            timing = measure(params, viewmat, K, W, H, ts, label)
            print(f"    FWD: {timing['fwd_median_ms']:.1f} ms  BWD: {timing['bwd_median_ms']:.1f} ms")
            entry["tile_sizes"][label] = timing
        results[ckpt_name] = entry
        torch.cuda.empty_cache()

    out = {
        "experiment_id": "phase8c-bwd-timing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }

    out_path = OUT_DIR / "phase8c_bwd_timing.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
