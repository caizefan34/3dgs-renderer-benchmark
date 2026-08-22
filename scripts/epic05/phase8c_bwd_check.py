#!/usr/bin/env python3
"""
Verify backward timing by comparing 3 methods:
1. My split approach (fwd then separate bwd)
2. Phase 8B combined approach (fwd+bwd together) - then subtract fwd
3. Check that gradients are actually computed
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

print("="*60)
print("Phase 8C — Backward timing verification")
print(f"GPU: {torch.cuda.get_device_name(0)}")

viewmat, K, W, H = make_camera()

for ckpt_name in ["room_iter5000", "room_iter30000"]:
    ckpt_path = CKPT_DIR / {
        "room_iter5000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
        "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
    }[ckpt_name]

    params = load_ckpt(str(ckpt_path))
    N = params["xyz"].shape[0]
    print(f"\n{ckpt_name}: N={N:,}")

    for ts in [16, 32]:
        print(f"\n--- tile_size={ts} ---")
        
        # Setup
        xyz = params["xyz"].detach().clone().requires_grad_(True)
        rot = params["rotations"].detach().clone().requires_grad_(True)
        scl = params["scales"].detach().clone().requires_grad_(True)
        opa = params["opacity"].detach().clone().requires_grad_(True)
        shs = params["shs"].detach().clone().requires_grad_(True)
        target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)

        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(2):
            r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                     viewmats=viewmat, Ks=K, width=W, height=H,
                                     tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            loss = ((r - target) ** 2).mean()
            loss.backward()
            torch.cuda.synchronize()
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None

        # METHOD 1: Split timing (fwd first, then bwd separately)
        print(f"  Method 1 (split fwd then bwd):")
        for rep in range(3):
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None

            ev_s.record()
            r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                     viewmats=viewmat, Ks=K, width=W, height=H,
                                     tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            ev_e.record()
            torch.cuda.synchronize()
            t_fwd = ev_s.elapsed_time(ev_e)

            ev_s.record()
            loss = ((r - target) ** 2).mean()
            loss.backward()
            ev_e.record()
            torch.cuda.synchronize()
            t_bwd = ev_s.elapsed_time(ev_e)
            print(f"    Rep {rep}: fwd={t_fwd:.2f}ms  bwd={t_bwd:.2f}ms  "
                  f"xyz_grad_norm={xyz.grad.norm().item():.4f}" if xyz.grad is not None else f"    Rep {rep}: NO GRAD!")
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None

        # METHOD 2: Combined timing (like Phase 8B)
        print(f"  Method 2 (combined fwd+bwd, then subtract fwd):")
        for rep in range(3):
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None

            ev_s.record()
            r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                     viewmats=viewmat, Ks=K, width=W, height=H,
                                     tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            loss = ((r - target) ** 2).mean()
            loss.backward()
            ev_e.record()
            torch.cuda.synchronize()
            t_total = ev_s.elapsed_time(ev_e)

            # Also get fwd separately for this comparison
            ev_s.record()
            r2, a2, m2 = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                        viewmats=viewmat, Ks=K, width=W, height=H,
                                        tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            ev_e.record()
            torch.cuda.synchronize()
            t_fwd_sep = ev_s.elapsed_time(ev_e)

            inferred_bwd = t_total - t_fwd_sep
            print(f"    Rep {rep}: total={t_total:.2f}ms  fwd_sep={t_fwd_sep:.2f}ms  "
                  f"inferred_bwd={inferred_bwd:.2f}ms")
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None

        torch.cuda.empty_cache()

print("\nDone.")
