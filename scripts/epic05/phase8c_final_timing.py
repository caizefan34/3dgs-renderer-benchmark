#!/usr/bin/env python3
"""
Phase 8C — FINAL timing. Exact Phase 8B protocol, 6 checkpoints, tile16+tile32.
Must reproduce Phase 8B results, then explain differences.

Key: Phase 8B protocol:
  Forward: BATCH=10, N_REPEAT=3, WARMUP=3 → 30 samples
  Fwd+Bwd (tile16): BATCH=3, N_REPEAT=2, WARMUP=2 → 6 samples
  Fwd+Bwd (tile32): BATCH=5, N_REPEAT=2, WARMUP=2 → 10 samples
  MEDIAN-based inference: backward = fwd+bwd_median - fwd_median
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

CKPT_MAP = {
    "room_iter5000":  "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
    "room_iter10000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter10000.pt",
    "room_iter15000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter15000.pt",
    "room_iter20000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter20000.pt",
    "room_iter25000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter25000.pt",
    "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
}

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

def robust_stats(arr):
    return {"mean_ms": float(np.mean(arr)), "median_ms": float(np.median(arr)),
            "std_ms": float(np.std(arr)), "min_ms": float(np.min(arr)), "max_ms": float(np.max(arr)),
            "cv": float(np.std(arr)/np.mean(arr)) if np.mean(arr)>0 else 0,
            "n": int(len(arr)), "samples": [float(x) for x in arr]}

def time_forward(params, viewmat, K, W, H, ts):
    xyz, rot, scl, opa, shs = [params[k] for k in ["xyz","rotations","scales","opacity","shs"]]
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                 viewmats=viewmat, Ks=K, width=W, height=H,
                                 tile_size=ts, packed=True, sh_degree=params["sh_degree"])
        torch.cuda.synchronize()
    times = []
    for _ in range(3):  # repeats
        for _ in range(10):  # batch
            ev_s.record()
            r, a, m = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                     viewmats=viewmat, Ks=K, width=W, height=H,
                                     tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            ev_e.record()
            torch.cuda.synchronize()
            times.append(ev_s.elapsed_time(ev_e))
    return robust_stats(np.array(times)), m

def time_fwd_bwd(params, viewmat, K, W, H, ts, is_tile16):
    xyz = params["xyz"].detach().clone().requires_grad_(True)
    rot = params["rotations"].detach().clone().requires_grad_(True)
    scl = params["scales"].detach().clone().requires_grad_(True)
    opa = params["opacity"].detach().clone().requires_grad_(True)
    shs = params["shs"].detach().clone().requires_grad_(True)
    target = torch.rand(1, H, W, 3, device=DEVICE, dtype=DTYPE)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    n_batch = 3 if is_tile16 else 5
    n_repeat = 2

    for _ in range(2):
        r, a, _ = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                 viewmats=viewmat, Ks=K, width=W, height=H,
                                 tile_size=ts, packed=True, sh_degree=params["sh_degree"])
        loss = ((r - target) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [xyz, rot, scl, opa, shs]:
            if p.grad is not None: p.grad = None

    times = []
    for _ in range(n_repeat):
        for _ in range(n_batch):
            for p in [xyz, rot, scl, opa, shs]:
                if p.grad is not None: p.grad = None
            ev_s.record()
            r, a, _ = rasterization(means=xyz, quats=rot, scales=scl, opacities=opa, colors=shs,
                                     viewmats=viewmat, Ks=K, width=W, height=H,
                                     tile_size=ts, packed=True, sh_degree=params["sh_degree"])
            loss = ((r - target) ** 2).mean()
            loss.backward()
            ev_e.record()
            torch.cuda.synchronize()
            times.append(ev_s.elapsed_time(ev_e))

    return robust_stats(np.array(times))

def main():
    print("="*70)
    print("Phase 8C — FINAL Exact Phase 8B Protocol Reproduction")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*70)

    viewmat, K, W, H = make_camera()
    results = {}

    for ckpt_name in sorted(CKPT_MAP.keys()):
        ckpt_path = CKPT_DIR / CKPT_MAP[ckpt_name]
        if not ckpt_path.exists():
            print(f"\nSKIP {ckpt_name}: not found")
            continue

        params = load_ckpt(str(ckpt_path))
        N = params["xyz"].shape[0]
        print(f"\n{'='*50}")
        print(f"  {ckpt_name}: N={N:,}")
        entry = {"num_gaussians": N, "sh_degree": params["sh_degree"], "tile_sizes": {}}

        for ts in [16, 32]:
            is_t16 = (ts == 16)
            label = f"tile{ts}"
            print(f"\n  [{label}]")

            fwd_stats, meta = time_forward(params, viewmat, K, W, H, ts)
            print(f"    FWD: median={fwd_stats['median_ms']:.1f}ms  mean={fwd_stats['mean_ms']:.1f}±{fwd_stats['std_ms']:.1f}ms  "
                  f"[{fwd_stats['min_ms']:.1f}, {fwd_stats['max_ms']:.1f}]  CV={fwd_stats['cv']:.3f}")

            fb_stats = time_fwd_bwd(params, viewmat, K, W, H, ts, is_t16)
            print(f"    F+B: median={fb_stats['median_ms']:.1f}ms  mean={fb_stats['mean_ms']:.1f}±{fb_stats['std_ms']:.1f}ms  "
                  f"[{fb_stats['min_ms']:.1f}, {fb_stats['max_ms']:.1f}]  CV={fb_stats['cv']:.3f}")

            # Infer backward: median
            bwd_median = max(0.001, fb_stats["median_ms"] - fwd_stats["median_ms"])
            bwd_mean = max(0.001, fb_stats["mean_ms"] - fwd_stats["mean_ms"])
            bwd_std = math.sqrt(max(0.001, fb_stats["std_ms"]**2 - fwd_stats["std_ms"]**2))
            bwd_cv = bwd_std / bwd_mean if bwd_mean > 0 else 0
            print(f"    INF BWD: median={bwd_median:.1f}ms  mean={bwd_mean:.1f}ms")

            entry["tile_sizes"][label] = {
                "forward": fwd_stats,
                "forward_plus_backward": fb_stats,
                "inferred_backward": {
                    "median_ms": bwd_median, "mean_ms": bwd_mean,
                    "std_ms": bwd_std, "cv": bwd_cv,
                },
            }

        # Ratios
        t16 = entry["tile_sizes"]["tile16"]
        t32 = entry["tile_sizes"]["tile32"]
        ratios = {}
        for phase in ["forward", "forward_plus_backward", "inferred_backward"]:
            m16 = t16[phase]["median_ms"]
            m32 = t32[phase]["median_ms"]
            r = m16 / m32 if m32 > 0 else float('inf')
            ratios[phase] = {"t16_ms": m16, "t32_ms": m32, "ratio_t16_t32": r}
            print(f"    RATIO ({phase}): {r:.1f}x  (t16={m16:.1f}ms, t32={m32:.1f}ms)")
        entry["ratios"] = ratios
        results[ckpt_name] = entry
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "phase8c_timing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
