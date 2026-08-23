#!/usr/bin/env python3
"""Phase 8 — Full real-scene microbenchmark across all available checkpoints (forward only)."""
from __future__ import annotations
import json, math, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from gsplat import rasterization

DEVICE = 'cuda'
BATCH, N_REPEAT, WARMUP = 10, 3, 2

ckpt_dir = Path('results/epic05/phase7')
checkpoints = {
    "room_t16_iter5000":  str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter5000.pt"),
    "room_t16_iter10000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter10000.pt"),
    "room_t16_iter15000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter15000.pt"),
    "room_t16_iter20000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter20000.pt"),
    "room_t16_iter25000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter25000.pt"),
    "room_t16_iter30000": str(ckpt_dir / "phase7_room_30k_v2_16" / "phase7_room_30k_v2_16_iter30000.pt"),
    "room_t32_iter5000":  str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter5000.pt"),
    "room_t32_iter10000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter10000.pt"),
    "room_t32_iter15000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter15000.pt"),
    "room_t32_iter20000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter20000.pt"),
    "room_t32_iter25000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter25000.pt"),
    "room_t32_iter30000": str(ckpt_dir / "phase7_room_30k_v2_t32_32" / "phase7_room_30k_v2_t32_32_iter30000.pt"),
}

W, H = 1920, 1080
fx = W / (2.0 * math.tan(math.radians(25)))
viewmat = torch.eye(4, device=DEVICE, dtype=torch.float32).unsqueeze(0)
viewmat[0, 2, 3] = -5.0
K = torch.tensor([[fx, 0, W/2.0], [0, fx, H/2.0], [0, 0, 1]],
                 device=DEVICE, dtype=torch.float32).unsqueeze(0)

all_results = {}

for name, path in checkpoints.items():
    if not Path(path).exists():
        print(f"SKIP: {name} — not found")
        continue
    print(f"\n{'='*60}")
    print(f"  LOAD: {name}")
    print(f"  PATH: {path}")

    cp = torch.load(path, map_location=DEVICE, weights_only=False)
    ms = cp['model_state']
    xyz = ms['xyz']
    rot = ms['rotations']
    scales = ms['scales']
    opac = ms['opacity'].squeeze(1)
    shs = ms['shs']
    N = xyz.shape[0]
    print(f"  Gaussians: {N}")
    print(f"  SH degree: {ms['sh_degree']}")

    ckpt_entry = {"num_gaussians": N, "sh_degree": ms['sh_degree'], "tile_sizes": {}}

    for ts in [16, 32]:
        print(f"  --- tile_size={ts} ---")

        for w in range(WARMUP):
            rendered, alpha, meta = rasterization(
                means=xyz, quats=rot, scales=scales, opacities=opac, colors=shs,
                viewmats=viewmat, Ks=K, width=W, height=H,
                tile_size=ts, packed=True, sh_degree=ms['sh_degree'])
            torch.cuda.synchronize()
        print(f"    Warmup done", flush=True)

        all_times = []
        for r in range(N_REPEAT):
            for b in range(BATCH):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                rendered, alpha, meta = rasterization(
                    means=xyz, quats=rot, scales=scales, opacities=opac, colors=shs,
                    viewmats=viewmat, Ks=K, width=W, height=H,
                    tile_size=ts, packed=True, sh_degree=ms['sh_degree'])
                end_ev.record()
                torch.cuda.synchronize()
                all_times.append(start_ev.elapsed_time(end_ev))
            print(f"    Repeat {r+1}/{N_REPEAT} done", flush=True)

        arr = np.array(all_times)
        tpg = meta['tiles_per_gauss']

        entry = {
            "forward_ms_mean": float(arr.mean()),
            "forward_ms_std": float(arr.std()),
            "forward_ms_cv": float(arr.std() / arr.mean()),
            "tiles_per_gauss_mean": float(tpg.float().mean().item()),
            "tiles_per_gauss_std": float(tpg.float().std().item()),
            "tiles_per_gauss_median": float(tpg.float().median().item()),
            "tiles_per_gauss_max": int(tpg.max().item()),
            "total_intersections": int(tpg.sum().item()),
            "tile_grid": f"{meta['tile_width']}x{meta['tile_height']}",
        }
        ckpt_entry["tile_sizes"][str(ts)] = entry
        print(f"    Forward: {entry['forward_ms_mean']:.2f} +/- {entry['forward_ms_std']:.2f} ms")
        print(f"    Tiles/Gaussian: {entry['tiles_per_gauss_mean']:.2f} +/- {entry['tiles_per_gauss_std']:.2f}")

    t16_ms = ckpt_entry["tile_sizes"]["16"]["forward_ms_mean"]
    t32_ms = ckpt_entry["tile_sizes"]["32"]["forward_ms_mean"]
    ratio = t16_ms / t32_ms
    ckpt_entry["ratio_t16_t32"] = ratio
    print(f"  >>> Ratio t16/t32: {ratio:.4f}")

    all_results[name] = ckpt_entry
    torch.cuda.empty_cache()

# Summary table
print(f"\n\n{'='*80}")
print(f"  SUMMARY: Real-Scene Snapshot Microbenchmark (room)")
print(f"{'='*80}")
print(f"  {'Checkpoint':<20} {'N':<12} {'t16(ms)':<12} {'t32(ms)':<12} {'Ratio':<10} {'t16 TPG':<10} {'t32 TPG':<10}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
for name, entry in sorted(all_results.items()):
    t16 = entry["tile_sizes"]["16"]
    t32_s = entry["tile_sizes"]["32"]
    print(f"  {name[:18]:<20} {entry['num_gaussians']:<12} {t16['forward_ms_mean']:<12.2f} {t32_s['forward_ms_mean']:<12.2f} {entry['ratio_t16_t32']:<10.2f} {t16['tiles_per_gauss_mean']:<10.1f} {t32_s['tiles_per_gauss_mean']:<10.1f}")

# Save
out_path = Path("results/epic05/phase8/real_scene_microbench.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
