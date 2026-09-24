#!/usr/bin/env python3
"""H5-1 capture (EXPERIMENTAL pipeline, runs in the higs-13scene-env on mx).
Uses the validated f2_b2 projection/intersection + oracle per-tile ordered lists
(authoritative native forward order, depth-then-id) and real per-Gaussian base
color.  Emits <scene>.npz for h5_run.py + the h5 CUDA micro-kernel."""
import argparse, json, sys
from pathlib import Path
import numpy as np

# strip corrupt user-site (~/.local) gsplat so the fork in higs-13scene-env wins
sys.path = [p for p in sys.path if "/.local/" not in p]
sys.path.insert(0, "/tmp")
from h3_fwd_0_oracle import f2_b2, oracle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["room", "bicycle", "garden"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--out", default="/tmp/h5")
    a = ap.parse_args()
    f = f2_b2(a.scene, a.cam, a.max_long_side, "cuda")
    fine = oracle(f)["fine"]  # per-tile ordered (depth,id), front-to-back
    n_tiles = f["tw"] * f["th"]
    tp = np.zeros(n_tiles + 1, dtype=np.int64)
    lens = np.asarray([len(x) for x in fine])
    tp[1:] = np.cumsum(lens)
    tile_ids = np.zeros(int(lens.sum()), dtype=np.int32)
    for t in range(n_tiles):
        tile_ids[tp[t]:tp[t + 1]] = fine[t]
    # real visible base SH0 color (visible id order) via the same gather path
    import torch
    from h2_bwd_0_structural import load_ply_scene, load_cameras
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    _CONFIGS = {
      "room": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",3114,2075),
      "bicycle": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",4946,3286),
      "garden": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",5187,3361),
    }
    ply, cams, nw, nh = _CONFIGS[a.scene]
    s = min(1.0, a.max_long_side / max(nw, nh))
    w, h = round(nw * s), round(nh * s)
    means, quats, scales, opacities, colors = load_ply_scene(ply, "cuda")
    vm, K = load_cameras(cams, w, h, "cuda")
    vm, K = vm[:, [a.cam]], K[:, [a.cam]]
    with torch.no_grad():
        ids, _, _ = _cull_gaussians_batched(means, quats, scales, vm, K, w, h, eps2d=.3,
                                            near_plane=.01, far_plane=1e10, radius_clip=0.,
                                            camera_model="pinhole")
        m, q, st, op, co = _gather_visible_native(means, quats, scales, opacities, colors, ids)
    c0 = co[:, 0, :3].float().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        Path(a.out) / (a.scene + ".npz"),
        tile_ptr=tp, tile_ids=tile_ids,
        m2d=f["m2d"].astype(np.float32), conics=f["conics"].astype(np.float32),
        opacity=f["opacity"].astype(np.float32), depth=f["depth"].astype(np.float32),
        color=c0, width=w, height=h, tw=f["tw"], th=f["th"], tile_size=128,
        nvis=f["nvis"], n_tiles=n_tiles, total_pairs=int(lens.sum()), cam=a.cam,
    )
    lt = np.diff(tp)[:n_tiles]
    print(json.dumps({"scene": a.scene, "W": w, "H": h, "nvis": int(f["nvis"]),
                      "n_tiles": int(n_tiles), "total_pairs": int(lens.sum()),
                      "per_tile_mean": float(lt.mean()), "p50": float(np.percentile(lt, 50)),
                      "p90": float(np.percentile(lt, 90)), "p99": float(np.percentile(lt, 99)),
                      "max": int(lt.max())}))

if __name__ == "__main__":
    main()