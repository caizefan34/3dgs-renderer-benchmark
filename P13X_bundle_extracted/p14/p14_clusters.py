"""P14 Section 3: camera-Gaussian clusters from the REAL C0 renderer.

For each scene at the P13-X verified 16K state:
  - V_c for ALL training cameras via exact C0 culling (no CPU approx).
  - 3 seed cameras at 25% / 50% / 75% of the sorted camera-ID order.
  - 8-camera cluster = seed + 7 highest visible-set Jaccard (tie: cam ID).
  - 8 audit cameras evenly sampled by camera ID from OUTSIDE the cluster.
Writes P14_CAMERA_CLUSTERS.json BEFORE any O1 result (protocol §3).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C


def jaccard(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).shape[0]
    union = a.shape[0] + b.shape[0] - inter
    return inter / union if union > 0 else 0.0


def extract_all_visible(ctx):
    vis = []
    for ci in range(ctx.n_cameras):
        cam, _ = ctx.cam_gt(ci)
        v = np.sort(ctx.visible_set(cam).astype(np.int32))
        vis.append(v)
    return vis


def build_scene_clusters(scene, gpu):
    t0 = time.time()
    ctx = C.SceneCtx(scene, gpu=gpu)
    n = ctx.n_cameras
    vis = extract_all_visible(ctx)
    sizes = [int(v.shape[0]) for v in vis]
    print(f"[p14:{scene}] V_c extracted for {n} cameras "
          f"({time.time()-t0:.0f}s); visible min/med/max = "
          f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}", flush=True)

    # sha256 per camera V_c for provenance
    vcs = [C.sha256_bytes(v.tobytes()) for v in vis]

    # seeds: 25/50/75% of the sorted camera-ID order (round half up)
    seeds = {}
    for p in (0.25, 0.50, 0.75):
        idx = int(np.floor((n - 1) * p + 0.5))
        seeds[p] = int(idx)

    cells = []
    for p, seed in seeds.items():
        jac = np.zeros(n, dtype=np.float64)
        for c in range(n):
            if c == seed:
                continue
            jac[c] = jaccard(vis[seed], vis[c])
        order = sorted(range(n), key=lambda c: (-jac[c], c))
        top7 = [c for c in order[:7]]
        cluster = sorted([seed] + top7)
        # cluster-internal jaccard matrix (record)
        jm = [[float(jaccard(vis[a], vis[b])) for b in cluster]
              for a in cluster]
        # audit: 8 evenly from outside the cluster
        outside = [c for c in range(n) if c not in set(cluster)]
        M = len(outside)
        audit = sorted(set(
            int(outside[int(np.floor(k * (M - 1) / 7 + 0.5))])
            for k in range(8)))
        assert len(audit) == 8, f"audit sampling collapsed: {audit}"
        cells.append({
            "seed_percent": p, "seed_camera": int(seed),
            "cluster_cameras": [int(x) for x in cluster],
            "cluster_jaccard_to_seed": {
                str(int(c)): float(jac[c]) for c in top7},
            "cluster_jaccard_matrix": jm,
            "audit_cameras": [int(x) for x in audit],
            "seed_top10_with_jaccard": [
                {"camera": int(c), "jaccard": float(jac[c])}
                for c in order[:10]],
        })
        print(f"[p14:{scene}] seed {seed} (p={p}): cluster={cluster} "
              f"jaccard=[{jac[top7[0]]:.3f}..{jac[top7[-1]]:.3f}] "
              f"audit={audit}", flush=True)

    # persist V_c arrays (int32) for exact reproduction
    npz_path = os.path.join(C.P14_ROOT, "raw", f"P14_VC_{scene}.npz")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    flat = np.concatenate(vis)
    offs = np.zeros(len(vis) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([v.shape[0] for v in vis])
    np.savez_compressed(npz_path, vis_flat=flat, offsets=offs, n=n)
    npz_sha = C.sha256_file(npz_path)

    return {
        "scene": scene, "n_cameras": int(n), "N_gaussians": int(ctx.N),
        "ckpt": C.CKPT[scene],
        "visible_counts": sizes,
        "visible_sha256": vcs,
        "vc_npz": {"path": npz_path, "sha256": npz_sha,
                   "dtype": "int32", "n_cameras": int(n)},
        "seed_rule": ("sorted camera-ID order (dataset order = camera ID); "
                      "seed index = floor((n-1)*p + 0.5), p in {0.25,0.50,"
                      "0.75} (round half up)"),
        "cluster_rule": ("seed + 7 cameras with highest Jaccard(V_seed, "
                         "V_c); tie-break camera ID ascending"),
        "audit_rule": ("8 cameras outside the cluster, evenly sampled by "
                       "camera ID: outside[floor(k*(M-1)/7 + 0.5)], k=0..7"),
        "vc_source": "exact C0 renderer culling (_cull_gaussians_batched "
                     "at the 16K state); no CPU footprint approximation",
        "cells": cells,
    }


def main():
    # single process: CUDA_VISIBLE_DEVICES is fixed at first CUDA init,
    # so both scenes run sequentially on the same idle GPU (clusters are
    # not timing-sensitive; GPU 4 verified idle at stage 0)
    gpus = {"room": "4", "bicycle": "4"}
    out = {"phase": "P14", "created_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_snapshot_start": C.gpu_snapshot(), "scenes": {}}
    for scene in C.SCENES:
        out["scenes"][scene] = build_scene_clusters(scene, gpus[scene])
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    out["gpu_snapshot_end"] = C.gpu_snapshot()
    out["ordering_guarantee"] = ("this file is written BEFORE any O1 "
                                 "measurement; clusters are frozen and "
                                 "may not change afterwards")
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts",
                          "P14_CAMERA_CLUSTERS.json"), out)
    print("[p14] P14_CAMERA_CLUSTERS.json written before any O1 result")


if __name__ == "__main__":
    main()
