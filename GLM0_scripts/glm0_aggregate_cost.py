#!/usr/bin/env python3
"""GLM0 Phase-1 helper: aggregate per-iteration cost samples into per-uid
lifetime visibility stats. Reads the per-iter npz files (uid, tiles) written
by the frozen C0 tracer, accumulates per-uid:
  vis_count   : #sampled iters the uid was visible (tiles>0)
  sum_tiles   : cumulative tile intersections (contribution proxy)
  vis_early   : visible in sampled iters < SPLIT
  vis_late    : visible in sampled iters >= SPLIT
  n_samples   : #sampled iters in [birth,death) the uid was alive (filled later)
Pure metadata aggregation; no training tensor touched.
"""
import os, numpy as np, glob, re, json, sys

SPLIT = 15000
RUNS = {
    "room":    "/root/epic05_d0/runs/TRACE_room/d0_cost_samples",
    "bicycle": "/root/epic05_d0/runs/TRACE_bicycle/d0_cost_samples",
    "garden":  "/root/glm0/runs/TRACE_garden/d0_cost_samples",
}
OUT = "/root/glm0/artifacts"
os.makedirs(OUT, exist_ok=True)

def scene_aggregate(scene, d):
    files = sorted(glob.glob(os.path.join(d, "sample_*.npz")))
    if not files:
        print(f"[{scene}] no cost samples in {d}"); return None
    iters = np.array([int(re.search(r"sample_(\d+)\.npz", f).group(1)) for f in files], dtype=np.int64)
    max_uid = 0
    for f in files:
        z = np.load(f); max_uid = max(max_uid, int(z["uid"].max()) + 1)
    vis_count = np.zeros(max_uid, dtype=np.int32)
    sum_tiles = np.zeros(max_uid, dtype=np.int64)
    vis_early = np.zeros(max_uid, dtype=np.int32)
    vis_late  = np.zeros(max_uid, dtype=np.int32)
    for it, f in zip(iters, files):
        z = np.load(f)
        u = z["uid"].astype(np.int64)
        t = z["tiles"].astype(np.int64)
        np.add.at(vis_count, u, 1)
        np.add.at(sum_tiles, u, t)
        if it < SPLIT:
            np.add.at(vis_early, u, 1)
        else:
            np.add.at(vis_late, u, 1)
    np.savez(os.path.join(OUT, f"cost_agg_{scene}.npz"),
             vis_count=vis_count, sum_tiles=sum_tiles,
             vis_early=vis_early, vis_late=vis_late, iters=iters)
    print(f"[{scene}] files={len(files)} iters=[{iters.min()},{iters.max()}] max_uid={max_uid} "
          f"total_vis_records={int(vis_count.sum())}")
    return {"scene": scene, "n_sampled_iters": int(len(iters)),
            "iter_min": int(iters.min()), "iter_max": int(iters.max()),
            "max_uid": int(max_uid), "total_vis_records": int(vis_count.sum())}

if __name__ == "__main__":
    scenes = sys.argv[1:] or list(RUNS)
    summary = {}
    for s in scenes:
        r = scene_aggregate(s, RUNS[s])
        if r: summary[s] = r
    json.dump(summary, open(os.path.join(OUT, "cost_agg_summary.json"), "w"), indent=2)
    print("DONE", json.dumps(summary))
