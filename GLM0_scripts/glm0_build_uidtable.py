#!/usr/bin/env python3
"""GLM0: build unified per-Gaussian lifecycle table per scene.
Joins uid_table (death/topology), birth_records (30 pre-birth feats),
and cost_agg (lifetime visibility). Computes derived lifecycle features.
Metadata-only; writes GLM0_WORK/uidtable_<scene>.parquet.
"""
import os, numpy as np, pandas as pd, json

RUNS = {"room":"/root/epic05_d0/runs/TRACE_room",
        "bicycle":"/root/epic05_d0/runs/TRACE_bicycle",
        "garden":"/root/glm0/runs/TRACE_garden"}
AGG = "/root/glm0/artifacts"
OUT = "/root/glm0/work"
os.makedirs(OUT, exist_ok=True)
T_END = 30000
SPLIT = 15000

def build(scene):
    ut = np.load(os.path.join(RUNS[scene], "d0_uid_table.npz"))
    br = np.load(os.path.join(RUNS[scene], "d0_birth_records.npz"), allow_pickle=True)
    ca = np.load(os.path.join(AGG, f"cost_agg_{scene}.npz"))
    uid = ut["uid"].astype(np.int64)
    birth = ut["birth_iter"].astype(np.int64)
    death = ut["death_iter"].astype(np.int64)
    death_eff = np.where(death < 0, T_END + 1, death)
    age = death_eff - birth  # lifetime in iters
    alive = (death < 0)
    # cost agg (indexed by uid)
    vc = ca["vis_count"]; st = ca["sum_tiles"]; ve = ca["vis_early"]; vl = ca["vis_late"]
    iters = ca["iters"]
    vis_count = vc[uid]; sum_tiles = st[uid]; vis_early = ve[uid]; vis_late = vl[uid]
    # alive sampled iters in [birth, death_eff)
    a_it = np.searchsorted(iters, death_eff, side="right") - np.searchsorted(iters, birth, side="left")
    a_it = np.maximum(a_it, 1)
    it_early = np.searchsorted(iters, SPLIT, side="right") - np.searchsorted(iters, birth, side="left")
    it_late = np.searchsorted(iters, death_eff, side="right") - np.searchsorted(iters, SPLIT, side="left")
    it_early = np.maximum(it_early, 0); it_late = np.maximum(it_late, 0)
    vis_frac = vis_count / a_it
    vis_frac_early = np.divide(vis_early, np.maximum(it_early,1), out=np.zeros_like(vis_early,dtype=float), where=it_early>0)
    vis_frac_late  = np.divide(vl,  np.maximum(it_late,1),  out=np.zeros_like(vl,dtype=float),  where=it_late>0)
    # birth features
    br_uid = br["uid"].astype(np.int64)
    feat = br["feat"]  # (30, N)
    fnames = list(br["feature_names"])
    fdf = pd.DataFrame(feat.T, columns=fnames)
    fdf.insert(0, "uid", br_uid)
    df = pd.DataFrame({
        "uid": uid, "parent_uid": ut["parent_uid"].astype(np.int64),
        "birth_iter": birth, "birth_type": ut["birth_type"].astype(np.int8),
        "death_iter": death, "terminal_reason": ut["terminal_reason"].astype(np.int8),
        "age": age.astype(np.int64), "alive": alive.astype(np.int8),
        "tiles_cum": ut["tiles_cum"].astype(np.int64),
        "cam_count": ut["cam_count"].astype(np.int32),
        "hist_n": ut["hist_n"].astype(np.int32),
        "vis_count": vis_count.astype(np.int32),
        "sum_tiles": sum_tiles.astype(np.int64),
        "vis_early": vis_early.astype(np.int32), "vis_late": vis_late.astype(np.int32),
        "alive_sampled_iters": a_it.astype(np.int32),
        "vis_frac": vis_frac, "vis_frac_early": vis_frac_early, "vis_frac_late": vis_frac_late,
    })
    df = df.merge(fdf, on="uid", how="left")
    df.to_parquet(os.path.join(OUT, f"uidtable_{scene}.parquet"), index=False)
    print(f"[{scene}] rows={len(df)} births={int((df.birth_type>0).sum())} "
          f"deaths={int((df.terminal_reason>0).sum())} alive={int(df.alive.sum())}")
    return df

if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or list(RUNS)):
        if os.path.isdir(RUNS[s]) and os.path.exists(os.path.join(RUNS[s],"d0_uid_table.npz")):
            build(s)
        else:
            print(f"[{s}] run dir not ready: {RUNS[s]}")
