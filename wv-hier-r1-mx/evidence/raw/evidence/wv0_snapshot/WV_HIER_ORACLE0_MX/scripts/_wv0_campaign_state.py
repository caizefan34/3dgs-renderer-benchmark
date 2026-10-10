#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX campaign runner (compact: metrics on-the-fly, no raw grads).
For one state: dense reference gradient per camera, then sparse estimator per
(granularity, budget, mask seed). Computes per-group gradient metrics vs dense,
variance/bias across masks, and structural-work counters. Stores compact summaries.

Usage: wv0_campaign_state.py <state_id> <resolution> <n_cams> <out.pt> <seed_base>
"""
import sys, os, json, time, math, gc
sys.path.insert(0, "/dev/shm")
import numpy as np, torch
try:
    torch.set_num_threads(int(os.environ.get("WV0_THREADS","8")))
except Exception:
    pass
from _wv0_core import (load_cameras, load_state_params, make_grad_enabled,
                       STATE_PATHS, RV1, ALL_GROUPS, PARAM_GROUPS,
                       GRANULARITIES, BUDGETS, N_MASKS)
from _wv0_grad import dense_reference_grad, sparse_tile_grad, tile_pixel_mask_fast

state_id = sys.argv[1]
res = sys.argv[2]
n_cams = int(sys.argv[3])
out_pt = sys.argv[4]
seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0

rel, scene, it = STATE_PATHS[state_id]
ckpt = os.path.join(RV1, rel)
import hashlib
cksha = hashlib.sha256(open(ckpt,"rb").read()).hexdigest()
raw, act = load_state_params(ckpt)
cams = load_cameras(scene, resolution=res)[:n_cams]
print(f"[{state_id}] N={act['N']} sh={act['active_sh']} cams={len(cams)} res={res} ckpt_sha={cksha[:12]}", flush=True)

def gn(t): return float(t.norm())
def rel_l2(gh, gs):  # ||gs-gh|| / (||gh||+eps)
    return float((gs-gh).norm()) / (float(gh.norm()) + 1e-12)
def cos(gh, gs):
    d = float((gh*gs).sum())
    return d / (float(gh.norm())*float(gs.norm()) + 1e-12)
def nmse(gh, gs):
    return float(((gs-gh)**2).sum()) / (float((gh**2).sum()) + 1e-12)

# ---- dense reference gradient per camera ----
dense = {}          # cam -> {group: grad tensor (cpu)}
dense_meta = None
t0 = time.time()
for ci, cam in enumerate(cams):
    g = make_grad_enabled(act)
    gd, render, alpha, meta, loss = dense_reference_grad(g, cam, act["active_sh"])
    dense[ci] = {k: gd[k].detach().cpu() for k in ALL_GROUPS}
    if ci == 0:
        dense_meta = {kk: (meta[kk].cpu() if torch.is_tensor(meta[kk]) else meta[kk])
                      for kk in ("radii","means2d","depths","width","height","tile_size")}
    del g; torch.cuda.synchronize()
print(f"[{state_id}] dense done {round(time.time()-t0,1)}s loss0={round(loss,3)}", flush=True)

# ---- structural work counters (cam0) ----
def compute_work_counters(meta, H, W):
    means2d = meta["means2d"][0]; radii = meta["radii"][0]; depths = meta["depths"][0]
    vis = (radii[:,0] > 0) & (radii[:,1] > 0) & (depths > 0)
    m2 = means2d[vis]; r = radii[vis].float(); Nvis = int(vis.sum())
    out = {"visible_gaussians": Nvis, "per_G": {}}
    for G in GRANULARITIES:
        th = math.ceil(H/G); tw = math.ceil(W/G)
        rx = r[:,0]; ry = r[:,1]
        tx0 = torch.floor((m2[:,0]-rx)/G).long().clamp(0, tw-1)
        tx1 = torch.floor((m2[:,0]+rx)/G).long().clamp(0, tw-1)
        ty0 = torch.floor((m2[:,1]-ry)/G).long().clamp(0, th-1)
        ty1 = torch.floor((m2[:,1]+ry)/G).long().clamp(0, th-1)
        n_tiles = (tx1-tx0+1).clamp(min=0) * (ty1-ty0+1).clamp(min=0)
        out["per_G"][str(G)] = {"dense_isects": int(n_tiles.sum().item()),
                                "n_tiles": int(th*tw),
                                "isects_per_visible": float(n_tiles.sum().item())/max(Nvis,1)}
    return out
work0 = compute_work_counters(dense_meta, cams[0]["H"], cams[0]["W"])
print(f"[{state_id}] work: vis={work0['visible_gaussians']} "
      f"G16={work0['per_G']['16']['dense_isects']} G8={work0['per_G']['8']['dense_isects']} "
      f"G4={work0['per_G']['4']['dense_isects']} G1={work0['per_G']['1']['dense_isects']}", flush=True)

# ---- sparse estimator: accumulate per (G,p) over masks & cams ----
# agg[G][p][group] = {sum_g (cpu tensor), sum_gnorm2 (float), per_mask_metrics: list of dict,
#                     sum_sel_px, sum_sel_tiles, n}
agg = {}
for G in GRANULARITIES:
    for p in BUDGETS:
        agg[(G,p)] = {k: {"sum_g": torch.zeros_like(dense[0][k]), "sum_gnorm2": 0.0,
                          "per_mask": []} for k in ALL_GROUPS}
        agg[(G,p)]["_sel_px"] = []; agg[(G,p)]["_sel_tiles"] = []

t0 = time.time()
for G in GRANULARITIES:
    for p in BUDGETS:
        pi = p
        for mi in range(N_MASKS):
            seed = seed_base + mi
            # per-mask: accumulate gradient sum across cams for variance, and metrics
            mask_metric_groups = {k: [] for k in ALL_GROUPS}
            for ci, cam in enumerate(cams):
                g = make_grad_enabled(act)
                gs, render, alpha, meta, counters, sloss, mask = sparse_tile_grad(
                    g, cam, act["active_sh"], G, seed=seed, pi=pi)
                for k in ALL_GROUPS:
                    gsc = gs[k].detach().cpu(); ghc = dense[ci][k]
                    agg[(G,p)][k]["sum_g"] += gsc
                    agg[(G,p)][k]["sum_gnorm2"] += float(gsc.norm()**2)
                    mask_metric_groups[k].append({
                        "rel_l2": rel_l2(ghc, gsc), "cos": cos(ghc, gsc), "nmse": nmse(ghc, gsc),
                        "gnorm": float(gsc.norm())})
                agg[(G,p)]["_sel_px"].append(counters["selected_pixels"])
                agg[(G,p)]["_sel_tiles"].append(counters["selected_tiles"])
                del g, gs; torch.cuda.synchronize()
            # store per-mask metric summary (mean over cams)
            for k in ALL_GROUPS:
                agg[(G,p)][k]["per_mask"].append({
                    "rel_l2": float(np.mean([m["rel_l2"] for m in mask_metric_groups[k]])),
                    "cos": float(np.mean([m["cos"] for m in mask_metric_groups[k]])),
                    "nmse": float(np.mean([m["nmse"] for m in mask_metric_groups[k]])),
                })
            if mi % 8 == 0:
                print(f"[{state_id}] G={G} p={p} seed={seed} {round(time.time()-t0,1)}s", flush=True)
        # finalize this (G,p): compute cross-mask mean grad, variance, bias
        for k in ALL_GROUPS:
            a = agg[(G,p)][k]
            n_tot = N_MASKS * len(cams)
            mean_g = a["sum_g"] / n_tot
            # Var = E||g||^2 - ||E g||^2  (per-sample variance of the estimator)
            E_gnorm2 = a["sum_gnorm2"] / n_tot
            var = E_gnorm2 - float(mean_g.norm()**2)
            a["mean_g"] = mean_g
            a["E_gnorm2"] = E_gnorm2
            a["var"] = max(var, 0.0)
            # bias vs dense (aggregate over cams: ||mean_g - mean_dense||)
            mean_dense = sum(dense[ci][k] for ci in range(len(cams))) / len(cams)
            a["bias_rel_l2"] = rel_l2(mean_dense, mean_g)
            a["bias_cos"] = cos(mean_dense, mean_g)
            a["mean_g"] = mean_g  # keep for potential further analysis (cpu)
            # free per-mask raw not needed beyond metrics
        gc.collect()

# ---- build compact summary ----
def stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std()), "p25": float(np.percentile(arr,25)),
            "p75": float(np.percentile(arr,75)), "p90": float(np.percentile(arr,90)),
            "n": int(arr.size)}

summary = {"state_id": state_id, "scene": scene, "iteration": it, "ckpt_path": ckpt,
           "ckpt_sha256": cksha, "N": act["N"], "active_sh": act["active_sh"],
           "resolution": res, "n_cams": len(cams), "cam_hw": [cams[0]["H"], cams[0]["W"]],
           "work": work0, "conditions": {}}
for G in GRANULARITIES:
    for p in BUDGETS:
        a = agg[(G,p)]
        cond = {"sel_px_mean": float(np.mean(a["_sel_px"])),
                "sel_tiles_mean": float(np.mean(a["_sel_tiles"])),
                "total_pixels": cams[0]["H"]*cams[0]["W"],
                "groups": {}}
        # structural work ratio vs dense (selected tiles only)
        # dense isects over selected tiles = dense_isects * (sel_px/total) approx (uniform)
        sel_frac = float(np.mean(a["_sel_px"])) / (cams[0]["H"]*cams[0]["W"])
        cond["sel_frac"] = sel_frac
        for k in ALL_GROUPS:
            gk = a[k]
            pm = gk["per_mask"]
            cond["groups"][k] = {
                "rel_l2": stats([m["rel_l2"] for m in pm]),
                "cos": stats([m["cos"] for m in pm]),
                "nmse": stats([m["nmse"] for m in pm]),
                "variance": gk["var"], "E_gnorm2": gk["E_gnorm2"],
                "bias_rel_l2": gk["bias_rel_l2"], "bias_cos": gk["bias_cos"],
            }
        summary["conditions"][f"G{G}_p{p}"] = cond
        # free mean_g / sum_g tensors (cpu) to keep memory bounded
        for k in ALL_GROUPS:
            a[k].pop("mean_g", None); a[k].pop("sum_g", None)
        gc.collect()

os.makedirs(os.path.dirname(out_pt), exist_ok=True)
torch.save(summary, out_pt)
print(f"[{state_id}] SAVED {out_pt} total {round(time.time()-t0,1)}s", flush=True)
