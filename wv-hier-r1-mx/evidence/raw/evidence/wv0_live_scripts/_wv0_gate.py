#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX gate evaluation + verdict.
Reads aggregated summary, computes R_E / R_W, Pareto, and evaluates GO/DROP gates.
"""
import sys, os, json, glob
import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/wv0_out"
GRANULARITIES = [16, 8, 4, 1]
GEOM_GROUPS = ["means","quats","scales"]

agg = torch.load(os.path.join(OUT, "WV0_GRADIENT_SUMMARY.pt"), weights_only=False)
grad = agg["grad"]; varbias = agg["varbias"]; work = agg["work"]; meta = agg["states"]

def geom_nmse(sid, ckey):
    v = grad[sid][ckey].get("geom_nmse_mean")
    return v

def geom_cos(sid, ckey):
    return grad[sid][ckey].get("geom_cos_mean")

# scene -> list of state ids
scenes = {}
for sid in grad:
    sc = meta[sid]["scene"]; scenes.setdefault(sc, []).append(sid)

# ---- per (scene, state, budget, G): geometry NMSE, cosine, W_int, W_blend ----
# W_int relative to dense = dense_isects (selected) / dense_isects (full) = sel_frac (uniform tiles)
# W_blend proxy: blend interactions ~ Gaussian-pixel intersections in selected region.
# We use the structural isect ratio as W_int; W_blend approximated by same sel_frac
# (both scale with selected-pixel fraction for uniform tile sampling). Report both.

def ckey(G,p): return f"G{G}_p{p}"

# ---- Direct G16 vs G8/G4 comparison (R_E, R_W) ----
comparison = {}  # (scene,budget) -> {G: {R_E, R_W, nmse, cos, W}}
for sid in grad:
    sc = meta[sid]["scene"]
    for p in [0.25, 0.40]:
        base_nmse = geom_nmse(sid, ckey(16,p))
        base_cos = geom_cos(sid, ckey(16,p))
        # W_int(G16) = sel_frac at G16 (same pixel budget, but tile count differs)
        # For uniform pixel budget p, sel_frac ~ p for all G (same expected pixels).
        # The structural work DIFFERENCE is in the isect count per selected pixel.
        # W_int(G) = (dense_isects_G / n_tiles_G) * sel_tiles_G / dense_isects_G_full
        #          = sel_frac (since isects/tile and tiles scale). So W_int ~ p for all G.
        # The REAL work saving of coarse G is that fewer, larger blocks reduce per-block overhead.
        # We report W_int as sel_frac (=p) AND the dense isect structure.
        for G in [8,4,1]:
            nm = geom_nmse(sid, ckey(G,p))
            cs = geom_cos(sid, ckey(G,p))
            R_E = nm/base_nmse if base_nmse else None
            R_W = None  # computed below with work ratios
            comparison.setdefault((sc,p), {}).setdefault(sid, {})[G] = {
                "nmse": nm, "cos": cs, "R_E": R_E,
                "nmse_G16": base_nmse, "cos_G16": base_cos,
            }

# ---- Work ratios: W_int(G) = dense_isects_G (selected fraction p) / dense_isects_G (full)
# Since selection is uniform over tiles, W_int(G) = p exactly for the isect-based work.
# But the structural insight: total dense isects at G1 is 100x G16. The SELECTED work
# at G1 (p=0.25) = 0.25 * 180M = 45M vs G16 selected = 0.25*1.4M = 0.36M.
# So G16 does 125x LESS structural work than G1 at the same pixel budget.
# W_int relative to DENSE (full image) for each G:
work_ratios = {}  # (sid,G) -> {W_int_dense, dense_isects}
for sid in work:
    for G in [str(g) for g in GRANULARITIES]:
        w = work[sid]["per_G"][G]
        # dense isects (full image). Selected work = p * dense_isects. W_int vs dense = p.
        work_ratios[(sid,int(G))] = {"dense_isects": w["dense_isects"], "n_tiles": w["n_tiles"],
                                     "isects_per_visible": w["isects_per_visible"]}

# ---- Gate evaluation ----
# Gate A: candidate G8 or G4 achieves NMSE_cand <= 0.70 * NMSE_G16 (>=30% reduction)
#   on geometry macro avg, across >=2 of 3 scenes AND >=2 stages per passing scene.
# Gate B: cosine improves materially (1-cos decreases) — directional improvement.
# Gate C: W_cand,dense <= 0.70 for blend or intersection work AND not catastrophic other.
#   For uniform pixel budget, W_int = p (e.g. 0.25/0.40) <= 0.70 always. So Gate C
#   is satisfied by the structural-work oracle at these budgets. The meaningful test is
#   whether the gradient improvement comes at acceptable work — but at matched pixel
#   budget the work is ~equal. The interior-PARETO question is: does G8/G4 sit between
#   G16 (cheap, noisy) and G1 (expensive, accurate) in the (W, NMSE) plane?
# Gate D: at least one high-load scene (garden/bicycle) passes.

def scene_state_count(scene): return len(scenes.get(scene, []))

gate_results = {}
for cand in [8, 4]:
    # Gate A: per scene, check if >=2 stages have NMSE_cand <= 0.70*NMSE_G16
    passing_scenes = {}
    for sc, sids in scenes.items():
        n_pass = 0; n_total = 0
        for sid in sids:
            for p in [0.25, 0.40]:
                b = geom_nmse(sid, ckey(16,p))
                c = geom_nmse(sid, ckey(cand,p))
                if b and c is not None:
                    n_total += 1
                    if c <= 0.70 * b: n_pass += 1
        passing_scenes[sc] = {"n_pass": n_pass, "n_total": n_total,
                              "gateA_scene_pass": n_pass >= 2}
    n_scenes_pass = sum(1 for v in passing_scenes.values() if v["gateA_scene_pass"])
    # Gate B: cosine improvement (1-cos_cand < 1-cos_G16)
    gateB = {}
    for sc, sids in scenes.items():
        diffs = []
        for sid in sids:
            for p in [0.25, 0.40]:
                b = geom_cos(sid, ckey(16,p)); c = geom_cos(sid, ckey(cand,p))
                if b is not None and c is not None:
                    diffs.append((1-c) - (1-b))  # negative = improvement
        gateB[sc] = float(np.mean(diffs)) if diffs else None
    gate_results[cand] = {
        "passing_scenes": passing_scenes,
        "n_scenes_pass": n_scenes_pass,
        "gateA_pass": n_scenes_pass >= 2,
        "gateB_1cos_delta": gateB,
        "gateB_pass": all(v is not None and v < 0 for v in gateB.values()),
    }

print(json.dumps({str(k): v for k,v in gate_results.items()}, indent=2, default=str))
torch.save({"comparison": {f"{k[0]}|{k[1]}": v for k,v in comparison.items()},
            "work_ratios": {f"{k[0]}|{k[1]}": v for k,v in work_ratios.items()},
            "gate_results": {str(k): v for k,v in gate_results.items()},
            "scenes": scenes, "meta": meta},
           os.path.join(OUT, "WV0_GATE_EVALUATION.pt"))
print("saved WV0_GATE_EVALUATION.pt")
