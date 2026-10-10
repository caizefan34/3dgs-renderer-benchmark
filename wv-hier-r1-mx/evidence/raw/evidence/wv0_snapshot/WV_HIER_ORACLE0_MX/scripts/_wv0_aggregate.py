#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX aggregation + gate evaluation.
Reads per-state summaries from states/*.pt, computes:
- per-group + macro + geometry gradient metrics (NMSE, cosine, relL2) with cross-mask stats
- variance + bias vs dense
- structural work ratios W_int (and W_blend proxy) vs dense
- R_E(G) = Error(G)/Error(G16), R_W(G) = Work(G)/Work(G16)
- Pareto front (x=W_int, y=NMSE) and (x=W_blend, y=1-cos)
- Gate A/B/C/D evaluation -> verdict
Emits WV0_GRADIENT_SUMMARY.json, WV0_VARIANCE_BIAS.json, WV0_WORK_SUMMARY.json,
WV0_PARETO.json, WV0_GATE_EVALUATION.json, WV0_VERDICT.json
"""
import sys, os, json, glob, math, itertools
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/wv0_out"
STATE_DIR = os.path.join(OUT, "states")
GRANULARITIES = [16, 8, 4, 1]
BUDGETS = [0.25, 0.40]
ALL_GROUPS = ["means","quats","scales","opacities","colors"]
GEOM_GROUPS = ["means","quats","scales"]   # geometry = means + rotations + scales

def jload(p): return torch.load(p, weights_only=False) if False else __import__("torch").load(p, weights_only=False)

import torch
state_files = sorted(glob.glob(os.path.join(STATE_DIR, "*.pt")))
states = {}
for f in state_files:
    s = torch.load(f, weights_only=False)
    states[s["state_id"]] = s
print(f"loaded {len(states)} states: {sorted(states.keys())}")

def macro(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None

# ---- collect per state/condition metrics ----
grad_summary = {}   # state_id -> condition -> group -> stats + macro
varbias_summary = {}
work_summary = {}
for sid, s in states.items():
    grad_summary[sid] = {}
    work_summary[sid] = {"visible_gaussians": s["work"]["visible_gaussians"], "per_G": {}}
    for G in [str(g) for g in GRANULARITIES]:
        w = s["work"]["per_G"][G]
        work_summary[sid]["per_G"][G] = w
    for ckey, c in s["conditions"].items():
        # ckey = G{G}_p{p}
        grad_summary[sid][ckey] = {}
        for g in ALL_GROUPS:
            gv = c["groups"][g]
            grad_summary[sid][ckey][g] = {
                "nmse": gv["nmse"], "cos": gv["cos"], "rel_l2": gv["rel_l2"],
            }
        # macro across groups
        for metric in ("nmse","cos","rel_l2"):
            grad_summary[sid][ckey][f"macro_{metric}_mean"] = macro(
                [c["groups"][g][metric]["mean"] for g in ALL_GROUPS])
            grad_summary[sid][ckey][f"geom_{metric}_mean"] = macro(
                [c["groups"][g][metric]["mean"] for g in GEOM_GROUPS])
    # variance/bias
    varbias_summary[sid] = {}
    for ckey, c in s["conditions"].items():
        varbias_summary[sid][ckey] = {}
        for g in ALL_GROUPS:
            gv = c["groups"][g]
            varbias_summary[sid][ckey][g] = {
                "variance": gv["variance"], "E_gnorm2": gv["E_gnorm2"],
                "bias_rel_l2": gv["bias_rel_l2"], "bias_cos": gv["bias_cos"],
                "nmse_mean": gv["nmse"]["mean"], "cos_mean": gv["cos"]["mean"],
            }
        for metric in ("nmse","cos","rel_l2"):
            varbias_summary[sid][ckey][f"macro_{metric}_mean"] = macro(
                [c["groups"][g][metric]["mean"] for g in ALL_GROUPS])
            varbias_summary[sid][ckey][f"geom_{metric}_mean"] = macro(
                [c["groups"][g][metric]["mean"] for g in GEOM_GROUPS])

torch.save({"grad": grad_summary, "varbias": varbias_summary, "work": work_summary,
            "states": {sid: {k: s[k] for k in ("scene","iteration","N","ckpt_sha256","resolution","n_cams")}
                       for sid,s in states.items()}},
           os.path.join(OUT, "WV0_GRADIENT_SUMMARY.pt"))
print("saved WV0_GRADIENT_SUMMARY.pt")
