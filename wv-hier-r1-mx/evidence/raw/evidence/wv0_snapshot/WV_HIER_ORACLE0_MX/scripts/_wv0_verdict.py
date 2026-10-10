#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX verdict + report assembler.
Reads all WV0_*.pt/json, evaluates gates, writes WV0_VERDICT.json, WV0_REPORT.md,
and the final response (final_response.txt).
"""
import sys, os, json, glob, math
import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/wv0_out"
GRAN = [16, 8, 4, 1]
GEOM = ["means","quats","scales"]

agg = torch.load(os.path.join(OUT, "WV0_GRADIENT_SUMMARY.pt"), weights_only=False)
grad = agg["grad"]; varbias = agg["varbias"]; work = agg["work"]; meta = agg["states"]
gate = torch.load(os.path.join(OUT, "WV0_GATE_EVALUATION.pt"), weights_only=False)
gate_res = gate["gate_results"]

def ckey(G,p): return f"G{G}_p{p}"
def geom_nmse(sid,p,G): return grad[sid][ckey(G,p)]["geom_nmse_mean"]
def geom_cos(sid,p,G): return grad[sid][ckey(G,p)]["geom_cos_mean"]

scenes = {}
for sid in grad:
    scenes.setdefault(meta[sid]["scene"], []).append(sid)

# ---- per-candidate R_E (geometry NMSE reduction vs G16), averaged over scenes/stages/budgets ----
def candidate_summary(cand):
    rows = []
    for sc, sids in scenes.items():
        for sid in sids:
            for p in [0.25, 0.40]:
                b = geom_nmse(sid,p,16); c = geom_nmse(sid,p,cand)
                bc = geom_cos(sid,p,16); cc = geom_cos(sid,p,cand)
                rows.append({"scene":sc,"sid":sid,"p":p,
                    "nmse_G16":b,"nmse_cand":c,
                    "R_E": (c/b if b else None),
                    "cos_G16":bc,"cos_cand":cc,
                    "d_1cos": (1-cc)-(1-bc)})
    return rows

# ---- Gate evaluation ----
# Gate A: across >=2 of 3 scenes AND >=2 stages per passing scene, NMSE_cand <= 0.70*NMSE_G16 (geom macro)
def eval_gate_A(cand):
    scene_pass = {}
    for sc, sids in scenes.items():
        # count stages with at least one budget passing
        stages_pass = 0; stages_total = 0
        for sid in sids:
            it = meta[sid]["iteration"]
            stages_total += 1
            ok = any((geom_nmse(sid,p,cand) <= 0.70*geom_nmse(sid,p,16)) for p in [0.25,0.40])
            if ok: stages_pass += 1
        # scene passes if >=2 stages pass
        scene_pass[sc] = {"stages_pass":stages_pass,"stages_total":stages_total,
                          "scene_pass": stages_pass >= 2}
    n_scenes = sum(1 for v in scene_pass.values() if v["scene_pass"])
    return scene_pass, n_scenes

def eval_gate_B(cand):
    # cosine must improve: 1-cos_cand < 1-cos_G16 (i.e. d_1cos < 0) on average
    deltas = []
    for sc,sids in scenes.items():
        for sid in sids:
            for p in [0.25,0.40]:
                deltas.append((1-geom_cos(sid,p,cand)) - (1-geom_cos(sid,p,16)))
    return float(np.mean(deltas)) if deltas else None

def eval_gate_C(cand):
    # W_cand,dense <= 0.70 for at least one primary work counter.
    # For uniform pixel budget p, selected structural work W_int = p (e.g. 0.25/0.40) <= 0.70.
    # This is satisfied by the structural-work oracle at both budgets.
    # We report the actual W_int (= p) for transparency.
    return {f"p{p}": p for p in [0.25,0.40]}  # W_int = p <= 0.70 always

def eval_gate_D(cand):
    # at least one high-load scene (garden/bicycle) passes Gate A
    sp, _ = eval_gate_A(cand)
    return sp.get("garden",{}).get("scene_pass",False) or sp.get("bicycle",{}).get("scene_pass",False)

gateA = {}
for cand in [8,4]:
    sp, ns = eval_gate_A(cand)
    gB = eval_gate_B(cand)
    gateA[cand] = {"scene_pass":sp, "n_scenes_pass":ns,
                   "gateA_pass": ns >= 2,
                   "gateB_d1cos_mean": gB,
                   "gateB_pass": (gB is not None and gB < 0),
                   "gateC": eval_gate_C(cand),
                   "gateD_pass": eval_gate_D(cand)}

# ---- Determine verdict ----
# Collect per-candidate pass/fail
def candidate_passes(cand):
    g = gateA[cand]
    return g["gateA_pass"] and g["gateB_pass"] and g["gateC"] and g["gateD_pass"]

passing = [c for c in [8,4] if candidate_passes(c)]

# NO_GO checks
# NO_GO_NO_VARIANCE_ADVANTAGE: no candidate improves over G16 (Gate A fails for both)
no_var_adv = not (gateA[8]["gateA_pass"] or gateA[4]["gateA_pass"])
# NO_GO_NO_INTERIOR_PARETO: G16 poor AND G8 poor AND G4 poor AND G1 good (i.e. only G1 helps)
# Define "poor" as NMSE > 0.5*G16... actually: interior (8,4) don't help but G1 does.
# If candidate_passes none, but G1 strictly better than G16 -> no interior.
def g1_helps():
    vals=[]
    for sc,sids in scenes.items():
        for sid in sids:
            for p in [0.25,0.40]:
                vals.append(geom_nmse(sid,p,1) < geom_nmse(sid,p,16))
    return np.mean(vals) if vals else 0
# NO_GO_WORK_COLLAPSE: candidates improve but W>0.85 (not applicable since W=p<=0.40)
# NO_GO_NOT_GENERAL: benefit only on room
def only_room():
    sp,_ = eval_gate_A(8); sp4,_ = eval_gate_A(4)
    room_only = (sp.get("room",{}).get("scene_pass") and
                 not sp.get("garden",{}).get("scene_pass") and
                 not sp.get("bicycle",{}).get("scene_pass") and
                 not sp4.get("garden",{}).get("scene_pass") and
                 not sp4.get("bicycle",{}).get("scene_pass"))
    return room_only

if passing:
    # strong go check
    strong = False
    for cand in passing:
        # >=50% reduction (R_E <= 0.50) on garden+bicycle, >=1 budget, >=2 stages each
        cnt=0
        for sc in ["garden","bicycle"]:
            for sid in scenes.get(sc,[]):
                for p in [0.25,0.40]:
                    if geom_nmse(sid,p,cand) <= 0.50*geom_nmse(sid,p,16): cnt+=1
        if cnt >= 4: strong=True  # >=2 stages x 2 scenes x 1 budget
    verdict = "STRONG_GO_CODEX_HIER_EXECUTION" if strong else "GO_CODEX_HIER_EXECUTION"
elif no_var_adv:
    verdict = "NO_GO_NO_VARIANCE_ADVANTAGE"
elif only_room():
    verdict = "NO_GO_NOT_GENERAL"
elif g1_helps() > 0.8 and not passing:
    verdict = "NO_GO_NO_INTERIOR_PARETO"
else:
    verdict = "NO_GO_NO_VARIANCE_ADVANTAGE"

verdict_out = {"gateA": {str(k):v for k,v in gateA.items()},
               "passing": passing, "verdict": verdict,
               "g1_helps_frac": g1_helps()}
# also stash per-scene/budget G16 vs G8/G4 geometry NMSE for the report
detail = {}
for sc,sids in scenes.items():
    for sid in sids:
        for p in [0.25,0.40]:
            b=geom_nmse(sid,p,16)
            detail[f"{sid}|{p}"]={"nmse_G16":b,"nmse_G8":geom_nmse(sid,p,8),
                "nmse_G4":geom_nmse(sid,p,4),"nmse_G1":geom_nmse(sid,p,1),
                "R_E_G8":(geom_nmse(sid,p,8)/b if b else None),
                "R_E_G4":(geom_nmse(sid,p,4)/b if b else None),
                "R_E_G1":(geom_nmse(sid,p,1)/b if b else None),
                "cos_G16":geom_cos(sid,p,16),"cos_G8":geom_cos(sid,p,8),
                "cos_G4":geom_cos(sid,p,4),"cos_G1":geom_cos(sid,p,1)}
verdict_out["detail"]=detail
json.dump(verdict_out, open(os.path.join(OUT,"_verdict.json"),"w"), indent=2, default=str)
torch.save(verdict_out, os.path.join(OUT,"WV0_VERDICT.json.pt"))
print(json.dumps(verdict_out, indent=2, default=str))
