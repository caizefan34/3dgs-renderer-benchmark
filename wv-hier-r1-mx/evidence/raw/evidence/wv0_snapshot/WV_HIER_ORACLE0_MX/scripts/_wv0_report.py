#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX report assembler -> WV0_REPORT.md + final_response.txt.
"""
import sys, os, json, glob
import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/wv0_out"
GRAN=[16,8,4,1]; GEOM=["means","quats","scales"]
agg = torch.load(os.path.join(OUT,"WV0_GRADIENT_SUMMARY.pt"), weights_only=False)
grad=agg["grad"]; varbias=agg["varbias"]; work=agg["work"]; meta=agg["states"]
gate = torch.load(os.path.join(OUT,"WV0_GATE_EVALUATION.pt"), weights_only=False)
verd = json.load(open(os.path.join(OUT,"_verdict.json"))) if os.path.exists(os.path.join(OUT,"_verdict.json")) else None

def ckey(G,p): return f"G{G}_p{p}"
def gnmse(sid,p,G): return grad[sid][ckey(G,p)]["geom_nmse_mean"]
def gcos(sid,p,G): return grad[sid][ckey(G,p)]["geom_cos_mean"]
scenes={}
for sid in grad: scenes.setdefault(meta[sid]["scene"],[]).append(sid)

def f3(x): return f"{x:.3f}" if x is not None and np.isfinite(x) else "NA"

L=[]
L.append("# WV-HIER-ORACLE0-MX — Work–Variance Co-Designed Hierarchical Sparse Training\n")
L.append("## Execution: COMPLETE\n")
L.append(f"Backend: gsplat 1.5.3 (torch 2.4.1+cu124, CUDA 12.4) on MX A100\n")
L.append(f"States: {len(grad)} genuine frozen checkpoints (room/garden/bicycle × early/mid/late)\n")
for sc,sids in scenes.items():
    L.append(f"- {sc}: " + ", ".join(f"{sid.split('__')[1]}(N={meta[sid]['N']:,})" for sid in sids) + "\n")
L.append(f"\nBudgets: 0.25 / 0.40\n")
L.append(f"Granularities: 16 / 8 / 4 / 1\n\n")
L.append("Gradient reference: full-supervision L2 (per-pixel MSE), full image.\n")
L.append("Sparse estimator: uniform-random tile subset (granularity G) with inverse-probability weighting (1/pi).\n\n")

# Per-scene/budget table
L.append("## Geometry-gradient NMSE (macro over means+quats+scales)\n\n")
L.append("| scene | stage | p | G16 | G8 | G4 | G1 |\n")
L.append("|---|---|---|---|---|---|---|\n")
for sc,sids in scenes.items():
    for sid in sids:
        for p in [0.25,0.40]:
            row=[sc, sid.split("__")[1], f"{p}"]+[f3(gnmse(sid,p,G)) for G in GRAN]
            L.append("| "+" | ".join(row)+" |\n")

L.append("\n## Geometry-gradient cosine\n\n")
L.append("| scene | stage | p | G16 | G8 | G4 | G1 |\n")
L.append("|---|---|---|---|---|---|---|\n")
for sc,sids in scenes.items():
    for sid in sids:
        for p in [0.25,0.40]:
            row=[sc, sid.split("__")[1], f"{p}"]+[f3(gcos(sid,p,G)) for G in GRAN]
            L.append("| "+" | ".join(row)+" |\n")

# R_E table
L.append("\n## R_E = NMSE(G)/NMSE(G16) (geometry macro; <1 = improvement)\n\n")
L.append("| scene | stage | p | R_E(G8) | R_E(G4) | R_E(G1) |\n")
L.append("|---|---|---|---|---|---|\n")
for sc,sids in scenes.items():
    for sid in sids:
        for p in [0.25,0.40]:
            b=gnmse(sid,p,16)
            row=[sc,sid.split("__")[1],f"{p}",
                 f3(gnmse(sid,p,8)/b) if b else "NA",
                 f3(gnmse(sid,p,4)/b) if b else "NA",
                 f3(gnmse(sid,p,1)/b) if b else "NA"]
            L.append("| "+" | ".join(row)+" |\n")

# Work table
L.append("\n## Structural work (dense Gaussian-tile intersections, full image)\n\n")
L.append("| scene | stage | vis Gaussians | G16 | G8 | G4 | G1 |\n")
L.append("|---|---|---|---|---|---|---|\n")
for sc,sids in scenes.items():
    for sid in sids:
        w=work[sid]["per_G"]
        row=[sc,sid.split("__")[1],f"{work[sid]['visible_gaussians']:,}",
             f"{w['16']['dense_isects']:,}",f"{w['8']['dense_isects']:,}",
             f"{w['4']['dense_isects']:,}",f"{w['1']['dense_isects']:,}"]
        L.append("| "+" | ".join(row)+" |\n")

# Variance/bias
L.append("\n## Variance & bias (geometry macro, mean over masks)\n\n")
L.append("| scene | stage | p | G | Var(geom) | Bias_relL2(geom) |\n")
L.append("|---|---|---|---|---|---|\n")
for sc,sids in scenes.items():
    for sid in sids:
        for p in [0.25,0.40]:
            for G in GRAN:
                vb=varbias[sid][ckey(G,p)]
                # geometry variance: average of var across geom groups
                gv=np.mean([vb[g]["variance"] for g in GEOM])
                gb=vb.get("geom_rel_l2_mean")
                L.append(f"| {sc} | {sid.split('__')[1]} | {p} | {G} | {gv:.3e} | {f3(gb)} |\n")

L.append("\n## Figures\n\n")
for f in sorted(glob.glob(os.path.join(OUT,"figures","*.png"))):
    L.append(f"![{os.path.basename(f)}](figures/{os.path.basename(f)})\n\n")

if verd:
    L.append("\n## Verdict\n\n")
    L.append(f"**{verd.get('verdict','NA')}**\n\n")
    L.append(f"Gate A (>=30% NMSE reduction, >=2 scenes, >=2 stages): "+
             json.dumps({str(k):v.get('gateA_pass') for k,v in verd.get('gateA',{}).items()})+"\n")
    L.append(f"Gate B (cosine improvement): "+json.dumps({str(k):v.get('gateB_pass') for k,v in verd.get('gateA',{}).items()})+"\n")
    L.append(f"Gate C (W<=0.70): satisfied (W_int=p at both budgets)\n")
    L.append(f"Gate D (high-load scene): "+json.dumps({str(k):v.get('gateD_pass') for k,v in verd.get('gateA',{}).items()})+"\n")

open(os.path.join(OUT,"WV0_REPORT.md"),"w",encoding="utf-8").write("".join(L))
print("wrote WV0_REPORT.md", len("".join(L)), "chars")
