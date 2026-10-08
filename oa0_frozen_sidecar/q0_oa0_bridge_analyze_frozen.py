#!/usr/bin/env python3
"""
Frozen Q0<->OA0 bridge analyzer.
Expected table has one row per Q0 cluster/group with:
scene, checkpoint, group, null_frac,
lowv_q1_update_energy_share, zero_grad_update_energy_share,
moment_conflict_update_energy_share, update_top1_energy_share
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr

P=["lowv_q1_update_energy_share","zero_grad_update_energy_share",
   "moment_conflict_update_energy_share","update_top1_energy_share"]
ap=argparse.ArgumentParser()
ap.add_argument("--table",required=True)
ap.add_argument("--q0-verdict",required=True,
 choices=["Q0_STRONG_QUOTIENT_SIGNAL","Q0_MODERATE_QUOTIENT_SIGNAL",
          "Q0_NO_MATERIAL_QUOTIENT_SIGNAL","Q0_SKETCH_UNSTABLE"])
ap.add_argument("--out",required=True)
a=ap.parse_args()
df=pd.read_csv(a.table) if not a.table.endswith(".parquet") else pd.read_parquet(a.table)
res={}
passing=[]
for feat in P:
    res[feat]={}
    signs=[]
    ok=True
    for scene in ["room","bicycle"]:
        x=df[df.scene==scene]
        rho,p=spearmanr(x[feat],x.null_frac)
        res[feat][scene]={"spearman":float(rho),"p":float(p)}
        if not np.isfinite(rho) or abs(rho)<.35: ok=False
        signs.append(np.sign(rho))
        for ck in [20000,25000]:
            y=x[x.checkpoint==ck]
            rr,_=spearmanr(y[feat],y.null_frac)
            res[feat][scene][str(ck)] = None if not np.isfinite(rr) else float(rr)
    same_scene_sign=(signs[0]!=0 and signs[0]==signs[1])
    temporal=True
    base=signs[0] if same_scene_sign else 0
    if base:
        for scene in ["room","bicycle"]:
            for ck in [20000,25000]:
                rr=res[feat][scene][str(ck)]
                if rr is None or np.sign(rr)!=base: temporal=False
    else: temporal=False
    if ok and same_scene_sign and temporal:
        passing.append(feat)
qok=a.q0_verdict in ["Q0_STRONG_QUOTIENT_SIGNAL","Q0_MODERATE_QUOTIENT_SIGNAL"]
verdict="OA0_Q0_GEOMETRY_LINK" if qok and len(passing)>=2 else "OA0_Q0_NO_GEOMETRY_LINK"
out={"verdict":verdict,"q0_verdict":a.q0_verdict,"passing_features":passing,"associations":res}
Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"verdict":verdict,"passing_features":passing},indent=2))
