#!/usr/bin/env python3
"""
Frozen OA0 downstream analyzer.
Do not change thresholds after OA0 data are observed.

Inputs:
  --features OA0_ANCHOR_FEATURES.csv|parquet
  --labels N1D_LABELS_FROZEN.csv
Output:
  OA0_ANALYSIS.json

Expected feature rows:
scene,checkpoint,anchor_id,group,
lowv_q1_update_energy_share,
zero_grad_update_energy_share,
moment_conflict_update_energy_share,
update_top1_energy_share
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PRED = [
    "lowv_q1_update_energy_share",
    "zero_grad_update_energy_share",
    "moment_conflict_update_energy_share",
    "update_top1_energy_share",
]
GROUPS = ["xyz", "scale", "quat", "opacity", "sh_dc", "sh_rest"]

def read_table(path):
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

def finite_or_none(x):
    return None if x is None or not np.isfinite(x) else float(x)

def transfer(train, test, cols):
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    ytr, yte = train["damping_rescue"].astype(int).to_numpy(), test["damping_rescue"].astype(int).to_numpy()
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"error":"single_class"}
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight=None, max_iter=10000, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    pr = clf.predict_proba(sc.transform(Xte))[:,1]
    return {
        "roc_auc": float(roc_auc_score(yte, pr)),
        "average_precision": float(average_precision_score(yte, pr)),
        "brier": float(brier_score_loss(yte, pr)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

f = read_table(args.features)
l = pd.read_csv(args.labels)
req = {"scene","checkpoint","anchor_id","group",*PRED}
miss = req - set(f.columns)
if miss: raise SystemExit(f"Missing features: {sorted(miss)}")
if not set(GROUPS).issubset(set(f["group"].unique())):
    raise SystemExit(f"Expected groups {GROUPS}; found {sorted(f['group'].unique())}")

m = f.merge(l, on=["scene","checkpoint","anchor_id"], how="inner", validate="many_to_one")
if len(m) != len(f): raise SystemExit("Feature/label join lost rows")
if not np.isfinite(m[PRED].to_numpy(float)).all():
    raise SystemExit("Nonfinite primary OA0 feature: hard data-quality failure")

assoc = {}
consistent_cells = {}
for group in GROUPS:
    assoc[group] = {}
    gg = m[m.group == group]
    for feat in PRED:
        assoc[group][feat] = {}
        for scene in ["room","bicycle"]:
            x = gg[gg.scene == scene]
            rho, p = spearmanr(x[feat], x["overshoot_gain"])
            q1, q5 = x[feat].quantile([.2,.8])
            low = x[x[feat] <= q1]
            high = x[x[feat] >= q5]
            assoc[group][feat][scene] = {
                "spearman": finite_or_none(rho),
                "p": finite_or_none(p),
                "top_minus_bottom_mean_overshoot": float(high.overshoot_gain.mean()-low.overshoot_gain.mean()),
                "top_native_harmful_rate": float(high.native_harmful.mean()),
                "bottom_native_harmful_rate": float(low.native_harmful.mean()),
                "top_damping_rescue_rate": float(high.damping_rescue.mean()),
                "bottom_damping_rescue_rate": float(low.damping_rescue.mean()),
            }

# Wide anchor features: exact four predictors x six groups.
wide = m.pivot_table(
    index=["scene","checkpoint","anchor_id","damping_rescue","overshoot_gain","native_harmful"],
    columns="group", values=PRED, aggfunc="first"
)
wide.columns = [f"{feat}__{group}" for feat,group in wide.columns]
wide = wide.reset_index()
cols = [f"{feat}__{group}" for feat in PRED for group in GROUPS]
missing = [c for c in cols if c not in wide.columns]
if missing: raise SystemExit(f"Missing wide cols: {missing}")

tr_r = wide[wide.scene=="room"]; te_b = wide[wide.scene=="bicycle"]
tr_b = wide[wide.scene=="bicycle"]; te_r = wide[wide.scene=="room"]
transfer_res = {
    "room_to_bicycle": transfer(tr_r,te_b,cols),
    "bicycle_to_room": transfer(tr_b,te_r,cols),
}

prevalence = {
    s: float(wide[wide.scene==s].damping_rescue.mean()) for s in ["room","bicycle"]
}

# Frozen gate: pick whether ANY single group-feature meets scene pooled and cell sign consistency.
strong_candidates=[]; moderate_candidates=[]
for group in GROUPS:
  for feat in PRED:
    vals={}
    ok=True
    signs=[]
    for scene in ["room","bicycle"]:
        rho=assoc[group][feat][scene]["spearman"]
        vals[scene]=rho
        if rho is None: ok=False
    if not ok: continue
    target_sign = np.sign(vals["room"] + vals["bicycle"])
    if target_sign == 0: target_sign=1
    cell_cons=0
    for scene in ["room","bicycle"]:
      for ck in [15000,20000,25000]:
        x=m[(m.group==group)&(m.scene==scene)&(m.checkpoint==ck)]
        rho,_=spearmanr(x[feat],x["overshoot_gain"])
        if np.isfinite(rho) and np.sign(rho)==target_sign: cell_cons+=1
    if min(abs(vals["room"]),abs(vals["bicycle"])) >= .35 and cell_cons >= 5:
        strong_candidates.append({"group":group,"feature":feat,"rho":vals,"consistent_cells":cell_cons})
    if min(abs(vals["room"]),abs(vals["bicycle"])) >= .25:
        moderate_candidates.append({"group":group,"feature":feat,"rho":vals,"consistent_cells":cell_cons})

auc_rb = transfer_res["room_to_bicycle"].get("roc_auc",-1)
auc_br = transfer_res["bicycle_to_room"].get("roc_auc",-1)

if (min(prevalence.values()) >= .20 and strong_candidates and auc_rb >= .70 and auc_br >= .70):
    verdict="OA0_STRONG_OPTIMIZER_PATHOLOGY"
elif (min(prevalence.values()) >= .10 and moderate_candidates and auc_rb >= .62 and auc_br >= .62):
    verdict="OA0_MODERATE_OPTIMIZER_PATHOLOGY"
else:
    verdict="OA0_NO_STABLE_OPTIMIZER_PATHOLOGY"

out = {
    "verdict":verdict,
    "damping_rescue_prevalence":prevalence,
    "associations":assoc,
    "transfer":transfer_res,
    "strong_candidates":strong_candidates,
    "moderate_candidates":moderate_candidates,
    "n_anchor_rows":int(len(wide)),
}
Path(args.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"verdict":verdict,"prevalence":prevalence,"transfer":transfer_res},indent=2))
