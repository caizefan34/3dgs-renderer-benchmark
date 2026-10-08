#!/usr/bin/env python3
import argparse, glob, json, math, statistics
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--units",required=True)
ap.add_argument("--out",required=True)
args=ap.parse_args()

rows=[]
for p in glob.glob(str(Path(args.units)/"**/*.json"), recursive=True):
    try:
        d=json.load(open(p))
    except Exception:
        continue
    if d.get("tau") != 2.0:  # primary verdict only
        continue
    if "scene" in d and "camera_id" in d and "metrics" in d and "step" in d:
        rows.append(d)

def med(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(xs) if xs else None

def quantile(xs,q):
    xs=sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:return None
    if len(xs)==1:return xs[0]
    pos=(len(xs)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi:return xs[lo]
    return xs[lo]*(hi-pos)+xs[hi]*(pos-lo)

scenes=sorted(set(r["scene"] for r in rows))
agg={}
for s in scenes:
    xs=[r for r in rows if r["scene"]==s]
    parity=all(bool(x.get("d1_parity_pass")) for x in xs)
    R=[x["step"].get("R_step") for x in xs if x["step"].get("reference_improving") and x["step"].get("R_step") is not None]
    aud=[x["step"].get("P_audit") for x in xs]
    a={
      "n":len(xs),
      "parity_all":parity,
      "interaction_median":med([x["oracle_interaction_ratio"] for x in xs]),
      "interaction_p75":quantile([x["oracle_interaction_ratio"] for x in xs],.75),
      "means_cos_median":med([x["metrics"]["means"]["cosine"] for x in xs]),
      "means_cos_p10":quantile([x["metrics"]["means"]["cosine"] for x in xs],.10),
      "scales_cos_median":med([x["metrics"]["scales"]["cosine"] for x in xs]),
      "quats_cos_median":med([x["metrics"]["quats"]["cosine"] for x in xs]),
      "opacity_cos_median":med([x["metrics"]["opacity"]["cosine"] for x in xs]),
      "sh0_cos_median":med([x["metrics"]["sh0"]["cosine"] for x in xs]),
      "R_step_median":med(R),
      "R_step_in_90_110":sum(1 for v in R if .90 <= v <= 1.10),
      "R_step_n":len(R),
      "audit_penalty_median":med(aud),
      "audit_penalty_gt_1pct":sum(1 for v in aud if v is not None and v > .01),
    }
    a["candidate"]=(
      a["n"]==12 and a["parity_all"] and
      a["interaction_median"] is not None and a["interaction_median"]<=.30 and
      a["interaction_p75"] is not None and a["interaction_p75"]<=.35 and
      a["means_cos_median"] is not None and a["means_cos_median"]>=.995 and
      a["means_cos_p10"] is not None and a["means_cos_p10"]>=.990 and
      a["scales_cos_median"] is not None and a["scales_cos_median"]>=.990 and
      a["quats_cos_median"] is not None and a["quats_cos_median"]>=.990 and
      a["opacity_cos_median"] is not None and a["opacity_cos_median"]>=.995 and
      a["sh0_cos_median"] is not None and a["sh0_cos_median"]>=.990 and
      a["R_step_median"] is not None and .95<=a["R_step_median"]<=1.05 and
      a["R_step_in_90_110"]>=10 and
      a["audit_penalty_median"] is not None and a["audit_penalty_median"]<=.002 and
      a["audit_penalty_gt_1pct"]<=1
    )
    a["hard_drop"]=(
      (a["interaction_median"] is not None and a["interaction_median"]>.50) or
      (a["means_cos_median"] is not None and a["means_cos_median"]<.980) or
      (a["opacity_cos_median"] is not None and a["opacity_cos_median"]<.980) or
      (a["R_step_median"] is not None and a["R_step_median"]<.80) or
      a["audit_penalty_gt_1pct"]>=4
    )
    agg[s]=a

# Invalid parity is stronger than scientific verdict.
if any(not a["parity_all"] for a in agg.values()):
    verdict="P17_EXPORT_INVALID"
elif any(a["hard_drop"] for s,a in agg.items() if s in ("room","bicycle")):
    verdict="P17_P0_HARD_DROP"
elif all(agg.get(s,{}).get("candidate",False) for s in ("room","bicycle")):
    if "garden" not in agg:
        verdict="P17_P0_STAGE_A_STRONG"
    elif agg["garden"]["candidate"]:
        verdict="P17_P0_GO_MULTIRATE_REALIZABILITY"
    else:
        verdict="P17_P0_HOLD_CROSS_SCENE"
else:
    verdict="P17_P0_HOLD_NO_STRONG_ORACLE"

out={"verdict":verdict,"primary_tau":2.0,"scenes":agg}
Path(args.out).write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"verdict":verdict,"scenes":agg},indent=2,allow_nan=False))
