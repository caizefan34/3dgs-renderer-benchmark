"""Aggregate P-TRANSITION units using the preregistered gates."""
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path

def median(xs):
    ys=sorted(xs); n=len(ys)
    return ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2

def ranks(vals):
    order=sorted(range(len(vals)), key=lambda i: vals[i])
    r=[0.0]*len(vals); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and vals[order[j+1]]==vals[order[i]]: j+=1
        rr=(i+j)/2+1
        for k in range(i,j+1): r[order[k]]=rr
        i=j+1
    return r

def spearman(x,y):
    rx,ry=ranks(x),ranks(y)
    mx,my=sum(rx)/len(rx),sum(ry)/len(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx=sum((a-mx)**2 for a in rx); dy=sum((b-my)**2 for b in ry)
    return num/math.sqrt(dx*dy) if dx>0 and dy>0 else 0.0

ap=argparse.ArgumentParser()
ap.add_argument("--inputs", required=True, help="directory containing transition_*.json")
ap.add_argument("--out", required=True)
args=ap.parse_args()

rows=[]
for p in sorted(glob.glob(str(Path(args.inputs)/"*.json"))):
    d=json.load(open(p))
    if d.get("status")!="P_TRANSITION_UNIT_COMPLETE":
        continue
    b=d["best_by_eta"]
    rows.append({
        "file":p, "scene":d["scene"], "iteration":int(d["iteration"]), "seed":int(d["seed"]),
        "R_eta":float(d["R_eta"]),
        "screen_lr":b["screen"]["lr"], "geometry_lr":b["geometry"]["lr"],
        "screen_eta":b["screen"]["eta"], "geometry_eta":b["geometry"]["eta"],
        "screen_audit_per_train":b["screen"]["audit_per_train"],
        "geometry_audit_per_train":b["geometry"]["audit_per_train"],
        "screen_memory_ratio":d["screen_memory_ratio"],
    })

groups=defaultdict(list)
for r in rows: groups[(r["scene"],r["iteration"])].append(r)

summary={}
for (scene,it), rs in sorted(groups.items()):
    summary[f"{scene}:{it}"]={
        "n":len(rs),
        "median_R_eta":median([r["R_eta"] for r in rs]),
        "seeds_R_gt_1":sum(r["R_eta"]>1.0 for r in rs),
        "screen_audit_no_worse_count":sum(
            r["screen_audit_per_train"] <= r["geometry_audit_per_train"] for r in rs
        ),
        "median_screen_memory_ratio":median([r["screen_memory_ratio"] for r in rs]),
        "rows":rs,
    }

scenes=sorted(set(r["scene"] for r in rows))
scene_verdicts={}
for s in scenes:
    keys={it:summary.get(f"{s}:{it}") for it in (5000,8000,11000,14000)}
    complete_2=all(keys[i] and keys[i]["n"]>=3 for i in (5000,14000))
    two=False
    if complete_2:
        two=(keys[5000]["median_R_eta"]<=0.90 and
             keys[14000]["median_R_eta"]>=1.15 and
             keys[14000]["seeds_R_gt_1"]>=2 and
             keys[14000]["screen_audit_no_worse_count"]>=2)
    complete_4=all(keys[i] and keys[i]["n"]>=3 for i in (5000,8000,11000,14000))
    four=False; rho=None; medians=None
    if complete_4 and two:
        stages=[5000,8000,11000,14000]
        medians=[keys[i]["median_R_eta"] for i in stages]
        rho=spearman(stages,[math.log(max(x,1e-30)) for x in medians])
        crossed=any(medians[i]<1.0 and medians[i+1]>1.0 for i in range(3))
        no_big_reverse=all(medians[i+1] >= medians[i]-0.20 for i in range(3))
        four=(rho>=0.8 and crossed and no_big_reverse)
    scene_verdicts[s]={
        "two_point_signal":two,
        "four_point_signal":four,
        "spearman_stage_log_R":rho,
        "stage_median_R_eta":medians,
        "has_5k_14k":complete_2,
        "has_all_4_stages":complete_4,
    }

both_two=(len(scene_verdicts)>=2 and all(v["two_point_signal"] for v in scene_verdicts.values()))
both_four=(len(scene_verdicts)>=2 and all(v["four_point_signal"] for v in scene_verdicts.values()))
if both_four:
    verdict="MECHANISM_SIGNAL_STAGE_DEPENDENT"
elif both_two:
    verdict="TWO_POINT_STAGE_CROSSOVER_SIGNAL"
elif any(not v["has_all_4_stages"] for v in scene_verdicts.values()):
    verdict="TWO_POINT_PRECHECK_ONLY"
else:
    verdict="HOLD_NO_REPRODUCIBLE_TRANSITION"

out={"verdict":verdict,"n_units":len(rows),"groups":summary,"scenes":scene_verdicts}
Path(args.out).write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({"verdict":verdict,"n_units":len(rows),"scenes":scene_verdicts},indent=2))
