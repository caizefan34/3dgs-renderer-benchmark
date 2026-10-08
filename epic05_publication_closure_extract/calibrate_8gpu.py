#!/usr/bin/env python3
"""Generate and validate P0: paired 2K room calibration on all 8 GPUs."""
import argparse, json, math, pathlib, subprocess, os, statistics, shlex
ap=argparse.ArgumentParser()
ap.add_argument("--trainer",required=True)
ap.add_argument("--python",default="/root/miniforge3/envs/gsplat/bin/python")
ap.add_argument("--out",default="/root/3dgs-publication-epic05/calibration")
a=ap.parse_args()
out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
rows=[]
for gpu in range(8):
    for arm,eps in [("b1a",0.1),("c0",0.3)]:
        rd=out/f"gpu{gpu}_{arm}_room_2k"; rd.mkdir(exist_ok=True)
        cmd=[a.python,a.trainer,"--arm",arm,"--scene","room","--iterations","2000",
             "--outdir",str(rd),"--gpu","0","--final-eval","subset",
             "--timing-grade","FUNCTIONAL_ONLY","--eps2d",str(eps)]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        with open(rd/"stdout.log","w") as f:
            rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc: raise SystemExit(f"calibration failed gpu{gpu} {arm}")
        r=json.load(open(rd/"results.json"))
        wall=r["timing"]["total_wall_s"]
        rows.append({"gpu":gpu,"arm":arm,"wall_s":wall})
json.dump(rows,open(out/"calibration_rows.json","w"),indent=2)

def vals(arm): return [x["wall_s"] for x in rows if x["arm"]==arm]
report={}
for arm in ("b1a","c0"):
    v=vals(arm); med=statistics.median(v)
    report[arm]={"median":med,"min":min(v),"max":max(v),
                 "max_rel_deviation":max(abs(x-med)/med for x in v)}
ratios=[]
for gpu in range(8):
    b=next(x["wall_s"] for x in rows if x["gpu"]==gpu and x["arm"]=="b1a")
    c=next(x["wall_s"] for x in rows if x["gpu"]==gpu and x["arm"]=="c0")
    ratios.append(b/c)
report["speedup_by_gpu"]=ratios
# Gate is deliberately about device homogeneity, not scientific speedup.
report["pass"]=all(report[a]["max_rel_deviation"] <= 0.05 for a in ("b1a","c0"))
json.dump(report,open(out/"calibration_report.json","w"),indent=2)
print(json.dumps(report,indent=2))
if not report["pass"]:
    raise SystemExit("P0 FAIL: >5% per-arm wall deviation across EPIC GPUs. Restrict or pair on validated GPUs.")
