#!/usr/bin/env python3
"""Aggregate normalized P16-RB run JSONs.

Expected normalized run JSON files under --runs recursively.
A P12_C0 record for every scene is required.
"""
import argparse, glob, json, math
from pathlib import Path
from collections import defaultdict

ap=argparse.ArgumentParser()
ap.add_argument("--runs",required=True)
ap.add_argument("--out",required=True)
args=ap.parse_args()

rows=[]
for p in glob.glob(str(Path(args.runs)/"**/*.json"),recursive=True):
    try:d=json.load(open(p))
    except Exception:continue
    req={"method","scene","training_wall_s","metrics_common","resolution","clean","direct_comparable_resolution"}
    if not req.issubset(d):continue
    rows.append(d)

scenes=("room","bicycle","garden")
ref={}
for s in scenes:
    xs=[r for r in rows if r["scene"]==s and r["method"]=="P12_C0" and r.get("clean")]
    if len(xs)!=1:
        raise SystemExit(f"Need exactly one clean P12_C0 normalized record for {s}, got {len(xs)}")
    ref[s]=xs[0]

def admissible(r):
    rr=ref[r["scene"]]["metrics_common"]; m=r["metrics_common"]
    return (m["psnr"] >= rr["psnr"]-0.10 and
            m["ssim"] >= rr["ssim"]-0.003 and
            m["lpips"] <= rr["lpips"]+0.005)

table=[]
for r in rows:
    q=admissible(r)
    comp=bool(r.get("direct_comparable_resolution",False))
    speed=None
    if r["method"]!="P12_C0" and q and comp and r.get("clean"):
        speed=ref[r["scene"]]["training_wall_s"]/r["training_wall_s"]
    table.append({
        "method":r["method"],"scene":r["scene"],
        "wall_s":r["training_wall_s"],
        "psnr":r["metrics_common"]["psnr"],
        "ssim":r["metrics_common"]["ssim"],
        "lpips":r["metrics_common"]["lpips"],
        "quality_admissible":q,
        "direct_comparable_resolution":comp,
        "speedup_vs_C0":speed,
        "native_iterations":r.get("native_iterations"),
        "num_gaussians_final":r.get("num_gaussians_final"),
        "peak_vram_mb":r.get("peak_vram_mb"),
        "clean":r.get("clean")
    })

methods=sorted(set(r["method"] for r in rows if r["method"]!="P12_C0"))
method_summary={}
strong=[]
for m in methods:
    xs=[x for x in table if x["method"]==m]
    bys={x["scene"]:x for x in xs}
    all3=all(s in bys for s in scenes)
    all_qual=all3 and all(bys[s]["quality_admissible"] and bys[s]["direct_comparable_resolution"] and bys[s]["clean"] for s in scenes)
    all_fast=all_qual and all((bys[s]["speedup_vs_C0"] or 0)>1.0 for s in scenes)
    method_summary[m]={"all_three_present":all3,"all_three_admissible":all_qual,"faster_all_three":all_fast}
    if all_fast: strong.append(m)

if strong:
    verdict="RESET_STRONG_BASELINE_FOUND"
else:
    any_front=any(x["quality_admissible"] and x["direct_comparable_resolution"] and
                  x["method"]!="P12_C0" and (x["speedup_vs_C0"] or 0)>1.0 for x in table)
    verdict="RESET_SCENE_DEPENDENT" if any_front else "RESET_P12_REMAINS_STRONG"

out={"verdict":verdict,"table":table,"methods":method_summary,"strong_methods":strong}
Path(args.out).write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"verdict":verdict,"strong_methods":strong},indent=2))
