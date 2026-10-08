#!/usr/bin/env python3
"""Consolidate raw census_*.json into FUSED_HOTSPOT0_KERNEL_CENSUS.json."""
import json, os, datetime
P="/mnt/storage_pool/liaoyuanjun/fused_profile0"; HOT=P+"/FUSED_HOTSPOT0"; RAW=HOT+"/raw"
SCENES=["room","bicycle","garden"]; STAGES=["s1_5k","s2_10k","s3_mature"]
GEN=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(p):
    with open(p) as f: return json.load(f)
per={}
for sc in SCENES:
    for st in STAGES:
        per[f"{sc}_{st}"]=load(f"{RAW}/census_{sc}_{st}.json")
# cross-scene mature kernel aggregation (by mangled name -> mean pct across 3 scenes)
def top_kernels(stage):
    agg={}
    for sc in SCENES:
        for k in per[f"{sc}_{stage}"]["kernels"]:
            nm=k["name"]
            agg.setdefault(nm,{"pct_iter":[], "pct_kernel":[], "self_us":[], "count":[]})
            agg[nm]["pct_iter"].append(k["pct_iteration_time"])
            agg[nm]["pct_kernel"].append(k["pct_kernel_time"])
            agg[nm]["self_us"].append(k["total_us_per_iter"])
            agg[nm]["count"].append(k["count_per_iter"])
    rows=[]
    for nm,d in agg.items():
        rows.append({"name":nm,"mean_pct_iteration_time_mature":round(sum(d["pct_iter"])/len(d["pct_iter"]),3),
                     "mean_pct_kernel_time_mature":round(sum(d["pct_kernel"])/len(d["pct_kernel"]),3),
                     "mean_self_us_per_iter_mature":round(sum(d["self_us"])/len(d["self_us"]),2)})
    rows.sort(key=lambda r:-r["mean_pct_iteration_time_mature"])
    return rows
out={"schema":"FUSED_HOTSPOT0_KERNEL_CENSUS","generated_utc":GEN,
     "method":"H2 torch.profiler chrome trace, per-kernel self_cuda_time aggregated by name; normalized against H1 clean PL2 iteration time. Profiler wall != production wall.",
     "counters_status":"COUNTERS_UNAVAILABLE (ncu unusable; census from chrome trace self-time).",
     "per_scene_stage":{k:{"iters_profiled":v["iters_profiled"],"trace":v["trace"],"T_iter_clean_ms":v["T_iter_clean_ms"],
                            "total_kernel_ms_per_iter":v["total_kernel_ms_per_iter"],"kernels":v["kernels"]} for k,v in per.items()},
     "mature_top_kernels_cross_scene":top_kernels("s3_mature")}
json.dump(out, open(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CENSUS.json","w"), indent=2)
print("KERNEL_CENSUS consolidated; mature top kernels:")
for r in out["mature_top_kernels_cross_scene"][:10]:
    print(f"  {r['mean_pct_iteration_time_mature']:5.2f}%it {r['mean_pct_kernel_time_mature']:5.1f}%kern  {r['name'][:70]}")