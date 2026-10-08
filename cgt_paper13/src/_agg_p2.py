import json, glob, os
import numpy as np
HOME="/mnt/storage_pool/liaoyuanjun/cgt_paper13"
DS={"bicycle":"mipnerf360","flowers":"mipnerf360","garden":"mipnerf360","stump":"mipnerf360",
 "treehill":"mipnerf360","bonsai":"mipnerf360","counter":"mipnerf360","kitchen":"mipnerf360","room":"mipnerf360",
 "train":"tanksandtemples","truck":"tanksandtemples","drjohnson":"deepblending","playroom":"deepblending"}
rows={}
for f in glob.glob(f"{HOME}/results/p2/cgt_*.json"):
    d=json.load(open(f))
    if d.get("status")=="OK": rows[d["scene"]]=d
scenes=sorted(rows.keys())
S=np.array([rows[s]["S_mixed"] for s in scenes])
def gm(a): return float(np.exp(np.mean(np.log(np.array(a)))))
agg={
 "stage":"CGT-PAPER13 P2 CGT mixed-forward benchmark (13-scene)",
 "date_utc":"2026-10-04","host":"MX A100-PCIE-40GB, runtime liaoyuanjun_mx0qual",
 "method":"frozen CGT-F0 (R=4, GATE=0.02, kernel-correctness 2e-5, TILE=16, seed 20261004)",
 "workload":"5 views/scene x 10 tiles/view = 50 tiles/scene, work-decile stratified (F0-style)",
 "n_scenes":len(scenes),
 "S_mixed_geomean":gm(S),"S_mixed_median":float(np.median(S)),
 "S_mixed_min":float(S.min()),"S_mixed_max":float(S.max()),
 "S_mixed_arithmetic_mean":float(S.mean()),
 "per_scene":{s:{"S_mixed":rows[s]["S_mixed"],"safe_fraction":rows[s]["certificate"]["safe_fraction"],
   "weighted_fallback":rows[s]["certificate"]["weighted_fallback_fraction"],
   "kernel_max_abs":rows[s]["correctness"]["kernel_max_abs"],
   "kernel_pass":rows[s]["correctness"]["kernel_all_passed"],
   "cgt_vs_canon_max_abs":rows[s]["correctness"]["cgt_vs_canonical_max_abs"],
   "M_median":rows[s]["M_median"],"T_build_ms":rows[s]["T_build_ms"],
   "T_teacher_ms":rows[s]["T_teacher_all_ms"],"T_fused_ms":rows[s]["T_fused_mixed_ms"]} for s in scenes},
 "dataset_aggregate":{},
 "certificate_pooled":{
   "safe_fraction_mean":float(np.mean([rows[s]["certificate"]["safe_fraction"] for s in scenes])),
   "weighted_fallback_mean":float(np.mean([rows[s]["certificate"]["weighted_fallback_fraction"] for s in scenes])),
   "rigorous_pass_rate_mean":float(np.mean([rows[s]["certificate"]["rigorous_pass_rate"] for s in scenes])),
 },
 "correctness_pooled":{
   "all_kernel_pass":all(rows[s]["correctness"]["kernel_all_passed"] for s in scenes),
   "max_kernel_max_abs":float(max(rows[s]["correctness"]["kernel_max_abs"] for s in scenes)),
   "cgt_vs_canon_max_abs_max":float(max(rows[s]["correctness"]["cgt_vs_canonical_max_abs"] for s in scenes)),
 },
}
for ds in ["mipnerf360","tanksandtemples","deepblending"]:
    ss=[rows[s]["S_mixed"] for s in scenes if DS[s]==ds]
    agg["dataset_aggregate"][ds]={"n":len(ss),"S_mixed_geomean":gm(ss),"S_mixed_median":float(np.median(ss))}
allss=[rows[s]["S_mixed"] for s in scenes]
agg["dataset_aggregate"]["all_13"]={"n":len(allss),"S_mixed_geomean":gm(allss),"S_mixed_median":float(np.median(allss))}
open(f"{HOME}/results/PAPER13_CGT_RESULTS.json","w").write(json.dumps(agg,indent=2))
print("S_mixed geomean=%.4f median=%.4f min=%.4f max=%.4f"%(agg["S_mixed_geomean"],agg["S_mixed_median"],agg["S_mixed_min"],agg["S_mixed_max"]))
print("all kernel pass:",agg["correctness_pooled"]["all_kernel_pass"],"max kernel_max_abs=%.2e"%agg["correctness_pooled"]["max_kernel_max_abs"])
for ds,a in agg["dataset_aggregate"].items(): print(f"  {ds:16s} n={a['n']} geomean={a['S_mixed_geomean']:.4f}")
print("Wrote PAPER13_CGT_RESULTS.json")