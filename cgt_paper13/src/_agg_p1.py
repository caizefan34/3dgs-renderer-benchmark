import json, glob, os
import numpy as np
HOME="/mnt/storage_pool/liaoyuanjun/cgt_paper13"
DS={"bicycle":"mipnerf360","flowers":"mipnerf360","garden":"mipnerf360","stump":"mipnerf360",
 "treehill":"mipnerf360","bonsai":"mipnerf360","counter":"mipnerf360","kitchen":"mipnerf360","room":"mipnerf360",
 "train":"tanksandtemples","truck":"tanksandtemples","drjohnson":"deepblending","playroom":"deepblending"}
rows={}
for f in glob.glob(f"{HOME}/results/p1/canonical_*.json"):
    d=json.load(open(f)); rows[d["scene"]]=d
scenes=sorted(rows.keys())
def gm(a): return float(np.exp(np.mean(np.log(np.array(a)))))
per={}
b16=[]; b32=[]
for s in scenes:
    pt=rows[s]["per_tile"]; t16=pt["16"]["median_ms"]; t32=pt["32"]["median_ms"]
    sel_t, sel_tms = (16, t16) if t16<=t32 else (32, t32)
    per[s]={"tile16_ms":t16,"tile32_ms":t32,"baseline_tile":sel_t,"baseline_ms":sel_tms,
            "baseline_fps":1000.0/sel_tms,"n_test_views":rows[s]["n_test_views"],
            "quality":rows[s].get("quality",{})}
    b16.append(t16); b32.append(t32)
g16=gm(b16); g32=gm(b32)
global_tile = 16 if g16<=g32 else 32
out={
 "stage":"CGT-PAPER13 P1 canonical gsplat full-frame baseline (13-scene)",
 "date_utc":"2026-10-04","host":"MX A100-PCIE-40GB, runtime liaoyuanjun_mx0qual",
 "renderer":"gsplat 1.5.3 rasterization (packed=True), native resolution",
 "timing":"per-view median over bounded test views (split every 8th), warmup+repeats",
 "scope_note":"n_test_views bounded to 12 for timing; per-frame E2E = median per-view render",
 "per_scene":per,
 "baseline_B_geomean_ms":gm([per[s]["baseline_ms"] for s in scenes]),
 "baseline_B_geomean_fps":gm([per[s]["baseline_fps"] for s in scenes]),
 "global_canonical_tile":global_tile,
 "global_tile16_geomean_ms":g16,"global_tile32_geomean_ms":g32,
 "dataset_aggregate":{},
 "n_test_views_per_scene":{s:rows[s]["n_test_views"] for s in scenes},
}
for ds in ["mipnerf360","tanksandtemples","deepblending"]:
    ss=[s for s in scenes if DS[s]==ds]
    out["dataset_aggregate"][ds]={"n":len(ss),
      "baseline_B_geomean_ms":gm([per[s]["baseline_ms"] for s in ss]),
      "baseline_B_geomean_fps":gm([per[s]["baseline_fps"] for s in ss])}
out["dataset_aggregate"]["all_13"]={"n":len(scenes),
  "baseline_B_geomean_ms":out["baseline_B_geomean_ms"],"baseline_B_geomean_fps":out["baseline_B_geomean_fps"]}
open(f"{HOME}/results/PAPER13_BASELINE_RESULTS.json","w").write(json.dumps(out,indent=2))
print("Baseline B (per-scene fastest tile):")
for s in scenes: print(f"  {s:10s} tile{per[s]['baseline_tile']} {per[s]['baseline_ms']:.3f}ms {per[s]['baseline_fps']:.1f}fps")
print(f"Baseline B geomean: {out['baseline_B_geomean_ms']:.3f}ms  {out['baseline_B_geomean_fps']:.1f}fps")
print(f"global tile: {global_tile} (t16 geo {g16:.3f} vs t32 geo {g32:.3f})")
print("Wrote PAPER13_BASELINE_RESULTS.json")