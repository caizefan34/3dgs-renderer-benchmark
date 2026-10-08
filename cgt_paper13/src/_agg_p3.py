import json, glob, os
import numpy as np
HOME="/mnt/storage_pool/liaoyuanjun/cgt_paper13"
DS={"bicycle":"mipnerf360","flowers":"mipnerf360","garden":"mipnerf360","stump":"mipnerf360",
 "treehill":"mipnerf360","bonsai":"mipnerf360","counter":"mipnerf360","kitchen":"mipnerf360","room":"mipnerf360",
 "train":"tanksandtemples","truck":"tanksandtemples","drjohnson":"deepblending","playroom":"deepblending"}
rows={}
for f in glob.glob(f"{HOME}/results/p3/e2e_*.json"):
    d=json.load(open(f))
    if d.get("status")=="OK": rows[d["scene"]]=d
scenes=sorted(rows.keys())
def gm(a): return float(np.exp(np.mean(np.log(np.array(a)))))
per={}
for s in scenes:
    d=rows[s]
    per[s]={"n_tiles_full":d["n_tiles_full_frame"],"Mt_full":d["Mt_full_frame"],
            "t_build_per_tile_ms":d["t_build_per_tile_ms"],
            "T_build_full_min":d["T_build_full_ms"]/60000.0,
            "T_cgt_frame_ms":d["T_cgt_frame_ms"],"T_canon_ms":d["T_canonical_frame_ms"],
            "render_ratio":d["T_cgt_frame_ms"]/d["T_canonical_frame_ms"],
            "S_novel":d["S_e2e_novel_view"],"N_break":d["N_break_frames"]}
build_min=np.array([per[s]["T_build_full_min"] for s in scenes])
render_ratio=np.array([per[s]["render_ratio"] for s in scenes])
out={
 "stage":"CGT-PAPER13 P3 E2E build-amortization oracle (13-scene, per-frame views=1)",
 "date_utc":"2026-10-04","host":"MX A100-PCIE-40GB, runtime liaoyuanjun_mx0qual",
 "method":"frozen CGT-F0 full production frame path: per-frame per-tile fit(R4)+certify build + fused kernel render",
 "extrapolation":"full-frame T_build/T_render extrapolated from a 50-tile sample (per-tile cost x n_tiles_full_frame)",
 "per_scene":per,
 "T_build_full_min_geomean":gm(build_min.tolist()),
 "T_build_full_min_min":float(build_min.min()),"T_build_full_min_max":float(build_min.max()),
 "render_ratio_geomean":gm(render_ratio.tolist()),
 "render_ratio_min":float(render_ratio.min()),"render_ratio_max":float(render_ratio.max()),
 "all_S_novel_zero":all(per[s]["S_novel"]==0.0 for s in scenes),
 "all_N_break_none":all(per[s]["N_break"] is None for s in scenes),
 "n_tiles_full_range":[int(min(per[s]["n_tiles_full"] for s in scenes)),int(max(per[s]["n_tiles_full"] for s in scenes))],
 "dataset_aggregate":{},
}
for ds in ["mipnerf360","tanksandtemples","deepblending"]:
    ss=[s for s in scenes if DS[s]==ds]
    out["dataset_aggregate"][ds]={"n":len(ss),
      "T_build_full_min_geomean":gm([per[s]["T_build_full_min"] for s in ss]),
      "render_ratio_geomean":gm([per[s]["render_ratio"] for s in ss])}
out["dataset_aggregate"]["all_13"]={"n":len(scenes),
  "T_build_full_min_geomean":out["T_build_full_min_geomean"],"render_ratio_geomean":out["render_ratio_geomean"]}
open(f"{HOME}/results/PAPER13_BUILD_AMORTIZATION.json","w").write(json.dumps(out,indent=2))
print("scene          n_tiles  T_build(min)  T_cgt_frame  T_canon  render_ratio  S_novel  N_break")
for s in scenes:
    p=per[s]
    print(f"  {s:12s} {p['n_tiles_full']:7d}  {p['T_build_full_min']:10.1f}  {p['T_cgt_frame_ms']:9.1f}ms  {p['T_canon_ms']:6.1f}ms  {p['render_ratio']:9.1f}x  {p['S_novel']:.3f}  {p['N_break']}")
print(f"\nT_build geomean={out['T_build_full_min_geomean']:.1f}min  range={out['T_build_full_min_min']:.1f}-{out['T_build_full_min_max']:.1f}min")
print(f"render_ratio geomean={out['render_ratio_geomean']:.2f}x  (CGT frame / canonical frame)")
print(f"all S_novel==0: {out['all_S_novel_zero']}  all N_break None: {out['all_N_break_none']}")
print("Wrote PAPER13_BUILD_AMORTIZATION.json")