#!/usr/bin/env python3
"""Finalize: regenerate AMDAHL_MIN_Q.json + CSV from fresh G1, fix ENVIRONMENT.txt, copy build scripts."""
import json, os, glob, csv, time, platform, shutil
HERE=os.path.dirname(os.path.abspath(__file__))
REC=os.path.join(HERE,"G1_RAW_RECORDS")
SCR=os.path.join(HERE,"SCRIPTS")
g1=json.load(open(os.path.join(HERE,"G1_REAL_SCENE_COMPRESSIBILITY.json")))
combos=g1["combos"]
GAMMA_GATE=2.0; TARGET=1.25
# AMDAHL_MIN_Q.json
amd={"gamma_gate":GAMMA_GATE,"target_speedup":TARGET,
 "note":"per-unique-tile work weighting. S_inference is a ZERO-OVERHEAD Amdahl upper bound (assumes per-component cost gamma x Gaussian and no tree/gating/compositing overhead). q_weighted (work-weighted compressible fraction) is the primary defensible metric; S is a loose upper bound, not a speedup claim.",
 "per_combo":{}}
for k,c in combos.items():
    amd["per_combo"][k]={"q_weighted_cumulative":c["q_weighted_cumulative"],
      "incompressible_work_frac":c["incompressible_work_frac"],
      "S_inference_actual":c["S_inference_actual"],
      "best_S_gamma_gate":c["best_S_gamma_gate"],"verdict":c["verdict"]}
json.dump(amd,open(os.path.join(HERE,"AMDAHL_MIN_Q.json"),"w"),indent=1)
print("AMDAHL_MIN_Q.json regenerated")
# G1_PER_VIEW_SUMMARY.csv
rows=[]
for fp in sorted(glob.glob(os.path.join(REC,"g1_*.json"))):
    b=os.path.basename(fp)
    if "smoke" in b or "test" in b or "stale" in b: continue
    d=json.load(open(fp))
    for r in d["records"]:
        if "psnr" not in r: continue
        rows.append(dict(scene=d["scene"],ckpt=d["ckpt"].split("/")[-1],view=r["view"],rank=r["rank"],
            tile=f"({r['tile_ti']},{r['tile_tj']})",n_gauss=r["n_gauss"],tile_work=r["tile_work"],
            psnr=round(r["psnr"],2),max_abs=round(r["max_abs"],5),jac_abs=f"{r['jac_abs']:.3e}",
            resp_range=f"{r['resp_range']:.3e}",teacher_jac_norm=f"{r['teacher_jac_norm']:.3e}"))
with open(os.path.join(HERE,"G1_PER_VIEW_SUMMARY.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("CSV rows:",len(rows))
# ENVIRONMENT.txt
prov=[]
prov.append("host: "+platform.node())
prov.append("date_utc: "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
prov.append("python: 3.10.21")
prov.append("torch: 2.7.1+cu118  gsplat: 1.5.3  numpy: 2.2.6  scipy: 1.15.3  cuda_available: True  ngpu: 8")
prov.append("gpu_used: 0,1,2 (G1 rerun), 3 (G2-rigorous), 3,5,6 (G2-dense), 0,3,4 (G3) -> union {0,1,2,3,4,5,6}")
prov.append("teacher_fix: tile_teacher px/py bug fixed (line 131); all G1/G2/G3 regenerated with corrected teacher")
prov.append("assets_root: /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/")
prov.append("data_root: /tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/")
prov.append("runtime: /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python")
prov.append("seed: 20261004")
prov.append("sampling: 6 views x 250 tiles (stratified by work decile) x 3 ranks")
os.makedirs(os.path.join(HERE,"PROVENANCE"),exist_ok=True)
open(os.path.join(HERE,"PROVENANCE","ENVIRONMENT.txt"),"w").write("\n".join(prov)+"\n")
print("ENVIRONMENT.txt updated")
# copy build scripts to SCRIPTS
for f in ["build_g2_agg.py","build_g2_rig_agg.py","build_g3_agg.py","build_report.py","update_verdict.py","rebuild_g1.sh"]:
    src=os.path.join(HERE,f)
    if os.path.exists(src): shutil.copy(src,os.path.join(SCR,f))
print("build scripts copied to SCRIPTS")
