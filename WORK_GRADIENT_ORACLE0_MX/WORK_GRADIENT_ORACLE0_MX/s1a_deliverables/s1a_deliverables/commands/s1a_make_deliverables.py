#!/usr/bin/env python3
"""Generate S1A deliverable CSVs + S1A_IDENTITY.json from raw measurement JSONs."""
import json, csv, hashlib, os, subprocess, datetime
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_deliverables"

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()

# ---- S1A_COST_CALIBRATION.csv ----
cost = json.load(open(f"{D}/S1A_COST_CALIBRATION.json"))
agg = cost["aggregate"]
with open(f"{OUT}/S1A_COST_CALIBRATION.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["config","fwd_ms_median","bwd_ms_median","fb_ms_median","n_isects","tile_ratio","speedup_FB_vs_full","isect_reduction"])
    full_fb = agg["full"]["fb_ms_median"]
    for name in ["full","sparse_p25","sparse_p40"]:
        a=agg[name]
        sp = full_fb/a["fb_ms_median"] if name!="full" else 1.0
        isr = 1 - a["n_isects_median"]/agg["full"]["n_isects_median"] if name!="full" else 0.0
        w.writerow([name,f"{a['fwd_ms_median']:.4f}",f"{a['bwd_ms_median']:.4f}",
                    f"{a['fb_ms_median']:.4f}",a["n_isects_median"],f"{a['tile_ratio_median']:.3f}",
                    f"{sp:.4f}",f"{isr:.4f}"])
print("wrote S1A_COST_CALIBRATION.csv")

# ---- S1A_PARETO.csv ----
# cost (F+B ms) vs gradient NMSE (means group) for each strategy/budget
gv = json.load(open(f"{D}/S1A_GRADIENT_VALUE.json"))
with open(f"{OUT}/S1A_PARETO.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["budget","strategy","fb_ms_cost","primary_nmse_means","isect_reduction","label"])
    # cost per strategy: sparse cost at that budget (from cost calibration)
    cost_map = {"0.25":{"fb_ms":agg["sparse_p25"]["fb_ms_median"],"isr":1-agg["sparse_p25"]["n_isects_median"]/agg["full"]["n_isects_median"]},
                "0.4":{"fb_ms":agg["sparse_p40"]["fb_ms_median"],"isr":1-agg["sparse_p40"]["n_isects_median"]/agg["full"]["n_isects_median"]}}
    for p in ["0.25","0.4"]:
        for sname in ["U_uniform","E_error","G_grad_only","O_cost_aware"]:
            r=gv["strategies"][p][sname]
            c=cost_map[p]
            w.writerow([p,sname,f"{c['fb_ms']:.4f}",f"{r['primary_nmse_means']:.4f}",f"{c['isr']:.4f}","DERIVED"])
print("wrote S1A_PARETO.csv")

# ---- S1A_IDENTITY.json ----
ident = {
  "schema":"S1A_IDENTITY",
  "round":"S1A",
  "mx_node":"bms-39468022-001",
  "python":"3.10.21","torch":"2.9.1+cu128","cuda":"12.8","gpu":"A100-PCIE-40GB cc[8,0]","gpu_index_used":6,
  "gsplat_upstream_commit":"77ab983ffe43420b2131669cb35776b883ca4c3c",
  "prebuilt_gsplat_scene_cuda_so":"e4b0bc5bf358ec43875a3b7e0d84591763968021b7c7eabcc6d60572490627c8",
  "binary_sha256_source":"S0R2 (verified)",
  "env_freeze_source":"S1A_ENV_FREEZE.json",
  "PYTHONNOUSERSITE":1,
  "TMPDIR":f"{D}",
  "TORCH_EXTENSIONS_DIR":f"{D}/torch_ext",
  "XDG_CACHE_HOME":f"{D}/xdg",
  "checkpoints_used":{
     "primary":"FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt (N=2,456,697, iter=5000)",
     "secondary":".../checkpoints/0010000.pt (N=3,849,060, iter=10000)"
  },
  "camera":"bicycle eval_cameras.json camera 0 (_DSC8679)",
  "kernel":"gsplat.experimental.rasterize_gaussian_higs_dynamic (HiGS differentiable sparse-tile, native tile_mask)",
  "timestamp":datetime.datetime.now().isoformat(),
  "authorization":"S1A only; NO S1B, NO 3K/30K, NO CUDA rebuild"
}
json.dump(ident, open(f"{OUT}/S1A_IDENTITY.json","w"), indent=2)
print("wrote S1A_IDENTITY.json")

# ---- SHA256_MANIFEST.json ----
manifest = {"schema":"S1A_SHA256_MANIFEST","files":{}}
for fn in ["S1A_IDENTITY.json","S1A_CHECKPOINT_COMPATIBILITY.json","S1A_HT_CORRECTNESS.json",
           "S1A_GRADIENT_VALUE.json","S1A_GRADIENT_VALUE.csv","S1A_COST_CALIBRATION.csv",
           "S1A_COST_CALIBRATION.json","S1A_PARETO.csv"]:
    p=f"{OUT}/{fn}"
    if os.path.exists(p): manifest["files"][fn]=sha256(p)
# include raw measurement JSONs
for fn in ["S1A_HT_CORRECTNESS.json","S1A_COST_CALIBRATION.json","S1A_GRADIENT_VALUE.json"]:
    p=f"{D}/{fn}"
    if os.path.exists(p): manifest["files"][f"raw/{fn}"]=sha256(p)
json.dump(manifest, open(f"{OUT}/SHA256_MANIFEST.json","w"), indent=2)
print("wrote SHA256_MANIFEST.json")
print(json.dumps(manifest["files"], indent=2))

