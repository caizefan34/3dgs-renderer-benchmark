#!/usr/bin/env python3
"""Finalize S1A deliverables: RUN_STATUS.json, copy report/scripts, gpu snapshot, commands log, manifest refresh."""
import json, os, shutil, hashlib, datetime, subprocess
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_deliverables"
LOCAL = "C:/Users/36570/3dgs-renderer-benchmark/WORK_GRADIENT_ORACLE0_MX/WORK_GRADIENT_ORACLE0_MX"

# ---- RUN_STATUS.json ----
status = {
  "schema":"RUN_STATUS",
  "round":"S1A",
  "state":"S1A_MATH_PASS",
  "sub_state":"render_sanity_FAILED_value_comparison_BLOCKED",
  "overall":"S1A_BLOCKED_INPUT",
  "timestamp":datetime.datetime.now().isoformat(),
  "mx_node":"bms-39468022-001","gpu":"A100-PCIE-40GB cc[8,0]","gpu_index":6,
  "gates":{
     "env_freeze":"PASS",
     "convention_check":"PASS",
     "render_sanity":"FAIL",
     "S1A_0_ht_math":"PASS",
     "S1A_1_gradient_value":"DERIVED_ONLY",
     "S1A_2_cost_calibration":"PASS"
  },
  "S1A_0":{"pass":True,"additivity_rel_err":"~1e-7","p1_equals_dense":"~1e-7",
           "mc_cos_p25":"0.995-0.999","mc_cos_p40":"0.996-0.999","groups":5},
  "S1A_1":{"label":"DERIVED","subset_oracle":True,"N":120000,"tiles":64,
           "nmse_reduction_O_vs_U_p25":0.38,"nmse_reduction_O_vs_U_p40":0.37,
           "E_worse_than_U":True,"O_equals_G":True,
           "note":"estimator-efficiency math; render sanity failed; NOT training value; does NOT support GO"},
  "S1A_2":{"pass":True,"isect_reduction_p25":0.726,"isect_reduction_p40":0.587,
           "speedup_p25":1.14,"speedup_p40":1.10,"fixed_overhead_fraction":0.84,
           "note":"structural isect win; modest speedup at small scale (fixed overhead dominates)"},
  "render_sanity":{"best_corr":0.253,"best_psnr":13.56,"cameras_tested":[0,2,5,8],
                   "benchmark_own_render_path_corr":0.084,
                   "root_cause_hypothesis":"undertrained iter5K/10K for fused repr, or eval_images are held-out views"},
  "verdict_rationale":"HT estimator math correct (S1A-0 PASS); sparse kernel saves real isects (S1A-2 PASS); BUT checkpoint does not reconstruct GT with available gsplat-HiGS renderer (render sanity FAIL) => formal image-reconstruction gradient-value comparison and GO blocked per task Section 2. PROMISING condition (oracle MSE -38%) met only on DERIVED single-state single-camera estimator math, not on >=2 states x 2 cameras with meaningful render. No S1B.",
  "go_decision":"NO_GO",
  "next_action_required":"obtain checkpoint/renderer combo passing RGB/alpha render sanity, then re-run S1A-1/S1A-2 on >=2 bicycle states x >=2 cameras",
  "s1b_started":False,
  "deliverables":["S1A_IDENTITY.json","S1A_CHECKPOINT_COMPATIBILITY.json","S1A_HT_CORRECTNESS.json",
                   "S1A_GRADIENT_VALUE.csv","S1A_COST_CALIBRATION.csv","S1A_PARETO.csv",
                   "S1A_REPORT.md","RUN_STATUS.json","SHA256_MANIFEST.json","raw/","logs/","commands/","gpu_snapshots/"]
}
json.dump(status, open(f"{OUT}/RUN_STATUS.json","w"), indent=2)
print("wrote RUN_STATUS.json")

# ---- copy report + scripts ----
shutil.copy(f"{LOCAL}/s1a_deliverables/S1A_REPORT.md".replace("C:/","/mnt/c/"), f"{OUT}/S1A_REPORT.md") if os.path.exists("/mnt/c") else None
# scripts live locally; they were already uploaded to $D. Copy the .py scripts into commands/ as the run log
scripts = ["s1a_ht_correctness.py","s1a_cost_calibration.py","s1a_gradient_value.py",
           "s1a_make_deliverables.py","s1a_rgb_sanity.py","s1a_cam_sweep.py","s1a_sh_test.py",
           "s1a_native_res.py","s1a_render_char.py","s1a_bench_render.py","s1a_diagnose.py"]
cmds = []
for s in scripts:
    src = f"{D}/{s}"
    if os.path.exists(src):
        shutil.copy(src, f"{OUT}/commands/{s}")
        cmds.append(f"python {src}")
with open(f"{OUT}/commands/RUN_COMMANDS.txt","w") as f:
    f.write("# S1A run commands (GPU 6, PYTHONNOUSERSITE=1)\n")
    f.write("ENV: PYTHONNOUSERSITE=1 TMPDIR=%s TORCH_EXTENSIONS_DIR=%s/torch_ext XDG_CACHE_HOME=%s/xdg\n" % (D,D,D))
    f.write("PY: /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python\n")
    f.write("EVERY script: sys.path.insert(0,'/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda')\n\n")
    for c in cmds: f.write(c + "\n")
print("copied scripts to commands/")

# ---- gpu snapshot ----
try:
    snap = subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    open(f"{OUT}/gpu_snapshots/gpu_snapshot.txt","w").write(
        f"# captured {datetime.datetime.now().isoformat()}\n"+snap.stdout)
    print("wrote gpu_snapshot.txt")
except Exception as e:
    print("gpu snapshot failed:", e)

# ---- refresh manifest with all files ----
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()
manifest={"schema":"S1A_SHA256_MANIFEST","files":{}}
for root,_,files in os.walk(OUT):
    for fn in files:
        if fn=="SHA256_MANIFEST.json": continue
        p=os.path.join(root,fn)
        rel=os.path.relpath(p,OUT).replace("\\","/")
        manifest["files"][rel]=sha256(p)
json.dump(manifest, open(f"{OUT}/SHA256_MANIFEST.json","w"), indent=2)
print("refreshed SHA256_MANIFEST.json with", len(manifest["files"]), "files")
print(json.dumps(manifest["files"], indent=2))
