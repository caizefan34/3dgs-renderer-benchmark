#!/usr/bin/env python3
"""Finalize S1A-R1 deliverables: RUN_STATUS.json, copy erratum, refresh manifest, copy renders."""
import json, os, hashlib, shutil, datetime
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_r1_deliverables"
LOCAL = "C:/Users/36570/3dgs-renderer-benchmark/WORK_GRADIENT_ORACLE0_MX/WORK_GRADIENT_ORACLE0_MX"
os.makedirs(OUT, exist_ok=True)

status = {
  "schema":"RUN_STATUS","round":"S1A_R1",
  "state":"S1A_R1_GRADIENT_ONLY",
  "timestamp":datetime.datetime.now().isoformat(),
  "mx_node":"bms-39468022-001","gpu":"A100-PCIE-40GB cc[8,0]","gpu_index":6,
  "torch":"2.9.1+cu128","cuda":"12.8","python":"3.10.21",
  "authorization":"S1A-R1 only; NO S1B start, NO 3K/30K new training, NO CUDA rebuild, NO env reinstall",
  "phases":{
     "phase1_formula_audit":"PASS (correct (1-p)/p confirmed vs MC; original misused (1-p)/p^2)",
     "phase2_render_threeway":"B/C PASS (5K PSNR17.1/15.6); A native BLOCKED (ABI mismatch)",
     "phase3_valid_gradient_value":"G +59.3%/O +58.3% vs U; O never beats G; E worse"
  },
  "phase1":{"formula_correct":"sum[(1-p_j)/p_j]||g_j||^2","formula_wrong":"sum[(1-p_j)/p_j^2]||g_j||^2",
            "mc_validation":"analytic 7.627e-3 vs mc 7.609e-3 (U, p=0.25) — 3 sig figs",
            "corrected_nmse_reduction_G_vs_U":{"p0.25":0.507,"p0.40":0.599},
            "math_correctness":"PASS"},
  "phase2":{"checkpoints":["5K(N=2456697)","10K(N=3849060)","15K(N=4430283,final.pt)"],
            "camera":"bicycle cam0 _DSC8679","resolution":"396x264 scale 0.08",
            "path_A_native_fastergs":"BLOCKED: _C.forward illegal memory access (ABI mismatch torch2.9/cu128)",
            "path_B_gsplat_lowlevel":{"5K":17.10,"10K":13.36,"15K":13.25},
            "path_C_higs_fulltile":{"5K":15.56,"10K":13.20,"15K":12.88},
            "full_gaussian_set":True,"truncation":"none","render_sanity_5K":"PASS (PSNR 17.1)"},
  "phase3":{"states":["5K","10K"],"cameras":[0,1],"formula":"correct (1-p)/p",
            "mean_reduction_vs_U":{"G":0.593,"O":0.583,"E":-0.378},
            "O_beats_G_any":False,"G_beats_U_all":True,"O_beats_U_all":True,
            "subset_oracle":True},
  "verdict":"S1A_R1_GRADIENT_ONLY",
  "verdict_rationale":"G (gradient-only) beats U ~58% consistently across 2 states x 2 cameras x 2 budgets with the CORRECT HT variance formula; BUT O (cost-aware oracle) never beats G (mean O-vs-G=-2.6%) because 84% of GPU cost is fixed overhead so the sqrt(c_j) term is null. The cost-aware oracle adds no advantage over pure gradient-magnitude sampling. Does not meet the S1B bar (which requires a cost-aware advantage).",
  "key_finding":"The original S1A cost-aware oracle hypothesis is NOT supported once the variance formula is corrected and the render is valid. A cheaper gradient-only sampler is equally effective.",
  "go_decision":"NO_S1B (cost-aware advantage absent; gradient-only advantage exists but is not the authorized S1B target)",
  "s1b_started":False,
  "deliverables":["S1A_VARIANCE_ERRATUM.md","S1A_R1_REPORT.md","S1A_R1_VARIANCE_ERRATUM.json",
                   "S1A_R1_RENDER_THREEWAY.json","S1A_R1_GRADIENT_VALUE.json",
                   "pertile_grad_norms_bicycle5K_cam0.npz","renders/","RUN_STATUS.json","SHA256_MANIFEST.json","raw/"]
}
json.dump(status, open(f"{OUT}/RUN_STATUS.json","w"), indent=2)
print("wrote RUN_STATUS.json")

# copy erratum from local workspace (it was written locally)
err_local = f"{LOCAL}/s1a_r1_deliverables/S1A_VARIANCE_ERRATUM.md"
if os.path.exists(err_local):
    shutil.copy(err_local, f"{OUT}/S1A_VARIANCE_ERRATUM.md")
    print("copied S1A_VARIANCE_ERRATUM.md")

# copy raw measurement JSONs from tmp root
for fn in ["S1A_R1_VARIANCE_ERRATUM.json","S1A_R1_RENDER_THREEWAY.json","S1A_R1_GRADIENT_VALUE.json"]:
    if os.path.exists(f"{D}/{fn}"):
        shutil.copy(f"{D}/{fn}", f"{OUT}/{fn}")
# also preserve original S1A wrong-formula data in raw/
for fn in ["S1A_GRADIENT_VALUE.json","S1A_HT_CORRECTNESS.json","S1A_COST_CALIBRATION.json"]:
    if os.path.exists(f"{D}/{fn}"):
        os.makedirs(f"{OUT}/raw",exist_ok=True)
        shutil.copy(f"{D}/{fn}", f"{OUT}/raw/{fn}")
print("copied JSONs + raw/")

# refresh manifest
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()
m={"schema":"S1A_R1_SHA256_MANIFEST","files":{}}
for r,_,fs in os.walk(OUT):
    for f in fs:
        if f=="SHA256_MANIFEST.json": continue
        p=os.path.join(r,f); rel=os.path.relpath(p,OUT).replace("\\","/")
        m["files"][rel]=sha256(p)
json.dump(m, open(f"{OUT}/SHA256_MANIFEST.json","w"), indent=2)
print("refreshed manifest:", len(m["files"]), "files")
print(json.dumps(m["files"], indent=2))
