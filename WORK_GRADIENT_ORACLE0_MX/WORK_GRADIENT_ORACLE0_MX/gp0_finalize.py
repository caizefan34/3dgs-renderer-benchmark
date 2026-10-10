#!/usr/bin/env python3
"""Finalize GRADIENT-PROXY-P0 deliverables: RUN_STATUS, copy report, refresh manifest."""
import json, os, hashlib, shutil, datetime
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/gp0_deliverables"; os.makedirs(OUT,exist_ok=True)
LOCAL="C:/Users/36570/3dgs-renderer-benchmark/WORK_GRADIENT_ORACLE0_MX/WORK_GRADIENT_ORACLE0_MX/gp0_deliverables"
status={
 "schema":"RUN_STATUS","round":"GRADIENT-PROXY-P0",
 "state":"DROP","timestamp":datetime.datetime.now().isoformat(),
 "mx_node":"bms-39468022-001","gpu":"A100-PCIE-40GB cc[8,0]","gpu_index":6,
 "torch":"2.9.1+cu128","cuda":"12.8","python":"3.10.21",
 "authorization":"P0 feasibility only; NO 3K/30K training, NO CUDA modification, NO runtime rebuild; NOT a Cost-Aware S1B pass",
 "input_state":"S1A_R1_GRADIENT_ONLY",
 "phases":{
   "phase1_consistency":"B vs C render near-identical (cam1 byte-identical, cam0 MAE0.02/PSNR32); dense grad NEAR-ORTHOGONAL (cos~0.005) -> per-tile grad renderer-sensitive, use HiGS(path C); PSNR ~10.3-10.7 FLAT (S1A-R1 17.10 was wrong, true 10.72); visible ~27% of total; RUN_VALIDATION=false so source identity unclear -> math proxy only, no training-quality claim",
   "phase2_proxies":"H1 historical EMA of true per-tile grad from prior states (design 5K/10K, eval 15K); H2 geometry-Jacobian = residual*conic_mass (forward projection only, no dense backward); H3 hybrid H1+H2+exploration floor 0.02; per-tile true grad via HiGS per-tile backward (genuine per-tile, not aggregate); unsampled tiles NOT zeroed (floor + carry-over)",
   "phase3_oracle":"CORRECT formula Var=sum[(1-p_j)/p_j]||g_j||^2; 9 combos (5K/10K/15K x cam0/1/2); held-out 15K+cam2. G +49%/+60% (heldout +44/+55) but NOT deployable. ALL deployable proxies FAIL >=25%: E -31/-35%, H1 -9/+12% (mixed), H2 -373/-546%, H2b -78/-89%, H3 -18/-16%. Correlation cheap-signal vs true grad <=0.6, ~0 on cam0, residual anti-correlated on cam0",
   "phase4_timing":"H2 proxy 34x sparse iter overhead (Python per-tile loop); oracle info cost 317x sparse (1389ms, 425 backwards) NOT hidden; no dense backward in proxy (confirmed); scheduling overhead 148.7ms vs sparse fwd+bwd 4.38ms"
 },
 "verdict":"DROP",
 "verdict_rationale":"No deployable proxy (H1/H2/H3) meets >=25% NMSE reduction vs U on primary group consistently across states/cameras/held-out. Cheap forward signals correlate <=0.6 with true per-tile gradient (~0 on cam0). H2 overhead 34x sparse (budget 5% FAIL). Oracle G strong (+49-60%) but 317x info cost, non-deployable. Conditions NOT relaxed despite strong G.",
 "go_criteria":{
   "1_proxy_vs_U_ge25pct":"FAIL (best H1 +10-12% heldout but -9% design; H2/H3 negative)",
   "2_direction_consistent":"FAIL",
   "3_no_key_group_degradation":"FAIL for H2/H2b/H3",
   "4_overhead_le_5pct":"FAIL (H2 34x)",
   "5_budget_consistent":"PASS",
   "6_no_leakage":"PASS"
 },
 "s1b_promotion":False,
 "s1b_note":"Cost-Aware S1B NOT rescued by cheap proxies; gradient-only oracle advantage real but non-deployable",
 "deliverables":["GRADIENT_PROXY_P0_REPORT.md","GP0_PHASE1_CONSISTENCY.json","GP0_PHASE3_ORACLE.json","GP0_PHASE4_TIMING.json","gp0_pertile_meta.json","gp0_pertile_*.npz (9)","RUN_STATUS.json","SHA256_MANIFEST.json"]
}
json.dump(status,open(f"{OUT}/RUN_STATUS.json","w"),indent=2)
print("wrote RUN_STATUS.json")
# copy report from local
rep=f"{LOCAL}/GRADIENT_PROXY_P0_REPORT.md"
if os.path.exists(rep): shutil.copy(rep,f"{OUT}/GRADIENT_PROXY_P0_REPORT.md"); print("copied report")
# manifest
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536),b""): h.update(b)
    return h.hexdigest()
m={"schema":"GP0_SHA256_MANIFEST","files":{}}
for r,_,fs in os.walk(OUT):
    for f in fs:
        if f=="SHA256_MANIFEST.json": continue
        p=os.path.join(r,f); rel=os.path.relpath(p,OUT).replace("\\","/")
        m["files"][rel]=sha256(p)
json.dump(m,open(f"{OUT}/SHA256_MANIFEST.json","w"),indent=2)
print("manifest files:",len(m["files"]))
for k in sorted(m["files"]): print(" ",k)
