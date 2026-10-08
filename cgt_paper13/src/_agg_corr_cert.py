import json, glob, os
import numpy as np
HOME="/mnt/storage_pool/liaoyuanjun/cgt_paper13"
def gm(a): return float(np.exp(np.mean(np.log(np.array(a)))))
# load P2 (mixed-forward + certificate + correctness)
p2={}
for f in glob.glob(f"{HOME}/results/p2/cgt_*.json"):
    d=json.load(open(f))
    if d.get("status")=="OK": p2[d["scene"]]=d
scenes=sorted(p2.keys())
# correctness aggregate (P2)
corr={
 "stage":"CGT-PAPER13 correctness (kernel + CGT-vs-canonical)",
 "kernel_correctness_gate_max_abs":2e-5,
 "kernel_correctness_all_pass":all(p2[s]["correctness"]["kernel_all_passed"] for s in scenes),
 "max_kernel_max_abs":float(max(p2[s]["correctness"]["kernel_max_abs"] for s in scenes)),
 "cgt_vs_canonical_max_abs_max":float(max(p2[s]["correctness"]["cgt_vs_canonical_max_abs"] for s in scenes)),
 "cgt_vs_canonical_psnr_median_geomean":gm([p2[s]["correctness"]["cgt_vs_canonical_psnr_median"] for s in scenes]),
 "per_scene_kernel":{s:{"kernel_max_abs":p2[s]["correctness"]["kernel_max_abs"],
    "kernel_pass":p2[s]["correctness"]["kernel_all_passed"],
    "cgt_vs_canon_max_abs":p2[s]["correctness"]["cgt_vs_canonical_max_abs"],
    "cgt_vs_canon_psnr_median":p2[s]["correctness"]["cgt_vs_canonical_psnr_median"]} for s in scenes},
 "note":"kernel correctness = fused kernel vs where(safe,student,teacher) reference (F0 2e-5 gate). cgt_vs_canonical = fused vs full teacher (rendering fidelity, student approx error bounded by cert GATE=0.02).",
 "quality_gate":{"dPSNR_max":0.15,"dSSIM_max":0.005,
   "note":"full-image dPSNR/dSSIM vs GT NOT_MEASURED in this oracle lane (mixed-forward tile-level; no GT LPIPS pipeline run). E2E quality gate NOT_REACHED due to P3 E2E failure."},
}
open(f"{HOME}/results/PAPER13_CORRECTNESS.json","w").write(json.dumps(corr,indent=2))
# certificate aggregate (P2)
cert={
 "stage":"CGT-PAPER13 certificate statistics (frozen rigorous interval cert, GATE_MAX_ABS=0.02)",
 "gate_max_abs":0.02,
 "safe_fraction_mean":float(np.mean([p2[s]["certificate"]["safe_fraction"] for s in scenes])),
 "safe_fraction_geomean":gm([p2[s]["certificate"]["safe_fraction"] for s in scenes]),
 "weighted_fallback_fraction_mean":float(np.mean([p2[s]["certificate"]["weighted_fallback_fraction"] for s in scenes])),
 "rigorous_pass_rate_mean":float(np.mean([p2[s]["certificate"]["rigorous_pass_rate"] for s in scenes])),
 "rigorous_max_abs_max":float(max(p2[s]["certificate"]["rigorous_max_abs_max"] for s in scenes)),
 "leaves_median_geomean":gm([p2[s]["certificate"]["leaves_median"] for s in scenes]),
 "per_scene":{s:{"safe_fraction":p2[s]["certificate"]["safe_fraction"],
    "weighted_fallback":p2[s]["certificate"]["weighted_fallback_fraction"],
    "rigorous_pass_rate":p2[s]["certificate"]["rigorous_pass_rate"],
    "rigorous_max_abs_median":p2[s]["certificate"]["rigorous_max_abs_median"],
    "rigorous_max_abs_max":p2[s]["certificate"]["rigorous_max_abs_max"],
    "leaves_median":p2[s]["certificate"]["leaves_median"],"leaves_max":p2[s]["certificate"]["leaves_max"]} for s in scenes},
}
open(f"{HOME}/results/PAPER13_CERTIFICATE_STATS.json","w").write(json.dumps(cert,indent=2))
print("Wrote CORRECTNESS + CERT_STATS")
print("kernel all pass:",corr["kernel_correctness_all_pass"],"max kernel_max_abs=%.2e"%corr["max_kernel_max_abs"])
print("safe_fraction mean=%.3f geomean=%.3f"%(cert["safe_fraction_mean"],cert["safe_fraction_geomean"]))
print("weighted_fallback mean=%.3f"%cert["weighted_fallback_fraction_mean"])