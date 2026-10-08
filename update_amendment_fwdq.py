#!/usr/bin/env python3
# Append the E1 forward-quality-1080p numerics result to AMENDMENT_E1_RESULT.json,
# following the existing per-experiment key-block pattern. Backup first. NUMERICS-ONLY.
import json, time, shutil
from pathlib import Path

P = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/AMENDMENT_E1_RESULT.json")
BAK = P.with_suffix(".json.bak_pre_fwdq_20261006")
if not BAK.exists():
    shutil.copy2(P, BAK)

d = json.loads(P.read_text())
now = time.time()
res = json.loads(Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/e1_fq_full_20261006_a/e1/E1_FORWARD_QUALITY_1080P.json").read_text())
views = res["views"]
deltas = [round(v["delta_psnr_db"], 2) for v in views]
d["forward_quality_experiment"] = "E1_FORWARD_QUALITY_1080P (forward-only quality-vs-GT at preregistered 1080p primary resolution; NUMERICS-ONLY; NOT the full paired-refinement E1 pilot; NOT a timing/performance claim)"
d["forward_quality_method"] = "render-only (no_grad) on unrefined iter_15000 checkpoint; baseline=gsplat fully_fused_projection+alpha-composite; comax=comax_eta_ckpt_clamp (power-clamp>=0, TF32-off) eta=1 frozen kNN-64 grouping; deterministic frozen-test subset stride=7 (6 views 0/56/112/168/224/280, defined BEFORE measuring); common.Metrics PSNR/SSIM/LPIPS; via run_on_mx.py on guarded idle GPU2"
d["forward_quality_result"] = {
    "baseline_mean": res["baseline_mean"],
    "comax_mean": res["comax_mean"],
    "delta_mean": res["delta_mean"],
    "per_view_delta_psnr_db": deltas,
    "quality_forward_pass": res["quality_forward_pass"],
    "peak_gpu_mem_mib": res["peak_gpu_mem_mib"],
    "nan_free": all(v["baseline_nan_frac"] == 0.0 and v["comax_nan_frac"] == 0.0 for v in views),
    "metrics_complete": res["metrics_complete"],
    "resolution": res["resolution"],
    "test_split_size": res["test_split_size"],
}
d["forward_quality_verdict"] = "FORWARD_QUALITY_GAP_CONFIRMED (frozen kNN-64 grouping ~7 dB below full-gaussian baseline at 1080p; forward/unrefined)"
d["forward_quality_key_finding"] = "comax-vs-GT 23.11 dB == H0 720p frozen 23.83 dB (resolution-consistent) => the ~7 dB gap vs baseline (30.10 dB) is the frozen kNN-64 grouping representation cost, not resolution/NaN/fast-math. Pre-refinement starting gap; the full E1 pilot (1000 steps, budget-infeasible) tests whether refinement closes it to -0.15 dB."
d["forward_quality_gpu"] = "GPU-25086196-7911-cc05-1452-1885d070d710 (index2, A100-40GB, allowed[1-5], protected[0,6,7] untouched); idle at launch (0 MiB/0% all samples); no co-resident"
d["forward_quality_performance_evidence_valid"] = False
d["forward_quality_no_timing_claim"] = True
d["forward_quality_no_coresident_touched"] = True
d["forward_quality_master_verdict_changed"] = False
d["forward_quality_master_verdict"] = "INCONCLUSIVE_BUDGET"
d["forward_quality_evidence"] = "COMAX0_E1_FORWARD_QUALITY_1080P/ (REPORT.md + run/{e1_fq_full_20261006_a,e1_fq_smoke_20261006_a}/{E1_FORWARD_QUALITY_1080P.json,GUARD.json,RUN_STATUS.json,console.log} + SOURCE_MANIFEST.sha256 + MANIFEST.sha256; sha256sum -c OK; all JSON parse)"
d["forward_quality_harness_sha256"] = "d1adda20a344ad57b712e8e0530c4d24c5268b8dfc0937c002d98969b1d4ca5f"
d["forward_quality_source_hashes"] = {
    "e1_forward_quality_1080p.py": "d1adda20a344ad57b712e8e0530c4d24c5268b8dfc0937c002d98969b1d4ca5f",
    "comax_eta_ckpt_clamp.py": "039fbe61f05a877dd195b0ce6a2f148a9f54116a8a9c822350cdcd8010fdaa7d",
    "common.py": "5bc3b43e09aa77da8f8e4f13f33a0fd69ac8984655b9d5b4262cbb0fc73a0c9c",
    "prereg.json": "2e7721544b40305c65c941779780a7e82b612c42363357a072350c8340708ca4",
}
d["forward_quality_budget_after_s"] = 37595.7
d["forward_quality_note"] = "NUMERICS-ONLY forward-quality evidence; does not advance the reducer; E1 full pilot still budget-infeasible, E2 timing still not legitimately runnable, E3 depends on E2. No delivery rebuild."
d["forward_quality_timestamp_unix"] = now
d["latest_update_unix5"] = now
d["latest_update_note5"] = "Added E1 forward-quality-1080p numerics (COMAX0_E1_FORWARD_QUALITY_1080P): frozen kNN-64 grouping comax is ~7 dB PSNR below full-gaussian gsplat baseline at 1080p (6-view mean delta -6.99 dB, range -5.00..-8.79), resolution-consistent with H0 720p (23.83), NaN-free, 822 MiB, quality_forward_pass=false. NUMERICS-ONLY, forward/unrefined, NOT the E1 refined gate, NOT a timing claim. master verdict UNCHANGED=INCONCLUSIVE_BUDGET; no delivery rebuild."

P.write_text(json.dumps(d, indent=1, ensure_ascii=False))
print("APPENDED keys now", len(d))
print("backup", BAK.exists())
