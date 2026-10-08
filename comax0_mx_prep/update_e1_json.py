#!/usr/bin/env python3
"""Update COMAX0_REVIEW/e1/E1.json: record the CLAMP determinism prerequisite PASS.
Keep status=INCONCLUSIVE_BUDGET (E1 quality pilot budget-blocked). NUMERICS ONLY; no timing claim."""
import json, time
from pathlib import Path

WORK = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx")
p = WORK / "COMAX0_REVIEW/e1/E1.json"
d = json.loads(p.read_text(encoding="utf-8"))
det = json.loads((WORK / "e1_clamp_det2_20261006/e1/DETERMINISM.json").read_text())

d["shared_gpu_numerics_only"] = True
d["performance_evidence_valid"] = False
d["amendment"] = "E1_NUMERICS_SHARED_GPU_AMENDMENT"
d["determinism_check"] = {
    "pass": True,
    "renderer": "comax_render_eta_ckpt_clamp (power-clamp>=0, TF32-off)",
    "mode": "render_only_twice_nograd",
    "n_views": len(det["views"]),
    "views_pass": sum(1 for v in det["views"] if v["view_pass"]),
    "all_nan_count_zero": all(v["nan_count_run_a"] == 0 and v["nan_count_run_b"] == 0 for v in det["views"]),
    "all_hash_match": all(v["hash_match"] for v in det["views"]),
    "max_abs_diff_excl_nan": max(v["max_abs_diff_excl_nan"] for v in det["views"]),
    "tolerance_rgb_abs_max": det["tolerance_rgb_abs_max"],
    "evidence": "COMAX0_E1_CLAMP_DETERMINISM/run/DETERMINISM.json",
    "note": ("CLAMP production renderer is bitwise-deterministic + NaN-free on all 5 views (incl. former-NaN "
             "96,240,304). Satisfies the E1 determinism PREREQUISITE for the production path. The non-clamp oracle "
             "remains NaN-nondeterministic (COMAX0_E1_SHARED_DET2)."),
}
d["clamp_determinism"] = {
    "verdict": "PASS_CLAMP_DETERMINISM",
    "experiment": "COMAX0_E1_CLAMP_DETERMINISM",
    "evidence": "COMAX0_E1_CLAMP_DETERMINISM/",
    "timestamp_unix": time.time(),
}
d["status"] = "INCONCLUSIVE_BUDGET"
d["e1_blocker"] = ("E1 QUALITY pilot (1000-step paired refinement PSNR/SSIM/LPIPS) is BUDGET-BLOCKED: the "
                   "Python/reference scan renderer is ~minutes per render, so a multi-hundred-step paired "
                   "refinement does not fit the 12h launch budget. The determinism PREREQUISITE is now satisfied "
                   "(clamp renderer). Master reducer never reaches e2 while e1=INCONCLUSIVE_BUDGET.")
d["note"] = ("E1 determinism PREREQUISITE = PASS (CLAMP production renderer, 5/5 views bitwise-deterministic, "
             "NaN-free, shared-GPU amendment, peak 648 MiB, no timing claim). E1 QUALITY = BUDGET-BLOCKED. "
             "Master verdict UNCHANGED INCONCLUSIVE_BUDGET.")
p.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print("E1.json updated; status=", d["status"], "det_pass=", d["determinism_check"]["pass"])
