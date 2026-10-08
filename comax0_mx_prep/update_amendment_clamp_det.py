#!/usr/bin/env python3
"""Append the CLAMP production-renderer determinism result to AMENDMENT_E1_RESULT.json.
NUMERICS ONLY; no timing claim; master verdict unchanged."""
import json, time
from pathlib import Path

WORK = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx")
p = WORK / "AMENDMENT_E1_RESULT.json"
d = json.loads(p.read_text(encoding="utf-8"))

det = json.loads((WORK / "e1_clamp_det2_20261006/e1/DETERMINISM.json").read_text())
rs = json.loads((WORK / "e1_clamp_det2_20261006/e1/RUN_STATUS.json").read_text())
man = (WORK / "COMAX0_E1_CLAMP_DETERMINISM/MANIFEST.sha256").read_text()

d["clamp_det_experiment"] = "COMAX0_E1_CLAMP_DETERMINISM (shared-GPU amendment; CLAMP production renderer determinism gate)"
d["clamp_det_method"] = ("render-only (subset_steps=0), comax_render_eta_ckpt_clamp (power-clamp>=0, TF32-off), "
                          "chunk_tiles=8, 1920x1080, 2 renders/view, deterministic representative subset "
                          "(frozen test globals 0,40,96,240,304: clean 0,40 + former-NaN 96,240,304)")
d["clamp_det_result"] = ("5/5 views deterministic: hash_match=true, nan_count_run_a=nan_count_run_b=0, "
                          "max_abs_diff_excl_nan=0.0, view_pass=true. Former-NaN views 96,240,304 now fully "
                          "finite + bitwise-deterministic. peak reserved 648 MiB. elapsed 2762.6s.")
d["clamp_det_verdict"] = "PASS_CLAMP_DETERMINISM"
d["clamp_det_key_finding"] = ("The power-clamp (power>=0) removes the inf->NaN source; the CLAMP production renderer "
                               "is bitwise-deterministic (NaN-free) on every tested view, including those that were "
                               "nondeterministic-NaN under the non-clamp oracle. The nondeterminism is a numerical-"
                               "overflow artifact of the oracle, NOT an inherent COMAX-envelope-scan property and NOT "
                               "a shared-GPU artifact.")
d["clamp_det_per_view"] = [{ "global_view": v["global_view"], "hash_match": v["hash_match"],
                             "nan_count_a": v["nan_count_run_a"], "nan_count_b": v["nan_count_run_b"],
                             "max_abs_diff_excl_nan": v["max_abs_diff_excl_nan"], "view_pass": v["view_pass"] }
                            for v in det["views"]]
d["clamp_det_gpu"] = {
    "index": rs.get("gpu_index"), "uuid": rs.get("gpu_uuid"),
    "selected_avg_util": 31.6, "selected_free_mib": 16939, "feasible": True,
    "co_resident_train_pid": rs.get("train_tf_pid"),
    "co_resident_note": "user train/train_tf.py owner-verified; co-resident ended on its own mid-run; determinism is worker self-consistency",
    "selection_rule": "lowest time-avg util -> lowest used mem -> lowest index; co-resident exclusively user's own train/*.py; render-only free-mem floor 6000 MiB",
    "peak_e1_mem_mib": rs.get("peak", {}).get("peak_e1_mem_mib"),
    "peak_gpu_mem_mib": rs.get("peak", {}).get("peak_gpu_mem_mib"),
}
d["clamp_det_performance_evidence_valid"] = False
d["clamp_det_no_timing_claim"] = True
d["clamp_det_no_coresident_touched"] = True
d["clamp_det_master_verdict_changed"] = False
d["clamp_det_evidence"] = "COMAX0_E1_CLAMP_DETERMINISM/ (REPORT.md, run/DETERMINISM.json, run/results.jsonl, run/RUN_STATUS.json, run/GUARD.json, SOURCE_MANIFEST.sha256, MANIFEST.sha256)"
d["clamp_det_manifest_sha256"] = man.strip()
d["clamp_det_note"] = ("Fresh shared-GPU determinism evidence under E1_NUMERICS_SHARED_GPU_AMENDMENT using the CLAMP "
                        "production renderer. PASSES where the non-clamp oracle (COMAX0_E1_SHARED_DET2) FAILED. The E1 "
                        "determinism PREREQUISITE is satisfied for the production/clamp renderer. E1 QUALITY (paired "
                        "refinement) remains BUDGET-BLOCKED (Python scan renderer ~minutes/render). Master verdict UNCHANGED.")
d["clamp_det_timestamp_unix"] = time.time()
d["latest_update_unix3"] = time.time()
d["latest_update_note3"] = ("Added CLAMP production-renderer determinism gate (COMAX0_E1_CLAMP_DETERMINISM): 5/5 views "
                            "bitwise-deterministic, NaN-free (incl. former-NaN 96,240,304); gate PASS; satisfies the E1 "
                            "determinism prerequisite for the production path; E1 quality still budget-blocked; master "
                            "verdict UNCHANGED INCONCLUSIVE_BUDGET; no timing claim.")

p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("AMENDMENT_E1_RESULT.json updated; keys:", len(d))
print("clamp_det_verdict:", d["clamp_det_verdict"])
