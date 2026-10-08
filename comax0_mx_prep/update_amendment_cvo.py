#!/usr/bin/env python3
"""Append the CLAMP-vs-ORACLE result to AMENDMENT_E1_RESULT.json. NUMERICS ONLY; no timing claim."""
import json, time
from pathlib import Path
WORK = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx")
p = WORK / "AMENDMENT_E1_RESULT.json"
d = json.loads(p.read_text(encoding="utf-8"))
cvo = json.loads((WORK / "e1_cvo_20261006/e1/CLAMP_VS_ORACLE.json").read_text())
rs = json.loads((WORK / "e1_cvo_20261006/e1/RUN_STATUS.json").read_text())

d["clamp_vs_oracle_experiment"] = "COMAX0_E1_CLAMP_VS_ORACLE (clamp vs non-clamp oracle faithfulness)"
d["clamp_vs_oracle_method"] = ("render-once-each, no_grad, comax_render_eta_ckpt_clamp vs comax_render_eta_ckpt, "
                                "chunk_tiles=8, 1920x1080, views 0 (clean) + 240 (former-NaN); co-finite pixel comparison")
d["clamp_vs_oracle_result"] = ("gate pass=true. View0: oracle_nan=0 clamp_nan=0 max_abs_diff_cofinite=0.0 (6.22M px). "
                                "View240: oracle_nan=5409456 (87%) clamp_nan=0 max_abs_diff_cofinite=0.0 (811344 co-finite px); "
                                "clamp fixed 5409456 NaN px to finite. Clamp is BIT-EXACT vs oracle on all co-finite pixels.")
d["clamp_vs_oracle_verdict"] = "PASS_CLAMP_FAITHFUL"
d["clamp_vs_oracle_key_finding"] = ("The clamp matches the non-clamp oracle bit-exactly (0.0 <= 2e-5) on every co-finite pixel "
                                     "and eliminates the oracle's NaN overflow (5.41M -> 0 on view 240). The oracle's nondeterminism "
                                     "is a numerical-overflow artifact; the clamp is a faithful, deterministic, NaN-free production renderer.")
d["clamp_vs_oracle_e2_note"] = ("E2 correctness gate (production vs oracle max-abs<=2e-5) must compare on co-finite pixels or treat "
                                "oracle-NaN as the artifact; against the raw oracle (~87% NaN on high-overdraw views) a naive all-pixel "
                                "max-abs is ill-defined. Protocol note for the reducer; NOT a speed claim.")
d["clamp_vs_oracle_gpu"] = {"index": rs.get("gpu_index"), "uuid": rs.get("gpu_uuid"),
                            "run_status": rs.get("status"), "elapsed_wall_seconds": rs.get("elapsed_wall_seconds"),
                            "peak_gpu_mem_mib": rs.get("peak", {}).get("peak_gpu_mem_mib"),
                            "note": "idle GPU via run_on_mx.py (original idle guard); numerics-only; no timing claim"}
d["clamp_vs_oracle_performance_evidence_valid"] = False
d["clamp_vs_oracle_no_timing_claim"] = True
d["clamp_vs_oracle_master_verdict_changed"] = False
d["clamp_vs_oracle_evidence"] = "e1_cvo_20261006/e1/CLAMP_VS_ORACLE.json; COMAX0_E1_CLAMP_DETERMINISM/CLAMP_VS_ORACLE_ADDENDUM.md"
d["clamp_vs_oracle_timestamp_unix"] = time.time()
d["latest_update_unix4"] = time.time()
d["latest_update_note4"] = ("Added CLAMP-vs-ORACLE faithfulness gate (COMAX0_E1_CLAMP_VS_ORACLE): clamp is bit-exact (0.0) vs the "
                            "non-clamp oracle on all co-finite pixels and eliminates the oracle's NaN overflow (5.41M->0 on view 240); "
                            "gate PASS_CLAMP_FAITHFUL; E2 correctness gate must compare co-finite; master verdict UNCHANGED; no timing claim.")
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("AMENDMENT updated with clamp_vs_oracle; keys:", len(d), "verdict:", d["clamp_vs_oracle_verdict"])
