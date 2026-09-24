#!/usr/bin/env python3
"""R1 refresh consistency audit — verifies all 9 required updates are consistent."""
import json
import pathlib
import py_compile

ok_all = True

def check(name, cond, detail=""):
    global ok_all
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        ok_all = False


print("=== 1. JSON PARSE ===")
files = [
    "artifacts/final-30k/manifest.json", "artifacts/final-30k/protocol.json",
    "artifacts/final-30k/candidate_registry.json", "artifacts/final-30k/aggregation_spec.json",
    "artifacts/final-30k/quality_metric_spec.json", "artifacts/final-30k/ttq_spec.json",
    "artifacts/research-registry/authoritative_results.json", "artifacts/research-registry/supersession_map.json",
    "artifacts/research-registry/candidate_status.json", "artifacts/research-registry/prohibited_claims.json",
]
data = {}
for f in files:
    try:
        data[f] = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        print(f"  OK   {f}")
    except Exception as e:
        print(f"  FAIL {f}: {e}")
        ok_all = False

cs = data["artifacts/research-registry/candidate_status.json"]["candidates"]
ar = data["artifacts/research-registry/authoritative_results.json"]
pc = data["artifacts/research-registry/prohibited_claims.json"]

print("=== 2. P2-1A ===")
check("1799facc INVALID", cs["P2_1A_1799FACC"]["status"] == "INVALID")
check("1799facc reason = FP16/FP32 entry-contract failure",
      "FP16/FP32 entry-contract failure" in cs["P2_1A_1799FACC"]["reason"])
check("1799facc .so hash", cs["P2_1A_1799FACC"]["so_sha256"].startswith("1799facc"))
check("R2 PREFLIGHT", cs["P2_1A_R2"]["status"] == "PREFLIGHT")
check("R2 role = contract-true projected-state adapter repair",
      "contract-true projected-state adapter repair" in cs["P2_1A_R2"]["role"])
check("overall P2-1A UNRESOLVED, no STRONG/MARGINAL/WEAK until R2",
      "UNRESOLVED" in cs["P2_1A"]["note"] and "STRONG/MARGINAL/WEAK" in cs["P2_1A"]["note"])
check("authoritative_results P2_1A variants consistent",
      ar["P2_1A"]["variants"]["P2_1A_1799FACC"]["status"] == "INVALID"
      and ar["P2_1A"]["variants"]["P2_1A_R2"]["status"] == "PREFLIGHT")

print("=== 3. P2-1A-R0 static evidence ===")
r0 = cs["P2_1A_R0"]
check("RESOURCE_DELTA_MEDIUM", r0["status"] == "RESOURCE_DELTA_MEDIUM")
check("FP16 raster 3 CTA/SM", r0["static_evidence"]["FP16_raster_cta_per_sm"] == 3)
check("FP32 raster 3 CTA/SM", r0["static_evidence"]["FP32_raster_cta_per_sm"] == 3)
check("occupancy 46.875% both", r0["static_evidence"]["occupancy_pct"] == "46.875% both")
check("FP32<8> spill AMPLIFIED", r0["static_evidence"]["FP32_8_spill"] == "AMPLIFIED")
check("FP32<16> spill NEW", r0["static_evidence"]["FP32_16_spill"] == "NEW")
check("no pending-corrected-audit residue",
      "pending corrected audit" not in r0["note"].replace("No longer 'pending corrected audit'", ""))

print("=== 4. P2-1C DROP ===")
c1c = cs["P2_1C"]
ev = ar["P2_1C"]["empirical_evidence"]
check("status DROP", c1c["status"] == "DROP")
check("gross dead groups 20.9/30.1/19.8",
      ev["gross_dead_groups_pct"] == {"room": 20.9, "bicycle": 30.1, "garden": 19.8})
check("entry-weighted 10.8/4.7/8.2",
      ev["entry_weighted_pct"] == {"room": 10.8, "bicycle": 4.7, "garden": 8.2})
check("last_id overlap 100%", ev["overlap_with_last_id_processing_skip_pct"] == 100)
check("frozen 128-G skippable 0.40/2.00/1.08",
      ev["frozen_128G_fully_skippable_work_pct"] == {"room": 0.40, "bicycle": 2.00, "garden": 1.08})
check("H5-0R timing +1.54/-0.01/0.00",
      ev["equivalent_measured_H5_0R_timing_pct"] == {"room": 1.54, "bicycle": -0.01, "garden": 0.00})
check("MARGINAL noted but decision DROP",
      "MARGINAL" in ar["P2_1C"]["oracle_preflight_gate"] and "DROP" in ar["P2_1C"]["status"])

print("=== 5. Superseded P2-0 reuse projections ===")
sm = data["artifacts/research-registry/supersession_map.json"]["superseded"]
ids = [e["id"] for e in sm]
check("13-20% backward reuse estimate superseded", "P2_0_BACKWARD_REUSE_ESTIMATE_13_20_PCT" in ids)
check("~9.2% F+B reuse estimate superseded", "P2_0_FB_REUSE_ESTIMATE_9_2_PCT" in ids)
check("1799facc .so superseded", "P2_1A_1799FACC_SO" in ids)
for e in sm:
    if e["id"].startswith("P2_0_"):
        check(f"{e['id']}: dead-group counts NOT superseded", "NOT_superseded" in e)
check("superseded entries not deleted", all(e.get("do_not_delete") for e in sm))

print("=== 6. P2_FINAL_V2 composition ===")
cr = data["artifacts/final-30k/candidate_registry.json"]["candidates"]["P2_FINAL_V2"]
check("composition = C0 V3 + P2-1A (R2-gated) + optional E2",
      "C0 V3 + P2-1A native hierarchical forward" in cr["description"] and "optional E2" in cr["description"])
check("P2-1C removed from final stack", "REMOVED" in cr["description"])
check("P2-1C toggle REMOVED", "REMOVED" in cr["runtime_feature_toggles"]["P2_1C_backward_reuse"])
check("acceptance gated on R2", "R2" in cr["acceptance_gate"])
check("authoritative_results composition matches",
      ar["P2_FINAL_V2"]["composition"] == "C0 V3 + P2-1A native hierarchical forward (IF R2 runtime passes) + optional E2")

print("=== 7. TTQ censoring ===")
tq = data["artifacts/final-30k/ttq_spec.json"]
crule = tq["threshold_derivation"]["censoring_rule"]
check("ttq = null", "ttq = null" in crule)
check("status = TTQ_NOT_REACHED", "TTQ_NOT_REACHED" in crule)
check("censor_time = total_wall_time_30000", "censor_time = total_wall_time_30000" in crule)
check("censor_time NOT an achieved TTQ", "NOT an achieved TTQ" in crule)
check("geomean over comparable reached/reached pairs only",
      "COMPARABLE reached/reached PAIRS only" in tq["ttq_speedup"]["aggregate"])
rr = json.dumps(tq["ttq_speedup"]["reporting_requirements"])
check("reach counts explicitly reported", "n_reached_pairs" in rr and "n_not_reached" in rr)
check("no censor_time substitution", "Do NOT substitute censor_time" in rr)
check("old 'TTQ = total wall time' rule removed",
      "TTQ = 30000-step wall time" not in json.dumps(tq))

print("=== 8. Garden policy ===")
mf = data["artifacts/final-30k/manifest.json"]["garden_quality_policy"]
check("garden in primary aggregate", "included in the primary" in mf["primary_aggregate"])
check("pre-labeled sensitivity analysis optional", "sensitivity analysis" in mf["sensitivity_analysis_optional"])
check("no one-sided exclusion", mf["prohibition"].startswith("Do NOT exclude garden only from unfavorable"))
check("frozen before results", mf["frozen_before_results"] is True)

print("=== 9. Timing attribution language ===")
why = pc["invalid_c0_timing_claims"][0]["why_invalid"]
check("instrumentation/protocol-state artifact", "instrumentation/protocol-state artifact" in why.lower() or "Instrumentation/protocol-state artifact" in why)
check("protocol/GPU-state-dependent variability", "protocol/GPU-state-dependent timing variability" in why)
check("explicit no-hardware-attribution disclaimer", "NO hardware-level causal attribution" in why)
check("prohibited-language entry added",
      any("Attributing the H8/C0 timing anomaly" in x for x in pc["language_discipline_prohibited"]))
# The old positive claim must not survive as an assertion of cause
old_claim_fragment = "Root cause: GPU clock/power-state oscillation"
check("old 'Root cause: clock/power-state' text removed from registry", old_claim_fragment not in why)

print("=== 10. Harness scripts compile ===")
for p in sorted(pathlib.Path("harness/final30k").glob("*.py")):
    py_compile.compile(str(p), doraise=True)
print("  PASS all harness scripts compile")

print("=== 11. Reports updated ===")
reg_md = pathlib.Path("reports/higs/authoritative-experiment-registry.md").read_text(encoding="utf-8")
hs_md = pathlib.Path("reports/final-30k/benchmark-harness-spec.md").read_text(encoding="utf-8")
check("registry report: R2 decision tree", "P2-1A R2" in reg_md and "RESOURCE_LIMITED" in reg_md)
check("registry report: P2-1C DROP in all branches", "P2-1C = DROP in all branches" in reg_md)
check("registry report: R0 section", "RESOURCE_DELTA_MEDIUM" in reg_md and "46.875% both" in reg_md)
check("registry report: no pending-residency residue",
      "Static residency remains pending" not in reg_md)
check("registry report: reuse projections marked superseded", "SUPERSEDED_FOR_PUBLICATION" in reg_md)
check("registry report: dead-group counts NOT superseded", "NOT superseded" in reg_md)
check("registry report: no clock/power causal claim",
      "Root cause: GPU clock/power-state oscillation" not in reg_md)
check("harness spec: censoring semantics", "ttq = null" in hs_md and "right-censoring bound" in hs_md)
check("harness spec: P2-1C removed", "P2-1C is REMOVED" in hs_md)
check("harness spec: garden policy", "Garden quality policy" in hs_md)
check("harness spec: no protocol redesign", "No protocol redesign" in hs_md)

print()
print("ALL CONSISTENCY CHECKS PASS" if ok_all else "CONSISTENCY FAILURES PRESENT")
raise SystemExit(0 if ok_all else 1)
