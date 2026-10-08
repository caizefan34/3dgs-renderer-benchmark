import json
import os
import statistics

# Read all units (tau=2 only, from main run)
units = {}
for scene in ("room", "bicycle"):
    d = f"/root/epic05_p17/P17_UNITS/{scene}"
    units[scene] = [json.load(open(os.path.join(d, f)))
                    for f in sorted(os.listdir(d)) if f.endswith(".json")]

# Gate check
results = {}
for scene, us in units.items():
    co = [u["oracle_interaction_ratio"] for u in us]
    med_c = statistics.median(co)
    p75_c = sorted(co)[int(0.75 * (len(co) - 1))]
    
    mcos = [u["metrics"]["means"]["cosine"] for u in us
            if u["metrics"]["means"]["cosine"] is not None]
    ocos = [u["metrics"]["opacity"]["cosine"] for u in us
            if u["metrics"]["opacity"]["cosine"] is not None]
    
    rs = [u["step"]["R_step"] for u in us
          if u["step"]["reference_improving"] and u["step"]["R_step"] is not None]
    n_improving = sum(1 for u in us if u["step"]["reference_improving"])
    
    pa = [u["step"]["P_audit"] for u in us]
    n_big = sum(1 for p in pa if p > 0.01)
    
    drop = []
    if med_c > 0.50: drop.append("C_oracle>0.5")
    if mcos and statistics.median(mcos) < 0.980: drop.append("means_cos<0.98")
    if ocos and statistics.median(ocos) < 0.980: drop.append("opacity_cos<0.98")
    if rs and statistics.median(rs) < 0.80: drop.append("R_step<0.8")
    if n_big >= 4: drop.append("P_audit>0.01 x4")
    
    cand = {}
    cand["work"] = med_c <= 0.30 and p75_c <= 0.35
    cand["means"] = mcos and statistics.median(mcos) >= 0.995 and \
        sorted(mcos)[int(0.1*(len(mcos)-1))] >= 0.990
    cand["opacity"] = ocos and statistics.median(ocos) >= 0.995
    cand["R_step"] = rs and 0.95 <= statistics.median(rs) <= 1.05 and \
        sum(1 for r in rs if 0.90 <= r <= 1.10) >= 10
    cand["audit"] = statistics.median(pa) <= 0.002 and n_big <= 1
    
    all_cand = all(cand.values())
    any_drop = len(drop) > 0
    
    if any_drop:
        verdict = "HARD_DROP"
    elif all_cand:
        verdict = "CANDIDATE"
    else:
        verdict = "HOLD"
    
    results[scene] = {
        "n_units": len(us),
        "parity_all_pass": all(u["d1_parity_pass"] for u in us),
        "C_oracle_median": med_c,
        "C_oracle_p75": p75_c,
        "means_cos_median": statistics.median(mcos) if mcos else None,
        "means_cos_p10": sorted(mcos)[int(0.1*(len(mcos)-1))] if mcos else None,
        "scales_cos_median": statistics.median(
            [u["metrics"]["scales"]["cosine"] for u in us
             if u["metrics"]["scales"]["cosine"] is not None]) if us else None,
        "quats_cos_median": statistics.median(
            [u["metrics"]["quats"]["cosine"] for u in us
             if u["metrics"]["quats"]["cosine"] is not None]) if us else None,
        "opacity_cos_median": statistics.median(ocos) if ocos else None,
        "sh0_cos_median": (lambda v: statistics.median(v) if v else None)(
            [u["metrics"]["sh0"]["cosine"] for u in us
             if u["metrics"]["sh0"]["cosine"] is not None]),
        "shN_cos_median": (lambda v: statistics.median(v) if v else None)(
            [u["metrics"]["shN"]["cosine"] for u in us
             if u["metrics"]["shN"]["cosine"] is not None]),
        "n_improving": n_improving,
        "n_nonimproving": len(us) - n_improving,
        "R_step_median": statistics.median(rs) if rs else None,
        "R_step_in_90_110": sum(1 for r in rs if 0.90 <= r <= 1.10) if rs else 0,
        "P_audit_median": statistics.median(pa) if pa else None,
        "P_audit_n_gt_001": n_big,
        "hard_drop_triggers": drop,
        "candidate_gates": cand,
        "verdict": verdict,
    }
    print(f"== {scene}: {verdict} ==")
    for k, v in results[scene].items():
        print(f"  {k}: {v}")

# Overall verdict
if any(r["verdict"] == "HARD_DROP" for r in results.values()):
    overall = "P17_P0_HARD_DROP"
elif all(r["verdict"] == "CANDIDATE" for r in results.values()):
    overall = "P17_P0_STAGE_A_STRONG"
else:
    overall = "P17_P0_HOLD_NO_STRONG_ORACLE"

print(f"\nOVERALL VERDICT: {overall}")

# Write P17_VERDICT.json
verdict_obj = {
    "verdict": overall,
    "scene_results": results,
    "notes": [
        "Primary tau=2.0 only; tau=1/4 are sensitivity outputs (P17_SENSITIVITY_TAU.json)",
        "Garden (Stage B) only runs if overall verdict is P17_P0_STAGE_A_STRONG",
        "HARD_DROP and HOLD are terminal — no garden, no P17-P1",
    ],
}
with open("/root/epic05_p17/P17_VERDICT.json", "w") as f:
    json.dump(verdict_obj, f, indent=2)
print("Wrote P17_VERDICT.json")
