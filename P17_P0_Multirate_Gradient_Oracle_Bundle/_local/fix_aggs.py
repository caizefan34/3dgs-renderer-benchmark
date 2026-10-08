import json
import os
import re
import statistics

log_re = re.compile(
    r"\b(room|bicycle) cam(\d+): C_oracle=([\d.eE+-]+) n_d2\+=\d+ "
    r"means_cos=([\d.eE+-]+) R_step=([\d.eE+-]+) P_audit=([\d.eE+-]+)\b")

def parse_log(path):
    out = []
    for m in log_re.finditer(open(path).read()):
        scene, cam, C, mc, R, P = m.groups()
        out.append({
            "scene": scene, "camera_id": int(cam),
            "oracle_interaction_ratio": float(C),
            "means_cos": float(mc),
            "R_step": float(R),
            "P_audit": float(P),
        })
    return out

main_logs = {
    "room": "/root/epic05_p17/room_main.log",
    "bicycle": "/root/epic05_p17/bicycle_main.log",
}

# P17_VERDICT
results = {}
for scene, path in main_logs.items():
    recs = parse_log(path)
    co = [r["oracle_interaction_ratio"] for r in recs]
    rs = [r["R_step"] for r in recs]
    pa = [r["P_audit"] for r in recs]
    mc = [r["means_cos"] for r in recs]
    drop = []
    if statistics.median(co) > 0.50: drop.append("C_oracle>0.5")
    if statistics.median(mc) < 0.980: drop.append("means_cos<0.98")
    if statistics.median(rs) < 0.80: drop.append("R_step<0.8")
    if sum(1 for p in pa if p > 0.01) >= 4: drop.append("P_audit>0.01 x4")
    results[scene] = {
        "n_units": len(recs),
        "C_oracle_median": statistics.median(co),
        "C_oracle_p75": sorted(co)[int(0.75 * (len(co) - 1))],
        "means_cos_median": statistics.median(mc),
        "means_cos_p10": sorted(mc)[int(0.1 * (len(mc) - 1))],
        "R_step_median": statistics.median(rs),
        "R_step_in_90_110": sum(1 for r in rs if 0.90 <= r <= 1.10),
        "P_audit_median": statistics.median(pa),
        "P_audit_n_gt_001": sum(1 for p in pa if p > 0.01),
        "hard_drop_triggers": drop,
        "verdict": "HARD_DROP" if drop else "NO_HARD_DROP",
    }
    print(f"== {scene}: {results[scene]['verdict']} ==")
    for k in ("n_units", "C_oracle_median", "means_cos_median",
              "R_step_median", "P_audit_median"):
        print(f"  {k}: {results[scene][k]}")

overall = ("P17_P0_HARD_DROP"
           if any(r["verdict"] == "HARD_DROP" for r in results.values())
           else "P17_P0_HOLD_NO_STRONG_ORACLE")
print(f"\nOVERALL: {overall}")

verdict_obj = {
    "verdict": overall,
    "primary_tau": 2.0,
    "scene_results": results,
    "notes": [
        "tau=1/4 are sensitivity outputs and cannot rescue a failed gate.",
        "HARD_DROP and HOLD are terminal: no garden (Stage B), no P17-P1.",
    ],
}
with open("/root/epic05_p17/P17_VERDICT.json", "w") as f:
    json.dump(verdict_obj, f, indent=2)
print("Wrote P17_VERDICT.json")

# P17_GRADIENT_AGG
agg = {"primary_tau": 2.0, "scenes": {}}
for scene, path in main_logs.items():
    recs = parse_log(path)
    co = [r["oracle_interaction_ratio"] for r in recs]
    mc = [r["means_cos"] for r in recs]
    rs = [r["R_step"] for r in recs]
    pa = [r["P_audit"] for r in recs]
    agg["scenes"][scene] = {
        "n_units": len(recs),
        "work": {
            "C_oracle_median": statistics.median(co),
            "C_oracle_p75": sorted(co)[int(0.75 * (len(co) - 1))],
        },
        "fidelity": {
            "means": {
                "cosine_median": statistics.median(mc),
                "cosine_p10": sorted(mc)[int(0.1 * (len(mc) - 1))],
            },
        },
        "step": {
            "R_step_median": statistics.median(rs),
            "R_step_in_90_110": sum(1 for r in rs if 0.90 <= r <= 1.10),
            "P_audit_median": statistics.median(pa),
            "P_audit_n_gt_001": sum(1 for p in pa if p > 0.01),
        },
    }
with open("/root/epic05_p17/P17_GRADIENT_AGG.json", "w") as f:
    json.dump(agg, f, indent=2)
print("Wrote P17_GRADIENT_AGG.json")

# P17_STEP_AGG
step_agg = {"step_t": 16001, "scenes": {}}
for scene, path in main_logs.items():
    recs = parse_log(path)
    step_agg["scenes"][scene] = {
        "R_step_values": [
            {"camera_id": r["camera_id"], "R_step": r["R_step"],
             "P_audit": r["P_audit"]} for r in recs
        ],
    }
with open("/root/epic05_p17/P17_STEP_AGG.json", "w") as f:
    json.dump(step_agg, f, indent=2)
print("Wrote P17_STEP_AGG.json")

# P17_WORK_ORACLE
work = {}
for scene, path in main_logs.items():
    recs = parse_log(path)
    work[scene] = {
        "C_oracle_median": statistics.median(
            [r["oracle_interaction_ratio"] for r in recs]),
        "C_oracle_values": [
            {"camera_id": r["camera_id"],
             "C_oracle": r["oracle_interaction_ratio"]} for r in recs
        ],
    }
with open("/root/epic05_p17/P17_WORK_ORACLE.json", "w") as f:
    json.dump(work, f, indent=2)
print("Wrote P17_WORK_ORACLE.json")
print("All done.")
