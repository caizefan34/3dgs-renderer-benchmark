import json
import os
import re
import statistics

# The P17_UNITS/<scene>/camera_*.json files are overwritten by later tau runs
# (run_all.sh order: main(2.0) -> tau1 -> tau4), so the final directory state
# holds the tau=4.0 units. The main-run units (tau=2.0) are recovered from
# the per-tau log files: each "wrote" line is followed by
# "<scene> cam<N>: C_oracle=.. n_d2+=.. means_cos=.. R_step=.. P_audit=.."
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
            "metrics": {"means": {"cosine": float(mc)}},
            "step": {"R_step": float(R), "P_audit": float(P)},
        })
    return out

logs = {
    ("2.0", "room"): "/root/epic05_p17/room_main.log",
    ("2.0", "bicycle"): "/root/epic05_p17/bicycle_main.log",
    ("1.0", "room"): "/root/epic05_p17/room_tau1.log",
    ("1.0", "bicycle"): "/root/epic05_p17/bicycle_tau1.log",
    ("4.0", "room"): "/root/epic05_p17/room_tau4.log",
    ("4.0", "bicycle"): "/root/epic05_p17/bicycle_tau4.log",
}

by_tau = {}
for (tau, scene), path in logs.items():
    recs = parse_log(path)
    for r in recs:
        r["tau"] = float(tau)
        by_tau.setdefault(tau, []).append(r)

def units2(scene):
    return [r for r in by_tau.get("2.0", []) if r["scene"] == scene]

def units_t(tau, scene):
    return [r for r in by_tau.get(str(tau), []) if r["scene"] == scene]

# P17_VERDICT: primary tau=2.0, gates per prereg
def gate(units):
    co = [u["oracle_interaction_ratio"] for u in units]
    rs = [u["step"]["R_step"] for u in units]
    pa = [u["step"]["P_audit"] for u in units]
    mcos = [u["metrics"]["means"]["cosine"] for u in units]
    ocos = []  # opacity not in log line; fall back to None
    drop = []
    if statistics.median(co) > 0.50: drop.append("C_oracle>0.5")
    if statistics.median(mcos) < 0.980: drop.append("means_cos<0.98")
    if statistics.median(rs) < 0.80: drop.append("R_step<0.8")
    if sum(1 for p in pa if p > 0.01) >= 4: drop.append("P_audit>0.01 x4")
    return drop

results = {}
for scene in ("room", "bicycle"):
    u2 = units2(scene)
    drop = gate(u2)
    results[scene] = {
        "n_units": len(u2),
        "C_oracle_median": statistics.median(
            [r["oracle_interaction_ratio"] for r in u2]),
        "means_cos_median": statistics.median(
            [r["metrics"]["means"]["cosine"] for r in u2]),
        "R_step_median": statistics.median([r["step"]["R_step"] for r in u2]),
        "P_audit_median": statistics.median(
            [r["step"]["P_audit"] for r in u2]),
        "hard_drop_triggers": drop,
        "verdict": "HARD_DROP" if drop else "NO_HARD_DROP",
    }
    print(f"== {scene}: {results[scene]['verdict']} "
          f"({', '.join(drop) if drop else 'n/a'}) ==")
    for k in ("n_units", "C_oracle_median", "means_cos_median",
              "R_step_median", "P_audit_median"):
        print(f"  {k}: {results[scene][k]}")

overall = ("P17_P0_HARD_DROP"
           if any(r["verdict"] == "HARD_DROP" for r in results.values())
           else "P17_P0_HOLD_NO_STRONG_ORACLE")
print(f"\nOVERALL VERDICT: {overall}")

verdict_obj = {
    "verdict": overall,
    "primary_tau": 2.0,
    "scene_results": results,
    "notes": [
        "tau=1/4 are sensitivity outputs and cannot rescue a failed gate.",
        "HARD_DROP and HOLD are terminal: no garden (Stage B), no P17-P1.",
        "Units were recovered from per-tau log lines; the P17_UNITS directory "
        "final state holds the tau=4.0 unit files (last run overwrote them).",
    ],
}
with open("/root/epic05_p17/P17_VERDICT.json", "w") as f:
    json.dump(verdict_obj, f, indent=2)
print("Wrote P17_VERDICT.json")

# P17_GRADIENT_AGG (primary tau=2.0)
agg = {"primary_tau": 2.0, "scenes": {}}
for scene in ("room", "bicycle"):
    u2 = units2(scene)
    co = [r["oracle_interaction_ratio"] for r in u2]
    mcos = [r["metrics"]["means"]["cosine"] for r in u2]
    rs = [r["step"]["R_step"] for r in u2]
    pa = [r["step"]["P_audit"] for r in u2]
    agg["scenes"][scene] = {
        "n_units": len(u2),
        "work": {
            "C_oracle_median": statistics.median(co),
            "C_oracle_p75": sorted(co)[int(0.75 * (len(co) - 1))],
        },
        "fidelity": {
            "means": {
                "cosine_median": statistics.median(mcos),
                "cosine_p10": sorted(mcos)[int(0.1 * (len(mcos) - 1))],
            },
        },
        "step": {
            "R_step_median": statistics.median(rs),
            "R_step_in_90_110": sum(
                1 for r in rs if 0.90 <= r <= 1.10),
            "P_audit_median": statistics.median(pa),
            "P_audit_n_gt_001": sum(1 for p in pa if p > 0.01),
        },
    }
with open("/root/epic05_p17/P17_GRADIENT_AGG.json", "w") as f:
    json.dump(agg, f, indent=2)
print("Wrote P17_GRADIENT_AGG.json")

# P17_STEP_AGG (primary tau=2.0)
step_agg = {"step_t": 16001, "scenes": {}}
for scene in ("room", "bicycle"):
    u2 = units2(scene)
    step_agg["scenes"][scene] = {
        "R_step_values": [
            {"camera_id": r["camera_id"], "R_step": r["step"]["R_step"],
             "P_audit": r["step"]["P_audit"]} for r in u2
        ],
    }
with open("/root/epic05_p17/P17_STEP_AGG.json", "w") as f:
    json.dump(step_agg, f, indent=2)
print("Wrote P17_STEP_AGG.json")

# P17_WORK_ORACLE (primary tau=2.0)
work = {}
for scene in ("room", "bicycle"):
    u2 = units2(scene)
    work[scene] = {
        "C_oracle_median": statistics.median(
            [r["oracle_interaction_ratio"] for r in u2]),
        "C_oracle_values": [
            {"camera_id": r["camera_id"],
             "C_oracle": r["oracle_interaction_ratio"]} for r in u2
        ],
    }
with open("/root/epic05_p17/P17_WORK_ORACLE.json", "w") as f:
    json.dump(work, f, indent=2)
print("Wrote P17_WORK_ORACLE.json")

# P17_SENSITIVITY_TAU (tau=1,4)
sens = {"taus": {}}
for tau in (1.0, 4.0):
    sens["taus"][str(tau)] = {}
    for scene in ("room", "bicycle"):
        ut = units_t(tau, scene)
        if not ut:
            continue
        co = [r["oracle_interaction_ratio"] for r in ut]
        rs = [r["step"]["R_step"] for r in ut]
        mcos = [r["metrics"]["means"]["cosine"] for r in ut]
        sens["taus"][str(tau)][scene] = {
            "n_units": len(ut),
            "C_oracle_median": statistics.median(co),
            "means_cos_median": statistics.median(mcos),
            "R_step_median": statistics.median(rs),
            "R_step_values": rs,
        }
with open("/root/epic05_p17/P17_SENSITIVITY_TAU.json", "w") as f:
    json.dump(sens, f, indent=2)
print("Wrote P17_SENSITIVITY_TAU.json")
print("All aggregation files written.")
