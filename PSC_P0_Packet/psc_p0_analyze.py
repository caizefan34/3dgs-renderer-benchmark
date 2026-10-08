#!/usr/bin/env python3
"""
Frozen mechanical verdict calculator for PSC-P0 v1.

Input JSON must contain:
{
  "timing": {"S_upper": {"8": float}},
  "primary": {
    "R_int_energy": {"2": float, "4": float, "8": float},
    "leave_one_out": {"8": {"P_hold_pos": float, "median_C_hold": float}}
  }
}
"""
import argparse, json, math, pathlib, sys

def finite01(x, name):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return float(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_json")
    ap.add_argument("--out", default="PSC_P0_VERDICT.json")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.metrics_json).read_text())
    s8 = finite01(data["timing"]["S_upper"]["8"], "S_upper_8")
    r2 = finite01(data["primary"]["R_int_energy"]["2"], "R_int_energy_B2")
    r4 = finite01(data["primary"]["R_int_energy"]["4"], "R_int_energy_B4")
    r8 = finite01(data["primary"]["R_int_energy"]["8"], "R_int_energy_B8")
    p8 = finite01(data["primary"]["leave_one_out"]["8"]["P_hold_pos"], "P_hold_pos_B8")
    c8 = finite01(data["primary"]["leave_one_out"]["8"]["median_C_hold"], "median_C_hold_B8")

    fired_drop = []
    if s8 < 1.30:
        fired_drop.append("S_upper_8 < 1.30")
    if r4 < 0.40:
        fired_drop.append("R_int_energy_B4 < 0.40")
    if p8 <= 0.55:
        fired_drop.append("P_hold_pos_B8 <= 0.55")

    strong_checks = {
        "S_upper_8 >= 1.50": s8 >= 1.50,
        "R_int_energy_B8 >= 0.65": r8 >= 0.65,
        "P_hold_pos_B8 >= 0.75": p8 >= 0.75,
        "R_int_energy_B8 >= 0.50 * R_int_energy_B2": r8 >= 0.50 * r2,
    }

    if fired_drop:
        verdict = "PSC_P0_DROP"
        fired = fired_drop
    elif all(strong_checks.values()):
        verdict = "PSC_P0_STRONG_PROMOTE"
        fired = [k for k,v in strong_checks.items() if v]
    else:
        verdict = "PSC_P0_INCONCLUSIVE"
        fired = [k for k,v in strong_checks.items() if v]

    out = {
        "schema_version": "PSC-P0-v1",
        "verdict": verdict,
        "inputs": {
            "S_upper_8": s8,
            "R_int_energy_B2": r2,
            "R_int_energy_B4": r4,
            "R_int_energy_B8": r8,
            "P_hold_pos_B8": p8,
            "median_C_hold_B8": c8,
        },
        "drop_rules_fired": fired_drop,
        "strong_checks": strong_checks,
        "fired_rules": fired,
        "precedence": ["DROP", "STRONG_PROMOTE", "INCONCLUSIVE"],
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
