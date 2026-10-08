#!/usr/bin/env python3
"""G2 收益预算 (benefit budget) per plan sec.7/sec.9.

Reads measured G1 compressibility (q_weighted, S_inference at gamma=1/2/4) and
applies the sec.7 cost-model stop-rule WITHOUT fabricating any GPU cost:

    S = 1 / (1 - p*f*(1 - 1/r) + h)

Mapping used (p*f*(1-1/r) is exactly the Amdahl forward-ratio reduction the G1
pipeline already measured):
  - p*f  -> q_weighted (work-weighted compressible fraction, min over ranks<=R)
  - r    -> gamma (per-component cost relative to one Gaussian; 1/2/4)
  - the measured S_inference(gamma) = 1/fwd_ratio is the ZERO-OVERHEAD (h=0)
    optimistic feasible upper bound.
  - h    -> real implementation overhead (tree traversal/gating/normalization/
            compositing/adaptive maintenance), as a fraction of baseline time.

Key G2 quantity: break-even overhead budget
    h_max(target, gamma) = 1/target - fwd_ratio(gamma) = 1/target - 1/S(gamma)
i.e. the largest overhead fraction the future renderer may add before the
end-to-end speedup drops to `target`. Negative => unreachable even at h=0.

sec.9 G2 stop rule: "if the optimistic feasible upper bound is below target,
stop that target." We evaluate at the plan 3x target and the pre-registered
1.25x gate target. This does NOT claim a speedup; it quantifies the overhead
budget that G3 (measured CUDA costs) must fit inside.
"""
import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "G1_REAL_SCENE_COMPRESSIBILITY.json")
GAMMAS = [1.0, 2.0, 4.0]
TARGETS = {"gate_1.25x": 1.25, "plan_3x": 3.0}

def main():
    d = json.load(open(SRC, encoding="utf-8"))
    combos = d["combos"]
    out = {
        "gate": "G2_benefit_budget",
        "formula": "S = 1/(1 - p*f*(1 - 1/r) + h);  p*f->q_weighted, r->gamma, h->overhead_fraction",
        "inputs_source": "G1_REAL_SCENE_COMPRESSIBILITY.json (measured, zero fabricated GPU cost)",
        "targets": TARGETS,
        "gammas": GAMMAS,
        "per_combo": {},
    }
    # collect h_max grids
    hmax_grid = {t: {g: [] for g in GAMMAS} for t in TARGETS}
    Smin_per_gamma = {g: [] for g in GAMMAS}
    for name, c in combos.items():
        S = {float(g): c["S_inference_actual"][g] for g in c["S_inference_actual"]}
        q = c["q_weighted_cumulative"]
        incomp = c["incompressible_work_frac"]
        rec = {
            "q_weighted_cumulative": q,
            "incompressible_work_frac": incomp,
            "S_zero_overhead": {str(g): S[g] for g in GAMMAS},
            "h_max": {},  # target -> gamma -> budget
        }
        for tname, tval in TARGETS.items():
            rec["h_max"][tname] = {}
            for g in GAMMAS:
                fwd = 1.0 / S[g]
                hmax = 1.0 / tval - fwd
                rec["h_max"][tname][str(g)] = hmax
                hmax_grid[tname][g].append(hmax)
            # gamma needed for target at h=0: smallest gamma s.t. S(gamma)>=target
        out["per_combo"][name] = rec
        for g in GAMMAS:
            Smin_per_gamma[g].append(S[g])

    # aggregate (conservative = min over combos)
    agg = {}
    for tname in TARGETS:
        agg[tname] = {}
        for g in GAMMAS:
            vals = hmax_grid[tname][g]
            agg[tname][str(g)] = {
                "min_over_combos": min(vals),
                "mean_over_combos": sum(vals) / len(vals),
                "max_over_combos": max(vals),
            }
    out["aggregate_h_max"] = agg

    # optimistic feasible upper bound = min over combos of S at gamma=1 (largest S)
    opt_bound = min(Smin_per_gamma[1.0])
    out["optimistic_upper_bound_gamma1_min_over_combos"] = opt_bound
    out["optimistic_upper_bound_gamma4_min_over_combos"] = min(Smin_per_gamma[4.0])

    # sec.9 G2 stop rule
    stop = {}
    for tname, tval in TARGETS.items():
        stop[tname] = {
            "optimistic_upper_bound": opt_bound,
            "below_target": opt_bound < tval,
            "verdict": ("STOP_TARGET_OPTIMISTIC_BOUND_BELOW_TARGET"
                        if opt_bound < tval else
                        "DO_NOT_STOP_OPTIMISTIC_BOUND_ABOVE_TARGET"),
        }
    out["g2_stop_rule"] = stop

    # overall: the tightest overhead budget that still clears 3x at gamma=2 (the
    # pre-registered gamma_gate) across all combos
    out["binding_constraint"] = {
        "note": ("Tightest (min over combos) break-even overhead to still clear "
                 "plan 3x at the pre-registered gamma_gate=2. G3 measured overhead "
                 "must be <= this for the 3x target to survive."),
        "h_max_3x_gamma2_min_over_combos": agg["plan_3x"]["2.0"]["min_over_combos"],
        "h_max_3x_gamma1_min_over_combos": agg["plan_3x"]["1.0"]["min_over_combos"],
        "h_max_3x_gamma4_min_over_combos": agg["plan_3x"]["4.0"]["min_over_combos"],
    }

    with open(os.path.join(HERE, "G2_BENEFIT_BUDGET.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote G2_BENEFIT_BUDGET.json")
    print("optimistic upper bound (gamma=1, min over combos) =", round(opt_bound, 3))
    print("h_max to clear 3x at gamma=2 (min over combos) =",
          round(agg["plan_3x"]["2.0"]["min_over_combos"], 4))
    for tname in TARGETS:
        print(tname, "stop verdict:", stop[tname]["verdict"])

if __name__ == "__main__":
    main()
