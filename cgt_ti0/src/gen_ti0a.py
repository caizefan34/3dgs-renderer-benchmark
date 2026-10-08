#!/usr/bin/env python3
"""Generate TI0A_UPDATE_COST.json + TI0A_DIRTY_WORK.json from the per-view
footprint-stable TI0-A measurement (authoritative result backing TI0_REPORT.md).
Frozen upstream: F0 fused kernel, g1 fit, g2 certificate. No modification.
"""
import json, os

OUT = "/mnt/storage_pool/liaoyuanjun/cgt_ti0/results"

per_view = {
    15:  {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 6, "dirty_fraction": 0.60, "wdirty": 0.366, "shift_med": 1.18e-2, "Tcan": 0.311, "Tcgt": 0.015, "refit_all": 3568.2, "cert_dirty": 666.0},
    125: {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 2, "dirty_fraction": 0.20, "wdirty": 0.160, "shift_med": 2.23e-4, "Tcan": 0.343, "Tcgt": 0.014, "refit_all": 829.8,  "cert_dirty": 496.0},
    129: {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 2, "dirty_fraction": 0.20, "wdirty": 0.217, "shift_med": 0.00,  "Tcan": 0.158, "Tcgt": 0.015, "refit_all": 693.6,  "cert_dirty": 484.0},
    133: {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 7, "dirty_fraction": 0.70, "wdirty": 0.644, "shift_med": 8.05e-3, "Tcan": 0.253, "Tcgt": 0.012, "refit_all": 3957.1, "cert_dirty": 1354.5},
    193: {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 2, "dirty_fraction": 0.20, "wdirty": 0.159, "shift_med": 4.99e-3, "Tcan": 0.121, "Tcgt": 0.017, "refit_all": 1400.7, "cert_dirty": 191.6},
    254: {"n_tiles": 10, "surv": 10, "match": 10, "dirty": 5, "dirty_fraction": 0.50, "wdirty": 0.404, "shift_med": 7.20e-3, "Tcan": 0.403, "Tcgt": 0.017, "refit_all": 2517.7, "cert_dirty": 698.3},
}
views = [15, 125, 129, 133, 193, 254]

T_canonical = 0.2821119874715805
T_cgt       = 0.015359999611973763
T_update_A  = 1959.1939449310303
T_update_B  = 1692.1075582504272
T_update_C  = 581.0040235519409

R_update_A = T_update_A / T_canonical
R_update_B = T_update_B / T_canonical
R_update_C = T_update_C / T_canonical
S_A = T_canonical / (T_update_A + T_cgt)
S_B = T_canonical / (T_update_B + T_cgt)
S_C = T_canonical / (T_update_C + T_cgt)

agg_dirty = sum(per_view[v]["dirty"] for v in views) / sum(per_view[v]["n_tiles"] for v in views)
agg_wdirty = sum(per_view[v]["wdirty"] for v in views) / len(views)
agg_shift_med = 0.006091690978623804
agg_shift_max = 0.009197480029438654

update = {
    "strategy_A_full_rebuild": "MEASURED",
    "strategy_B_affected_tiles": "MEASURED",
    "strategy_C_incremental_coeffs": "MEASURED",
    "method": "per-view footprint-stable: theta_t footprints reused at theta_t1; ONE real reference_v1 Adam step (densification OFF at iter 15000)",
    "T_update_A_ms": T_update_A,
    "T_update_B_ms": T_update_B,
    "T_update_C_ms": T_update_C,
    "notes": "T_update per-view average (ms). A=full per-tile refit (fit_cgt L-BFGS-B 200 iters + certify_tile over all tiles); B=certify-all + refit-dirty-only (realistic); C=certify-all + reuse-passing (optimistic lower bound on update cost). CGT representation is STATELESS across optimizer steps => A is the production route; B/C are lower bounds."
}

ratios = {
    "R_update_A": R_update_A,
    "R_update_B": R_update_B,
    "R_update_C": R_update_C,
    "T_effective_CGT_A_ms": T_update_A + T_cgt,
    "T_effective_CGT_B_ms": T_update_B + T_cgt,
    "T_effective_CGT_C_ms": T_update_C + T_cgt,
    "S_effective_A": S_A,
    "S_effective_B": S_B,
    "S_effective_C": S_C,
}

dirty_work = {
    "aggregate": {
        "dirty_fraction": agg_dirty,
        "weighted_dirty": agg_wdirty,
        "teacher_shift_median": agg_shift_med,
        "teacher_shift_max": agg_shift_max,
    },
    "per_view": {str(v): per_view[v] for v in views},
}

def verdict(S):
    if S <= 1.0: return "DROP_UPDATE_COST"
    if S < 1.15: return "MARGINAL_UPDATE_HEADROOM"
    return "PROCEED_TI0_B"

S_primary = S_B
v = verdict(S_primary)

result = {
    "stage": "TI0-A",
    "ts": "2026-10-04 20:06:15",
    "scene": "room",
    "ckpt": "iter_15000.pt",
    "n_gaussians": 901818,
    "views": views,
    "rank": 4,
    "fit_iters": 200,
    "clean_gpu": True,
    "device": 0,
    "method": "per-view footprint-stable (theta_t footprints reused at theta_t1)",
    "update": update,
    "ratios": ratios,
    "dirty_work": dirty_work["aggregate"],
    "verdict_TI0_A": v,
    "S_primary": S_primary,
    "gate_thresholds": {"PASS_ge": 1.15, "MARGINAL_ge": 1.0, "DROP_le": 1.0},
}

with open(os.path.join(OUT, "TI0A_UPDATE_COST.json"), "w") as f:
    json.dump(result, f, indent=2)
with open(os.path.join(OUT, "TI0A_DIRTY_WORK.json"), "w") as f:
    json.dump(dirty_work, f, indent=2)

print("WROTE TI0A_UPDATE_COST.json verdict=%s S_B=%.6e R_update_B=%.1f" % (v, S_B, R_update_B))
print("WROTE TI0A_DIRTY_WORK.json dirty_fraction=%.3f weighted_dirty=%.4f" % (agg_dirty, agg_wdirty))
