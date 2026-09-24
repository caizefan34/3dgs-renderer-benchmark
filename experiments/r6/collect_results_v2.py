#!/usr/bin/env python3
"""
Collect ALL R6 profiling results (v1 + repaired) into r6-profile-results-v2.json.

This is the AUTHORITATIVE machine-readable source. All report tables must be
generated from this file.

v1 results (SUPERSEDED): r6_1_*.json, r6_4_*.json, r6_5_*.json
v2 results (REPAIRED): r6_a_direct_*.json, r6_c_repair_*.json, r6_b_correctness_*.json
"""
import json, glob, os, sys
import numpy as np

prof_dir = "/mnt/storage_pool/liaoyuanjun/r6_profiling"
results = {
    "version": "v2",
    "v1_status": "SUPERSEDED",
    "v2_changes": [
        "R6-A: Replaced footprint-based warp estimate with direct pixel-level simulation. R_atomic corrected from 5.5-7.6 to ~2.8-3.0.",
        "R6-A: Fixed dimensional labels: 'warps_per_gaussian' was actually 'within_tile_warps_per_gaussian' (per-tile, not total).",
        "R6-A: Added 'total_warps_per_gaussian' = tiles_per_gaussian 脳 within_tile_warps.",
        "R6-C: Fixed fundamental oracle error: T_C_conservative was T_opt + 0.5*traffic (ADDS instead of SAVES). Repaired to bandwidth-limited traffic model.",
        "R6-C: Conservative E2E corrected from 4.8-17.2% to 0.5-2.7% (10x overestimate in v1).",
        "R6-B: Added correctness validation framework (stale gradient test, gradient comparison, topology safety audit).",
    ],
    "r6_1_backward_decomposition": {},  # v1 (unchanged, timing is correct)
    "r6_a_direct_atomic": {},           # v2 (REPAIRED)
    "r6_a_original_estimated": {},      # v1 (SUPERSEDED, kept for provenance)
    "r6_c_repaired_oracle": {},         # v2 (REPAIRED)
    "r6_c_original_oracle": {},         # v1 (SUPERSEDED, kept for provenance)
    "r6_b_correctness": {},             # v2 (NEW)
    "gate_verdicts_v2": {},
    "summary": {}
}

scenes = ["room", "bicycle", "garden"]
stages = ["5000", "15000", "30000"]

# === R6-1 backward decomposition (v1, timing is correct, keep as-is) ===
for f in sorted(glob.glob(f"{prof_dir}/r6_1_*.json")):
    if "smoke" in f: continue
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_1_", "").replace(".json", "")
    results["r6_1_backward_decomposition"][name] = {
        "N_total": d["N_total"], "r_touch": d["r_touch"],
        "T_bwd_ms": d["T_bwd_ms"]["mean"], "T_iter_ms": d["T_iter_ms"]["mean"],
        "raster_bwd_ms": d["kernel_decomposition"]["raster_bwd"]["per_iter_ms"],
        "memset_zero_ms": d["kernel_decomposition"]["memset_zero"]["per_iter_ms"],
        "T_zero_pct_bwd": d["kernel_decomposition"]["T_zero_pct_bwd"],
        "T_raster_pct_bwd": d["kernel_decomposition"]["T_raster_pct_bwd"],
        "grad_buffer_bytes": d["grad_buffer_bytes"]["grand_total"],
    }

# === R6-A direct atomic (v2 REPAIRED) ===
for f in sorted(glob.glob(f"{prof_dir}/r6_a_direct_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_a_direct_", "").replace(".json", "")
    agg = d["aggregate"]
    results["r6_a_direct_atomic"][name] = {
        "N_total": d["N_total"],
        "method": "direct_pixel_simulation",
        "n_isects": agg["n_isects_mean"],
        "n_visible": agg["n_visible_mean"],
        "cross_tile_dup": agg["cross_tile_dup_mean"],
        "within_tile_warps": agg["within_tile_warps_mean"],
        "within_tile_warps_std": agg["within_tile_warps_std"],
        "R_atomic": agg["R_atomic_mean"],
        "R_atomic_std": agg["R_atomic_std"],
        "reduction_factor": agg["reduction_factor_mean"],
        "total_warps_per_gaussian": agg["cross_tile_dup_mean"] * agg["R_atomic_mean"],
    }

# === R6-A original estimated (v1 SUPERSEDED, for provenance) ===
for f in sorted(glob.glob(f"{prof_dir}/r6_4_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_4_", "").replace(".json", "")
    if name.endswith("_5k"): continue
    agg = d["aggregate"]
    results["r6_a_original_estimated"][name] = {
        "N_total": d["N_total"],
        "method": "footprint_estimate_SUPERSEDED",
        "R_atomic_estimated": agg["R_atomic_mean"],
        "tiles_per_gauss": agg["tiles_per_gauss_mean"],
        "within_tile_warps_estimated": agg.get("within_tile_warps_per_gauss_mean",
                                                 agg.get("warps_per_gauss_mean", 0)),
        "status": "SUPERSEDED 鈥?footprint overestimates by ~2.7x",
    }

# === R6-C repaired oracle (v2 REPAIRED) ===
for f in sorted(glob.glob(f"{prof_dir}/r6_c_repair_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_c_repair_", "").replace(".json", "")
    results["r6_c_repaired_oracle"][name] = {
        "N_total": d["N_total"],
        "T_optimizer_ms": d["T_optimizer_ms"]["mean"],
        "T_iter_ms": d["T_iter_ms"]["mean"],
        "grad_bytes": d["grad_bytes"],
        "intermediate_grad_bytes": d["intermediate_grad_bytes"],
        "adam_total_traffic_bytes": d["adam_total_traffic_bytes"],
        "grad_fraction_of_adam": d["grad_fraction_of_adam"],
        "T_saved_lower_ms": d["T_saved_lower_ms"],
        "T_saved_conservative_ms": d["T_saved_conservative_ms"],
        "T_saved_optimistic_ms": d["T_saved_optimistic_ms"],
        "T_saved_lower_pct_iter": d["T_saved_lower_pct_iter"],
        "T_saved_conservative_pct_iter": d["T_saved_conservative_pct_iter"],
        "T_saved_optimistic_pct_iter": d["T_saved_optimistic_pct_iter"],
        "original_broken_oracle_pct_iter": d["original_T_C_conservative_pct_iter"],
    }

# === R6-C original oracle (v1 SUPERSEDED, for provenance) ===
for f in sorted(glob.glob(f"{prof_dir}/r6_5_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_5_", "").replace(".json", "")
    results["r6_c_original_oracle"][name] = {
        "N_total": d["N_total"],
        "T_C_conservative_pct_iter": d["T_C_conservative_pct_iter"],
        "avoidable_grad_traffic_bytes": d["avoidable_grad_traffic_bytes"],
        "status": "SUPERSEDED 鈥?oracle adds T_opt + traffic instead of computing savings",
    }

# === R6-B correctness ===
for f in sorted(glob.glob(f"{prof_dir}/r6_b_correctness_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_b_correctness_", "").replace(".json", "")
    results["r6_b_correctness"][name] = {
        "N_total": d["N_total"],
        "variant": d.get("variant", "baseline"),
        "stale_gradient_test_pass": d["stale_gradient_test"]["PASS"],
        "gradient_comparison_pass": d["gradient_comparison"]["PASS"],
        "overall_pass": d["overall_pass"],
        "n_stale_gaussians": d["stale_gradient_test"]["n_stale_gaussians"],
    }

# === Gate verdicts v2 ===
for name in [f"{s}_{st}" for s in scenes for st in stages]:
    r1 = results["r6_1_backward_decomposition"].get(name, {})
    ra = results["r6_a_direct_atomic"].get(name, {})
    rc = results["r6_c_repaired_oracle"].get(name, {})

    if not r1:
        continue

    # R6-B gates (unchanged 鈥?timing is correct)
    b_gate1 = r1.get("T_zero_pct_bwd", 0) >= 5.0
    b_gate2 = r1.get("memset_zero_ms", 0) / r1.get("T_iter_ms", 1) * 100 >= 3.0

    # R6-A gates (REPAIRED with direct measurement)
    a_gate1 = r1.get("T_raster_pct_bwd", 0) >= 40.0
    a_gate2 = ra.get("R_atomic", 0) >= 2.0 if ra else False
    # A-GATE-3: conservative E2E 鈮?5%
    # Need to recompute with corrected R_atomic
    # Using hardware model: T_atomic = 5 脳 R_atomic 脳 n_isects / 30e9 脳 1000
    # Savings = T_atomic 脳 (1 - 1/R_atomic) 脳 0.70
    if ra and r1:
        R_atomic = ra["R_atomic"]
        n_isects = ra["n_isects"]
        T_raster = r1["raster_bwd_ms"]
        T_iter = r1["T_iter_ms"]
        total_atomics = 5 * R_atomic * n_isects
        T_atomic_cons = total_atomics / 30e9 * 1000  # ms
        ideal_atomics = 5 * n_isects
        T_atomic_ideal = ideal_atomics / 30e9 * 1000
        savings_cons = max(T_atomic_cons - T_atomic_ideal, 0) * 0.70
        a_gate3 = savings_cons / T_iter * 100 >= 5.0
        a_e2e_cons = savings_cons / T_iter * 100
    else:
        a_gate3 = False
        a_e2e_cons = 0

    # R6-C gate (REPAIRED)
    c_gate = rc.get("T_saved_conservative_pct_iter", 0) >= 5.0 if rc else False

    results["gate_verdicts_v2"][name] = {
        "R6-B": {"B-GATE-1": b_gate1, "B-GATE-2": b_gate2,
                  "T_zero_pct_bwd": r1.get("T_zero_pct_bwd", 0)},
        "R6-A": {"A-GATE-1": a_gate1, "A-GATE-2": a_gate2, "A-GATE-3": a_gate3,
                  "R_atomic_direct": ra.get("R_atomic", 0),
                  "within_tile_warps_direct": ra.get("within_tile_warps", 0),
                  "S_atomic_cons_pct_iter": a_e2e_cons},
        "R6-C": {"C-GATE": c_gate,
                  "T_saved_cons_pct_iter": rc.get("T_saved_conservative_pct_iter", 0),
                  "T_saved_opt_pct_iter": rc.get("T_saved_optimistic_pct_iter", 0),
                  "original_broken_pct_iter": rc.get("original_broken_oracle_pct_iter", 0)},
    }

# === Summary ===
for cand in ["R6-B", "R6-A", "R6-C"]:
    passing = []
    for name, gates in results["gate_verdicts_v2"].items():
        if cand == "R6-B":
            if gates["R6-B"]["B-GATE-1"]:
                passing.append(name)
        elif cand == "R6-A":
            if gates["R6-A"]["A-GATE-1"] and gates["R6-A"]["A-GATE-2"]:
                passing.append(name)
        elif cand == "R6-C":
            if gates["R6-C"]["C-GATE"]:
                passing.append(name)
    results["summary"][cand] = {
        "profiles_passing": len(passing),
        "profiles_total": len(results["gate_verdicts_v2"]),
        "passing_names": passing,
    }

out_path = f"{prof_dir}/r6-profile-results-v2.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
print(f"\nGate Summary (v2 REPAIRED):")
for cand, s in results["summary"].items():
    print(f"  {cand}: {s['profiles_passing']}/{s['profiles_total']} profiles passing")
print(f"\nR6-B correctness:")
for name, r in results["r6_b_correctness"].items():
    print(f"  {name}: overall_pass={r['overall_pass']}")
