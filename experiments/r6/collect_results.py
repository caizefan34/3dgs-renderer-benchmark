#!/usr/bin/env python3
"""Collect all R6 profiling results into a single machine-readable JSON."""
import json, glob, os, sys
import numpy as np

prof_dir = "/mnt/storage_pool/liaoyuanjun/r6_profiling"
results = {"r6_1": {}, "r6_4": {}, "r6_5": {}}

# R6-1 backward decomposition
for f in sorted(glob.glob(f"{prof_dir}/r6_1_*.json")):
    if "smoke" in f: continue
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_1_", "").replace(".json", "")
    results["r6_1"][name] = {
        "N_total": d["N_total"], "N_visible": d["N_visible"],
        "r_touch": d["r_touch"], "sh_degree": d["sh_degree"],
        "T_bwd_ms": d["T_bwd_ms"]["mean"],
        "T_fwd_ms": d["T_fwd_ms"]["mean"],
        "T_iter_ms": d["T_iter_ms"]["mean"],
        "raster_bwd_ms": d["kernel_decomposition"]["raster_bwd"]["per_iter_ms"],
        "memset_zero_ms": d["kernel_decomposition"]["memset_zero"]["per_iter_ms"],
        "sh_bwd_ms": d["kernel_decomposition"]["sh_bwd"]["per_iter_ms"],
        "proj_bwd_ms": d["kernel_decomposition"]["proj_bwd"]["per_iter_ms"],
        "other_bwd_ms": d["kernel_decomposition"]["other_bwd"]["per_iter_ms"],
        "T_zero_pct_bwd": d["kernel_decomposition"]["T_zero_pct_bwd"],
        "T_raster_pct_bwd": d["kernel_decomposition"]["T_raster_pct_bwd"],
        "grad_buffer_bytes": d["grad_buffer_bytes"]["grand_total"],
    }

# R6-4 warp duplicate
for f in sorted(glob.glob(f"{prof_dir}/r6_4_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_4_", "").replace(".json", "")
    if name.endswith("_5k"): continue  # skip duplicate
    agg = d["aggregate"]
    results["r6_4"][name] = {
        "N_total": d["N_total"],
        "R_atomic": agg["R_atomic_mean"],
        "R_atomic_std": agg["R_atomic_std"],
        "tiles_per_gaussian": agg["tiles_per_gauss_mean"],
        "warps_per_gaussian": agg["warps_per_gauss_mean"],
        "n_isects": agg["n_isects_mean"],
    }

# R6-5 fusion oracle
for f in sorted(glob.glob(f"{prof_dir}/r6_5_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_5_", "").replace(".json", "")
    results["r6_5"][name] = {
        "N_total": d["N_total"],
        "T_optimizer_ms": d["T_optimizer_ms"]["mean"],
        "T_optimizer_pct_iter": d["T_optimizer_pct_iter"],
        "T_C_conservative_ms": d["T_C_conservative_ms"],
        "T_C_conservative_pct_iter": d["T_C_conservative_pct_iter"],
        "avoidable_grad_traffic_bytes": d["avoidable_grad_traffic_bytes"],
        "T_iter_ms": d["T_iter_ms"]["mean"],
    }

# Compute derived metrics for R6-B oracle
for name, d in results["r6_1"].items():
    r_touch = d["r_touch"]
    T_zero = d["memset_zero_ms"]
    T_iter = d["T_iter_ms"]
    T_bwd = d["T_bwd_ms"]
    # Upper bound speedup if ALL zero-init eliminated
    d["S_zero_upper"] = T_iter / max(T_iter - T_zero, 0.001)
    d["T_zero_pct_iter"] = T_zero / T_iter * 100
    # Conservative: only untouched buffers can be lazily zeroed
    d["T_zero_conservative_savings_ms"] = T_zero * (1 - r_touch)
    d["S_zero_conservative"] = T_iter / max(T_iter - T_zero * (1 - r_touch), 0.001)
    d["T_zero_conservative_pct_iter"] = T_zero * (1 - r_touch) / T_iter * 100

# Compute R6-A oracle (hardware-model estimation)
# A100 L2 atomic throughput: ~45 G 32-bit atomics/sec (peak, no contention)
# Conservative: 30 G/sec (with contention)
# Each (warp, Gaussian) = 5 cache line atomics (not 11, since same-buffer atomics coalesce)
a100_atomic_throughput_conservative = 30e9  # 30 G ops/sec
a100_atomic_throughput_moderate = 45e9  # 45 G ops/sec
for name, d in results["r6_1"].items():
    r4 = results["r6_4"].get(name, {})
    if not r4: continue
    R_atomic = r4["R_atomic"]
    n_isects = r4["n_isects"]
    T_raster = d["raster_bwd_ms"]
    T_iter = d["T_iter_ms"]
    # Total atomics = 5 cache lines × R_atomic × n_isects
    total_atomics = 5 * R_atomic * n_isects
    # Atomic time estimates
    T_atomic_conservative = total_atomics / a100_atomic_throughput_conservative * 1000  # ms
    T_atomic_moderate = total_atomics / a100_atomic_throughput_moderate * 1000
    # Ideal atomics (block-aggregated) = 5 × n_isects
    ideal_atomics = 5 * n_isects
    T_atomic_ideal = ideal_atomics / a100_atomic_throughput_conservative * 1000
    # Savings
    savings_conservative = max(T_atomic_conservative - T_atomic_ideal, 0) * 0.70  # 30% overhead
    savings_moderate = max(T_atomic_moderate - T_atomic_ideal, 0) * 0.80  # 20% overhead
    d["R_atomic"] = R_atomic
    d["T_atomic_conservative_ms"] = T_atomic_conservative
    d["T_atomic_moderate_ms"] = T_atomic_moderate
    d["atomic_fraction_conservative"] = T_atomic_conservative / max(T_raster, 0.001)
    d["atomic_fraction_moderate"] = T_atomic_moderate / max(T_raster, 0.001)
    d["S_atomic_conservative_pct_iter"] = savings_conservative / T_iter * 100
    d["S_atomic_moderate_pct_iter"] = savings_moderate / T_iter * 100

# Gate verdicts
gate_verdicts = {"R6-B": {}, "R6-A": {}, "R6-C": {}}
for name, d in results["r6_1"].items():
    # B-GATE-1: T_zero >= 5% T_bwd
    b1 = d["T_zero_pct_bwd"] >= 5.0
    # B-GATE-2: T_zero >= 3% T_iter
    b2 = d["T_zero_pct_iter"] >= 3.0
    # B-GATE-3: sparse-tail (check 30K vs 15K: T_zero%T_bwd increases as work decreases)
    gate_verdicts["R6-B"][name] = {"B-GATE-1": b1, "B-GATE-2": b2,
                                    "T_zero_pct_bwd": d["T_zero_pct_bwd"],
                                    "T_zero_pct_iter": d["T_zero_pct_iter"]}

for name, d in results["r6_1"].items():
    r4 = results["r6_4"].get(name, {})
    if not r4: continue
    # A-GATE-1: rasterizer is dominant backward kernel
    a1 = d["T_raster_pct_bwd"] >= 40.0
    # A-GATE-2: R_atomic >= 2
    a2 = r4["R_atomic"] >= 2.0
    # A-GATE-3: conservative E2E >= 5%
    a3 = d.get("S_atomic_conservative_pct_iter", 0) >= 5.0
    gate_verdicts["R6-A"][name] = {"A-GATE-1": a1, "A-GATE-2": a2, "A-GATE-3": a3,
                                    "R_atomic": r4["R_atomic"],
                                    "S_atomic_conservative_pct_iter": d.get("S_atomic_conservative_pct_iter", 0),
                                    "S_atomic_moderate_pct_iter": d.get("S_atomic_moderate_pct_iter", 0)}

for name, d in results["r6_5"].items():
    # C-GATE: T_C_conservative >= 5% T_iter
    c1 = d["T_C_conservative_pct_iter"] >= 5.0
    gate_verdicts["R6-C"][name] = {"C-GATE": c1,
                                    "T_C_conservative_pct_iter": d["T_C_conservative_pct_iter"]}

results["gate_verdicts"] = gate_verdicts

# Summary statistics
scenes = ["room", "bicycle", "garden"]
stages = ["5000", "15000", "30000"]
results["summary"] = {}
for cand in ["R6-B", "R6-A", "R6-C"]:
    passing = []
    for name in gate_verdicts[cand]:
        gates = gate_verdicts[cand][name]
        if cand == "R6-B":
            if gates["B-GATE-1"] or gates["B-GATE-2"]:
                passing.append(name)
        elif cand == "R6-A":
            if gates["A-GATE-1"] and gates["A-GATE-2"]:
                passing.append(name)
        elif cand == "R6-C":
            if gates["C-GATE"]:
                passing.append(name)
    results["summary"][cand] = {
        "profiles_passing": len(passing),
        "profiles_total": len(gate_verdicts[cand]),
        "passing_names": passing,
    }

out_path = f"{prof_dir}/r6-profile-results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
print(f"\nGate Summary:")
for cand, s in results["summary"].items():
    print(f"  {cand}: {s['profiles_passing']}/{s['profiles_total']} profiles passing")
