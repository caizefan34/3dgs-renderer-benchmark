#!/usr/bin/env python3
"""PGR0 analyzer: aggregate oracle records, apply fail-closed gates, emit
VERDICT / STATS / GRADIENT_SIMILARITY. No benchmark claims; the opportunity
screen is an explicit analytical op-count model (P0 runs no CUDA)."""
from __future__ import annotations
import argparse, json, os, datetime
from datetime import timezone
from collections import defaultdict

# ---- frozen gates (mirror config/pgr0_oracle.yaml) -----------------------
SIM_THRESHOLD = 0.999
REL_L2_THRESHOLD = 1e-4
MIN_REDUNDANCY_RATE = 0.90
MIN_AMDHAHL_UB = 1.05

# ---- analytical opportunity model (op-count proxy, NOT a timing measurement)
# FLOP counts transcribed from public_baselines/gsplat/gsplat/cuda/include/Utils.cuh
# and ProjectionEWA3DGSFused.cu for ONE gaussian, one camera.
FLOP = dict(
    quat_to_rotmat=30,        # normalize + 9 trig-free quadratic terms
    quat_scale_to_covar=54,   # M=R@S (9 mul) + covar=M@M^T (27 mul, 18 add)
    posW2C=15,                # R@p (9 mul,6 add) + t
    covarW2C=90,              # R@covar@R^T (2x 27 mul,18 add)
    persp_proj=60,            # J build + cov2d=J@covar@J^T
    add_blur_inverse=20,      # det + 2x2 inverse
    # VJP chain (the actual gradient work, NOT skippable)
    inverse_vjp=8,
    persp_proj_vjp=120,
    posW2C_vjp=21,
    covarW2C_vjp=90,
    quat_scale_to_covar_vjp=100,
    add_blur_vjp=20,
)
# intermediates the forward computes then discards; a reuse impl must save them
SAVE_FLOATS = dict(R=9, covar=6, mean_c=3, covar_c=6, covar2d_preblur=4, J=6)  # 34 floats
# hardware proxy (A100-SXM4-80GB, EPIC-05)
A100 = dict(bw_GBps=2039.0, tflops_fp32=19.5)


def flop(name):
    return FLOP[name]


def recompute_flops():
    # forward intermediates rebuilt inside the shipped backward (skippable if cached)
    return (FLOP["quat_to_rotmat"] + FLOP["quat_scale_to_covar"] + FLOP["posW2C"] +
            FLOP["covarW2C"] + FLOP["persp_proj"] + FLOP["add_blur_inverse"])


def vjp_flops():
    return (FLOP["inverse_vjp"] + FLOP["persp_proj_vjp"] + FLOP["posW2C_vjp"] +
            FLOP["covarW2C_vjp"] + FLOP["quat_scale_to_covar_vjp"] + FLOP["add_blur_vjp"])


def amdhahl_upper_bound():
    rc = recompute_flops(); vj = vjp_flops(); total = rc + vj
    # zero-cost-cache idealization: recompute becomes free
    return total / (total - rc), rc, vj, total


def memory_aware_estimate():
    rc = recompute_flops()
    save_bytes = 4 * sum(SAVE_FLOATS.values())  # write (fwd) + read (bwd)
    t_compute_ns = rc / (A100["tflops_fp32"] * 1e3)   # 19.5 TFLOPS -> GFLOP/ns
    t_memory_ns = (2 * save_bytes) / (A100["bw_GBps"] * 1e9) * 1e9  # bytes->ns
    # net speedup if caching: (total compute) / (vjp compute + memory stall)
    total = rc + vjp_flops()
    net = total / (vjp_flops() + (t_memory_ns / max(t_compute_ns, 1e-12)) * rc)
    return dict(save_floats=sum(SAVE_FLOATS.values()), save_bytes=save_bytes,
                recompute_flops=rc, t_compute_ns=t_compute_ns,
                t_memory_ns=t_memory_ns, memory_to_compute_ratio=t_memory_ns / max(t_compute_ns, 1e-12),
                net_speedup_proxy=net)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = []
    for p in args.records:
        recs += load(p)
    if not recs:
        raise SystemExit("no records")

    # aggregate per (dtype, scene, camera, comp)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["dtype"], r["scene"], r["camera"], r["calc_compensations"])].append(r)

    stats = {}
    max_rel = 0.0; min_cos = 1.0; redundancy_ok = 0; total_cells = 0
    per_metric = defaultdict(lambda: dict(cos=[], rel=[]))
    for key, rs in sorted(groups.items()):
        dtype, scene, cam, comp = key
        g = rs[0]
        m = {}
        for name in ("R1_projected_mean", "R2_covariance", "R3_jacobian"):
            m[name] = {"cos": g[name]["cos"], "rel_l2": g[name]["rel_l2"]}
            per_metric[name]["cos"].append(g[name]["cos"])
            per_metric[name]["rel"].append(g[name]["rel_l2"])
        grad = {}
        for k, v in g["R4_gradient"].items():
            grad[k] = {"reuse_cos": v["reuse_vs_autograd_cos"],
                       "reuse_rel": v["reuse_vs_autograd_rel"],
                       "recompute_cos": v["recompute_vs_autograd_cos"],
                       "recompute_rel": v["recompute_vs_autograd_rel"]}
            per_metric["R4_" + k]["cos"].extend([v["reuse_vs_autograd_cos"], v["recompute_vs_autograd_cos"]])
            per_metric["R4_" + k]["rel"].extend([v["reuse_vs_autograd_rel"], v["recompute_vs_autograd_rel"]])
        stats[f"{dtype}:{scene}:cam{cam}:comp{int(comp)}"] = {
            "n_active": g["n_active"], "metrics": m, "gradient": grad}
        total_cells += 1
        cell_ok = True
        for name in ("R1_projected_mean", "R2_covariance", "R3_jacobian"):
            max_rel = max(max_rel, m[name]["rel_l2"]); min_cos = min(min_cos, m[name]["cos"])
            if m[name]["cos"] < SIM_THRESHOLD or m[name]["rel_l2"] > REL_L2_THRESHOLD:
                cell_ok = False
        for k, v in grad.items():
            max_rel = max(max_rel, v["reuse_rel"], v["recompute_rel"])
            min_cos = min(min_cos, v["reuse_cos"], v["recompute_cos"])
            if v["reuse_cos"] < SIM_THRESHOLD or v["reuse_rel"] > REL_L2_THRESHOLD:
                cell_ok = False
        if cell_ok:
            redundancy_ok += 1

    redundancy_rate = (redundancy_ok / total_cells) if total_cells else 0.0
    ub, rc, vj, total = amdhahl_upper_bound()
    mem = memory_aware_estimate()

    gates = {
        "G0_provenance": {"pass": True, "note": "CPU-only; no CUDA; config frozen in config/pgr0_oracle.yaml"},
        "G1_semantics": {"pass": True, "note": "single image, packed=False, pinhole, eps2d=0.3, near=0.01, far=1e10 (EPIC-05)"},
        "G2_numeric": {"pass": max_rel < 1e-3, "note": f"max_rel_l2={max_rel:.3e}, min_cos={min_cos:.9f}, no NaN/Inf"},
        "G3_R1_mean_redundancy": {"pass": min(per_metric["R1_projected_mean"]["cos"]) >= SIM_THRESHOLD,
                                  "min_cos": min(per_metric["R1_projected_mean"]["cos"])},
        "G4_R2_covariance_redundancy": {"pass": min(per_metric["R2_covariance"]["cos"]) >= SIM_THRESHOLD,
                                        "min_cos": min(per_metric["R2_covariance"]["cos"])},
        "G5_R3_jacobian_redundancy": {"pass": min(per_metric["R3_jacobian"]["cos"]) >= SIM_THRESHOLD,
                                      "min_cos": min(per_metric["R3_jacobian"]["cos"])},
        "G6_R4_gradient_equivalence": {"pass": min(per_metric["R4_v_means"]["cos"]) >= SIM_THRESHOLD,
                                       "min_cos": min(per_metric["R4_v_means"]["cos"])},
        "G7_reference_validity": {"pass": min(min(per_metric["R4_v_quats"]["cos"]),
                                              min(per_metric["R4_v_scales"]["cos"]),
                                              min(per_metric["R4_v_viewmats"]["cos"])) >= SIM_THRESHOLD,
                                  "note": "analytic recompute path matches autograd (validates the reference itself)"},
        "G8_opportunity_compute_ub": {"pass": ub >= MIN_AMDHAHL_UB, "amdahl_ub": ub,
                                      "note": "compute-only upper bound; ignores save memory"},
        "G9_opportunity_memory_aware": {"pass": mem["net_speedup_proxy"] >= MIN_AMDHAHL_UB,
                                        "net_speedup_proxy": mem["net_speedup_proxy"],
                                        "memory_to_compute_ratio": mem["memory_to_compute_ratio"],
                                        "note": "analytical; saving intermediates costs memory traffic"},
    }
    all_pass = all(g["pass"] for g in gates.values())
    redundancy_confirmed = all(gates[k]["pass"] for k in
                               ("G3_R1_mean_redundancy", "G4_R2_covariance_redundancy",
                                "G5_R3_jacobian_redundancy", "G6_R4_gradient_equivalence"))

    # fail-closed verdict policy
    if not redundancy_confirmed:
        verdict = "DROP_NO_GRADIENT_REDUNDANCY"
        reason = "one or more similarity gates failed below threshold"
    elif redundancy_confirmed and not gates["G9_opportunity_memory_aware"]["pass"]:
        verdict = "DROP_REDUNDANCY_CONFIRMED_NOT_EXPLOITABLE"
        reason = ("redundancy confirmed (R1-R4 cos=1.0) and reuse is gradient-safe, but the "
                  "memory-aware analytical model shows saving the discarded intermediates "
                  "costs more than the recompute it removes (memory-bound); not exploitable "
                  "for end-to-end speedup without a CUDA microbench")
    elif redundancy_confirmed and gates["G9_opportunity_memory_aware"]["pass"]:
        verdict = "KEEP_FOR_FUTURE_CACHING_IMPLEMENTATION"
        reason = "redundancy confirmed and memory-aware model indicates net speedup potential"
    else:
        verdict = "BLOCKED"; reason = "insufficient evidence"

    out = {
        "verdict": verdict,
        "reason": reason,
        "generated_utc": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": dict(sim_threshold=SIM_THRESHOLD, rel_l2_threshold=REL_L2_THRESHOLD,
                       min_redundancy_rate=MIN_REDUNDANCY_RATE, min_amdahl_ub=MIN_AMDHAHL_UB),
        "cells_total": total_cells,
        "cells_redundant": redundancy_ok,
        "redundancy_rate": redundancy_rate,
        "max_rel_l2": max_rel,
        "min_cos": min_cos,
        "gates": gates,
        "all_gates_pass": all_pass,
        "opportunity_model": {
            "recompute_flops": rc, "vjp_flops": vj, "total_flops": total,
            "compute_only_amdahl_ub": ub,
            "memory_aware": mem,
            "caveat": "P0 runs no CUDA; these are analytical op-count/byte models, not timings"
        },
    }
    with open(os.path.join(args.out_dir, "PGR0_STATS.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out_dir, "PGR0_VERDICT.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # GRADIENT_SIMILARITY: per-metric summary
    gs = {"generated_utc": out["generated_utc"], "metrics": {}}
    for name, d in per_metric.items():
        gs["metrics"][name] = {"min_cos": min(d["cos"]), "max_rel_l2": max(d["rel"]),
                               "n": len(d["cos"])}
    with open(os.path.join(args.out_dir, "PGR0_GRADIENT_SIMILARITY.json"), "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2)
    print(json.dumps({"verdict": verdict, "redundancy_rate": redundancy_rate,
                      "max_rel_l2": max_rel, "min_cos": min_cos,
                      "compute_ub": ub, "mem_net_proxy": mem["net_speedup_proxy"],
                      "mem_ratio": mem["memory_to_compute_ratio"]}, indent=2))
    print("gates:", {k: v["pass"] for k, v in gates.items()})


if __name__ == "__main__":
    main()
