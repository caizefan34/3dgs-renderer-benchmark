#!/usr/bin/env python3
"""Offline *analytic only* work/gradient oracle over observed tile data.

Input is a SINGLE genuine frozen-state/camera/candidate-pool+gradient-group CSV.
The optimizer assumes ADDITIVE tile-cost proxy and independent Bernoulli masks.
This program NEVER measures real CUDA cost and NEVER certifies speedup.
"""
import argparse
import csv
import json
import math
from pathlib import Path


LABEL = "DERIVED_OFFLINE_PROXY_COST_NOT_RUNTIME"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def probabilities(scores, costs, budget, p_min=0.05):
    if not (0.0 < p_min < 1.0):
        raise ValueError("0 < p_min < 1 required")
    if not costs or len(scores) != len(costs):
        raise ValueError("nonempty scores and costs must align")
    if any(c <= 0.0 or not math.isfinite(c) for c in costs):
        raise ValueError("costs must be positive finite values")
    if any(s < 0.0 or not math.isfinite(s) for s in scores):
        raise ValueError("scores must be nonnegative finite values")
    minimum = p_min * sum(costs)
    if budget + 1e-12 < minimum:
        raise ValueError(f"budget {budget} below mandatory inclusion floor {minimum}")
    if budget >= sum(costs):
        return [1.0] * len(costs)
    # Numerical floor only, not an empirical tile-value observation.
    s = [max(x, 1e-15) for x in scores]

    def make(k):
        return [max(p_min, min(1.0, k * z)) for z in s]

    low, high = 0.0, 1.0
    while dot(make(high), costs) < budget:
        high *= 2.0
        if not math.isfinite(high):
            raise RuntimeError("could not find finite probability scale")
    for _ in range(180):
        mid = (low + high) / 2.0
        if dot(make(mid), costs) < budget:
            low = mid
        else:
            high = mid
    return make(high)


def analytic_variance(grad_norms, probabilities_):
    return sum(((1.0 / p) - 1.0) * (g*g)
               for g, p in zip(grad_norms, probabilities_))


def read_data(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"tile_id", "grad_norm", "cost_proxy", "error_score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("required columns: " + ", ".join(sorted(required)))
        rows = list(reader)
    if len(rows) < 4:
        raise ValueError("require >=4 genuine tile rows; header-only example is not experimental data")
    ids = [r["tile_id"] for r in rows]
    if any(not x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("tile_id must be unique and nonempty within one case")
    try:
        g = [float(r["grad_norm"]) for r in rows]
        c = [float(r["cost_proxy"]) for r in rows]
        e = [float(r["error_score"]) for r in rows]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("non-numeric or missing measurement column") from exc
    if any(not math.isfinite(x) or x < 0.0 for x in g+e):
        raise ValueError("grad_norm/error_score must be finite and >=0")
    if any(not math.isfinite(x) or x <= 0.0 for x in c):
        raise ValueError("cost_proxy must be finite and >0")
    return ids, g, c, e


def analyze(ids, g, c, e, budget_fraction, p_min, case_label):
    if not (p_min <= budget_fraction <= 1.0):
        raise ValueError("budget-fraction must lie in [p-min,1]")
    budget = budget_fraction * sum(c)
    scores = {
        "U_uniform_Bernoulli": [1.0] * len(c),
        "E_error_proxy_Bernoulli": [max(z, 1e-6) for z in e],
        "O_unavailable_true_gradient_oracle": [gg / math.sqrt(cc) for gg, cc in zip(g, c)],
    }
    results = {}
    for name, values in scores.items():
        probs = probabilities(values, c, budget, p_min)
        results[name] = {
            "analytic_variance": analytic_variance(g, probs),
            "spent_additive_proxy_cost": dot(probs, c),
            "expected_active_tiles": sum(probs),
            "p_by_tile": dict(zip(ids, probs)),
        }
    o = results["O_unavailable_true_gradient_oracle"]["analytic_variance"]
    bas = min(results["U_uniform_Bernoulli"]["analytic_variance"],
              results["E_error_proxy_Bernoulli"]["analytic_variance"])
    reduction = None if bas == 0 else (bas - o) / bas
    return {
        "evidence_class": LABEL,
        "case_label": case_label,
        "tile_count": len(ids),
        "budget_fraction_of_additive_proxy_cost": budget_fraction,
        "p_min": p_min,
        "cost_definition": "INPUT_DATA_USER_DEFINED_PROXY_NOT_REAL_WALL",
        "sampling": "independent_Bernoulli_HT",
        "cannot_claim": ["real_gpu_speedup", "full_frame_performance_from_subset", "deployability_of_O", "quality_or_30K"],
        "results": results,
        "oracle_reduction_vs_best_U_or_E": reduction,
        "note": "The E arm is Bernoulli error-proxy, NOT original with-replacement _error_guided_mask",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--case-label", required=True)
    p.add_argument("--budget-fraction", type=float, default=0.25)
    p.add_argument("--p-min", type=float, default=0.05)
    args = p.parse_args()
    data = read_data(args.input)
    report = analyze(*data, budget_fraction=args.budget_fraction,
                     p_min=args.p_min, case_label=args.case_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("ANALYTIC_ONLY", args.output, "variance_reduction_vs_best_U_or_E=", report["oracle_reduction_vs_best_U_or_E"])
    print("NOT_REAL_CUDA_TIMING; NOT_A_PAPER_GO")


if __name__ == "__main__":
    main()
