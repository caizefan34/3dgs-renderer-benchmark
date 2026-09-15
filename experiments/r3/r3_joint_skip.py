#!/usr/bin/env python3
"""
R3 Joint Tile-Gaussian Skip-Set Analysis
(Red-team req 2, 3, 4)

A single "interaction bit" per (tile, Gaussian) pair suppresses ALL four
derivative families (color, opacity, mean2d, conic) simultaneously.

The bit can be set when EVERY family's certified bound contribution for
that pair is below the budget threshold:

    B^fam_it <= epsilon * total_B^fam

where total_B^fam = sum over all pairs of B^fam_it.

Philosophy:
  * The forward pass computes alpha/T/buffer traversal once per (tile,pixel)
    irrespective of which derivatives are computed.
  * If a pair is removed from the derivative pass entirely, ALL families'
    bounds must still hold within the certified error budget.
  * EXACT_ZERO_SUPPORT_CULLING (alpha_max < 1/255) is PRIOR ART
    (Speedy-Splat / AccuTile already do this); we must never count it as
    Candidate C novelty.
  * LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING = pairs whose derivative work
    is skipped under the certified bound budget despite being inside the
    ordinary nonzero support. THIS is Candidate C novelty.

Metrics reported per budget epsilon:

  - JOINT_SKIP_PAIR_FRACTION           : (# skipped pairs) / (total pairs)
  - JOINT_SKIP_WEIGHTED_WORK_FRACTION  : sum(W_it for skipped) / sum(all W_it)

  where W_it = number of pixel lanes in tile t for Gaussian i on which
  derivative work executes.  Since actual per-pair pixel-lane counts are not
  visible through canonical gsplat autograd, we use the offline tile-map
  instrumentation (available from the forward pass's isect data):
  W_it = number of tile pixels where the Gaussian is active (alpha>=1/255).

This module only *evaluates* the opportunity: it never modifies the runtime.
"""

import json
import os
import numpy as np


def compute_joint_skip_analysis(
    pair_records,
    budgets=(0.001, 0.005, 0.01, 0.02, 0.05),
):
    """Compute joint skip-set statistics across all 4 derivative families.

    Args:
        pair_records: list of dicts with keys
            - 'tile_idx': integer tile index
            - 'gauss_idx': integer Gaussian index
            - 'w_color': W_color_it, pixel lanes where color derivative executes
            - 'w_unclamped': W_unclamped_it, pixel lanes where
              opacity/mean2d/conic derivative also executes
            - 'B_color': per-pair color bound (tight variant)
            - 'B_opacity': per-pair opacity bound
            - 'B_mean2d': per-pair mean2d bound (sigmamin_tight variant)
            - 'B_conic': per-pair conic bound (sigmamin_tight variant)
            - 'is_exact_zero': bool, whether pair has exact-zero `< 1/255`
              support (prior-art Speedy-Splat/AccuTile class)
    Returns:
        dict of per-budget statistics.
    """
    if len(pair_records) == 0:
        return {"error": "No pair records"}

    fams = ["B_color", "B_opacity", "B_mean2d", "B_conic"]

    # Total per-family bound over ALL pairs (denominator for epsilon)
    totals = {f: sum(float(r[f]) for r in pair_records) for f in fams}

    total_lanes = sum(int(r.get("w_color", r.get("n_lanes", 0))) for r in pair_records)
    total_pairs = len(pair_records)

    # Rank pairs by max normalized contribution across families
    def max_norm(r):
        return max(float(r[f]) / totals[f] if totals[f] > 0 else 0.0 for f in fams)

    ranked = sorted(pair_records, key=max_norm)

    results = {}
    for eps in budgets:
        cumulative = {f: 0.0 for f in fams}
        skipped_pairs = 0
        skipped_lanes = 0
        exact_zero_in_skip = 0
        loss_cond_in_skip = 0

        for r in ranked:
            # Check if adding this pair keeps ALL families within budget
            would_be = {
                f: cumulative[f] + float(r[f]) for f in fams
            }
            if all(would_be[f] <= eps * totals[f] for f in fams):
                for f in fams:
                    cumulative[f] = would_be[f]
                skipped_pairs += 1
                skipped_lanes += int(r.get("w_color", r.get("n_lanes", 0)))
                if r["is_exact_zero"]:
                    exact_zero_in_skip += 1
                else:
                    loss_cond_in_skip += 1
            else:
                # Greedy: stop at first violation (sorted ascending)
                break

        results[f"eps_{eps*100:.1f}pct"] = {
            "JOINT_SKIP_PAIR_FRACTION": skipped_pairs / total_pairs,
            "JOINT_SKIP_PAIRS": skipped_pairs,
            "JOINT_SKIP_WEIGHTED_WORK_FRACTION": skipped_lanes / total_lanes if total_lanes > 0 else 0.0,
            "JOINT_SKIP_LANES": skipped_lanes,
            "EXACT_ZERO_SUPPORT_CULLING": exact_zero_in_skip / total_pairs,
            "EXACT_ZERO_CULLED_PAIRS": exact_zero_in_skip,
            "LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING": loss_cond_in_skip / total_pairs,
            "LOSS_CONDITIONED_CULLED_PAIRS": loss_cond_in_skip,
            "per_family_budget_used": {
                f: cumulative[f] / totals[f] if totals[f] > 0 else 0.0 for f in fams
            },
            "TOTAL_PAIRS": total_pairs,
            "TOTAL_LANES": total_lanes,
        }

    return {
        "budgets": results,
        "totals": totals,
        "total_lanes": total_lanes,
        "total_pairs": total_pairs,
    }


def load_pair_records(npz_path):
    """Load pair records from the runner's saved .npz artifact."""
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    # Support both old (n_lanes) and new (w_color, w_unclamped) formats
    has_w_color = "w_color" in data and "w_unclamped" in data
    if not has_w_color and "n_lanes" not in data:
        return None
    n = len(data["tile_idx"])
    records = []
    for i in range(n):
        r = {
            "tile_idx": int(data["tile_idx"][i]),
            "gauss_idx": int(data["gauss_idx"][i]),
            "B_color": float(data["B_color"][i]),
            "B_opacity": float(data["B_opacity"][i]),
            "B_mean2d": float(data["B_mean2d"][i]),
            "B_conic": float(data["B_conic"][i]),
            "is_exact_zero": bool(data["is_exact_zero"][i]),
        }
        if has_w_color:
            r["w_color"] = int(data["w_color"][i])
            r["w_unclamped"] = int(data["w_unclamped"][i])
        else:
            r["w_color"] = int(data["n_lanes"][i])
            r["w_unclamped"] = int(data["n_lanes"][i])
        records.append(r)
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Path to pair_records.npz from runner")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    records = load_pair_records(args.input)
    if records is None:
        print(f"ERROR: no valid pair records in {args.input}")
        return 1

    result = compute_joint_skip_analysis(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {args.output}")

    # Print summary
    if "budgets" in result:
        for key, entry in result["budgets"].items():
            print(
                f"  {key}: pairs={entry['JOINT_SKIP_PAIR_FRACTION']*100:.1f}% "
                f"work={entry['JOINT_SKIP_WEIGHTED_WORK_FRACTION']*100:.1f}% "
                f"(loss-cond {entry['LOSS_CONDITIONED_NONZERO_SKIP'] if False else entry.get('LOSS_CONDITIONED_NONZERO_SKIP', 0):.1f}%)"
            )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
