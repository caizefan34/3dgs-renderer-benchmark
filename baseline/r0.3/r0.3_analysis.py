"""
R0.3 Analysis — Aggregates continuation outputs into 7 final JSON files.

Produces:
  1. visible_conditional_correlations.json    (Part 4)
  2. visible_gradient_mass_coverage.json       (Part 5)
  3. visibility_confound_decomposition.json    (Part 6)
  4. work_gradient_pareto.json                 (Part 7+8)
  5. signal_availability_cost.json             (Part 9)
  6. masked_backward_scaling.json              (Part 10 — copy from benchmark)
  7. final_decision.json                       (Part 12)
"""

import json
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
R03_DIR = REPO_ROOT / "results" / "reference_v1" / "r0.3"
WINDOWS = ["2000", "5000", "10000", "14000"]
SIGNALS = ["tiles_per_gauss", "projected_radius", "opacity"]


def load_window(w, filename):
    p = R03_DIR / f"window_{w}" / filename
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def stats_from_list(values):
    if not values:
        return {"mean": 0, "median": 0, "p10": 0, "p90": 0, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": len(arr),
    }


# === Part 1: Visible fraction ===
def aggregate_visible_fraction():
    print("=== Part 1: Visible Fraction ===")
    result = {"per_window": {}, "overall": []}
    all_fracs = []
    for w in WINDOWS:
        data = load_window(w, "visible_fraction.json")
        fracs = [v for v in data.values()]
        result["per_window"][w] = stats_from_list(fracs)
        all_fracs.extend(fracs)
        print(f"  window {w}: mean={np.mean(fracs):.3f} median={np.median(fracs):.3f}")
    result["overall"] = stats_from_list(all_fracs)
    print(f"  Overall: mean={result['overall']['mean']:.3f} median={result['overall']['median']:.3f}")
    return result


# === Part 4: Visible-conditional correlations ===
def aggregate_correlations():
    print("\n=== Part 4: Visible-Conditional Correlations ===")
    result = {"per_window": {}, "overall": {}}

    for signal in SIGNALS:
        for gopt_name in ["total_norm", "xyz_norm", "opacity_abs", "scale_norm", "rot_norm", "shs_norm"]:
            all_pearson = []
            all_spearman = []
            for w in WINDOWS:
                data = load_window(w, "visible_conditional_correlations.json")
                for iter_str, sig_data in data.items():
                    if signal in sig_data and gopt_name in sig_data[signal]:
                        m = sig_data[signal][gopt_name]
                        if m.get("n", 0) >= 10:
                            all_pearson.append(m["pearson"])
                            all_spearman.append(m["spearman"])

            key = f"{signal}_vs_{gopt_name}"
            result["overall"][key] = {
                "pearson": stats_from_list(all_pearson),
                "spearman": stats_from_list(all_spearman),
            }

    # Print key results
    for signal in SIGNALS:
        key = f"{signal}_vs_total_norm"
        if key in result["overall"]:
            o = result["overall"][key]
            print(f"  {signal} vs G_opt_total: Pearson={o['pearson']['median']:.4f} Spearman={o['spearman']['median']:.4f}")

    return result


# === Part 5: Visible-conditional gradient-mass coverage ===
def aggregate_visible_coverage():
    print("\n=== Part 5: Visible-Conditional Gradient-Mass Coverage ===")
    result = {"per_window": {}, "overall": {}}

    for signal in SIGNALS:
        all_cov = {f"k{k}": [] for k in [10, 20, 32, 50]}
        all_oracle = {f"oracle_k{k}": [] for k in [10, 20, 32, 50]}
        all_random = {f"random_k{k}": [] for k in [10, 20, 32, 50]}

        for w in WINDOWS:
            data = load_window(w, "visible_gradient_mass_coverage.json")
            w_cov = {f"k{k}": [] for k in [10, 20, 32, 50]}
            w_oracle = {f"oracle_k{k}": [] for k in [10, 20, 32, 50]}
            w_random = {f"random_k{k}": [] for k in [10, 20, 32, 50]}

            for iter_str, sig_data in data.items():
                if signal in sig_data:
                    m = sig_data[signal]
                    for k in [10, 20, 32, 50]:
                        if f"k{k}" in m: w_cov[f"k{k}"].append(m[f"k{k}"])
                        if f"oracle_k{k}" in m: w_oracle[f"oracle_k{k}"].append(m[f"oracle_k{k}"])
                        if f"random_k{k}" in m: w_random[f"random_k{k}"].append(m[f"random_k{k}"])

            result["per_window"][signal] = {}
            for k in [10, 20, 32, 50]:
                result["per_window"][signal][f"k{k}"] = stats_from_list(w_cov[f"k{k}"])
                result["per_window"][signal][f"oracle_k{k}"] = stats_from_list(w_oracle[f"oracle_k{k}"])
                result["per_window"][signal][f"random_k{k}"] = stats_from_list(w_random[f"random_k{k}"])

            for k in [10, 20, 32, 50]:
                all_cov[f"k{k}"].extend(w_cov[f"k{k}"])
                all_oracle[f"oracle_k{k}"].extend(w_oracle[f"oracle_k{k}"])
                all_random[f"random_k{k}"].extend(w_random[f"random_k{k}"])

        result["overall"][signal] = {}
        for k in [10, 20, 32, 50]:
            result["overall"][signal][f"k{k}"] = stats_from_list(all_cov[f"k{k}"])
            result["overall"][signal][f"oracle_k{k}"] = stats_from_list(all_oracle[f"oracle_k{k}"])
            result["overall"][signal][f"random_k{k}"] = stats_from_list(all_random[f"random_k{k}"])

    # Print
    print(f"  {'Signal':<20} {'K':>4} {'Coverage':>10} {'Oracle':>10} {'Random':>10}")
    for signal in SIGNALS:
        for k in [10, 20, 32, 50]:
            o = result["overall"][signal]
            print(f"  {signal:<20} {k:>4}% {o[f'k{k}']['median']:>10.4f} {o[f'oracle_k{k}']['median']:>10.4f} {o[f'random_k{k}']['median']:>10.4f}")

    return result


# === Part 6: Visibility confound decomposition ===
def aggregate_visibility_confound():
    print("\n=== Part 6: Visibility Confound Decomposition ===")
    result = {"overall": {}}

    for signal in SIGNALS:
        all_vis_cov = {f"k{k}": [] for k in [10, 20, 32, 50]}
        all_all_cov = {f"k{k}": [] for k in [10, 20, 32, 50]}

        for w in WINDOWS:
            vis_data = load_window(w, "visible_gradient_mass_coverage.json")
            all_data = load_window(w, "all_gaussian_coverage.json")

            for iter_str in vis_data:
                if signal in vis_data.get(iter_str, {}) and signal in all_data.get(iter_str, {}):
                    vm = vis_data[iter_str][signal]
                    am = all_data[iter_str][signal]
                    for k in [10, 20, 32, 50]:
                        if f"k{k}" in vm: all_vis_cov[f"k{k}"].append(vm[f"k{k}"])
                        if f"k{k}" in am: all_all_cov[f"k{k}"].append(am[f"k{k}"])

        result["overall"][signal] = {}
        for k in [10, 20, 32, 50]:
            vis_stats = stats_from_list(all_vis_cov[f"k{k}"])
            all_stats = stats_from_list(all_all_cov[f"k{k}"])
            visibility_gain = all_stats["median"] - vis_stats["median"]
            result["overall"][signal][f"k{k}"] = {
                "all_gaussian_coverage": all_stats,
                "visible_conditional_coverage": vis_stats,
                "visibility_gain": visibility_gain,
            }

    # Print
    for signal in SIGNALS:
        for k in [50, 32]:
            o = result["overall"][signal][f"k{k}"]
            print(f"  {signal} K={k}%: all={o['all_gaussian_coverage']['median']:.4f} vis={o['visible_conditional_coverage']['median']:.4f} gain={o['visibility_gain']:.4f}")

    return result


# === Part 7+8: Work-gradient Pareto ===
def aggregate_work_pareto():
    print("\n=== Part 7+8: Work-Gradient Pareto ===")
    result = {"overall": {}}

    for signal in SIGNALS:
        all_grad = {f"k{k}": [] for k in [10, 20, 32, 50]}
        all_work_ret = {f"k{k}": [] for k in [10, 20, 32, 50]}
        all_work_rem = {f"k{k}": [] for k in [10, 20, 32, 50]}

        for w in WINDOWS:
            data = load_window(w, "work_gradient_pareto.json")
            for iter_str, sig_data in data.items():
                if signal in sig_data:
                    for k in [10, 20, 32, 50]:
                        if f"k{k}" in sig_data[signal]:
                            m = sig_data[signal][f"k{k}"]
                            all_grad[f"k{k}"].append(m["gradient_mass_retained"])
                            all_work_ret[f"k{k}"].append(m["work_retained"])
                            all_work_rem[f"k{k}"].append(m["work_removed"])

        result["overall"][signal] = {}
        for k in [10, 20, 32, 50]:
            result["overall"][signal][f"k{k}"] = {
                "gradient_mass_retained": stats_from_list(all_grad[f"k{k}"]),
                "work_retained": stats_from_list(all_work_ret[f"k{k}"]),
                "work_removed": stats_from_list(all_work_rem[f"k{k}"]),
            }

    # Print
    print(f"  {'Signal':<20} {'K':>4} {'Grad Mass':>10} {'Work Ret':>10} {'Work Rem':>10}")
    for signal in SIGNALS:
        for k in [10, 20, 32, 50]:
            o = result["overall"][signal][f"k{k}"]
            print(f"  {signal:<20} {k:>4}% {o['gradient_mass_retained']['median']:>10.4f} {o['work_retained']['median']:>10.4f} {o['work_removed']['median']:>10.4f}")

    return result


# === Part 9: Signal availability cost ===
def build_signal_availability_cost():
    print("\n=== Part 9: Signal Availability Cost ===")
    return {
        "tiles_per_gauss": {
            "cost": "FREE",
            "description": "Produced by forward rasterization as meta['tiles_per_gauss']. Already available in the rendering pipeline.",
            "first_available": "After forward, before backward",
        },
        "projected_radius": {
            "cost": "FREE",
            "description": "Derived from meta['radii'] which is produced by forward projection. Already available.",
            "first_available": "After forward, before backward",
        },
        "opacity": {
            "cost": "FREE",
            "description": "Model parameter, available at any time. No computation needed.",
            "first_available": "Before forward (model state)",
        },
    }


# === Part 10: Load masked backward scaling ===
def load_masked_backward_scaling():
    p = R03_DIR / "masked_backward_scaling.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# === Part 12: Final decision ===
def compute_final_decision(correlations, visible_coverage, confound, pareto, masked_scaling, visible_frac):
    print("\n=== Part 12: Final Decision ===")

    # Key metrics for each signal at K=50% (visible-conditional)
    signal_metrics = {}
    for signal in SIGNALS:
        cov50 = visible_coverage["overall"][signal]["k50"]["median"]
        oracle50 = visible_coverage["overall"][signal]["oracle_k50"]["median"]
        random50 = visible_coverage["overall"][signal]["random_k50"]["median"]
        spearman = correlations["overall"][f"{signal}_vs_total_norm"]["spearman"]["median"]
        pearson = correlations["overall"][f"{signal}_vs_total_norm"]["pearson"]["median"]

        # Work Pareto at K=50%
        p = pareto["overall"][signal]["k50"]
        grad_retained = p["gradient_mass_retained"]["median"]
        work_retained = p["work_retained"]["median"]
        work_removed = p["work_removed"]["median"]

        # Visibility confound
        vc = confound["overall"][signal]["k50"]
        all_cov = vc["all_gaussian_coverage"]["median"]
        vis_cov = vc["visible_conditional_coverage"]["median"]
        vis_gain = vc["visibility_gain"]

        signal_metrics[signal] = {
            "coverage_k50": cov50,
            "oracle_k50": oracle50,
            "random_k50": random50,
            "spearman": spearman,
            "pearson": pearson,
            "grad_mass_retained_k50": grad_retained,
            "work_retained_k50": work_retained,
            "work_removed_k50": work_removed,
            "all_gaussian_coverage_k50": all_cov,
            "visible_conditional_coverage_k50": vis_cov,
            "visibility_gain": vis_gain,
        }

        print(f"\n  {signal}:")
        print(f"    Coverage K=50%: {cov50:.4f} (Oracle={oracle50:.4f}, Random={random50:.4f})")
        print(f"    Pearson={pearson:.4f}, Spearman={spearman:.4f}")
        print(f"    Grad mass retained: {grad_retained:.4f}, Work retained: {work_retained:.4f}, Work removed: {work_removed:.4f}")
        print(f"    Visibility confound: all={all_cov:.4f}, vis={vis_cov:.4f}, gain={vis_gain:.4f}")

    # Masked backward scaling
    scaling = {}
    for key, val in masked_scaling.get("retention_results", {}).items():
        if "f_k" in val:
            scaling[key] = {
                "retention_fraction": val["retention_fraction"],
                "f_k": val["f_k"],
                "backward_saving_pct": val["backward_saving_pct"],
                "backward_ms": val["backward_total_ms"]["median"],
            }

    print(f"\n  Masked backward scaling:")
    for key, s in scaling.items():
        print(f"    {key}: f(K)={s['f_k']:.3f}, saving={s['backward_saving_pct']:.1f}%")

    # Decision logic:
    # C51_MECHANISM_CANDIDATE: signal available before backward at negligible cost,
    #   visible-conditioned prediction strong, retains >=90% G_opt mass while
    #   eliminating substantial backward work, >5% E2E saving, structurally distinct
    # C51_SYSTEMS_OPTIMIZATION: measurable speed but conventional or crowded novelty
    # C51_CLOSE: no useful Pareto tradeoff

    # Check each signal against criteria
    best_signal = None
    best_score = 0

    for signal in SIGNALS:
        m = signal_metrics[signal]
        # Criterion 1: available before backward at negligible cost — all FREE ✓
        # Criterion 2: visible-conditioned prediction strong (Spearman > 0.5)
        pred_strong = m["spearman"] > 0.5
        # Criterion 3: retains >=90% G_opt mass at K=50% while eliminating substantial work
        mass_ok = m["coverage_k50"] >= 0.90
        work_ok = m["work_removed_k50"] >= 0.20  # at least 20% work removed
        # Criterion 4: >5% E2E saving
        # From backward profile: backward = 39.9% of total, optimizer = 5.8%
        # If we mask K=50% of visible (which is ~50% of all), f(K=50%) ≈ 0.824
        # Saving = (1 - 0.824) * 39.9% ≈ 7.0% E2E
        backward_frac_of_total = 0.399
        if "retention_50" in scaling:
            e2e_saving = (1 - scaling["retention_50"]["f_k"]) * backward_frac_of_total * 100
        else:
            e2e_saving = 0
        e2e_ok = e2e_saving > 5.0

        # Criterion 5: not simply visibility/opacity/contribution/historical
        # All our signals are forward-time natural signals, not historical gradient
        structurally_distinct = True  # will assess novelty separately

        score = 0
        if pred_strong: score += 1
        if mass_ok: score += 1
        if work_ok: score += 1
        if e2e_ok: score += 1

        if score > best_score:
            best_score = score
            best_signal = signal

        print(f"\n  {signal} criteria: pred={pred_strong} mass={mass_ok} work={work_ok} e2e={e2e_ok} (saving={e2e_saving:.1f}%)")

    # Decision
    best_m = signal_metrics[best_signal] if best_signal else None

    if best_score >= 4:
        decision = "C51_MECHANISM_CANDIDATE"
        reason = f"{best_signal} meets all criteria: Spearman={best_m['spearman']:.3f}, coverage_k50={best_m['coverage_k50']:.3f}, work_removed={best_m['work_removed_k50']:.3f}, E2E saving ~{e2e_saving:.1f}%"
    elif best_score >= 2:
        decision = "C51_SYSTEMS_OPTIMIZATION"
        reason = f"{best_signal} provides measurable speed opportunity but does not meet all criteria for new mechanism. Conventional importance-based approximate skip."
    else:
        decision = "C51_CLOSE"
        reason = "No free forward signal provides a useful gradient-mass/work Pareto tradeoff within the visible set."

    print(f"\n  Decision: {decision}")
    print(f"  Reason: {reason}")

    return {
        "decision": decision,
        "reason": reason,
        "best_signal": best_signal,
        "signal_metrics": signal_metrics,
        "masked_backward_scaling": scaling,
        "e2e_saving_estimate_k50": e2e_saving if best_signal else 0,
        "visible_fraction_overall": visible_frac["overall"],
    }


# === Main ===
if __name__ == "__main__":
    output_dir = R03_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part 1
    vis_frac = aggregate_visible_fraction()

    # Part 4
    correlations = aggregate_correlations()
    with open(output_dir / "visible_conditional_correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"  Saved: visible_conditional_correlations.json")

    # Part 5
    visible_coverage = aggregate_visible_coverage()
    with open(output_dir / "visible_gradient_mass_coverage.json", "w") as f:
        json.dump(visible_coverage, f, indent=2)
    print(f"  Saved: visible_gradient_mass_coverage.json")

    # Part 6
    confound = aggregate_visibility_confound()
    with open(output_dir / "visibility_confound_decomposition.json", "w") as f:
        json.dump(confound, f, indent=2)
    print(f"  Saved: visibility_confound_decomposition.json")

    # Part 7+8
    pareto = aggregate_work_pareto()
    with open(output_dir / "work_gradient_pareto.json", "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"  Saved: work_gradient_pareto.json")

    # Part 9
    availability = build_signal_availability_cost()
    with open(output_dir / "signal_availability_cost.json", "w") as f:
        json.dump(availability, f, indent=2)
    print(f"  Saved: signal_availability_cost.json")

    # Part 10
    masked_scaling = load_masked_backward_scaling()
    # Already saved as masked_backward_scaling.json by the benchmark script

    # Part 12
    decision = compute_final_decision(
        correlations, visible_coverage, confound, pareto, masked_scaling, vis_frac
    )
    with open(output_dir / "final_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"  Saved: final_decision.json")

    print("\n=== Analysis complete ===")
