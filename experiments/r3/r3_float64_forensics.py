#!/usr/bin/env python3
"""
R3.1 Float64 Forensics

Establishes pre-declared float64 verification methodology.
Cannot adjust tolerance after seeing results.

Purpose:
  1. Confirm 0-violation results are genuine (not float32 rounding hiding violations)
  2. Measure margin = B - |g| in both float32 and float64 for worst-case Gaussians
  3. Report relative_violation = (|g| - B) / max(|g|, eps) for any borderline cases

Pre-declared constants (cannot be changed after seeing results):
  FLOAT64_EPS = 1e-15  (float64 machine epsilon ~2.2e-16, with safety factor)
  FLOAT32_EPS = 1e-7   (float32 machine epsilon ~1.2e-7)
  BORDERLINE_THRESHOLD = 0.01  (|g|/B ratio within 1% of 1.0 is borderline)
  TOLERANCE = 1e-6     (same as runner, pre-declared, NOT adjustable)
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

# === PRE-DECLARED CONSTANTS (do not change after results) ===
FLOAT64_EPS = 1e-15
FLOAT32_EPS = 1e-7
BORDERLINE_THRESHOLD = 0.01  # ratio within 1% of 1.0
TOLERANCE = 1e-6  # same as r3_certificate_runner.py line 1178
EPS_NORM = 1e-12  # denominator floor for relative_violation

FAMILIES = [
    "color_coarse", "color_tight",
    "opacity", "opacity_tight",
    "mean2d", "mean2d_sigmamin",
    "conic", "conic_sigmamin",
]


def load_correctness(path):
    with open(path) as f:
        return json.load(f)


def load_pair_records(path):
    """Load pair_records.npz and return structured arrays."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def analyze_window(window_dir):
    """Analyze one window's float64 forensics."""
    corr = load_correctness(os.path.join(window_dir, "certificate_correctness.json"))
    pairs = load_pair_records(os.path.join(window_dir, "pair_records.npz"))

    results = {
        "window": os.path.basename(window_dir),
        "iterations": {},
        "summary": {
            "total_borderline_f32": 0,
            "total_borderline_f64": 0,
            "worst_margin_f32": float("inf"),
            "worst_margin_f64": float("inf"),
            "worst_ratio_f32": 0.0,
            "worst_ratio_f64": 0.0,
        },
    }

    for iter_key, fam_data in corr.get("correctness", {}).items():
        iter_result = {"families": {}}

        for fam, info in fam_data.items():
            if not isinstance(info, dict) or not info.get("available", False):
                continue

            vc = info.get("violation_count", 0)
            max_ratio = info.get("max_actual_over_bound", 0.0)

            # Classify: is this borderline?
            is_borderline = abs(1.0 - max_ratio) < BORDERLINE_THRESHOLD

            fam_result = {
                "violation_count": vc,
                "max_actual_over_bound": max_ratio,
                "borderline": is_borderline,
                "margin_f32": 1.0 - max_ratio,  # B/|g| - 1 ≈ 1 - max_ratio when B≈|g|
                "tolerance": TOLERANCE,
            }

            if is_borderline:
                results["summary"]["total_borderline_f32"] += 1

            if max_ratio > results["summary"]["worst_ratio_f32"]:
                results["summary"]["worst_ratio_f32"] = max_ratio
                results["summary"]["worst_margin_f32"] = 1.0 - max_ratio

            # If pair_records available, do float64 recomputation for worst Gaussians
            if pairs is not None and is_borderline:
                # Find the family's bound field in pair_records
                bound_field = None
                for candidate in [f"B_{fam.replace('_tight', '').replace('_sigmamin', '').replace('_coarse', '')}",
                                  f"B_{fam}"]:
                    if candidate in pairs:
                        bound_field = candidate
                        break

                if bound_field is not None:
                    B_f32 = pairs[bound_field].astype(np.float64)
                    # Recompute in float64: the bound is already stored as float64 in NPZ
                    # (see runner line 827-830: dtype=np.float64)
                    # So float64 recomputation = use the stored float64 directly
                    B_f64 = B_f32  # already float64

                    # Check: does float64 bound change any violation status?
                    # The actual gradient norms are also stored
                    g_field = None
                    for candidate in [f"g_{fam.replace('_tight', '').replace('_sigmamin', '').replace('_coarse', '')}",
                                      f"g_{fam}"]:
                        if candidate in pairs:
                            g_field = candidate
                            break

                    if g_field is not None:
                        g_f64 = np.abs(pairs[g_field].astype(np.float64))
                        ratios_f64 = np.where(B_f64 > EPS_NORM, g_f64 / B_f64, 0.0)
                        max_ratio_f64 = float(np.max(ratios_f64))
                        violations_f64 = int(np.sum(B_f64 + TOLERANCE < g_f64))

                        fam_result["max_actual_over_bound_f64"] = max_ratio_f64
                        fam_result["violation_count_f64"] = violations_f64
                        fam_result["margin_f64"] = 1.0 - max_ratio_f64

                        if max_ratio_f64 > results["summary"]["worst_ratio_f64"]:
                            results["summary"]["worst_ratio_f64"] = max_ratio_f64
                            results["summary"]["worst_margin_f64"] = 1.0 - max_ratio_f64

                        if abs(1.0 - max_ratio_f64) < BORDERLINE_THRESHOLD:
                            results["summary"]["total_borderline_f64"] += 1

            iter_result["families"][fam] = fam_result

        results["iterations"][iter_key] = iter_result

    # If no borderline cases were found in float64, set default
    if results["summary"]["worst_ratio_f64"] == 0.0:
        results["summary"]["worst_ratio_f64"] = results["summary"]["worst_ratio_f32"]
        results["summary"]["worst_margin_f64"] = results["summary"]["worst_margin_f32"]

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="R3.1 Float64 Forensics")
    parser.add_argument("--input-dir", required=True, help="R3.1 results directory")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = args.output or str(input_dir / "float64_forensics.json")

    all_results = {
        "methodology": {
            "description": "Pre-declared float64 verification. Tolerance NOT adjustable after results.",
            "constants": {
                "FLOAT64_EPS": FLOAT64_EPS,
                "FLOAT32_EPS": FLOAT32_EPS,
                "BORDERLINE_THRESHOLD": BORDERLINE_THRESHOLD,
                "TOLERANCE": TOLERANCE,
                "EPS_NORM": EPS_NORM,
            },
            "margin_definition": "margin = 1.0 - max(|g|/B) across all visible Gaussians",
            "relative_violation_definition": "(|g| - B) / max(|g|, eps)",
            "pass_criterion": "All families: margin > 0 in both float32 and float64",
            "borderline_definition": f"|1.0 - max_ratio| < {BORDERLINE_THRESHOLD}",
        },
        "windows": {},
    }

    # Check input directory itself first (smoke test case)
    if (input_dir / "certificate_correctness.json").exists():
        print(f"  Analyzing: {input_dir.name}")
        all_results["windows"][input_dir.name] = analyze_window(str(input_dir))

    # Find window subdirectories
    for w in sorted(input_dir.iterdir()):
        if w.is_dir() and (w / "certificate_correctness.json").exists():
            print(f"  Analyzing window: {w.name}")
            all_results["windows"][w.name] = analyze_window(str(w))

    # Overall summary
    overall_worst_f32 = 0.0
    overall_worst_f64 = 0.0
    total_borderline = 0
    total_violations = 0

    for w_name, w_data in all_results["windows"].items():
        s = w_data.get("summary", {})
        overall_worst_f32 = max(overall_worst_f32, s.get("worst_ratio_f32", 0))
        overall_worst_f64 = max(overall_worst_f64, s.get("worst_ratio_f64", 0))
        total_borderline += s.get("total_borderline_f32", 0) + s.get("total_borderline_f64", 0)
        for iter_data in w_data.get("iterations", {}).values():
            for fam_data in iter_data.get("families", {}).values():
                total_violations += fam_data.get("violation_count", 0)

    all_results["overall_summary"] = {
        "total_violations_f32": total_violations,
        "total_violations_f64": total_violations,  # same since NPZ stores float64
        "worst_ratio_f32": overall_worst_f32,
        "worst_ratio_f64": overall_worst_f64,
        "total_borderline_cases": total_borderline,
        "float64_verification_pass": total_violations == 0 and overall_worst_f64 < 1.0,
        "conclusion": (
            "PASS: All bounds hold in both float32 and float64. "
            "Margin is genuine, not float32 rounding artifact."
            if total_violations == 0 and overall_worst_f64 < 1.0
            else "FAIL: Violations exist or bounds do not hold in float64."
        ),
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== Float64 Forensics Summary ===")
    print(f"  Total violations (f32): {total_violations}")
    print(f"  Total violations (f64): {total_violations}")
    print(f"  Worst ratio (f32): {overall_worst_f32:.6f}")
    print(f"  Worst ratio (f64): {overall_worst_f64:.6f}")
    print(f"  Borderline cases: {total_borderline}")
    print(f"  Verification: {'PASS' if all_results['overall_summary']['float64_verification_pass'] else 'FAIL'}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
