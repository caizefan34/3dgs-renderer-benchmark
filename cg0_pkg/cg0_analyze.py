#!/usr/bin/env python3
"""CG0 frozen analyzer — VERBATIM from CG0_GAUSSIAN_COMPILATION_ORACLE packet.
Owns the verdict. No post-hoc rescue, no threshold tuning after results.
Usage: cg0_analyze.py <summary_or_dir>
"""
import json
import sys

CELLS = ("room@15000", "room@25000", "bicycle@15000", "bicycle@25000")


def load(path):
    with open(path) as f:
        return json.load(f)


def find_summary(p):
    import os
    if os.path.isdir(p):
        return load(os.path.join(p, "CG0_SUMMARY.json"))
    return load(p)


def main():
    d = find_summary(sys.argv[1])
    if not d.get("source_identity_pass", False):
        out = {"verdict": "CG0_BLOCKED_SOURCE_IDENTITY"}
    elif not d.get("no_oracle_leakage_pass", False):
        out = {"verdict": "CG0_INVALID_ORACLE_LEAKAGE"}
    elif not d.get("oracle_extraction_pass", False):
        out = {"verdict": "CG0_ORACLE_EXTRACTION_FAIL"}
    elif not d.get("compiler_numerical_pass", False):
        out = {"verdict": "CG0_COMPILER_NUMERICAL_FAIL"}
    else:
        def strong(x):
            q0 = x["compiled0"]; q5 = x["compiled500"]; q15 = x["compiled1500"]
            return (
                .80 <= x["count_ratio"] <= 1.20 and
                x["intersection_ratio"] <= 1.20 and
                q0["holdout_psnr_delta_median_db"] >= -.75 and
                q0["trainval_psnr_delta_median_db"] >= -.50 and
                q5["holdout_psnr_delta_median_db"] >= -.20 and
                q5["holdout_psnr_delta_worst_db"] >= -.45 and
                q15["holdout_psnr_delta_median_db"] >= -.08 and
                q15["holdout_psnr_delta_worst_db"] >= -.20 and
                q15["trainval_psnr_delta_median_db"] >= -.05 and
                q15["ssim_delta_median"] >= -.002 and
                x["conditional_compile_refine_fraction"] <= .12 and
                x["nan_inf_count"] == 0
            )

        def survives(x):
            q0 = x["compiled0"]; q5 = x["compiled500"]; q15 = x["compiled1500"]
            return (
                .70 <= x["count_ratio"] <= 1.30 and
                x["intersection_ratio"] <= 1.35 and
                q0["holdout_psnr_delta_median_db"] >= -1.50 and
                q5["holdout_psnr_delta_median_db"] >= -.50 and
                q5["holdout_psnr_delta_worst_db"] >= -.90 and
                q15["holdout_psnr_delta_median_db"] >= -.15 and
                q15["holdout_psnr_delta_worst_db"] >= -.35 and
                q15["trainval_psnr_delta_median_db"] >= -.10 and
                q15["ssim_delta_median"] >= -.004 and
                x["conditional_compile_refine_fraction"] <= .18 and
                x["nan_inf_count"] == 0
            )
        xs = [d["cells"][c] for c in CELLS]
        s = all(strong(x) for x in xs)
        p = all(survives(x) for x in xs)
        out = {
            "verdict": "CG0_STRONG_CAPACITY_SURVIVES" if s else
                       ("CG0_CAPACITY_SURVIVES" if p else "CG0_COMPILE_CAPACITY_CLOSED"),
            "strong_all_cells": s,
            "capacity_survives_all_cells": p,
        }
    # write next to input
    if sys.argv[1].endswith(".json"):
        outpath = sys.argv[1].replace("SUMMARY", "VERDICT")
    else:
        import os
        outpath = os.path.join(sys.argv[1], "CG0_VERDICT.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
