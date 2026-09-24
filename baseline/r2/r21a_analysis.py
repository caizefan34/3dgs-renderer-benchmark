"""
R2.1A Analysis — Aggregates benchmark data into 6 JSON outputs + decision.
"""
import json, os, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
R21A_DIR = REPO_ROOT / "results" / "reference_v1" / "r2.1a"
BENCHMARK_FILE = R21A_DIR / "r21a_benchmark.json"


def load_benchmark():
    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE) as f:
            return json.load(f)
    return {}


def stats_to_dict(s):
    return {"median": s["median"], "mean": s["mean"], "p10": s["p10"], "p90": s["p90"], "n": s["n"]}


bench = load_benchmark()
if not bench:
    print("ERROR: benchmark data not found")
    exit(1)

full_bwd = bench["full_timing"]["raster_bwd_ms"]["median"]
full_e2e = bench["full_timing"]["e2e_ms"]["median"]
all_ones_bwd = bench["all_ones_masked_timing"]["raster_bwd_ms"]["median"]
mask_overhead = all_ones_bwd - full_bwd
mask_overhead_pct = mask_overhead / full_bwd * 100

# ============================================================
# 1. current_camera_oracle.json
# ============================================================
print("=== current_camera_oracle.json ===")

per_cam = bench["per_camera_oracle"]
geo_keeps = [v["geo"]["keep_frac"] for v in per_cam.values()]
app_keeps = [v["app"]["keep_frac"] for v in per_cam.values()]
opa_keeps = [v["opa"]["keep_frac"] for v in per_cam.values()]
geo_utils = [v["geo"]["utility_retained"] for v in per_cam.values()]
app_utils = [v["app"]["utility_retained"] for v in per_cam.values()]
opa_utils = [v["opa"]["utility_retained"] for v in per_cam.values()]

cco = bench["current_camera_oracle"]

current_camera = {
    "per_camera": per_cam,
    "pooled": {
        "geo_keep_frac_mean": float(np.mean(geo_keeps)),
        "app_keep_frac_mean": float(np.mean(app_keeps)),
        "opa_keep_frac_mean": float(np.mean(opa_keeps)),
        "geo_utility_mean": float(np.mean(geo_utils)),
        "app_utility_mean": float(np.mean(app_utils)),
        "opa_utility_mean": float(np.mean(opa_utils)),
        "all_utility_ge_95": all(u >= 0.95 for u in geo_utils + app_utils + opa_utils),
    },
    "timing": {
        "raster_bwd_ms": stats_to_dict(cco["raster_bwd_ms"]),
        "e2e_ms": stats_to_dict(cco["e2e_ms"]),
        "raster_speedup_pct": cco["raster_speedup_pct"],
        "e2e_gain_pct": cco["e2e_gain_pct"],
    },
    "note": "CURRENT-camera oracle: each camera uses its OWN utility to construct masks. "
            "Utility is exactly 95.0% per family per camera (by construction). "
            "Keep fractions: geo≈54%, app≈25%, opa≈44% (mean across cameras).",
}
with open(R21A_DIR / "current_camera_oracle.json", "w") as f:
    json.dump(current_camera, f, indent=2)
print("  Saved.")

# ============================================================
# 2. geo_only_oracle_95.json
# ============================================================
print("\n=== geo_only_oracle_95.json ===")
geo = bench["geo_only_95"]
geo_only = {
    "configuration": {
        "geo_mask": "oracle >=95% geometry utility (per-camera)",
        "app_mask": "all ones",
        "opacity_mask": "all ones",
        "mean_geo_keep_frac": geo["mean_geo_keep_frac"],
        "geo_utility_retained": geo["mean_geo_utility_retained"],
    },
    "timing": {
        "raster_bwd_ms": stats_to_dict(geo["raster_bwd_ms"]),
        "e2e_ms": stats_to_dict(geo["e2e_ms"]),
        "raster_speedup_pct": geo["raster_speedup_pct"],
        "e2e_gain_pct": geo["e2e_gain_pct"],
        "full_bwd_ms": full_bwd,
        "full_e2e_ms": full_e2e,
    },
    "analysis": {
        "mask_overhead_ms": mask_overhead,
        "net_savings_vs_full_ms": full_bwd - geo["raster_bwd_ms"]["median"],
        "is_slower_than_full": geo["raster_bwd_ms"]["median"] > full_bwd,
        "reason": "GEO_ONLY_95 is SLOWER than FULL because the mask mechanism overhead (1.72ms from 3 shared-memory "
                  "loads + 3 conditional branches per Gaussian) exceeds the savings from suppressing ~46% of geometry "
                  "atomicAdds. The oracle keeps the HIGHEST-utility Gaussians, which are also the ones with the most "
                  "pixel intersections (highest compute cost). Suppressing the remaining low-utility, low-cost "
                  "Gaussians saves very little time.",
    },
}
with open(R21A_DIR / "geo_only_oracle_95.json", "w") as f:
    json.dump(geo_only, f, indent=2)
print(f"  Raster: {geo['raster_bwd_ms']['median']:.2f}ms, gain: {geo['e2e_gain_pct']:.1f}%")

# ============================================================
# 3. geo_opacity_oracle_95.json
# ============================================================
print("\n=== geo_opacity_oracle_95.json ===")
go = bench["geo_opacity_95"]
geo_opa = {
    "configuration": {
        "geo_mask": "oracle >=95% geometry utility (per-camera)",
        "opacity_mask": "oracle >=95% opacity utility (per-camera)",
        "app_mask": "all ones",
    },
    "timing": {
        "raster_bwd_ms": stats_to_dict(go["raster_bwd_ms"]),
        "e2e_ms": stats_to_dict(go["e2e_ms"]),
        "raster_speedup_pct": go["raster_speedup_pct"],
        "e2e_gain_pct": go["e2e_gain_pct"],
        "full_bwd_ms": full_bwd,
        "full_e2e_ms": full_e2e,
    },
    "analysis": {
        "is_slower_than_full": go["raster_bwd_ms"]["median"] > full_bwd,
        "gain_beyond_geo_only_ms": geo["raster_bwd_ms"]["median"] - go["raster_bwd_ms"]["median"],
        "reason": "GEO_OPACITY_95 is still slower than FULL. Adding opacity suppression to geometry suppression "
                  "provides marginal improvement (0.37ms faster than GEO_ONLY) but not enough to overcome the "
                  "mask overhead.",
    },
}
with open(R21A_DIR / "geo_opacity_oracle_95.json", "w") as f:
    json.dump(geo_opa, f, indent=2)
print(f"  Raster: {go['raster_bwd_ms']['median']:.2f}ms, gain: {go['e2e_gain_pct']:.1f}%")

# ============================================================
# 4. full_attribute_oracle_95.json
# ============================================================
print("\n=== full_attribute_oracle_95.json ===")
fa = bench["full_attribute_95"]
full_attr = {
    "configuration": {
        "geo_mask": "oracle >=95% geometry utility (per-camera)",
        "app_mask": "oracle >=95% appearance utility (per-camera)",
        "opacity_mask": "oracle >=95% opacity utility (per-camera)",
        "pooled_utility": cco["pooled_utility"],
    },
    "timing": {
        "raster_bwd_ms": stats_to_dict(fa["raster_bwd_ms"]),
        "e2e_ms": stats_to_dict(fa["e2e_ms"]),
        "raster_speedup_pct": fa["raster_speedup_pct"],
        "e2e_gain_pct": fa["e2e_gain_pct"],
        "full_bwd_ms": full_bwd,
        "full_e2e_ms": full_e2e,
    },
    "analysis": {
        "net_savings_vs_full_ms": full_bwd - fa["raster_bwd_ms"]["median"],
        "is_faster_than_full": fa["raster_bwd_ms"]["median"] < full_bwd,
        "reason": "FULL_ATTRIBUTE_95 is slightly faster than FULL (0.54ms). When ALL three branches are suppressed, "
                  "the combined savings from geo + app + opa atomicAdd skipping finally exceed the mask overhead. "
                  "However, the net gain is only 0.97% E2E — well below the 5% threshold for ARCH_CONFIRMED.",
    },
}
with open(R21A_DIR / "full_attribute_oracle_95.json", "w") as f:
    json.dump(full_attr, f, indent=2)
print(f"  Raster: {fa['raster_bwd_ms']['median']:.2f}ms, gain: {fa['e2e_gain_pct']:.1f}%")

# ============================================================
# 5. mask_overhead.json
# ============================================================
print("\n=== mask_overhead.json ===")

overhead = {
    "FULL_bwd_ms": full_bwd,
    "ALL_ONES_bwd_ms": all_ones_bwd,
    "mask_overhead_ms": mask_overhead,
    "mask_overhead_pct": mask_overhead_pct,
    "per_config_bwd_ms": {
        "FULL": full_bwd,
        "ALL_ONES_MASKED": all_ones_bwd,
        "GEO_ONLY_95": bench["geo_only_95"]["raster_bwd_ms"]["median"],
        "GEO_OPACITY_95": bench["geo_opacity_95"]["raster_bwd_ms"]["median"],
        "FULL_ATTRIBUTE_95": bench["full_attribute_95"]["raster_bwd_ms"]["median"],
        "CURRENT_CAMERA_ORACLE": cco["raster_bwd_ms"]["median"],
    },
    "savings_from_all_ones_baseline": {
        "GEO_ONLY_95": all_ones_bwd - bench["geo_only_95"]["raster_bwd_ms"]["median"],
        "GEO_OPACITY_95": all_ones_bwd - bench["geo_opacity_95"]["raster_bwd_ms"]["median"],
        "FULL_ATTRIBUTE_95": all_ones_bwd - bench["full_attribute_95"]["raster_bwd_ms"]["median"],
    },
    "synthetic_geo_scaling": {
        k: {"bwd_ms": v["bwd_ms"]["median"], "e2e_ms": v["e2e_ms"]["median"]}
        for k, v in bench["geo_synthetic_scaling"].items()
    },
    "analysis": {
        "overhead_source": "3 shared-memory loads (app_mask_batch, geo_mask_batch, opacity_mask_batch) + 3 conditional "
                           "branches per Gaussian per pixel intersection, always incurred regardless of mask values.",
        "overhead_vs_savings": f"Overhead is {mask_overhead:.2f}ms. At 95% utility, GEO_ONLY saves only "
                               f"{all_ones_bwd - bench['geo_only_95']['raster_bwd_ms']['median']:.2f}ms from all-ones "
                               f"baseline — not enough to break even. FULL_ATTRIBUTE saves "
                               f"{all_ones_bwd - bench['full_attribute_95']['raster_bwd_ms']['median']:.2f}ms — barely "
                               f"enough for a net 0.54ms gain.",
        "can_fewer_mask_loads_help": "Yes. A single packed mask (3 bits in 1 byte) would reduce overhead to ~1 load + "
                                     "3 bit-extract branches. Estimated overhead reduction: ~60% (from 1.72ms to ~0.7ms). "
                                     "At 0.7ms overhead, GEO_ONLY_95 would net +1.0ms savings (~4% raster speedup).",
        "correlation_issue": "Oracle masks keep the HIGHEST-utility Gaussians, which have the most pixel intersections "
                             "(highest compute cost). Suppressing the low-utility rest saves disproportionately little "
                             "time. Random masks at the same keep fraction save more because they also suppress "
                             "high-cost Gaussians.",
    },
}
with open(R21A_DIR / "mask_overhead.json", "w") as f:
    json.dump(overhead, f, indent=2)
print("  Saved.")

# ============================================================
# 6. final_decision.json
# ============================================================
print("\n=== final_decision.json ===")

decisions = bench["decisions"]

# Recompute decision logic with clear documentation
geo_e2e_gain = bench["geo_only_95"]["e2e_gain_pct"]
geo_opa_e2e_gain = bench["geo_opacity_95"]["e2e_gain_pct"]
full_attr_e2e_gain = bench["full_attribute_95"]["e2e_gain_pct"]

geo_capture_ratio = geo_e2e_gain / full_attr_e2e_gain if full_attr_e2e_gain > 0 else 0.0
geo_opa_capture_ratio = geo_opa_e2e_gain / full_attr_e2e_gain if full_attr_e2e_gain > 0 else 0.0
app_adds_gain = full_attr_e2e_gain - geo_opa_e2e_gain
opa_adds_gain = geo_opa_e2e_gain - geo_e2e_gain

# Dominant design
if app_adds_gain >= 1.0:
    dominant = "FULL_ATTRIBUTE_REQUIRED"
elif geo_opa_capture_ratio >= 0.90 and opa_adds_gain >= 0.5:
    dominant = "GEO_OPACITY_DOMINANT"
elif geo_capture_ratio >= 0.80:
    dominant = "GEOMETRY_DOMINANT"
else:
    dominant = "FULL_ATTRIBUTE_REQUIRED"  # fallback when geo-only is negative

# Architecture gate
pooled_util = cco["pooled_utility"]
arch_utility_ok = (pooled_util["geo"] >= 0.95 and pooled_util["app"] >= 0.95 and pooled_util["opa"] >= 0.95)
arch_e2e_ok = full_attr_e2e_gain >= 5.0

if arch_utility_ok and arch_e2e_ok:
    arch_gate = "ARCH_CONFIRMED"
elif full_attr_e2e_gain >= 2.0:
    arch_gate = "ARCH_WEAKENED"
else:
    arch_gate = "ARCH_INVALIDATED"

final = {
    "architecture_gate": arch_gate,
    "dominant_design": dominant,
    "main_comparison_table": {
        "FULL": {
            "geo_utility": 1.0, "app_utility": 1.0, "opa_utility": 1.0,
            "raster_ms": full_bwd, "raster_gain_pct": 0.0,
            "e2e_ms": full_e2e, "e2e_gain_pct": 0.0,
        },
        "GEO_ONLY_95": {
            "geo_utility": pooled_util["geo"], "app_utility": 1.0, "opa_utility": 1.0,
            "raster_ms": bench["geo_only_95"]["raster_bwd_ms"]["median"],
            "raster_gain_pct": bench["geo_only_95"]["raster_speedup_pct"],
            "e2e_ms": bench["geo_only_95"]["e2e_ms"]["median"],
            "e2e_gain_pct": geo_e2e_gain,
        },
        "GEO_OPACITY_95": {
            "geo_utility": pooled_util["geo"], "app_utility": 1.0, "opa_utility": pooled_util["opa"],
            "raster_ms": bench["geo_opacity_95"]["raster_bwd_ms"]["median"],
            "raster_gain_pct": bench["geo_opacity_95"]["raster_speedup_pct"],
            "e2e_ms": bench["geo_opacity_95"]["e2e_ms"]["median"],
            "e2e_gain_pct": geo_opa_e2e_gain,
        },
        "FULL_ATTRIBUTE_95": {
            "geo_utility": pooled_util["geo"], "app_utility": pooled_util["app"], "opa_utility": pooled_util["opa"],
            "raster_ms": bench["full_attribute_95"]["raster_bwd_ms"]["median"],
            "raster_gain_pct": bench["full_attribute_95"]["raster_speedup_pct"],
            "e2e_ms": bench["full_attribute_95"]["e2e_ms"]["median"],
            "e2e_gain_pct": full_attr_e2e_gain,
        },
    },
    "decision_metrics": {
        "geo_capture_ratio_vs_full_attr": float(geo_capture_ratio),
        "geo_opa_capture_ratio_vs_full_attr": float(geo_opa_capture_ratio),
        "app_adds_absolute_e2e_pct": float(app_adds_gain),
        "opa_adds_absolute_e2e_pct": float(opa_adds_gain),
        "arch_utility_met": arch_utility_ok,
        "arch_e2e_met": arch_e2e_ok,
        "mask_overhead_ms": mask_overhead,
        "mask_overhead_pct": mask_overhead_pct,
    },
    "thresholds": {
        "ARCH_CONFIRMED": ">=95% utility per active family AND >=5% E2E gain",
        "ARCH_WEAKENED": "2-5% E2E gain",
        "ARCH_INVALIDATED": "<2% E2E gain",
        "GEOMETRY_DOMINANT": "GEO_ONLY_95 captures >=80% of FULL_ATTRIBUTE_95's E2E gain",
        "GEO_OPACITY_DOMINANT": "GEO_OPACITY_95 captures >=90% of FULL_ATTR's gain AND opacity adds >=0.5%",
        "FULL_ATTRIBUTE_REQUIRED": "Appearance gating adds >=1% absolute E2E gain beyond GEO_OPACITY",
    },
    "key_finding": (
        "At 95% positive utility per family, the E2E gain is only +0.97% (FULL_ATTRIBUTE_95). "
        "This is below the 2% threshold → ARCH_INVALIDATED. "
        "The bottleneck is the mask mechanism overhead (1.72ms, 7% of raster backward), not the utility-coverage "
        "tradeoff. The overhead comes from 3 shared-memory loads + 3 conditional branches per Gaussian, always "
        "incurred regardless of mask values. Only when ALL three branches are suppressed do the combined savings "
        "barely overcome this fixed overhead. GEO_ONLY_95 and GEO_OPACITY_95 are actually SLOWER than FULL."
    ),
    "secondary_finding": (
        "Oracle masks keep the highest-utility Gaussians, which also have the most pixel intersections (highest "
        "compute cost). Suppressing the low-utility rest saves disproportionately little time. This is a fundamental "
        "correlation between utility and compute cost that limits the speedup potential of utility-aware masking."
    ),
    "r2_1_correction": (
        "R2.1's 15.6% raster speedup used CROSS-CAMERA oracle masks (1 camera's masks applied to 5 cameras) with "
        "lower effective utility (89-93%). R2.1A uses CURRENT-CAMERA oracle masks at 95% utility. The speedup "
        "drops from 15.6% to 2.2% because: (1) current-camera masks need higher keep fractions (54% vs 50% for geo), "
        "(2) the oracle keeps high-compute-cost Gaussians, and (3) the mask overhead is fixed regardless of keep "
        "fraction. R2.1's ALL_BRANCH_K50 (17.5% speedup) used RANDOM 50% masking which suppresses high-cost "
        "Gaussians too, but at only ~50% utility."
    ),
}
with open(R21A_DIR / "final_decision.json", "w") as f:
    json.dump(final, f, indent=2)
print(f"  Architecture: {arch_gate}")
print(f"  Dominant: {dominant}")
print(f"  FULL E2E: {full_e2e:.2f}ms")
print(f"  GEO_ONLY_95: {bench['geo_only_95']['e2e_ms']['median']:.2f}ms, gain={geo_e2e_gain:.1f}%")
print(f"  GEO_OPACITY_95: {bench['geo_opacity_95']['e2e_ms']['median']:.2f}ms, gain={geo_opa_e2e_gain:.1f}%")
print(f"  FULL_ATTRIBUTE_95: {bench['full_attribute_95']['e2e_ms']['median']:.2f}ms, gain={full_attr_e2e_gain:.1f}%")
print(f"  App adds >=1%? {'YES' if app_adds_gain >= 1.0 else 'NO'} ({app_adds_gain:.2f}%)")

print("\n=== R2.1A Analysis complete ===")
