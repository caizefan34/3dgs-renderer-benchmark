#!/usr/bin/env python3
"""Split h4_0r_summary.json into individual deliverable JSONs."""
import json
from pathlib import Path

src = Path("artifacts/higs-h4-0r/h4_0r_summary.json")
out_dir = Path("artifacts/higs-h4-0r")
data = json.loads(src.read_text())

scenes = data["scenes"]
pooled = data["pooled"]

# 1. corrected_counts.json — baseline work counts per scene + pooled
corrected = {}
for scene, sc in scenes.items():
    t = sc["totals"]
    corrected[scene] = {
        "U": t["U"], "V": t["V"],
        "V_over_U": t["V"] / max(t["U"], 1),
        "sigma_evals": t["baseline_sigma_evals"],
        "sigma_neg": t["sigma_neg"],
        "exp_evals": t["baseline_exp_calls"],
        "alpha_rej": t["alpha_rej"],
        "alpha_acc": t["alpha_acc"],
        "comp": t["comp"],
        "term_px": t["term_px"],
        "support_occ": t["occ"],
        "n_gaussians": sc["n_gaussians"],
        "n_entries_f4": sc["n_entries_f4"],
    }
pt = pooled["totals"]
corrected["pooled"] = {
    "U": pt["U"], "V": pt["V"],
    "V_over_U": pt["V"] / max(pt["U"], 1),
    "sigma_evals": pt["baseline_sigma_evals"],
    "sigma_neg": pt["sigma_neg"],
    "exp_evals": pt["baseline_exp_calls"],
    "alpha_rej": pt["alpha_rej"],
    "alpha_acc": pt["alpha_acc"],
    "comp": pt["comp"],
    "term_px": pt["term_px"],
    "support_occ": pt["occ"],
    "n_gaussians": pt.get("n_gauss", sum(s["n_gaussians"] for s in scenes.values())),
    "n_entries_f4": sum(s["n_entries_f4"] for s in scenes.values()),
}
(out_dir / "corrected_counts.json").write_text(json.dumps(corrected, indent=2))

# 2. exp_accounting.json — exp calls per scheme
exp_acc = {}
for scene, sc in scenes.items():
    exp_acc[scene] = {
        "baseline_exp_calls": sc["totals"]["baseline_exp_calls"],
        "schemes": {name: {"exp_after": s["exp_after"], "exp_saved_frac": s["exp_saved_frac"]}
                     for name, s in sc["schemes"].items()},
    }
exp_acc["pooled"] = {
    "baseline_exp_calls": pt["baseline_exp_calls"],
    "schemes": {name: {"exp_after": s["exp_after"], "exp_saved_frac": s["exp_saved_frac"]}
                 for name, s in pooled["schemes"].items()},
}
(out_dir / "exp_accounting.json").write_text(json.dumps(exp_acc, indent=2))

# 3. support_threshold.json — exact alpha-support equation
support = {
    "equation": "opacity * exp(-sigma) >= 1/255  =>  sigma <= log(255 * opacity) when opacity >= 1/255",
    "conic_quadratic": "sigma = 0.5*(A*dx^2 + C*dy^2) + B*dx*dy",
    "support_ellipse": "sigma <= log(255*op) defines an ellipse in (dx,dy) space centered at (gx,gy)",
    "sigma_neg_count": pt["sigma_neg"],
    "sigma_neg_explanation": (
        "gsplat fully_fused_projection clips each Gaussian's 2D radius to "
        "the tile grid, so only pixels within the projected ellipse are assigned. "
        "The conic (A,B,C) is positive-definite for all valid Gaussians, "
        "so sigma >= 0 for all tile-assigned (pixel,gaussian) pairs. "
        "The sigma < 0 check in the kernel is a safety net, not a workload reducer."
    ),
    "positive_definite_invariant": (
        "The conic matrix [[A, B/2], [B/2, C]] is positive-definite because "
        "A > 0 and C > 0 (from the 2D covariance inverse of a valid 3D Gaussian). "
        "This guarantees sigma >= 0 for ALL pixel positions, not just clipped ones. "
        "Therefore removing sigma < 0 from the fast path is semantically safe."
    ),
    "radius_clipping": (
        "gsplat additionally clips the projected radius to the tile boundary, "
        "which further restricts the pixel set but is redundant with the "
        "positive-definite invariant for sigma >= 0."
    ),
}
(out_dir / "support_threshold.json").write_text(json.dumps(support, indent=2))

# 4. mask_storage.json
mask_storage = {"per_scene": {}, "pooled": {}}
for scene, sc in scenes.items():
    mask_storage["per_scene"][scene] = sc["mask_storage"]
mask_storage["pooled"] = pooled["mask_storage"]
(out_dir / "mask_storage.json").write_text(json.dumps(mask_storage, indent=2))

# 5. block_efficiency.json — K/A ratios
block_eff = {}
for scene, sc in scenes.items():
    block_eff[scene] = {name: {"K": s["K"], "A": sc["totals"]["alpha_acc"],
                                "K_over_A": s["K_over_A"],
                                "saved_vs_visited": s["saved_vs_visited"]}
                         for name, s in sc["schemes"].items()}
block_eff["pooled"] = {name: {"K": s["K"], "A": pt["alpha_acc"],
                               "K_over_A": s["K_over_A"],
                               "saved_vs_visited": s["saved_vs_visited"]}
                        for name, s in pooled["schemes"].items()}
(out_dir / "block_efficiency.json").write_text(json.dumps(block_eff, indent=2))

# 6. higs32_mask_oracle.json
h32 = {}
for scene, sc in scenes.items():
    h32[scene] = sc["higs32_mask_oracle"]
(out_dir / "higs32_mask_oracle.json").write_text(json.dumps(h32, indent=2))

# 7. final_gate.json
gate = {
    "criteria": {
        "b16_saves_90pct_3of3": {
            "room": scenes["room"]["schemes"]["B16"]["saved_vs_visited"],
            "bicycle": scenes["bicycle"]["schemes"]["B16"]["saved_vs_visited"],
            "garden": scenes["garden"]["schemes"]["B16"]["saved_vs_visited"],
            "pass": False,
            "note": "B16 saves 36.3%, 53.8%, 30.7% — far below 90% threshold",
        },
        "k_b16_over_a_le_1_25_2of3": {
            "room": scenes["room"]["schemes"]["B16"]["K_over_A"],
            "bicycle": scenes["bicycle"]["schemes"]["B16"]["K_over_A"],
            "garden": scenes["garden"]["schemes"]["B16"]["K_over_A"],
            "pass": True,
            "note": "room=1.131 (pass), bicycle=1.257 (fail), garden=1.102 (pass) → 2/3 pass",
        },
        "mask_gen_plausible_advantage": {
            "room": scenes["room"]["higs32_mask_oracle"]["ratio_avoided_to_gen"],
            "bicycle": scenes["bicycle"]["higs32_mask_oracle"]["ratio_avoided_to_gen"],
            "garden": scenes["garden"]["higs32_mask_oracle"]["ratio_avoided_to_gen"],
            "pass": True,
            "note": "43x, 67x, 39x — mask generation cost is negligible vs avoided work",
        },
    },
    "decision": "H4_0R_WEAK",
    "reason": (
        "Criterion 1 fails dramatically: B16 removes only 30.7-53.8% of V "
        "(threshold: ≥90%). The H4-0 overestimate (97%+) was caused by: "
        "(1) using U instead of V in the savings formula, (2) using sigma>=0 "
        "instead of sigma<=log(255*op) as the support condition, (3) counting "
        "all tile-G pairs as 'visited' without respecting termination. "
        "With corrected accounting, the block mask approach eliminates ~36-54% "
        "of the sequential walk, which is meaningful but not transformative. "
        "The K/A ratio is good (≤1.26), meaning B16 blocks are near pixel-perfect, "
        "but the absolute savings are far below the H4-0R_STRONG threshold."
    ),
    "h4_production_authorized": False,
    "recommendation": (
        "B16 block masks provide a real but moderate 30-54% reduction in "
        "per-pixel Gaussian evaluations. This is below the 90% threshold for "
        "H4_0R_STRONG. Consider: (1) whether the 30-54% savings justify the "
        "implementation complexity for a standalone H4 kernel, or (2) whether "
        "block masks are better integrated as a pre-filter in the existing "
        "F5 kernel rather than a separate macro rasterizer. The 32-G transpose "
        "representation (39-67x cost ratio) is attractive for implementation "
        "regardless."
    ),
}
(out_dir / "final_gate.json").write_text(json.dumps(gate, indent=2))

print("All deliverables written to", out_dir)
for p in sorted(out_dir.glob("*.json")):
    print(f"  {p.name}: {p.stat().st_size} bytes")
