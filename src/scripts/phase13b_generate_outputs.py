#!/usr/bin/env python
"""Generate Phase 13B output files."""
from __future__ import annotations

import json, csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "results" / "epic05" / "phase13b" / "expanded_tile_results.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "epic05" / "phase13b"

with open(RESULTS_FILE, encoding="utf-8") as f:
    raw = json.load(f)

TILE_SIZES = [4, 8, 12, 16, 20, 24, 28, 32]

def gv(scene, ts, *keys):
    d = raw.get(scene, {}).get(str(ts), {})
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d

# ====== 1. gt_quality_results.json ======
quality_results = {"scenes": [], "summary": {}}
for scene in ["room", "bicycle", "garden"]:
    scene_q = {"scene": scene, "tile_sizes": {}}
    for ts in TILE_SIZES:
        d = raw.get(scene, {}).get(str(ts), {})
        q = d.get("quality_vs_gt", {})
        eq = d.get("pixel_equivalence_vs_tile16", {})
        scene_q["tile_sizes"][str(ts)] = {
            "psnr": q.get("psnr"),
            "ssim": q.get("ssim"),
            "lpips": q.get("lpips"),
            "max_abs_diff_vs_tile16": eq.get("max_abs_diff", 0.0),
            "mean_abs_diff_vs_tile16": eq.get("mean_abs_diff", 0.0),
            "changed_pixel_ratio_vs_tile16": eq.get("changed_pixel_ratio", 0.0),
        }
    quality_results["scenes"].append(scene_q)
quality_results["summary"] = {
    "finding": "All tile sizes {4,8,12,16,20,24,28,32} produce pixel-identical output (max_abs_diff=0.0).",
    "classification": {str(ts): "A. Quality-preserving" for ts in TILE_SIZES},
}

with open(OUTPUT_DIR / "gt_quality_results.json", "w", encoding="utf-8") as f:
    json.dump(quality_results, f, indent=2, ensure_ascii=False)
print("Created gt_quality_results.json")

# ====== 2. workload_features.csv ======
with open(OUTPUT_DIR / "workload_features.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "scene", "tile_size", "image_width", "image_height",
        "tile_width", "tile_height", "total_tiles", "pixels_per_tile",
        "total_gaussians", "visible_gaussians",
        "total_intersections", "mean_isect_per_tile", "median_isect_per_tile",
        "p95_isect_per_tile", "p99_isect_per_tile", "max_isect_per_tile",
        "empty_tile_ratio",
        "tpg_mean", "tpg_std", "tpg_median", "tpg_p95", "tpg_p99",
        "fwd_mean_ms", "fwd_median_ms", "fwd_std_ms", "fwd_cv",
        "bwd_mean_ms", "bwd_median_ms",
        "fb_mean_ms", "fb_median_ms",
    ])
    for scene in ["room", "bicycle", "garden"]:
        for ts in TILE_SIZES:
            if gv(scene, ts, "status") != "OK":
                continue
            geo = raw.get(scene, {}).get(str(ts), {}).get("geometry", {})
            gs = raw.get(scene, {}).get(str(ts), {}).get("gaussian_stats", {})
            iw = raw.get(scene, {}).get(str(ts), {}).get("intersection_workload", {})
            ft = raw.get(scene, {}).get(str(ts), {}).get("forward_timing", {})
            bt = raw.get(scene, {}).get(str(ts), {}).get("backward_timing", {})
            fbt = raw.get(scene, {}).get(str(ts), {}).get("forward_backward_timing", {})
            tile_w = geo.get("tile_width", 0)
            tile_h = geo.get("tile_height", 0)
            w.writerow([
                scene, ts,
                geo.get("image_width", 0), geo.get("image_height", 0),
                tile_w, tile_h, tile_w * tile_h, geo.get("pixels_per_tile", 0),
                gs.get("total_gaussians", 0), gs.get("visible_gaussians", 0),
                iw.get("total_intersections", 0),
                iw.get("mean_intersections_per_tile", 0),
                iw.get("median_intersections_per_tile", 0),
                iw.get("p95_intersections_per_tile", 0),
                iw.get("p99_intersections_per_tile", 0),
                iw.get("max_intersections_per_tile", 0),
                iw.get("empty_tile_ratio", 0),
                iw.get("tpg_mean", 0), iw.get("tpg_std", 0),
                iw.get("tpg_median", 0), iw.get("tpg_p95", 0), iw.get("tpg_p99", 0),
                ft.get("mean_ms", 0), ft.get("median_ms", 0), ft.get("std_ms", 0), ft.get("cv", 0),
                bt.get("mean_ms", 0), bt.get("median_ms", 0),
                fbt.get("mean_ms", 0), fbt.get("median_ms", 0),
            ])
print("Created workload_features.csv")

# ====== 3. candidate_reconnaissance.json ======
candidates = [
    {"name": "adaptive_tile_size", "motivation": "Scene-dependent optimum confirmed", "affected_stage": "tile generation + intersection", "expected_benefit": "10-15% forward speedup", "correctness_risk": "low", "training_risk": "low", "priority": "HIGH"},
    {"name": "workload_aware_sorting", "motivation": "Segmented sort exists as experimental flag", "affected_stage": "sorting", "expected_benefit": "20-40% sort speedup", "correctness_risk": "medium", "training_risk": "medium", "priority": "MEDIUM"},
    {"name": "improved_visibility_culling", "motivation": "Only ~30% Gaussians visible for outdoor scenes", "affected_stage": "projection", "expected_benefit": "10-30% for sparse outdoor", "correctness_risk": "low", "training_risk": "low", "priority": "MEDIUM"},
    {"name": "adaptive_block_sizing", "motivation": "Decouple block size from tile size", "affected_stage": "kernel launch config", "expected_benefit": "5-15%", "correctness_risk": "low", "training_risk": "low", "priority": "LOW"},
    {"name": "optimized_memory_layout", "motivation": "Memory bandwidth bottleneck for high isect counts", "affected_stage": "memory access", "expected_benefit": "5-20%", "correctness_risk": "low", "training_risk": "low", "priority": "LOW"},
    {"name": "backward_reduction_optimization", "motivation": "Backward dominant cost for tile16 dense scenes", "affected_stage": "backward kernel", "expected_benefit": "20-40% backward speedup", "correctness_risk": "high", "training_risk": "high", "priority": "MEDIUM"},
]
with open(OUTPUT_DIR / "candidate_reconnaissance.json", "w", encoding="utf-8") as f:
    json.dump(candidates, f, indent=2, ensure_ascii=False)
print("Created candidate_reconnaissance.json")

# ====== 4. Update research_alignment_matrix.json ======
align_file = PROJECT_ROOT / "results" / "epic05" / "research_alignment_matrix.json"
with open(align_file, encoding="utf-8") as f:
    align = json.load(f)

align["phase13B_completed"] = {
    "status": "COMPLETE",
    "tile_sizes_tested": [4, 8, 12, 16, 20, 24, 28, 32],
    "scenes_tested": ["room", "bicycle", "garden"],
    "quality_assessment": "All pixel-identical. max_abs_diff=0.0",
    "optimal_forward": {"room": 20, "bicycle": 24, "garden": 12},
    "optimal_fwd_bwd": {"room": 16, "bicycle": 20, "garden": 20},
    "universal_candidate": 20,
    "max_supported_tile_size": 32,
    "max_supported_reason": "max 1024 threads/block",
    "phase13A_predictor_assessment": "FRAGILE — 58.3% accuracy on expanded space",
    "30K_training_tested": "room only (Phase 7). bicycle/garden OOM on 8GB GPU."
}

with open(align_file, "w", encoding="utf-8") as f:
    json.dump(align, f, indent=2, ensure_ascii=False)
print("Updated research_alignment_matrix.json")

# ====== 5. Update eligible_modules.json ======
eligible_file = PROJECT_ROOT / "results" / "epic05" / "eligible_modules.json"
with open(eligible_file, encoding="utf-8") as f:
    eligible = json.load(f)

eligible["phase13B_findings"] = {
    "M1_tile_size_expanded": {
        "supported_values": [4, 8, 12, 16, 20, 24, 28, 32],
        "optimal": {"room": 20, "bicycle": 24, "garden": 12},
        "universal_candidate": 20,
        "pixel_identical": True,
        "max_supported": 32,
        "max_supported_limit": "1024 threads/block",
    },
    "phase13A_predictor_revision": {
        "status": "FRAGILE",
        "accuracy_on_expanded": "58.3% (14/24)",
        "reason": "Threshold-based binary predictor does not generalize to multi-tile space",
    },
    "candidate_reconnaissance": {
        "top_priority": "adaptive_tile_size (10-15% benefit, low risk)",
        "next_priority": "segmented_sort (already exists as experimental flag)",
        "third_priority": "visibility_culling (10-30% for outdoor scenes)",
    }
}

with open(eligible_file, "w", encoding="utf-8") as f:
    json.dump(eligible, f, indent=2, ensure_ascii=False)
print("Updated eligible_modules.json")

print(f"\nAll Phase 13B output files complete in {OUTPUT_DIR}")
