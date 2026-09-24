#!/usr/bin/env python3
"""Generate all H1 CSV deliverables from per-camera JSON results."""
import json, glob, os, csv, sys
import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"

def load_results():
    results = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*_cam*.json"))):
        results.append(json.load(open(f)))
    return results

results = load_results()
print("Loaded %d result files" % len(results))

# ---- run_manifest.csv ----
with open(os.path.join(OUT_DIR, "run_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "camera_id", "method", "width", "height",
                "warmup", "measure", "result_file"])
    for r in results:
        scene = r["scene"]
        cam_idx = r["camera_idx"]
        cam_id = r["camera_id"]
        fname = "%s_cam%d.json" % (scene, cam_idx)
        for method in ["B1", "B2"]:
            w.writerow([scene, cam_idx, cam_id, method, r["width"], r["height"],
                        r["warmup"], r["measure"], fname])
    print("Wrote run_manifest.csv")

# ---- stage_timings_raw.csv ----
with open(os.path.join(OUT_DIR, "stage_timings_raw.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "method", "stage", "median_ms", "mean_ms",
                "std_ms", "min_ms", "max_ms", "n_measure"])
    for r in results:
        scene = r["scene"]
        ci = r["camera_idx"]
        # B1 decomposed stages
        st = r.get("B1_forward_stages_repeated", {})
        for k, v in st.items():
            w.writerow([scene, ci, "B1_decomposed", k, v["median_ms"], v["mean_ms"],
                        v["std_ms"], v["min_ms"], v["max_ms"], r["measure"]])
        # B1 total timings
        for tk in ["B1_forward_decomposed_timing", "B1_forward_autograd_timing",
                    "B1_fwd_bwd_timing"]:
            if tk in r:
                t = r[tk]
                w.writerow([scene, ci, "B1", tk, t.get("median_ms", 0),
                            t.get("mean_ms", 0), t.get("std_ms", 0),
                            t.get("min_ms", 0), t.get("max_ms", 0),
                            t.get("n_measure", r["measure"])])
        # B2 total timings
        for tk in ["B2_forward_timing", "B2_fwd_bwd_timing", "B2_state_prep_timing"]:
            if tk in r:
                t = r[tk]
                w.writerow([scene, ci, "B2", tk, t.get("median_ms", 0),
                            t.get("mean_ms", 0), t.get("std_ms", 0),
                            t.get("min_ms", 0), t.get("max_ms", 0),
                            t.get("n_measure", r["measure"])])
    print("Wrote stage_timings_raw.csv")

# ---- stage_timings_summary.csv ----
# Aggregate per scene: median of medians across cameras
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for r in results:
    scene = r["scene"]
    st = r.get("B1_forward_stages_repeated", {})
    for k, v in st.items():
        agg[(scene, "B1_decomposed")][k].append(v["median_ms"])
    for tk in ["B1_forward_decomposed_timing", "B1_forward_autograd_timing", "B1_fwd_bwd_timing"]:
        if tk in r:
            agg[(scene, "B1")][tk].append(r[tk]["median_ms"])
    for tk in ["B2_forward_timing", "B2_fwd_bwd_timing", "B2_state_prep_timing"]:
        if tk in r:
            agg[(scene, "B2")][tk].append(r[tk]["median_ms"])

with open(os.path.join(OUT_DIR, "stage_timings_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "method", "stage", "median_across_cams_ms",
                "mean_across_cams_ms", "std_across_cams_ms", "n_cameras"])
    for (scene, method), stages in sorted(agg.items()):
        for stage, vals in sorted(stages.items()):
            arr = np.array(vals)
            w.writerow([scene, method, stage, float(np.median(arr)),
                        float(np.mean(arr)), float(np.std(arr)), len(vals)])
    print("Wrote stage_timings_summary.csv")

# ---- workload_metrics.csv ----
with open(os.path.join(OUT_DIR, "workload_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    cols = ["scene", "camera_idx", "method", "N_total", "N_visible", "N_isects",
            "total_tiles", "active_tiles", "tile_occupancy",
            "mean_isects_per_tile", "median_isects_per_tile",
            "p95_isects_per_tile", "p99_isects_per_tile", "max_isects_per_tile",
            "tiles_per_gaussian", "pixels_per_gaussian",
            "image_width", "image_height", "culling_ratio"]
    w.writerow(cols)
    for r in results:
        scene = r["scene"]
        ci = r["camera_idx"]
        wl1 = r.get("B1_workload", {})
        wl2 = r.get("B2_workload", {})
        for method, wl in [("B1", wl1), ("B2", wl2)]:
            row = [scene, ci, method]
            for c in cols[3:]:
                v = wl.get(c, "UNAVAILABLE")
                row.append(v)
            w.writerow(row)
    print("Wrote workload_metrics.csv")

# ---- workload_crossover.csv ----
with open(os.path.join(OUT_DIR, "workload_crossover.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "metric", "B1", "B2", "delta", "ratio"])
    crossover_keys = ["N_total", "N_visible", "N_isects", "total_tiles",
                      "active_tiles", "tile_occupancy", "mean_isects_per_tile",
                      "p95_isects_per_tile", "p99_isects_per_tile",
                      "tiles_per_gaussian", "pixels_per_gaussian"]
    for r in results:
        scene = r["scene"]
        ci = r["camera_idx"]
        wl1 = r.get("B1_workload", {})
        wl2 = r.get("B2_workload", {})
        for k in crossover_keys:
            v1 = wl1.get(k, None)
            v2 = wl2.get(k, None)
            if v1 is not None and v2 is not None and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                delta = v2 - v1
                ratio = v2 / v1 if v1 != 0 else float("inf")
            else:
                delta = "N/A"
                ratio = "N/A"
            w.writerow([scene, ci, k, v1, v2, delta, ratio])
    print("Wrote workload_crossover.csv")

# ---- matched_state_correctness.csv ----
with open(os.path.join(OUT_DIR, "matched_state_correctness.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "check_type", "metric", "value", "pass"])
    for r in results:
        scene = r["scene"]
        ci = r["camera_idx"]
        fc = r.get("forward_correctness", {})
        for k, v in fc.items():
            if isinstance(v, (int, float)):
                if k == "render_psnr_db":
                    passed = "PASS" if v >= 40.0 else "FAIL"
                elif k == "render_max_abs":
                    passed = "PASS" if v < 1e-4 else "FAIL"
                elif k == "alpha_max_abs":
                    passed = "PASS" if v < 1e-4 else "FAIL"
                else:
                    passed = "INFO"
                w.writerow([scene, ci, "forward", k, v, passed])
        bc = r.get("backward_correctness", {})
        for param, grads in bc.items():
            if isinstance(grads, dict):
                for k in ["max_abs", "mean_abs", "cosine", "relative_l2", "zero_nonzero_disagreement"]:
                    if k in grads:
                        w.writerow([scene, ci, "backward_%s" % param, k, grads[k], "INFO"])
    print("Wrote matched_state_correctness.csv")

print("\nAll CSVs generated in %s" % OUT_DIR)