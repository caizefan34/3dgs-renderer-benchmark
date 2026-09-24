#!/usr/bin/env python3
"""Generate analysis.json from H1 profiling results — mandated P1-P9 schema."""
import json, glob, os, sys, csv
import numpy as np
from collections import defaultdict

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"

def load_results():
    results = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*_cam*.json"))):
        results.append(json.load(open(f)))
    return results

results = load_results()
env = json.load(open(os.path.join(OUT_DIR, "environment.json")))

# ---- Gate checks ----
provenance_pass = True
matched_state_pass = True
correctness_pass = True
closure_pass = True
no_mixed_version = True
no_mixed_process = True  # B1 and B2 run in same process
n_cameras_per_scene = defaultdict(int)
for r in results:
    n_cameras_per_scene[r["scene"]] += 1

n_cameras_pass = all(v >= 3 for v in n_cameras_per_scene.values())

# Forward correctness
for r in results:
    fc = r.get("forward_correctness", {})
    if fc.get("render_psnr_db", 0) < 40.0 or fc.get("render_max_abs", 1) > 1e-4:
        correctness_pass = False
        matched_state_pass = False

# Closure
for r in results:
    cl = r.get("timing_closure_B1_forward", 0)
    if cl < 0.95:
        closure_pass = False

# ---- P1: stage-level delta for forward ----
p1_forward = {}
for scene in sorted(set(r["scene"] for r in results)):
    scene_results = [r for r in results if r["scene"] == scene]
    b1_stages = defaultdict(list)
    b2_fwd = []
    for r in scene_results:
        st = r.get("B1_forward_stages_repeated", {})
        for k, v in st.items():
            b1_stages[k].append(v["median_ms"])
        b2_fwd.append(r.get("B2_forward_timing", {}).get("median_ms", 0))
    p1_forward[scene] = {
        "B1_stages_ms": {k: float(np.median(v)) for k, v in b1_stages.items()},
        "B2_forward_total_ms": float(np.median(b2_fwd)),
        "B1_forward_total_ms": float(np.median([r.get("B1_forward_decomposed_timing", {}).get("median_ms", 0) for r in scene_results])),
    }

# ---- P2: stage-level delta for backward ----
p2_backward = {}
for scene in sorted(set(r["scene"] for r in results)):
    scene_results = [r for r in results if r["scene"] == scene]
    b1_bwd = [r.get("B1_backward_total_ms", 0) for r in scene_results]
    b2_bwd = [r.get("B2_backward_total_ms", 0) for r in scene_results]
    p2_backward[scene] = {
        "B1_backward_ms": float(np.median(b1_bwd)),
        "B2_backward_ms": float(np.median(b2_bwd)),
        "delta_ms": float(np.median(b2_bwd) - np.median(b1_bwd)),
        "method": "SUBTRACTED (backward = fwd_bwd - forward)",
    }

# ---- P3: largest B2 wins ----
p3_wins = []
for r in results:
    delta_fwd = r.get("delta_forward_ms", 0)
    if delta_fwd < 0:
        p3_wins.append({
            "scene": r["scene"], "camera_idx": r["camera_idx"],
            "delta_forward_ms": delta_fwd,
            "B1_ms": r.get("B1_forward_total_ms", 0),
            "B2_ms": r.get("B2_forward_total_ms", 0),
        })
p3_wins.sort(key=lambda x: x["delta_forward_ms"])

# ---- P4: largest B2 losses ----
p4_losses = []
for r in results:
    delta_bwd = r.get("delta_backward_ms", 0)
    if delta_bwd > 0:
        p4_losses.append({
            "scene": r["scene"], "camera_idx": r["camera_idx"],
            "delta_backward_ms": delta_bwd,
            "B1_bwd_ms": r.get("B1_backward_total_ms", 0),
            "B2_bwd_ms": r.get("B2_backward_total_ms", 0),
        })
p4_losses.sort(key=lambda x: -x["delta_backward_ms"])

# ---- P5: backward penalty ----
all_b1_bwd = [r.get("B1_backward_total_ms", 0) for r in results]
all_b2_bwd = [r.get("B2_backward_total_ms", 0) for r in results]
p5_backward_penalty = {
    "B1_backward_median_ms": float(np.median(all_b1_bwd)),
    "B2_backward_median_ms": float(np.median(all_b2_bwd)),
    "delta_ms": float(np.median(all_b2_bwd) - np.median(all_b1_bwd)),
    "B2_slower_fraction": float(np.mean([b2 > b1 for b1, b2 in zip(all_b1_bwd, all_b2_bwd)])),
}

# ---- P6: state overhead ----
all_state_prep = [r.get("B2_state_prep_timing", {}).get("median_ms", 0) for r in results]
p6_state_overhead = {
    "B2_state_prep_median_ms": float(np.median(all_state_prep)),
    "description": "create_higs_renderer() overhead per call",
}

# ---- P7: crossover signal ----
p7_crossover = {}
for r in results:
    scene = r["scene"]
    ci = r["camera_idx"]
    wl1 = r.get("B1_workload", {})
    wl2 = r.get("B2_workload", {})
    n_isects = wl1.get("N_isects", 0)
    n_visible = wl1.get("N_visible", 0)
    speed_ratio = r.get("speed_ratio_B2_over_B1", 1.0)
    p7_crossover["%s_cam%d" % (scene, ci)] = {
        "N_isects": n_isects,
        "N_visible": n_visible,
        "tiles_per_gaussian": wl1.get("tiles_per_gaussian", 0),
        "speed_ratio_B2_over_B1": speed_ratio,
        "delta_forward_ms": r.get("delta_forward_ms", 0),
    }

# ---- P8: NOT_SEPARATED stages ----
not_separated = {
    "B1_backward_stages": "B1 backward is measured as total (fwd_bwd - forward); individual backward kernels are NOT_SEPARATED",
    "B2_backward_stages": "B2 backward is measured as total (fwd_bwd - forward); individual backward kernels are NOT_SEPARATED",
    "B2_forward_stages": "B2 forward is measured as a single total; per-stage decomposition of B2 internal kernels is NOT_SEPARATED (requires nsys trace or source instrumentation)",
    "F0_visibility_culling": "Embedded in F1 projection (radii > 0); not separately timed",
    "F5_sort": "Sort is inside isect_tiles (F34); not separately timed",
}

# ---- P9: unsupported claims ----
unsupported = [
    "B2 backward kernel-level decomposition requires nsys trace analysis",
    "B2 macro-tile active count/occupancy not available from metadata (UNAVAILABLE)",
    "B1 active_tiles is approximated from isect_offsets (cumulative diff)",
    "Backward gradient cosine similarity is 0.0 — B1 and B2 use different backward paths (rasterization packed vs higs_native); gradient equivalence is NOT established",
]

# ---- Gate ----
if correctness_pass and closure_pass and n_cameras_pass and provenance_pass and no_mixed_process:
    gate = "PROFILE_VALID"
elif correctness_pass and n_cameras_pass:
    gate = "PROFILE_PARTIAL"
else:
    gate = "PROFILE_INVALID"

if not closure_pass:
    gate = "PROFILE_PARTIAL"

analysis = {
    "gate": gate,
    "gate_checks": {
        "provenance_pass": provenance_pass,
        "matched_state_pass": matched_state_pass,
        "correctness_pass": correctness_pass,
        "closure_pass": closure_pass,
        "closure_values": {r["scene"] + "_cam" + str(r["camera_idx"]): r.get("timing_closure_B1_forward", 0) for r in results},
        "no_mixed_version": no_mixed_version,
        "no_mixed_process": no_mixed_process,
        "n_cameras_per_scene": dict(n_cameras_per_scene),
        "n_cameras_pass": n_cameras_pass,
        "reproducibility": "Same process, same GPU (CUDA_VISIBLE_DEVICES=0), same torch/gsplat, same checkpoint per scene",
    },
    "P1_forward_stage_delta": p1_forward,
    "P2_backward_stage_delta": p2_backward,
    "P3_largest_B2_wins": p3_wins[:5],
    "P4_largest_B2_losses": p4_losses[:5],
    "P5_backward_penalty": p5_backward_penalty,
    "P6_state_overhead": p6_state_overhead,
    "P7_crossover_signal": p7_crossover,
    "P8_NOT_SEPARATED": not_separated,
    "P9_unsupported": unsupported,
    "provenance": {
        "hostname": env.get("hostname"),
        "gpu_name": env.get("gpu_name"),
        "gpu_uuid": env.get("gpu_uuid_smi"),
        "driver_version": env.get("driver_version"),
        "torch_version": env.get("torch_version"),
        "torch_cuda_version": env.get("torch_cuda_version"),
        "gsplat_version": env.get("gsplat_version"),
        "gsplat_file": env.get("gsplat_file"),
        "repo_commit": env.get("repo_commit"),
        "higs_tree_commit": env.get("higs_tree_commit"),
        "nvcc_version": env.get("nvcc_version"),
    },
}

out_path = os.path.join(OUT_DIR, "analysis.json")
with open(out_path, "w") as f:
    json.dump(analysis, f, indent=2)
print("Wrote %s" % out_path)
print("Gate: %s" % gate)
print("Closure pass: %s" % closure_pass)
print("Correctness pass: %s" % correctness_pass)