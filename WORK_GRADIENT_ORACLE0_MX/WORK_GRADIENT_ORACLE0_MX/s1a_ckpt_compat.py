#!/usr/bin/env python3
"""S1A_CHECKPOINT_COMPATIBILITY: record convention findings + render sanity verdict."""
import json

out = {
  "schema": "S1A_CHECKPOINT_COMPATIBILITY",
  "phase": "S1A",
  "primary_checkpoint": {
    "scene": "bicycle", "run": "fp1_bicycle_2026-10-06-01-01-33",
    "file": "0005000.pt", "N": 2456697, "iteration": 5000, "sh_degree": 3,
    "role": "EARLY_5K (priority per task)"
  },
  "secondary_checkpoint": {
    "scene": "bicycle", "file": "0010000.pt", "N": 3849060, "iteration": 10000,
    "role": "MIDDLE_10K (resource-permitting)"
  },
  "parameter_activations": {
    "opacity": "sigmoid(_opacities); checkpoint stores logit [N,1]",
    "scale": "exp(_scales); checkpoint stores log-scale [N,3]",
    "rotation": "normalize(_rotations); quaternion [N,4]",
    "means": "raw world position [N,3]; NORMALIZE_RECENTER=false so no recenter",
    "sh": "standard gsplat SH; colors=cat([sh_coefficients_0 (DC), sh_coefficients_rest]) [N,16,3]"
  },
  "pca_convention": {
    "apply_pca_config": True,
    "pca_applied_in_renderer": False,
    "evidence": "grep for pca/PCA across faster-gaussian-splatting/ and FasterGSFusedCudaBackend/ returns ZERO hits. APPLY_PCA is a parsed-but-unused config field.",
    "verdict": "PCA does NOT rotate colors; colors are standard gsplat SH (base 0.5, C0=0.2820947917). Confirmed by render brightness matching GT."
  },
  "camera_convention": {
    "json_rotation": "c2w (camera-to-world), OpenCV/COLMAP; position = camera center in world",
    "generator": "_mx_colmap_to_json.py: R_w2c from COLMAP qvec; position = -R_w2c.T @ t_w2c; rotation_c2w = R_w2c.T",
    "benchmark_load": "Rw2c=R.T; vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p (standard c2w->w2c)",
    "nerfidg_internal": "uses colmap2opengl=diag(1,-1,-1,1) on poses for its own rasterizer"
  },
  "geometry_projection_diagnostic": {
    "means_bbox_x": [-83.4, 117.4], "means_bbox_y": [-42.1, 4.0], "means_bbox_z": [-64.7, 58.8],
    "frac_in_front_opencv_z": 0.761,
    "in_bounds_projected": 659493,
    "verdict": "geometry IS in renderable COLMAP world space; as-is benchmark camera places 76% of gaussians in front (OpenCV +z). NOT a coordinate-flip problem."
  },
  "render_sanity_test": {
    "method": "gsplat rasterize_gaussian_higs_dynamic, bicycle 5K ckpt (sub 1.2M), cam id=0, 1236x821, sh_degree=3, real GT image",
    "best_variant_corr": 0.2942,
    "best_variant": "flip_z (cam z-flip)",
    "as_is_corr": 0.0984,
    "full_grid_max_corr": 0.2942,
    "all_16_combos_max": 0.2942,
    "verdict": "FAIL - gsplat HiGS renderer does NOT reproduce FasterGS training render (max corr 0.30 across all 16 means/camera axis-flip combos). Geometry projects correctly but appearance/structure does not match."
  },
  "fastergs_native_backend": {
    "path": "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend/_C.cpython-310-x86_64-linux-gnu.so",
    "importable": True,
    "abi": "imports with higs-13scene-env torch 2.9.1+cu128 after preloading libc10/libtorch",
    "ops": ["forward", "backward"],
    "signature": "6 main tensors + 6 moment tensors + densify_info + RasterizerSettings tuple (w2c, cam_position, bg, active_sh_bases, W, H, fx, fy, cx, cy, near, far, current_mean_lr, adam_step_count)",
    "direct_forward_attempt": "arg-count mismatch; requires exact moments/settings layout - not driven to completion",
    "verdict": "native trainer rasterizer present and ABI-loadable but not yet driven to a successful render"
  },
  "gate_decision": {
    "render_sanity_passed": False,
    "fallback_applied": "Per task section 2: when PCA/convention cannot be confirmed to reproduce training renders, continue PURE-MATH HT smoke test only; do NOT perform formal image-reconstruction gradient-value comparison; do NOT issue scientific GO.",
    "what_remains_valid": "S1A-0 HT gradient-estimator mathematical correctness (renderer-agnostic; validates unbiasedness of HT tile-gradient sampling for ANY fixed loss landscape).",
    "what_is_blocked": "S1A-1 gradient-VALUE comparison and S1A-2 GPU-cost comparison tied to real image reconstruction quality (require a render that matches GT).",
    "status": "S1A_BLOCKED_INPUT (render sanity not passed); S1A-0 pure-math proceeds"
  }
}
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_CHECKPOINT_COMPATIBILITY.json","w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["gate_decision"], indent=2))
print("saved S1A_CHECKPOINT_COMPATIBILITY.json")
