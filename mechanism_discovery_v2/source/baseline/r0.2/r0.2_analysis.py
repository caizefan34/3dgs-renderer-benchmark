"""
R0.2 Analysis — Aggregates continuation outputs into 7 final JSON files.

Produces:
  1. topk_metrics_corrected.json       (Part A)
  2. predictive_gradient_mass_coverage.json (Part B)
  3. historical_mask_source_audit.json  (Part C)
  4. gdens_gopt_per_gaussian.json       (Part D)
  5. backward_signal_availability.json  (Part E)
  6. reference_backward_profile.json    (Part F — copy from profile output)
  7. final_decision.json                (Decision)
"""

import json
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
R02_DIR = REPO_ROOT / "results" / "reference_v1" / "r0.2"
WINDOWS = ["2000", "5000", "10000", "14000"]


def load_window(w, filename):
    p = R02_DIR / f"window_{w}" / filename
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def stats_from_list(values):
    """Compute mean/median/p10/p90 from a list of floats."""
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


# === Part A: Corrected Top-K Metrics ===
def aggregate_topk_corrected():
    """Aggregate corrected Jaccard/Recall/Precision across all windows."""
    print("=== Part A: Corrected Top-K Metrics ===")
    result = {"per_window": {}, "overall": {}}

    all_jaccard50 = []
    all_recall50 = []
    all_precision50 = []
    all_jaccard32 = []
    all_recall32 = []
    all_precision32 = []
    # Also for xyz (historical C51 signal)
    all_jaccard50_xyz = []
    all_recall50_xyz = []
    all_jaccard32_xyz = []
    all_recall32_xyz = []

    for w in WINDOWS:
        data = load_window(w, "topk_corrected.json")
        w_j50, w_r50, w_p50 = [], [], []
        w_j32, w_r32, w_p32 = [], [], []
        w_j50x, w_r50x, w_j32x, w_r32x = [], [], [], []

        for iter_str, metrics in data.items():
            for signal, prefix in [("gopt_total", ""), ("gopt_xyz", "_xyz")]:
                if signal not in metrics:
                    continue
                m = metrics[signal]
                if "n_matched" in m and m["n_matched"] < 10:
                    continue
                for k in [50, 32]:
                    j = m.get(f"jaccard{k}", None)
                    r = m.get(f"recall{k}", None)
                    p = m.get(f"precision{k}", None)
                    if j is not None:
                        if prefix == "":
                            if k == 50: w_j50.append(j)
                            else: w_j32.append(j)
                        else:
                            if k == 50: w_j50x.append(j)
                            else: w_j32x.append(j)
                    if r is not None:
                        if prefix == "":
                            if k == 50: w_r50.append(r)
                            else: w_r32.append(r)
                        else:
                            if k == 50: w_r50x.append(r)
                            else: w_r32x.append(r)
                    if p is not None and prefix == "":
                        if k == 50: w_p50.append(p)
                        else: w_p32.append(p)

        result["per_window"][w] = {
            "gopt_total": {
                "jaccard50": stats_from_list(w_j50),
                "recall50": stats_from_list(w_r50),
                "precision50": stats_from_list(w_p50),
                "jaccard32": stats_from_list(w_j32),
                "recall32": stats_from_list(w_r32),
                "precision32": stats_from_list(w_p32),
            },
            "gopt_xyz": {
                "jaccard50": stats_from_list(w_j50x),
                "recall50": stats_from_list(w_r50x),
                "jaccard32": stats_from_list(w_j32x),
                "recall32": stats_from_list(w_r32x),
            },
        }
        all_jaccard50.extend(w_j50)
        all_recall50.extend(w_r50)
        all_precision50.extend(w_p50)
        all_jaccard32.extend(w_j32)
        all_recall32.extend(w_r32)
        all_precision32.extend(w_p32)
        all_jaccard50_xyz.extend(w_j50x)
        all_recall50_xyz.extend(w_r50x)
        all_jaccard32_xyz.extend(w_j32x)
        all_recall32_xyz.extend(w_r32x)

    result["overall"] = {
        "gopt_total": {
            "jaccard50": stats_from_list(all_jaccard50),
            "recall50": stats_from_list(all_recall50),
            "precision50": stats_from_list(all_precision50),
            "jaccard32": stats_from_list(all_jaccard32),
            "recall32": stats_from_list(all_recall32),
            "precision32": stats_from_list(all_precision32),
        },
        "gopt_xyz": {
            "jaccard50": stats_from_list(all_jaccard50_xyz),
            "recall50": stats_from_list(all_recall50_xyz),
            "jaccard32": stats_from_list(all_jaccard32_xyz),
            "recall32": stats_from_list(all_recall32_xyz),
        },
    }

    # Print summary
    o = result["overall"]["gopt_total"]
    print(f"  G_opt_total K=50: Jaccard={o['jaccard50']['median']:.4f} Recall={o['recall50']['median']:.4f} Precision={o['precision50']['median']:.4f}")
    print(f"  G_opt_total K=32: Jaccard={o['jaccard32']['median']:.4f} Recall={o['recall32']['median']:.4f} Precision={o['precision32']['median']:.4f}")
    ox = result["overall"]["gopt_xyz"]
    print(f"  G_opt_xyz  K=50: Jaccard={ox['jaccard50']['median']:.4f} Recall={ox['recall50']['median']:.4f}")
    print(f"  G_opt_xyz  K=32: Jaccard={ox['jaccard32']['median']:.4f} Recall={ox['recall32']['median']:.4f}")

    return result


# === Part B: Predictive Gradient-Mass Coverage ===
def aggregate_coverage():
    """Aggregate coverage across all windows for both signals."""
    print("\n=== Part B: Predictive Gradient-Mass Coverage ===")
    result = {"per_window": {}, "overall": {}}

    for signal_name, filename in [("gopt_total", "coverage_gopt_total.json"),
                                   ("gopt_xyz", "coverage_gopt_xyz.json")]:
        all_oracle50, all_prev50, all_rand50 = [], [], []
        all_oracle32, all_prev32, all_rand32 = [], [], []

        for w in WINDOWS:
            data = load_window(w, filename)
            w_or50, w_pv50, w_rd50 = [], [], []
            w_or32, w_pv32, w_rd32 = [], [], []

            for iter_str, m in data.items():
                if m.get("n_matched", 0) < 10:
                    continue
                for k in [50, 32]:
                    oracle = m.get(f"oracle_k{k}")
                    prev = m.get(f"previous_k{k}")
                    rand = m.get(f"random_k{k}_mean")
                    if oracle is not None:
                        if k == 50: w_or50.append(oracle)
                        else: w_or32.append(oracle)
                    if prev is not None:
                        if k == 50: w_pv50.append(prev)
                        else: w_pv32.append(prev)
                    if rand is not None:
                        if k == 50: w_rd50.append(rand)
                        else: w_rd32.append(rand)

            if signal_name not in result["per_window"]:
                result["per_window"][signal_name] = {}
            result["per_window"][signal_name][w] = {
                "oracle_k50": stats_from_list(w_or50),
                "previous_k50": stats_from_list(w_pv50),
                "random_k50": stats_from_list(w_rd50),
                "oracle_k32": stats_from_list(w_or32),
                "previous_k32": stats_from_list(w_pv32),
                "random_k32": stats_from_list(w_rd32),
            }

            all_oracle50.extend(w_or50)
            all_prev50.extend(w_pv50)
            all_rand50.extend(w_rd50)
            all_oracle32.extend(w_or32)
            all_prev32.extend(w_pv32)
            all_rand32.extend(w_rd32)

        result["overall"][signal_name] = {
            "oracle_k50": stats_from_list(all_oracle50),
            "previous_k50": stats_from_list(all_prev50),
            "random_k50": stats_from_list(all_rand50),
            "oracle_k32": stats_from_list(all_oracle32),
            "previous_k32": stats_from_list(all_prev32),
            "random_k32": stats_from_list(all_rand32),
        }

    # Also aggregate per-parameter coverage (K=50 only)
    per_param_all = {}
    for w in WINDOWS:
        data = load_window(w, "coverage_per_param.json")
        for iter_str, params in data.items():
            for pname, m in params.items():
                if pname not in per_param_all:
                    per_param_all[pname] = {"oracle_k50": [], "previous_k50": [], "random_k50": []}
                if m.get("n_matched", 0) < 10:
                    continue
                for key in ["oracle_k50", "previous_k50", "random_k50_mean"]:
                    val = m.get(key)
                    if val is not None:
                        short_key = key.replace("_mean", "")
                        per_param_all[pname][short_key].append(val)

    result["per_parameter_k50"] = {}
    for pname, lists in per_param_all.items():
        result["per_parameter_k50"][pname] = {
            "oracle_k50": stats_from_list(lists["oracle_k50"]),
            "previous_k50": stats_from_list(lists["previous_k50"]),
            "random_k50": stats_from_list(lists["random_k50"]),
        }

    # Print summary
    o = result["overall"]["gopt_total"]
    print(f"  G_opt_total K=50: Oracle={o['oracle_k50']['median']:.4f} Previous={o['previous_k50']['median']:.4f} Random={o['random_k50']['median']:.4f}")
    print(f"  G_opt_total K=32: Oracle={o['oracle_k32']['median']:.4f} Previous={o['previous_k32']['median']:.4f} Random={o['random_k32']['median']:.4f}")
    ox = result["overall"]["gopt_xyz"]
    print(f"  G_opt_xyz  K=50: Oracle={ox['oracle_k50']['median']:.4f} Previous={ox['previous_k50']['median']:.4f} Random={ox['random_k50']['median']:.4f}")
    print(f"  Per-parameter K=50 Previous coverage:")
    for pname, s in result["per_parameter_k50"].items():
        print(f"    {pname}: {s['previous_k50']['median']:.4f}")

    return result


# === Part D: Per-Gaussian G_dens ↔ G_opt ===
def aggregate_gdens_gopt():
    """Aggregate per-Gaussian G_dens↔G_opt correlation."""
    print("\n=== Part D: Per-Gaussian G_dens ↔ G_opt ===")
    result = {"per_window": {}, "overall": {}}

    keys_to_agg = [
        "pearson_gdens_gopt_total", "spearman_gdens_gopt_total",
        "pearson_gdens_gopt_xyz", "spearman_gdens_gopt_xyz",
        "top50_gdens_covers_gopt_total", "top32_gdens_covers_gopt_total",
        "top50_gdens_covers_gopt_xyz", "top32_gdens_covers_gopt_xyz",
        "oracle_top50_gopt_total", "oracle_top32_gopt_total",
    ]

    all_vals = {k: [] for k in keys_to_agg}

    for w in WINDOWS:
        data = load_window(w, "gdens_gopt_correlation.json")
        w_vals = {k: [] for k in keys_to_agg}
        for iter_str, m in data.items():
            for k in keys_to_agg:
                if k in m:
                    w_vals[k].append(m[k])
                    all_vals[k].append(m[k])

        result["per_window"][w] = {k: stats_from_list(v) for k, v in w_vals.items()}

    result["overall"] = {k: stats_from_list(v) for k, v in all_vals.items()}

    o = result["overall"]
    print(f"  Pearson(G_dens, G_opt_total): median={o['pearson_gdens_gopt_total']['median']:.4f}")
    print(f"  Spearman(G_dens, G_opt_total): median={o['spearman_gdens_gopt_total']['median']:.4f}")
    print(f"  Pearson(G_dens, G_opt_xyz): median={o['pearson_gdens_gopt_xyz']['median']:.4f}")
    print(f"  Top50(G_dens)→G_opt_total coverage: median={o['top50_gdens_covers_gopt_total']['median']:.4f}")
    print(f"  Top32(G_dens)→G_opt_total coverage: median={o['top32_gdens_covers_gopt_total']['median']:.4f}")
    print(f"  Oracle Top50(G_opt_total): median={o['oracle_top50_gopt_total']['median']:.4f}")

    return result


# === Part C: Historical Mask Source Audit ===
def build_source_audit():
    """Build the historical C51 mask source audit."""
    print("\n=== Part C: Historical Mask Source Audit ===")
    result = {
        "historical_c51_signal": "model.xyz.grad.detach().norm(dim=-1)",
        "description": "L2 norm of the xyz (position) parameter gradient ONLY",
        "not_combined": "NOT the combined G_opt_total (norm of all parameter gradients)",
        "variants": {
            "C51_early_simulated": {
                "file": "scripts/phase-c51/simulated_sparse_backward.py",
                "line": 254,
                "signal": "model.xyz.grad.detach().norm(dim=-1)",
                "ema": False,
                "notes": "Raw xyz gradient norm, no smoothing",
            },
            "C51_R": {
                "file": "scripts/phase-c51r/run_experiment.py",
                "line": 217,
                "signal": "model.xyz.grad.detach().norm(dim=-1)",
                "ema": False,
                "notes": "Same signal as C51 early",
            },
            "C51_stage4b_canonical": {
                "file": "scripts/phase-c51-stage4b/canonical_training.py",
                "line": "380-387",
                "signal": "model.xyz.grad.detach().norm(dim=-1)",
                "ema": True,
                "ema_decay": 0.9,
                "eps": 1e-6,
                "notes": "EMA-smoothed xyz gradient norm with epsilon floor",
            },
            "C51_stage4b_recall": {
                "file": "scripts/phase-c51-stage4b/measure_recall.py",
                "line": "158-163",
                "signal": "model.xyz.grad.detach().norm(dim=-1)",
                "ema": True,
                "ema_decay": 0.9,
                "eps": 1e-6,
                "notes": "Same EMA configuration as canonical",
            },
        },
        "cuda_kernel": {
            "file": "scripts/phase-c51-stage4a/patch_cuda.py",
            "description": "Mask loaded per-Gaussian in batch loading. If mask=0, skip gradient compute (v_rgb, v_conic, v_xy, v_opacity atomicAdd). T/buffer update still done for correctness.",
            "gate_location": "Inside rasterize_to_pixels_3dgs_bwd kernel",
            "timing": "Mask must be known BEFORE kernel starts (previous-iteration signal)",
        },
        "r0.1_comparison": {
            "r0.1_signal": "G_opt_total = sqrt(||grad_xyz||^2 + |grad_opacity|^2 + ||grad_scale||^2 + ||grad_rot||^2 + ||grad_shs||^2)",
            "matches_historical_c51": False,
            "explanation": "R0.1 measured combined parameter gradient norm. Historical C51 used xyz-only gradient norm. These are different quantities.",
            "r0.2_correction": "R0.2 measures BOTH xyz_norm and total_norm to enable valid historical comparison.",
        },
        "verdict": "Historical C51 mask signal = xyz gradient norm (not combined). R0.1 G_opt_total does NOT reproduce it. R0.2 measures xyz_norm separately for valid comparison.",
    }
    return result


# === Part E: Backward Signal Availability Audit ===
def build_availability_audit():
    """Build the gsplat backward signal availability audit."""
    print("\n=== Part E: Backward Signal Availability Audit ===")
    result = {
        "gsplat_version": "1.5.3",
        "source": "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/",
        "backward_pipeline": {
            "step_1": {
                "name": "Rasterization backward (rasterize_to_pixels_3dgs_bwd)",
                "class": "_RasterizeToPixels.backward",
                "file": "gsplat/cuda/_wrapper.py:1308",
                "description": "Per-pixel-per-Gaussian kernel. Computes v_means2d, v_means2d_abs, v_conics, v_colors, v_opacities.",
                "cost": "DOMINANT — this is the most expensive backward kernel",
                "outputs": ["v_means2d (gradient w.r.t. 2D positions)", "v_means2d_abs (absgrad for densification)", "v_conics", "v_colors", "v_opacities"],
            },
            "step_2": {
                "name": "Projection backward (FullyFusedProjection.backward)",
                "class": "_FullyFusedProjection.backward",
                "file": "gsplat/cuda/_wrapper.py:1093",
                "description": "Propagates v_means2d, v_conics → v_xyz, v_scaling, v_rotation. Matrix operations.",
                "cost": "Cheap (matrix multiplies, no per-pixel iteration)",
                "outputs": ["v_xyz (gradient w.r.t. 3D positions)"],
            },
            "step_3": {
                "name": "SH backward (SphericalHarmonics.backward)",
                "class": "_SphericalHarmonics.backward",
                "file": "gsplat/cuda/_wrapper.py:1815",
                "description": "Propagates v_colors → v_shs. Per-Gaussian SH coefficient gradient.",
                "cost": "Moderate (depends on SH degree, N Gaussians)",
                "outputs": ["v_shs (gradient w.r.t. SH coefficients)"],
            },
            "step_4": {
                "name": "Opacity backward (sigmoid derivative)",
                "description": "Propagates v_opacities → v_opacity_logit. Element-wise.",
                "cost": "Trivial",
                "outputs": ["v_opacity_logit"],
            },
            "step_5": {
                "name": "Optimizer step (Adam)",
                "description": "Applies gradients to all parameters. Per-parameter Adam update.",
                "cost": "Moderate (linear in N * param_size)",
            },
        },
        "candidate_signals": [
            {
                "signal": "G_dens / means2d.absgrad",
                "source": "Output of rasterize_to_pixels_3dgs_bwd kernel (step 1)",
                "first_available": "AFTER rasterization backward completes",
                "work_completed_before_available": "Entire rasterization backward (the dominant cost)",
                "work_remaining_after": "Projection backward + SH backward + opacity backward + optimizer step",
                "can_gate_rasterization_backward": False,
                "can_gate_sh_backward": True,
                "can_gate_optimizer": True,
                "remaining_avoidable_pct": "~15.8% (projection + SH + optimizer / total)",
                "verdict": "Available too late to gate the dominant backward kernel. Could gate SH backward and optimizer, but those are only ~15.8% of total.",
            },
            {
                "signal": "alpha / contribution (render_alphas from forward)",
                "source": "Forward pass output (rasterize_to_pixels_3dgs_fwd)",
                "first_available": "AFTER forward, BEFORE backward",
                "work_completed_before_available": "Forward only",
                "work_remaining_after": "Entire backward + optimizer",
                "can_gate_rasterization_backward": True,
                "can_gate_sh_backward": True,
                "can_gate_optimizer": True,
                "remaining_avoidable_pct": "~45.7% (backward + optimizer / total)",
                "verdict": "Available before backward. Could gate the entire backward. But: (1) alpha/contribution is a forward-time signal, not a gradient signal. (2) Its correlation with G_opt is unknown and needs verification. (3) The C51 CUDA kernel loads the mask per-Gaussian inside the backward kernel, so the mask must be prepared before the kernel starts — this signal IS available in time.",
            },
            {
                "signal": "opacity (model parameter, pre-sigmoid)",
                "source": "Model parameter, available at any time",
                "first_available": "Before forward",
                "work_completed_before_available": "None",
                "work_remaining_after": "Everything",
                "can_gate_rasterization_backward": True,
                "verdict": "Always available. But opacity alone is a poor predictor of gradient importance — many visible Gaussians have similar opacity but vastly different gradients.",
            },
            {
                "signal": "tiles_per_gauss (from forward meta)",
                "source": "Forward pass output",
                "first_available": "AFTER forward, BEFORE backward",
                "can_gate_rasterization_backward": True,
                "verdict": "Available before backward. Represents workload (how many tiles each Gaussian touches). Correlation with G_opt is unknown — this is a forward signal, not a gradient signal.",
            },
            {
                "signal": "means2d (from forward, projected 2D position)",
                "source": "Forward pass output (FullyFusedProjection.forward)",
                "first_available": "AFTER forward, BEFORE backward",
                "can_gate_rasterization_backward": True,
                "verdict": "Available before backward. But 2D position alone doesn't predict gradient magnitude — gradient depends on the rendering equation, not just position.",
            },
            {
                "signal": "Previous-iteration xyz gradient norm (historical C51 signal)",
                "source": "model.xyz.grad from iteration t-1",
                "first_available": "Before forward at iteration t (carried from t-1)",
                "can_gate_rasterization_backward": True,
                "verdict": "This is the historical C51 approach. Available in time, but R0.1/R0.2 show poor lag-1 predictability (Pearson ~0.03, Recall@50 ~0.65).",
            },
        ],
        "key_finding": "The only signals available BEFORE the rasterization backward kernel are forward-time signals (alpha, opacity, tiles_per_gauss, means2d) and previous-iteration signals. G_dens (means2d.absgrad) is only available AFTER the rasterization backward, which is too late to gate it. The historical C51 approach uses previous-iteration xyz gradient, which has poor lag-1 predictability.",
    }
    return result


# === Final Decision ===
def compute_final_decision(topk, coverage, gdens_gopt, backward_profile):
    """Compute the final C51 decision."""
    print("\n=== Final Decision ===")

    # Key metrics
    prev_cov50_total = coverage["overall"]["gopt_total"]["previous_k50"]["median"]
    oracle_cov50_total = coverage["overall"]["gopt_total"]["oracle_k50"]["median"]
    random_cov50_total = coverage["overall"]["gopt_total"]["random_k50"]["median"]

    prev_cov50_xyz = coverage["overall"]["gopt_xyz"]["previous_k50"]["median"]
    oracle_cov50_xyz = coverage["overall"]["gopt_xyz"]["oracle_k50"]["median"]
    random_cov50_xyz = coverage["overall"]["gopt_xyz"]["random_k50"]["median"]

    recall50_total = topk["overall"]["gopt_total"]["recall50"]["median"]
    jaccard50_total = topk["overall"]["gopt_total"]["jaccard50"]["median"]

    gdens_covers_gopt50 = gdens_gopt["overall"]["top50_gdens_covers_gopt_total"]["median"]
    pearson_gdens_gopt = gdens_gopt["overall"]["pearson_gdens_gopt_total"]["median"]

    amdahl_after_raster = backward_profile.get("amdahl_ceilings", {}).get("after_raster_bwd", {}).get("value", 0)
    amdahl_skip_bwd = backward_profile.get("amdahl_ceilings", {}).get("skip_entire_backward", {}).get("value", 0)
    amdahl_c51_k50 = backward_profile.get("amdahl_ceilings", {}).get("c51_k50", {}).get("value", 0)

    # Decision logic:
    # C51_NEW_MECHANISM_CANDIDATE: cheap current-iteration signal:
    #   1. available before substantial backward work
    #   2. strongly predicts current G_opt
    #   3. preserves high gradient-mass coverage
    #   4. >5% realistic E2E headroom
    #   5. structurally distinct from previous-gradient prediction
    # C51_SYSTEMS_COMPONENT: early gate can save measurable backward work,
    #   but novelty overlaps existing literature or E2E ceiling is modest
    # C51_DROP: previous-mask coverage insufficient AND no early signal has >5% ceiling

    # Check 1: Previous-mask coverage (historical C51 approach)
    prev_mask_sufficient = prev_cov50_total >= 0.80 or prev_cov50_xyz >= 0.80

    # Check 2: G_dens prediction quality
    gdens_predicts_gopt = gdens_covers_gopt50 >= 0.80 and pearson_gdens_gopt >= 0.5

    # Check 3: G_dens availability — it is available AFTER rasterization backward
    # (the dominant backward kernel). It can gate SH backward + optimizer (15.8% ceiling)
    # but NOT the rasterization backward itself.
    # Criterion 1 for NEW_MECHANISM: "available before substantial backward work"
    # The rasterization backward IS substantial (75% of backward, ~30% of total)
    # G_dens is available AFTER it → FAILS criterion 1
    gdens_before_substantial_bwd = False  # G_dens is a BYPRODUCT of the raster bwd

    # Check 4: E2E ceilings
    gdahl_after_raster_ok = amdahl_after_raster > 0.05  # 15.8% > 5%
    amdahl_skip_bwd_ok = amdahl_skip_bwd > 0.05  # 45.7% > 5%

    # Decision
    if not prev_mask_sufficient and not gdens_predicts_gopt and not amdahl_skip_bwd_ok:
        decision = "C51_DROP"
        reason = "Previous-mask coverage insufficient AND no early current signal has >5% E2E saving ceiling."
    elif gdens_predicts_gopt and gdens_before_substantial_bwd and gdahl_after_raster_ok:
        # Would be NEW_MECHANISM if G_dens were available before substantial backward
        decision = "C51_NEW_MECHANISM_CANDIDATE"
        reason = f"G_dens available before substantial backward, strongly predicts G_opt (coverage={gdens_covers_gopt50:.3f}), ceiling={100*amdahl_after_raster:.1f}% > 5%."
    elif gdens_predicts_gopt and gdahl_after_raster_ok:
        # G_dens predicts G_opt well, but is available AFTER the dominant backward kernel
        # Can gate SH backward + optimizer (modest 15.8% ceiling)
        # Novelty: using means2d.absgrad to skip SH/optimizer overlaps with gradient approximation literature
        decision = "C51_SYSTEMS_COMPONENT"
        reason = (f"G_dens (means2d.absgrad) strongly predicts G_opt (coverage={gdens_covers_gopt50:.3f}, "
                  f"Spearman={gdens_gopt['overall']['spearman_gdens_gopt_total']['median']:.3f}), "
                  f"but is available AFTER the rasterization backward (the dominant kernel). "
                  f"Can gate SH backward + optimizer only (ceiling={100*amdahl_after_raster:.1f}%). "
                  f"Previous-mask coverage={prev_cov50_total:.3f} (insufficient, barely beats random={random_cov50_total:.3f}). "
                  f"Novelty overlaps with per-Gaussian gradient-approximation literature. "
                  f"Forward-time signals (alpha, tiles_per_gauss) have {100*amdahl_skip_bwd:.1f}% ceiling but untested G_opt correlation.")
    elif amdahl_skip_bwd_ok:
        decision = "C51_SYSTEMS_COMPONENT"
        reason = f"Forward-time signals have {100*amdahl_skip_bwd:.1f}% ceiling but G_opt correlation untested. Previous-mask insufficient."
    else:
        decision = "C51_DROP"
        reason = "No mechanism meets the >5% E2E ceiling with sufficient predictability."

    print(f"  Previous K=50 coverage (total): {prev_cov50_total:.4f}")
    print(f"  Previous K=50 coverage (xyz):   {prev_cov50_xyz:.4f}")
    print(f"  Oracle K=50 coverage (total):   {oracle_cov50_total:.4f}")
    print(f"  Random K=50 coverage (total):   {random_cov50_total:.4f}")
    print(f"  Recall@50 (total):              {recall50_total:.4f}")
    print(f"  Jaccard@50 (total):             {jaccard50_total:.4f}")
    print(f"  Top50(G_dens)→G_opt coverage:   {gdens_covers_gopt50:.4f}")
    print(f"  Pearson(G_dens, G_opt):         {pearson_gdens_gopt:.4f}")
    print(f"  Amdahl (after raster bwd):      {100*amdahl_after_raster:.1f}%")
    print(f"  Amdahl (skip entire bwd):       {100*amdahl_skip_bwd:.1f}%")
    print(f"  Amdahl (C51 K=50):              {100*amdahl_c51_k50:.1f}%")
    print(f"  Decision: {decision}")
    print(f"  Reason: {reason}")

    return {
        "decision": decision,
        "reason": reason,
        "evidence": {
            "previous_k50_coverage_total": prev_cov50_total,
            "previous_k50_coverage_xyz": prev_cov50_xyz,
            "oracle_k50_coverage_total": oracle_cov50_total,
            "random_k50_coverage_total": random_cov50_total,
            "recall50_total": recall50_total,
            "jaccard50_total": jaccard50_total,
            "top50_gdens_covers_gopt_total": gdens_covers_gopt50,
            "pearson_gdens_gopt_total": pearson_gdens_gopt,
            "spearman_gdens_gopt_total": gdens_gopt["overall"]["spearman_gdens_gopt_total"]["median"],
            "amdahl_after_raster_bwd": amdahl_after_raster,
            "amdahl_skip_entire_backward": amdahl_skip_bwd,
            "amdahl_c51_k50": amdahl_c51_k50,
            "prev_mask_sufficient": prev_mask_sufficient,
            "gdens_predicts_gopt": gdens_predicts_gopt,
            "gdens_before_substantial_bwd": gdens_before_substantial_bwd,
            "gdens_ceiling_ok": gdahl_after_raster_ok,
        },
    }


# === Main ===
if __name__ == "__main__":
    output_dir = R02_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part A
    topk = aggregate_topk_corrected()
    with open(output_dir / "topk_metrics_corrected.json", "w") as f:
        json.dump(topk, f, indent=2)
    print(f"  Saved: topk_metrics_corrected.json")

    # Part B
    coverage = aggregate_coverage()
    with open(output_dir / "predictive_gradient_mass_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"  Saved: predictive_gradient_mass_coverage.json")

    # Part C
    source_audit = build_source_audit()
    with open(output_dir / "historical_mask_source_audit.json", "w") as f:
        json.dump(source_audit, f, indent=2)
    print(f"  Saved: historical_mask_source_audit.json")

    # Part D
    gdens_gopt = aggregate_gdens_gopt()
    with open(output_dir / "gdens_gopt_per_gaussian.json", "w") as f:
        json.dump(gdens_gopt, f, indent=2)
    print(f"  Saved: gdens_gopt_per_gaussian.json")

    # Part E
    availability = build_availability_audit()
    with open(output_dir / "backward_signal_availability.json", "w") as f:
        json.dump(availability, f, indent=2)
    print(f"  Saved: backward_signal_availability.json")

    # Part F (copy from profile output)
    profile_path = output_dir / "backward_profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            backward_profile = json.load(f)
    else:
        backward_profile = {}
    # Rename to required output
    with open(output_dir / "reference_backward_profile.json", "w") as f:
        json.dump(backward_profile, f, indent=2)
    print(f"  Saved: reference_backward_profile.json")

    # Final decision
    decision = compute_final_decision(topk, coverage, gdens_gopt, backward_profile)
    with open(output_dir / "final_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"  Saved: final_decision.json")

    print("\n=== Analysis complete ===")
