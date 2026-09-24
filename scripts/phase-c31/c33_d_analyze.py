#!/usr/bin/env python3
"""C33-D: Workload predictability analysis from recorded observation data."""
import json, sys
import numpy as np
from scipy.stats import spearmanr as spearmanr_fn

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase-c31/c33_d_workload_data.json"
with open(path) as f:
    data = json.load(f)
recs = data["records"]
n_cam = int(data["config"]["n_cameras"])
print(f"Records: {len(recs)}")
print(f"Camera cycles: {len(recs)/n_cam:.2f}")

# Extract arrays
cam_ids = np.array([r["camera_id"] for r in recs], dtype=int)
n_vis = np.array([r["n_visible"] for r in recs], dtype=float)
n_isect = np.array([r["n_intersections"] for r in recs], dtype=float)
n_tiles = np.array([r["n_active_tiles"] for r in recs], dtype=float)
fwd = np.array([r["fwd_ms"] for r in recs], dtype=float)
bwd = np.array([r["bwd_ms"] for r in recs], dtype=float)
total_ren = np.array([r["total_render_ms"] for r in recs], dtype=float)
n_g = np.array([r["n_gaussians"] for r in recs], dtype=float)

# ─── D1: W_t vs W_(t+1) correlation ───
print("\n=== D1: Consecutive iteration correlation ===")
for name, arr in [("n_visible", n_vis), ("n_intersections", n_isect),
                  ("n_active_tiles", n_tiles), ("fwd_ms", fwd),
                  ("bwd_ms", bwd), ("total_render_ms", total_ren)]:
    r_p = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    rho, _ = spearmanr_fn(arr[:-1], arr[1:])
    print(f"  {name:20s} Pearson r={r_p:.4f}  Spearman rho={rho:.4f}")

# ─── D2: Lag-k autocorrelation ───
print("\n=== D2: Autocorrelation (lag-1,2,3,5) ===")
for name, arr in [("n_visible", n_vis), ("n_intersections", n_isect),
                  ("n_active_tiles", n_tiles), ("total_render_ms", total_ren)]:
    mu, var = float(np.mean(arr)), float(np.var(arr))
    parts = []
    for lag in [1, 2, 3, 5]:
        ac = float(np.mean((arr[:-lag] - mu) * (arr[lag:] - mu)) / var)
        parts.append(f"l{lag}={ac:.4f}")
    print(f"  {name:20s} {' | '.join(parts)}")

# ─── D3: Camera locality vs training-state locality ───
print("\n=== D3: Camera vs training-state locality ===")
# Per-camera stats from cycle 1 → predict cycle 2
cycle1 = recs[:n_cam]
cycle2 = recs[n_cam:2*n_cam] if len(recs) >= 2*n_cam else recs[n_cam:]
if len(cycle2) >= n_cam:
    # Same-camera: predict cycle2[i] from cycle1[i]
    same_errs = [abs(cycle2[i]["n_visible"] - cycle1[i]["n_visible"]) for i in range(n_cam)]
    # Cross-camera: predict cycle2[i] from cycle1[i+1] (adjacent camera)
    cross_errs = [abs(cycle2[i]["n_visible"] - cycle1[(i+1)%n_cam]["n_visible"]) for i in range(n_cam)]
    print(f"  Same-camera error (mean): {np.mean(same_errs):.0f} Gs (median: {np.median(same_errs):.0f})")
    print(f"  Adjacent-camera error:    {np.mean(cross_errs):.0f} Gs (median: {np.median(cross_errs):.0f})")
    # Random-camera prediction
    rand_errs = [abs(cycle2[i]["n_visible"] - cycle1[np.random.randint(n_cam)]["n_visible"]) for i in range(n_cam)]
    print(f"  Random-camera error:      {np.mean(rand_errs):.0f} Gs")
    print(f"  → {'Camera locality dominates' if np.median(same_errs) < np.median(cross_errs) else 'Training-state change dominates'}")
    print(f"    Same/adjacent ratio: {np.median(same_errs)/max(np.median(cross_errs),1):.2f}")

# ─── D4: Prediction error comparison ───
print("\n=== D4: Predictor comparison (n_visible) ===")
# Last-value predictor
lv_errs = np.abs(n_vis[1:] - n_vis[:-1])
print(f"  Last-value |  mean={np.mean(lv_errs):.0f}  median={np.median(lv_errs):.0f}  P90={np.percentile(lv_errs, 90):.0f}  "
      f"P95={np.percentile(lv_errs, 95):.0f}  max={np.max(lv_errs):.0f}")
print(f"  Relative:   mean={np.mean(lv_errs/(n_vis[:-1]+1))*100:.1f}%  median={np.median(lv_errs/(n_vis[:-1]+1))*100:.1f}%")

# Camera-mean predictor (from cycle 1)
if len(cycle2) >= n_cam:
    cam_means = {}
    for r in cycle1:
        c = r["camera_id"]
        cam_means[c] = cam_means.get(c, []) + [r["n_visible"]]
    cam_preds = {c: np.mean(v) for c, v in cam_means.items()}
    cam_errs = [abs(r["n_visible"] - cam_preds[r["camera_id"]]) for r in cycle2]
    print(f"  Camera-mean | mean={np.mean(cam_errs):.0f}  median={np.median(cam_errs):.0f}  P90={np.percentile(cam_errs, 90):.0f}")

# EMA predictor (alpha=0.3)
ema_errs = []; ema = n_vis[0]
for i in range(1, len(n_vis)):
    ema_errs.append(abs(n_vis[i] - ema))
    ema = 0.3 * n_vis[i] + 0.7 * ema
ema_errs = np.array(ema_errs)
print(f"  EMA(0.3)    | mean={np.mean(ema_errs):.0f}  median={np.median(ema_errs):.0f}  P90={np.percentile(ema_errs, 90):.0f}")

# ─── D5: Prediction horizon ───
print("\n=== D5: Prediction horizon (naive last-value) ===")
for h in [1, 2, 4, 8, 16]:
    errs = np.abs(n_vis[h:] - n_vis[:-h])
    print(f"  Horizon={h:3d}  median={np.median(errs):.0f}  P90={np.percentile(errs, 90):.0f}  "
          f"max={np.max(errs):.0f}  relative median={np.median(errs/(n_vis[:-h]+1))*100:.1f}%")

# ─── D6: Topology breakpoints ───
print("\n=== D6: Topology event breakpoints ===")
denf_steps = [r["step"] for r in recs if r["denf_cloned"]+r["denf_split"]+r["pruned"] > 0]
print(f"  Densification/prune steps: {denf_steps[:10]}{'...' if len(denf_steps) > 10 else ''}")
for ds in denf_steps[:5]:
    idx = next(i for i, r in enumerate(recs) if r["step"] == ds)
    if idx > 0 and idx < len(recs) - 1:
        pre = abs(recs[idx]["n_visible"] - recs[idx-1]["n_visible"])
        post = abs(recs[idx+1]["n_visible"] - recs[idx]["n_visible"])
        g_before = recs[idx-1]["n_gaussians"]
        g_after = recs[idx]["n_gaussians"]
        print(f"  Step {ds:5d}: pre-denf err={pre:6.0f}  post-denf err={post:6.0f}  "
              f"G: {g_before:,}→{g_after:,}  Δ={g_after-g_before:+}")

# ─── Render time analysis ───
print("\n=== Render time correlation ===")
print(f"  n_visible vs fwd_ms r={np.corrcoef(n_vis, fwd)[0,1]:.4f}")
print(f"  n_visible vs bwd_ms r={np.corrcoef(n_vis, bwd)[0,1]:.4f}")
print(f"  n_intersections vs total_render r={np.corrcoef(n_isect, total_ren)[0,1]:.4f}")

# ─── Summary statistics ───
print("\n=== Summary ===")
print(f"  n_visible:  mean={np.mean(n_vis):.0f}  median={np.median(n_vis):.0f}  "
      f"min={np.min(n_vis):.0f}  max={np.max(n_vis):.0f}  std={np.std(n_vis):.0f}")
print(f"  n_isect:    mean={np.mean(n_isect):.0f}  median={np.median(n_isect):.0f}  "
      f"min={np.min(n_isect):.0f}  max={np.max(n_isect):.0f}")
print(f"  total_render: mean={np.mean(total_ren):.2f}ms  median={np.median(total_ren):.2f}ms  "
      f"min={np.min(total_ren):.2f}ms  max={np.max(total_ren):.2f}ms")
print(f"  n_gaussians: final={int(n_g[-1])}  (start={int(n_g[0])})")
