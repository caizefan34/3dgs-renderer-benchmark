#!/usr/bin/env python3
"""
Phase 8E — Per-Kernel Forward CUDA Timing v5 (Final)

Measure 5 clean forward stages in a single rasterization-equivalent pass:
  1. projection     — fully_fused_projection (1 CUDA kernel)
  2. sh_eval         — spherical_harmonics (1 CUDA kernel)
  3. intersect_sort  — isect_tiles(sort=True) = pass1 + CPU cumsum + pass2 + radix_sort
  4. offset          — isect_offset_encode (1 CUDA kernel)
  5. rasterize       — rasterize_to_pixels (1 CUDA kernel)

Sort isolation in a SEPARATE phase with GPU cooldown:
  - sort_false_times = isect_tiles(sort=False) * 10  [pass1+cumsum+pass2]
  - GPU cooldown (5s idle)
  - sort_true_times  = isect_tiles(sort=True)  * 10  [pass1+cumsum+pass2+radix_sort]
  - sort_estimated = median(true) - median(false)
  
For tile16 and tile32 across 6 frozen room checkpoints (iter5000–iter30000).

Protocol:
  - Single process, single CUDA stream
  - Same fixed camera (1920×1080, pinhole, fov=50°), same frozen Gaussian state
  - 3 warmup, 10 measurement repeats per stage
  - torch.cuda.Event around each stage, synchronize before event read
  - Separate sort isolation with 5s GPU cooldown between sort-false and sort-true
  - GPU temperature monitoring before/after each checkpoint
"""

import json, math, sys, os, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)

DEVICE = "cuda"
DTYPE = torch.float32
CKPT_DIR = REPO_ROOT / "results" / "epic05" / "phase7"
OUT_DIR = REPO_ROOT / "results" / "epic05"
REPORT_DIR = REPO_ROOT / "reports" / "epic05"

CKPT_MAP = {
    "room_iter5000":  "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt",
    "room_iter10000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter10000.pt",
    "room_iter15000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter15000.pt",
    "room_iter20000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter20000.pt",
    "room_iter25000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter25000.pt",
    "room_iter30000": "phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter30000.pt",
}

WARMUP = 3
N_REPEAT = 10  # repeats for main 5 stages
SORT_REPEAT = 10  # repeats for sort isolation

COOLDOWN_S = 5  # seconds between sort-false and sort-true phases


def make_camera(W=1920, H=1080):
    fx = fy = W / (2.0 * math.tan(math.radians(25)))
    viewmat = torch.eye(4, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    t = torch.eye(4, device=DEVICE, dtype=DTYPE)
    t[2, 3] = -5.0
    viewmat[0] = t
    K = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device=DEVICE, dtype=DTYPE,
    ).unsqueeze(0)
    return viewmat, K, W, H


def load_ckpt(path):
    cp = torch.load(path, map_location=DEVICE, weights_only=False)
    ms = cp["model_state"]
    opac = ms["opacity"].detach().clone()
    if opac.dim() == 2 and opac.shape[1] == 1:
        opac = opac.squeeze(1)
    return {
        "xyz": ms["xyz"].detach().clone(),
        "rotations": ms["rotations"].detach().clone(),
        "scales": ms["scales"].detach().clone(),
        "opacity": opac,
        "shs": ms["shs"].detach().clone(),
        "sh_degree": ms["sh_degree"],
    }


def robust_stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "cv": float(np.std(arr, ddof=1) / np.mean(arr)) if np.mean(arr) > 0 and len(arr) > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n": int(len(arr)),
    }


def gpu_metric(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return None


# ════════════════════════════════════════════
# Main 5-stage timing (single pass)
# ════════════════════════════════════════════

def time_5_stages_single_pass(params, viewmat, K, W, H, ts):
    """Measure 5 forward stages in sequence using torch.cuda.Event.
    
    Returns {stage: [ms, ...]} per stage across N_REPEAT runs.
    The pipeline is re-run for each repeat — no state carried over.
    """
    xyz = params["xyz"]; rot = params["rotations"]; scl = params["scales"]
    opa = params["opacity"]; shs = params["shs"]; sh_deg = params["sh_degree"]
    B, C = 1, 1
    N = xyz.shape[0]
    I = B * C
    tw = math.ceil(W / float(ts))
    th = math.ceil(H / float(ts))
    campos = torch.inverse(viewmat)[:, :3, 3]

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    times = {k: [] for k in ["projection", "sh_eval", "intersect_sort", "offset", "rasterize"]}

    total_rounds = WARMUP + N_REPEAT
    for rnd in range(total_rounds):
        # ── 1. Projection ──
        ev_s.record()
        proj_res = fully_fused_projection(
            xyz, None, rot, scl, viewmat, K, W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=True, sparse_grad=False, calc_compensations=False,
            camera_model="pinhole", opacities=opa,
        )
        ev_e.record()
        torch.cuda.synchronize()
        if rnd >= WARMUP:
            times["projection"].append(ev_s.elapsed_time(ev_e))

        (bi, ci, gi, radii, means2d, depths, conics, _) = proj_res
        op_packed = opa.view(B, N)[bi, gi]
        image_ids = bi * C + ci

        # ── 2. SH evaluation ──
        dirs = (xyz.view(B, N, 3)[bi, gi] - campos.view(B, C, 3)[bi, ci])
        masks = (radii > 0).all(dim=-1)
        shs_view = shs.view(B, N, -1, 3)[bi, gi]
        ev_s.record()
        colors = spherical_harmonics(sh_deg, dirs, shs_view, masks=masks)
        colors = torch.clamp_min(colors + 0.5, 0.0)
        ev_e.record()
        torch.cuda.synchronize()
        if rnd >= WARMUP:
            times["sh_eval"].append(ev_s.elapsed_time(ev_e))

        # ── 3. Intersect + sort (pass1 + cumsum + pass2 + radix_sort) ──
        ev_s.record()
        tpg, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, ts, tw, th,
            sort=True, segmented=False, packed=True, n_images=I,
            image_ids=image_ids, gaussian_ids=gi,
        )
        ev_e.record()
        torch.cuda.synchronize()
        if rnd >= WARMUP:
            times["intersect_sort"].append(ev_s.elapsed_time(ev_e))

        # ── 4. Offset ──
        ev_s.record()
        isect_offsets = isect_offset_encode(isect_ids, I, tw, th)
        isect_offsets = isect_offsets.view(B, C, th, tw)
        ev_e.record()
        torch.cuda.synchronize()
        if rnd >= WARMUP:
            times["offset"].append(ev_s.elapsed_time(ev_e))

        # ── 5. Rasterize ──
        ev_s.record()
        render_colors, render_alphas = rasterize_to_pixels(
            means2d, conics, colors, op_packed,
            W, H, ts, isect_offsets, flatten_ids,
            backgrounds=None, masks=None, packed=True, absgrad=False,
        )
        ev_e.record()
        torch.cuda.synchronize()
        if rnd >= WARMUP:
            times["rasterize"].append(ev_s.elapsed_time(ev_e))

    return times


# ════════════════════════════════════════════
# Sort isolation (separate, with cooldown)
# ════════════════════════════════════════════

def time_sort_isolated(params, viewmat, K, W, H, ts):
    """Separate sort isolation with GPU cooldown between false and true phases.
    
    Phase A: isect_tiles(sort=False) * SORT_REPEAT  (pass1+cumsum+pass2)
    Cooldown: 5 seconds idle
    Phase B: isect_tiles(sort=True)  * SORT_REPEAT  (+ radix_sort)
    
    Returns (false_times_ms, true_times_ms, sort_estimated_ms)
    """
    xyz = params["xyz"]; rot = params["rotations"]; scl = params["scales"]
    opa = params["opacity"]; shs = params["shs"]; sh_deg = params["sh_degree"]
    B, C = 1, 1
    N = xyz.shape[0]
    I = B * C
    tw = math.ceil(W / float(ts))
    th = math.ceil(H / float(ts))
    campos = torch.inverse(viewmat)[:, :3, 3]

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    def run_intersect(sort_flag, n_repeat):
        results = []
        for rep in range(WARMUP + n_repeat):
            pr = fully_fused_projection(
                xyz, None, rot, scl, viewmat, K, W, H,
                eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
                packed=True, sparse_grad=False, calc_compensations=False,
                camera_model="pinhole", opacities=opa,
            )
            (bi, ci, gi, radii, means2d, depths, conics, _) = pr
            op_packed = opa.view(B, N)[bi, gi]
            image_ids = bi * C + ci

            # SH eval (might be needed for the intersect call? No, but keeping same state)
            dirs = (xyz.view(B, N, 3)[bi, gi] - campos.view(B, C, 3)[bi, ci])
            masks = (radii > 0).all(dim=-1)
            shs_view = shs.view(B, N, -1, 3)[bi, gi]
            colors = spherical_harmonics(sh_deg, dirs, shs_view, masks=masks)
            colors = torch.clamp_min(colors + 0.5, 0.0)

            ev_s.record()
            tpg, isect_ids, flatten_ids = isect_tiles(
                means2d, radii, depths, ts, tw, th,
                sort=sort_flag, segmented=False, packed=True, n_images=I,
                image_ids=image_ids, gaussian_ids=gi,
            )
            ev_e.record()
            torch.cuda.synchronize()
            if rep >= WARMUP:
                results.append(ev_s.elapsed_time(ev_e))
        return results

    # Phase A: sort=False (pass1+cumsum+pass2 only)
    no_sort_times = run_intersect(False, SORT_REPEAT)

    # COOLDOWN
    temp_a = gpu_metric("temperature.gpu")
    time.sleep(COOLDOWN_S)
    torch.cuda.empty_cache()
    _ = torch.randn(1, device=DEVICE)  # touch GPU to ensure stable state

    # Phase B: sort=True (pass1+cumsum+pass2+radix_sort)
    with_sort_times = run_intersect(True, SORT_REPEAT)

    # Estimated sort time (median-based to be robust)
    false_med = float(np.median(no_sort_times))
    true_med = float(np.median(with_sort_times))
    sort_est = max(0.0, true_med - false_med)

    return no_sort_times, with_sort_times, sort_est


# ════════════════════════════════════════════
# Integrity audit
# ════════════════════════════════════════════

def audit_timing(stage_times):
    audit = {}
    for stage_name, times in stage_times.items():
        arr = np.array(times)
        if len(arr) < 2:
            continue
        sd = np.std(arr, ddof=1)
        audit[stage_name] = {
            "cv": float(sd / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "range_ms": float(np.max(arr) - np.min(arr)),
            "bimodal_suspected": bool(
                len(arr) >= 4 and sd > 0 and (np.max(arr) - np.min(arr)) > 3.0 * sd
            ),
        }
    return audit


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def main():
    print("=" * 75)
    print("Phase 8E — Per-Kernel Forward CUDA Timing v5 (Final)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Main: 5 stages, {WARMUP}+{N_REPEAT} rounds each")
    print(f"Sort isolation: {COOLDOWN_S}s cooldown between phases")
    print(f"Resolution: 1920×1080, pinhole, packed=True, SH deg=3")
    print("=" * 75)

    viewmat, K, W, H = make_camera()
    temp0 = gpu_metric("temperature.gpu")
    clock0 = gpu_metric("clocks.gr")
    print(f"\nGPU baseline: temp={temp0}°C, clock={clock0} MHz")

    results = {}

    for ckpt_name in sorted(CKPT_MAP.keys()):
        ckpt_path = CKPT_DIR / CKPT_MAP[ckpt_name]
        if not ckpt_path.exists():
            print(f"\n  SKIP {ckpt_name}: not found at {ckpt_path}")
            continue

        params = load_ckpt(str(ckpt_path))
        N = params["xyz"].shape[0]
        temp_ckpt = gpu_metric("temperature.gpu")
        print(f"\n{'=' * 55}")
        print(f"  {ckpt_name}: N={N:,}  sh_deg={params['sh_degree']}  GPU={temp_ckpt}°C")

        entry = {"num_gaussians": N, "sh_degree": params["sh_degree"], "tile_sizes": {}}

        for ts in [16, 32]:
            label = f"tile{ts}"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # ── Main 5-stage timing ──
            stage_times = time_5_stages_single_pass(params, viewmat, K, W, H, ts)

            # ── Sort isolation (separate phase) ──
            ns_times, ws_times, sort_est = time_sort_isolated(params, viewmat, K, W, H, ts)

            temp_post = gpu_metric("temperature.gpu")

            # ── Compute stats ──
            stats = {}
            for sn, tl in stage_times.items():
                stats[sn] = robust_stats(tl)
                s = stats[sn]
                print(f"\n    {sn:20s}:")
                print(f"      median={s['median_ms']:8.3f}ms  mean={s['mean_ms']:8.3f}±{s['std_ms']:.3f}  "
                      f"CV={s['cv']:.4f}  [{s['min_ms']:.3f}, {s['max_ms']:.3f}]")

            stats["intersect_no_sort"] = robust_stats(ns_times)
            stats["intersect_no_sort"]["note"] = "isect_tiles(sort=False): pass1+cumsum+pass2"
            stats["intersect_with_sort"] = robust_stats(ws_times)
            stats["intersect_with_sort"]["note"] = "isect_tiles(sort=True): pass1+cumsum+pass2+radix_sort"
            stats["sort_estimated"] = {
                "median_ms": sort_est,
                "note": "median(intersect_with_sort) - median(intersect_no_sort)",
            }
            s_ns = stats["intersect_no_sort"]
            s_ws = stats["intersect_with_sort"]
            print(f"    {'intersect_no_sort':20s}: median={s_ns['median_ms']:8.3f}ms")
            print(f"    {'intersect_with_sort':20s}: median={s_ws['median_ms']:8.3f}ms")
            print(f"    {'sort_estimated':20s}:   median={sort_est:8.3f}ms")

            # Forward total
            fwd_stages = ["projection", "sh_eval", "intersect_sort", "offset", "rasterize"]
            fwd_total = sum(stats[s]["median_ms"] for s in fwd_stages)
            stats["_forward_total"] = {"median_ms": fwd_total}
            print(f"    {'FORWARD TOTAL':20s}: {fwd_total:.3f}ms  (sum of medians)")

            # Audit
            audit = audit_timing(stage_times)

            entry["tile_sizes"][label] = {
                "stats": stats,
                "audit": audit,
                "forward_total_median_ms": fwd_total,
                "temp_before_main_c": temp_ckpt,
                "temp_after_sort_c": temp_post,
            }

            print(f"    GPU temp: {temp_ckpt}°C → {temp_post}°C")

        # ── Ratios ──
        t16 = entry["tile_sizes"]["tile16"]
        t32 = entry["tile_sizes"]["tile32"]
        ratios = {}

        stage_list = [
            "projection", "sh_eval", "intersect_sort", "offset", "rasterize",
            "intersect_no_sort", "intersect_with_sort", "sort_estimated",
        ]
        print("")
        for sn in stage_list:
            if sn not in t16["stats"] or sn not in t32["stats"]:
                continue
            m16 = t16["stats"][sn]["median_ms"]
            m32 = t32["stats"][sn]["median_ms"]
            r = m16 / m32 if m32 > 1e-6 else float("inf")
            fwd16 = t16["forward_total_median_ms"]
            fwd32 = t32["forward_total_median_ms"]
            pct16 = m16 / fwd16 * 100 if fwd16 > 0 else 0
            pct32 = m32 / fwd32 * 100 if fwd32 > 0 else 0
            ratios[sn] = {
                "t16_median_ms": m16,
                "t32_median_ms": m32,
                "ratio_t16_t32": r,
                "t16_pct_of_forward": round(pct16, 2),
                "t32_pct_of_forward": round(pct32, 2),
            }
            print(f"    RATIO {sn:20s}: {r:8.2f}x  "
                  f"(t16={m16:8.3f}ms [{pct16:.0f}%], t32={m32:8.3f}ms [{pct32:.0f}%])")

        fwd_r = t16["forward_total_median_ms"] / max(t32["forward_total_median_ms"], 0.001)
        ratios["forward_total"] = {
            "t16_median_ms": t16["forward_total_median_ms"],
            "t32_median_ms": t32["forward_total_median_ms"],
            "ratio_t16_t32": fwd_r,
        }
        print(f"    RATIO {'forward_total':20s}: {fwd_r:8.2f}x  "
              f"(t16={t16['forward_total_median_ms']:.3f}ms, t32={t32['forward_total_median_ms']:.3f}ms)")

        entry["ratios"] = ratios
        results[ckpt_name] = entry
        torch.cuda.empty_cache()

    temp1 = gpu_metric("temperature.gpu")
    clock1 = gpu_metric("clocks.gr")
    print(f"\nGPU final: temp={temp1}°C, clock={clock1} MHz")

    results["_meta"] = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "warmup": WARMUP,
            "repeat": N_REPEAT,
            "sort_repeat": SORT_REPEAT,
            "cooldown_s": COOLDOWN_S,
        },
        "gpu_audit": {
            "temp_before_c": temp0,
            "temp_after_c": temp1,
            "clock_before_mhz": clock0,
            "clock_after_mhz": clock1,
        },
    }

    out_json = OUT_DIR / "phase8e_forward_kernel_timing.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")
    print("=" * 75)
    print("Phase 8E complete.")


if __name__ == "__main__":
    main()
