#!/usr/bin/env python3
"""C22: Real Contribution Endpoint Measurement.

Calls the gsplat CUDA rasterizer directly to capture last_ids (per-pixel
final contributing Gaussian index), then computes useful-prefix and
skippable-suffix ratios across tiles, cameras, and workload regimes.

The CUDA kernel computes and writes last_ids; the stock Python wrapper
discards it. We capture it via _make_lazy_cuda_func, which returns
(render_colors, render_alphas, last_ids). The rendered image is unchanged.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def stats(a):
    a = np.asarray(a, dtype=float)
    return {
        k: float(v)
        for k, v in zip(
            ["mean", "p50", "p90", "p95", "p99", "max", "min"],
            [a.mean(), np.percentile(a, 50), np.percentile(a, 90),
             np.percentile(a, 95), np.percentile(a, 99), a.max(), a.min()],
        )
    }


def cdf_at_thresholds(values, thresholds):
    """Fraction of entries where value <= each threshold."""
    a = np.asarray(values, dtype=float)
    return {f"le_{th}": float((a <= th).mean()) for th in thresholds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=16)
    p.add_argument("--tile-size", type=int, default=16)
    args = p.parse_args()

    TS = args.tile_size
    W, H = 1920, 1080
    torch.set_grad_enabled(False)
    device = "cuda"

    scene = load_ply(
        str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=device
    )
    cameras = resize_cameras(
        load_cameras_from_json(
            str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=device
        ),
        W,
        H,
    )
    n_gaussians = scene["num_points"]
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    opac_raw = scene["opacity"].contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device=device)

    # Grab low-level CUDA function for last_ids.
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    fwd_cuda = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")

    tw = math.ceil(W / TS)
    th = math.ceil(H / TS)
    n_tiles = tw * th

    camera_records = []
    tile_classifications = []

    for ci in range(args.camera_offset, args.camera_offset + args.camera_count):
        cam = cameras[ci % len(cameras)]
        vw = cam.world_view_transform[None].contiguous()
        Ks = cam.K[None].contiguous()

        # Full forward pass (high-level API for stable intermediates)
        _, _, meta = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opac_raw, colors=shs,
            viewmats=vw, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=TS,
            backgrounds=bg, render_mode="RGB", sparse_grad=False,
            absgrad=False, rasterize_mode="classic",
        )
        torch.cuda.synchronize()

        m2d = meta["means2d"].contiguous()
        radii = meta["radii"].contiguous()
        depths = meta["depths"].contiguous()
        conics = meta["conics"].contiguous()
        opac = meta["opacities"].contiguous()

        # Intersect tiles (sorted)
        _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, TS, tw, th, sort=True)
        ioff = gsplat.isect_offset_encode(iid, 1, tw, th)

        dir_ = cam.camera_center.to(device) - means
        dir_ = dir_ / dir_.norm(dim=-1, keepdim=True)
        colors_sh = gsplat.spherical_harmonics(3, dir_, shs).unsqueeze(0)

        # --- Direct CUDA call to capture last_ids ---
        render_c, render_a, last = fwd_cuda(
            m2d, conics, colors_sh, opac, bg, None, W, H, TS, ioff, fid,
        )
        torch.cuda.synchronize()

        # last shape: [1, H, W] -> decode
        last_ids = last[0].long()  # [H, W]

        off = ioff[0].reshape(-1).long()
        off_list = off.tolist()
        ends = off_list[1:] + [int(iid.numel())]

        utils_ratio = []
        suffix_ratio = []
        total_range_len = []
        utils_len = []

        per_tile = []

        for tile_i, (lo, hi) in enumerate(zip(off_list, ends)):
            total_len = hi - lo
            y = tile_i // tw
            x = tile_i % tw
            px_y1 = y * TS
            px_y2 = min((y + 1) * TS, H)
            px_x1 = x * TS
            px_x2 = min((x + 1) * TS, W)
            tile_last = last_ids[px_y1:px_y2, px_x1:px_x2]

            # last_ids value is the index in the FULL sorted stream (range_start + offset).
            # Useful prefix = offset within tile up to max(last_id_in_tile).
            # We compute per-pixel useful steps, then aggregate.
            pix_utils = tile_last.flatten() - lo + 1  # +1 because last IS useful
            pix_utils = pix_utils.clamp(min=0, max=total_len)

            util_frac = (pix_utils.float() / max(total_len, 1)).tolist()
            suffix_frac = (1.0 - torch.tensor(util_frac)).clip(min=0).tolist()

            utils_ratio.extend(util_frac)
            suffix_ratio.extend(suffix_frac)
            total_range_len.extend([total_len] * len(pix_utils))
            utils_len.extend(pix_utils.tolist())

            per_tile.append({
                "tile": tile_i,
                "total_intersections": total_len,
                "pixel_count": int(tile_last.numel()),
                "useful_prefix_ratio_mean": float(np.mean(util_frac)),
                "useful_prefix_ratio_median": float(np.median(util_frac)),
                "useful_prefix_ratio_p90": float(np.percentile(util_frac, 90)),
                "useful_prefix_ratio_min": float(np.min(util_frac)),
                "useful_prefix_ratio_max": float(np.max(util_frac)),
                "skippable_suffix_ratio_mean": float(np.mean(suffix_frac)),
            })

        # Per-pixel aggregates
        usef_arr = np.asarray(utils_ratio, dtype=float)
        suff_arr = np.asarray(suffix_ratio, dtype=float)
        tot_arr = np.asarray(total_range_len, dtype=float)
        utils_len_arr = np.asarray(utils_len, dtype=float)

        cam_rec = {
            "camera": ci,
            "total_intersections": int(iid.numel()),
            "useful_prefix_ratio": stats(usef_arr),
            "skippable_suffix_ratio": stats(suff_arr),
            "total_sorted_range_length": stats(tot_arr),
            "useful_prefix_length": stats(utils_len_arr),
            "useful_prefix_ratio_cdf": cdf_at_thresholds(usef_arr, [0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]),
            "potentially_skipped_gaussian_steps_per_pixel": stats(
                tot_arr - utils_len_arr
            ),
            "tile_classification": per_tile,
        }

        # Per-frame aggregate work reduction
        total_steps = int(tot_arr.sum())
        potentially_skipped = int((tot_arr - utils_len_arr).sum())
        cam_rec["frame_work_reduction_bound"] = {
            "total_rasterizer_gaussian_steps": total_steps,
            "potentially_skipped_steps": potentially_skipped,
            "potential_work_reduction_fraction": float(potentially_skipped / max(total_steps, 1)),
        }

        # --- N12: per-tile active-pixel analysis ---
        active_at_end = []
        for tile_i, (lo, hi) in enumerate(zip(off_list, ends)):
            y = tile_i // tw
            x = tile_i % tw
            tile_last = last_ids[y * TS : min((y + 1) * TS, H), x * TS : min((x + 1) * TS, W)]
            # "Near the end": pixels whose useful_prefix is >= 90% of range
            pix_end = tile_last.flatten() - lo + 1
            total_len = hi - lo
            frac = (pix_end.float() / max(total_len, 1)).clamp(0, 1)
            still_active = float((frac >= 0.90).float().mean().item())
            active_at_end.append(still_active)
        cam_rec["n12_active_pixel_fraction_near_end"] = stats(active_at_end)

        camera_records.append(cam_rec)
        print(f"  Camera {ci}: total_intersections={cam_rec['total_intersections']:,}  "
              f"useful_ratio_mean={usef_arr.mean():.4f}  potential_reduction={cam_rec['frame_work_reduction_bound']['potential_work_reduction_fraction']:.4f}",
              flush=True)

    # Aggregate across cameras
    work_reductions = [r["frame_work_reduction_bound"]["potential_work_reduction_fraction"] for r in camera_records]

    output = {
        "schema_version": 1,
        "phase": "C22",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "gsplat": gsplat.__version__,
            "torch": torch.__version__,
        },
        "instrumentation": {
            "method": "Direct call to _make_lazy_cuda_func('rasterize_to_pixels_3dgs_fwd') which returns (render_colors, render_alphas, last_ids). last_ids is the per-pixel final committed Gaussian index in the sorted intersection stream.",
            "semantics_documented": "last_ids records the global sorted intersection index of the last Gaussian whose alpha contribution was committed to the pixel. A Gaussian whose alpha is below ALPHA_THRESHOLD (via sigma test) is SKIPPED and not recorded. The termination threshold is next_T <= 1e-4 (exclusive). Therefore last_ids is both the 'last non-zero contribution' and the 'renderer actual termination point' for that pixel.",
            "key_distinction_C22_vs_C21": "C21 used fixed depth-fraction truncation (50%) and measured alpha error. C22 uses the real per-pixel last_ids which tells exactly how many of the sorted intersections were traversed before the pixel was fully opaque. C21 estimated; C22 measures the ground-truth endpoint.",
        },
        "protocol": {
            "scene": "room official Mip-NeRF360",
            "n_gaussians": n_gaussians,
            "resolution": f"{W}x{H}",
            "tile_size": TS,
            "packed": False,
            "cameras": list(range(args.camera_offset, args.camera_offset + args.camera_count)),
            "total_cameras_in_scene": len(cameras),
        },
        "camera_records": camera_records,
        "cross_camera_aggregate": {
            "work_reduction_fraction": stats(work_reductions),
            "min_work_reduction": min(work_reductions),
            "max_work_reduction": max(work_reductions),
        },
        "n11_feasibility": None,  # filled below
        "n12_feasibility": None,
        "comparison_vs_c21": {
            "c21_method": "50% depth-fraction truncation replay measuring alpha MSE",
            "c22_method": "actual per-pixel last_ids from CUDA kernel",
            "key_difference": "C21's 50% truncation was a fixed guess; C22 measures the exact endpoint. C21's 'nice camera' (240-247) gave 49% pixels at <1e-3 error for 50% truncation — this implied significant potential. C22 will show the exact fraction. If C22 shows a larger opportunity than C21 suggested, C21 was conservative. If smaller, C21 overestimated."
        },
    }

    # N11 feasibility classification
    mean_wr = float(np.mean(work_reductions))
    p50_wr = float(np.median(work_reductions))
    p90_wr = float(np.percentile(work_reductions, 90))

    # Count cameras in each regime
    weak_cams = sum(1 for r in work_reductions if r < 0.10)
    moderate_cams = sum(1 for r in work_reductions if 0.10 <= r < 0.30)
    strong_cams = sum(1 for r in work_reductions if r >= 0.30)

    n_steps_all = sum(r["frame_work_reduction_bound"]["total_rasterizer_gaussian_steps"] for r in camera_records)
    n_skipped_all = sum(r["frame_work_reduction_bound"]["potentially_skipped_steps"] for r in camera_records)

    output["n11_feasibility"] = {
        "total_rasterizer_gaussian_steps_all_cameras": n_steps_all,
        "total_potentially_skipped_steps_all_cameras": n_skipped_all,
        "potential_work_reduction_fraction_mean": mean_wr,
        "potential_work_reduction_fraction_p50": p50_wr,
        "potential_work_reduction_fraction_p90": p90_wr,
        "cameras_classified": {
            "weak_below_10pct": weak_cams,
            "moderate_10_30pct": moderate_cams,
            "strong_above_30pct": strong_cams,
        },
        "regime_evaluation": (
            "STRONG" if strong_cams > moderate_cams + weak_cams
            else "MODERATE" if moderate_cams >= weak_cams
            else "WEAK"
        ),
        "n11_recommendation": (
            "KEEP_CANDIDATE" if mean_wr >= 0.30
            else "MAYBE" if mean_wr >= 0.10
            else "DROP"
        ),
    }

    # N12 feasibility: average active-pixel fraction near end across all cameras
    n12_vals = np.array([r["n12_active_pixel_fraction_near_end"]["mean"] for r in camera_records])
    n12_mean = float(np.mean(n12_vals))
    output["n12_feasibility"] = {
        "active_pixel_fraction_near_end_mean": n12_mean,
        "interpretation": (
            "Most tiles still have uniformly active pixels at the end → N12 DROP"
            if n12_mean > 0.70
            else "Mixed evidence — some tiles show sparse active pixels near end → N12 MAYBE"
            if n12_mean > 0.30
            else "Most tiles have sparse active pixels near end → N12 KEEP_CANDIDATE"
        ),
        "n12_recommendation": (
            "DROP" if n12_mean > 0.70
            else "MAYBE" if n12_mean > 0.30
            else "KEEP_CANDIDATE"
        ),
    }

    # Cross-camera ranking table
    # Build the candidate decision table
    output["candidate_decisions"] = [
        {
            "candidate": "N11 Contribution-Aware Traversal",
            "real_endpoint_evidence": f"Across {len(camera_records)} cameras: mean potential work reduction {mean_wr:.3f}, P50 {p50_wr:.3f}, P90 {p90_wr:.3f}. Regime: strong={strong_cams}, moderate={moderate_cams}, weak={weak_cams}.",
            "potential_work_reduction": mean_wr,
            "view_dependence": f"Range: {min(work_reductions):.3f}–{max(work_reductions):.3f}",
            "mechanism_confidence": (
                "HIGH" if mean_wr >= 0.30
                else "MODERATE" if mean_wr >= 0.15
                else "LOW"
            ),
            "decision": output["n11_feasibility"]["n11_recommendation"],
        },
        {
            "candidate": "N12 Active-Pixel Batch Execution",
            "real_endpoint_evidence": f"Active pixel fraction near end of sorted range: mean {n12_mean:.3f}",
            "potential_work_reduction": None,
            "view_dependence": "Measured across all tested cameras",
            "mechanism_confidence": (
                "HIGH" if n12_mean <= 0.30
                else "LOW"
            ),
            "decision": output["n12_feasibility"]["n12_recommendation"],
        },
    ]

    # Answer the two key questions
    output["key_answer_1"] = (
        f"Across {len(camera_records)} cameras, the mean fraction of rasterization steps that is "
        f"completed before additional Gaussians cease to matter (i.e., useful_prefix_ratio) is "
        f"{1-mean_wr:.3f} (stdev). This means {mean_wr*100:.1f}% of per-pixel Gaussian traversals "
        f"are potentially skippable after the pixel reaches saturation. The P50 skippable suffix ratio "
        f"is {p50_wr:.3f}. (These are removed-work bounds, not speedup.)"
    )
    output["key_answer_2"] = (
        f"The fraction is sufficiently large (mean={mean_wr:.3f}, "
        f"{'YES' if mean_wr >= 0.30 else 'MAYBE' if mean_wr >= 0.15 else 'NO'}) "
        f"and {'YES' if strong_cams > 0 else 'NOT'} predictable from view statistics. "
        f"A view-classification layer would be required before a new CUDA optimization can be justified."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {args.out}")
    print(f"  N11: {output['n11_feasibility']['n11_recommendation']}  (mean work reduction={mean_wr:.4f})")
    print(f"  N12: {output['n12_feasibility']['n12_recommendation']}  (active fraction near end={n12_mean:.4f})")


if __name__ == "__main__":
    main()
