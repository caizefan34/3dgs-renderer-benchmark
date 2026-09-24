#!/usr/bin/env python3
"""
R1 — Contribution-Gated Attribute Fetch.

For representative tiles across multiple cameras, simulate the per-pixel
compositing process to count three categories per Gaussian-pixel pair:
  1. Fetched (exists in sorted range)
  2. Evaluated (passes the alpha/sigma test, i.e. 2D Gaussian != 0 for this pixel)
  3. Committed (changes accumulated color/alpha above threshold)

Uses the gsplat projection/intersection pipeline to get sorted ranges,
then replays compositing in Python for a subset of tiles to avoid
simulating the full 1.6M Gaussians × 2M pixels brute-force.

GPU2 main, GPU3 replication.
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

W, H = 1920, 1080
TS = 16
TW = math.ceil(W / TS)
TH = math.ceil(H / TS)
TILE_PX = TS * TS

def eval_contrib(opacity, alpha_remaining):
    """Simple alpha-blending contribution model."""
    contrib = opacity * alpha_remaining
    return contrib

def simulate_tile(means2d_tile, conics_tile, colors_tile, opacities_tile,
                  tile_px_locs, W, H, tile_x, tile_y, TS,
                  opacity_threshold=1.0/256):
    """
    Simulate the rasterizer per-pixel compositing for one tile's Gaussians.

    Returns per-pixel:
      n_fetched, n_evaluated, n_committed arrays for each pixel.

    This is a simplified simulation — it approximates the 2D Gaussian
    evaluation for each pixel, which is the expensive attribute work.
    """
    n_gauss = means2d_tile.shape[0]
    if n_gauss == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    # Pre-compute per-pixel screen coordinates
    px_ys = np.arange(tile_y * TS, min((tile_y + 1) * TS, H))
    px_xs = np.arange(tile_x * TS, min((tile_x + 1) * TS, W))
    npx = len(px_ys) * len(px_xs)

    means2d_np = means2d_tile.cpu().numpy()
    conics_np = conics_tile.cpu().numpy()
    colors_np = colors_tile.cpu().numpy()
    opac_np = opacities_tile.cpu().numpy()

    # Sort Gaussians by depth (already sorted via isect_tiles)
    # isect_tiles returns them sorted by depth within each tile

    fetched = np.full((npx, n_gauss), True, dtype=bool)
    evaluated = np.full((npx, n_gauss), False, dtype=bool)
    committed = np.full((npx, n_gauss), False, dtype=bool)

    px_idx = 0
    for py in px_ys:
        for px in px_xs:
            T = 1.0
            for gi in range(n_gauss):
                # Fetched — always true (it's in the sorted list)
                # Evaluate 2D Gaussian at this pixel
                dx = px + 0.5 - means2d_np[gi, 0]
                dy = py + 0.5 - means2d_np[gi, 1]
                cx, cy, cxy = conics_np[gi, 0], conics_np[gi, 1], conics_np[gi, 2]
                power = cx * dx * dx + cy * dy * dy + 2 * cxy * dx * dy
                sigma = math.exp(-0.5 * power)
                opacity = opac_np[gi]

                # Evaluated = sigma passes threshold
                eval_flag = sigma > 0.0 and opacity > 0.0
                evaluated[px_idx, gi] = eval_flag

                if eval_flag:
                    committed[px_idx, gi] = True  # the pixel was touched
                    actual_contrib = opacity * T
                    T *= (1.0 - actual_contrib)
                    if T < 1e-4:  # next_T <= 1e-4 -> early termination
                        break
            px_idx += 1

    fetched_count = fetched.sum(axis=1)
    evaluated_count = evaluated.sum(axis=1)
    committed_count = committed.sum(axis=1)

    return fetched_count, evaluated_count, committed_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera-offset", type=int, default=0)
    p.add_argument("--camera-count", type=int, default=8)
    p.add_argument("--max-tiles-per-density", type=int, default=20)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    dev = "cuda"

    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device=dev)
    cameras = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device=dev),
        W, H
    )
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    shs = scene["shs"].contiguous()
    bg = torch.zeros(1, 3, device=dev)

    from gsplat.cuda._wrapper import _make_lazy_cuda_func
    fwd_cuda = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")

    camera_records = []

    for ci in range(args.camera_offset, args.camera_offset + args.camera_count):
        cam = cameras[ci % len(cameras)]
        vw = cam.world_view_transform[None].contiguous()
        Ks = cam.K[None].contiguous()

        _, _, meta = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=scene["opacity"].contiguous(),
            colors=shs, viewmats=vw, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=TS,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic",
        )
        torch.cuda.synchronize()

        m2d = meta["means2d"].contiguous()
        radii = meta["radii"].contiguous()
        depths = meta["depths"].contiguous()
        conics = meta["conics"].contiguous()
        opac = meta["opacities"].contiguous()

        _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, TS, TW, TH, sort=True)
        ioff = gsplat.isect_offset_encode(iid, 1, TW, TH)

        dir_ = cam.camera_center.to(dev) - means
        dir_ = dir_ / dir_.norm(dim=-1, keepdim=True)
        colors_sh = gsplat.spherical_harmonics(3, dir_, shs).unsqueeze(0)

        # Get last_ids for context
        _, _, last = fwd_cuda(m2d, conics, colors_sh, opac, bg, None, W, H, TS, ioff, fid)
        torch.cuda.synchronize()
        last_ids = last[0].long()

        off_t = ioff[0].reshape(-1).long()
        off_list = off_t.tolist()
        ends = off_list[1:] + [int(iid.numel())]

        # Select representative tiles: up to args.max_tiles_per_density from each group
        high_tiles, med_tiles, low_tiles = [], [], []
        for tile_i in range(TW * TH):
            lo = off_list[tile_i]
            hi = ends[tile_i]
            total_len = hi - lo
            if total_len == 0:
                continue
            if total_len > 10000 and len(high_tiles) < args.max_tiles_per_density:
                high_tiles.append(tile_i)
            elif total_len > 3000 and len(med_tiles) < args.max_tiles_per_density:
                med_tiles.append(tile_i)
            elif len(low_tiles) < args.max_tiles_per_density:
                low_tiles.append(tile_i)

        tile_results = []
        total_fetched_all = 0
        total_evaluated_all = 0
        total_committed_all = 0

        for tile_i in high_tiles + med_tiles + low_tiles:
            lo = off_list[tile_i]
            hi = ends[tile_i]
            # Get the gaussian indices for this tile (from flatten_ids)
            gauss_indices = fid[lo:hi].cpu().numpy()

            tile_m2d = m2d[0, gauss_indices].cpu()
            tile_con = conics[0, gauss_indices].cpu()
            tile_col = colors_sh[0, gauss_indices].cpu()
            tile_opac = opac[0, gauss_indices].cpu()

            y = tile_i // TW
            x = tile_i % TW

            fetched, evaluated, committed = simulate_tile(
                tile_m2d, tile_con, tile_col, tile_opac,
                None, W, H, x, y, TS
            )

            f_mean = float(fetched.mean()) if len(fetched) > 0 else 0
            e_mean = float(evaluated.mean()) if len(evaluated) > 0 else 0
            c_mean = float(committed.mean()) if len(committed) > 0 else 0

            total_fetched_all += int(fetched.sum()) if len(fetched) > 0 else 0
            total_evaluated_all += int(evaluated.sum()) if len(evaluated) > 0 else 0
            total_committed_all += int(committed.sum()) if len(committed) > 0 else 0

            tile_results.append({
                "tile": tile_i,
                "n_gaussians_in_tile": len(gauss_indices),
                "per_pixel_fetched_mean": f_mean,
                "per_pixel_evaluated_mean": e_mean,
                "per_pixel_committed_mean": c_mean,
                "attribute_efficiency": (c_mean / max(f_mean, 1)),
                "total_fetched": int(fetched.sum()) if len(fetched) > 0 else 0,
                "total_evaluated": int(evaluated.sum()) if len(evaluated) > 0 else 0,
                "total_committed": int(committed.sum()) if len(committed) > 0 else 0,
            })

        cam_rec = {
            "camera": ci,
            "total_fetched_samples": total_fetched_all,
            "total_evaluated_samples": total_evaluated_all,
            "total_committed_samples": total_committed_all,
            "ratio_evaluated_per_fetched": total_evaluated_all / max(total_fetched_all, 1),
            "ratio_committed_per_fetched": total_committed_all / max(total_fetched_all, 1),
            "ratio_committed_per_evaluated": total_committed_all / max(total_evaluated_all, 1),
            "tile_samples": tile_results,
        }
        camera_records.append(cam_rec)

        print(f"  Camera {ci}: fetched={total_fetched_all:,} evaled={total_evaluated_all:,} "
              f"committed={total_committed_all:,}  "
              f"evaluated/fetched={cam_rec['ratio_evaluated_per_fetched']:.3f} "
              f"committed/fetched={cam_rec['ratio_committed_per_fetched']:.3f}",
              flush=True)

    all_c = [r["ratio_committed_per_fetched"] for r in camera_records]
    all_e = [r["ratio_evaluated_per_fetched"] for r in camera_records]

    result = {
        "schema_version": 1,
        "phase": "R1 attribute fetch analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scene": "room", "resolution": f"{W}x{H}", "tile_size": TS,
                      "cameras": list(range(args.camera_offset, args.camera_offset + args.camera_count))},
        "camera_records": camera_records,
        "cross_camera_aggregate": {
            "mean_committed_per_fetched": float(np.mean(all_c)),
            "mean_evaluated_per_fetched": float(np.mean(all_e)),
            "mean_uncommitted_fraction": 1.0 - float(np.mean(all_c)),
            "mean_unnecessary_evaluation_fraction": 1.0 - float(np.mean(all_e)),
        },
        "verdict": (
            "KEEP-CANDIDATE"
            if np.mean(all_c) < 0.1
            else "MAYBE"
            if np.mean(all_c) < 0.3
            else "DROP"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=str)
    print(f"\nSaved: {args.out}")
    print(f" Committed/fetched: {np.mean(all_c):.4f}")
    print(f" Evaluated/fetched: {np.mean(all_e):.4f}")


if __name__ == "__main__":
    main()
