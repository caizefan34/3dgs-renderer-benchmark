#!/usr/bin/env python3
"""
Phase C17-C2 — C1 Minimal Verification (Python-only)

Purpose:
  - Simulate C1's IEEE 754 depth-truncation sort key entirely in numpy
  - For each tile: compare baseline alpha-composited color vs C1-reordered color
  - Report PSNR estimate, max per-pixel error, and per-tile ordering change stats
  - No CUDA / no gsplat dependency needed (synthetic scene only)

Usage:
  python tools/c1_minimal_verification.py

Output:
  results/phase-c17-c2/c1_verification_results.json
"""

import numpy as np
import math, json, time, sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "phase-c17-c2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

TILE_SIZE = 16

# ── Scenes ──────────────────────────────────────────────────────────────
SCENES = {
    "room":    {"n": 10000, "d_min": 0.2,  "d_max": 6.0,   "h": 540, "w": 960},
    "bicycle": {"n": 10000, "d_min": 0.5,  "d_max": 50.0,  "h": 540, "w": 960},
    "garden":  {"n": 10000, "d_min": 0.3,  "d_max": 30.0,  "h": 540, "w": 960},
}


def gen_gaussians(p, n_gaussians=None):
    """
    Generate a synthetic scene and compute per-tile intersections.
    Returns flat arrays of all intersections with their depths, tile_ids,
    Gaussian indices, and RGBA colors.
    """
    n = n_gaussians or p["n"]
    h, w = p["h"], p["w"]
    dmin, dmax = p["d_min"], p["d_max"]
    tw = math.ceil(w / TILE_SIZE)
    th = math.ceil(h / TILE_SIZE)
    tnb = int(math.floor(math.log2(tw * th))) + 1
    imgb = int(math.floor(math.log2(1))) + 1  # single image → 1 bit

    t0 = time.time()

    # Random depths (log-uniform for realistic distribution)
    depths = np.exp(np.random.uniform(np.log(dmin), np.log(dmax), n)).astype(np.float32)

    # Random positions (beta distribution → slight center bias, like real scenes)
    mx = (w * np.random.beta(1.5, 1.5, n)).astype(np.float32)
    my = (h * np.random.beta(1.5, 1.5, n)).astype(np.float32)

    # Projected radius: larger for closer Gaussians
    r = np.clip((64.0 / (depths + 0.01)).astype(np.int32), 1, w // 4)

    # Random colors (RGBA) — per Gaussian
    colors = np.random.rand(n, 4).astype(np.float32)
    colors[:, 3] = np.clip(0.1 + 0.9 * np.random.rand(n), 0.05, 0.95)  # alpha

    # Filter valid (r > 0)
    valid = r > 0
    dv = depths[valid]
    mxv = mx[valid]
    myv = my[valid]
    rv = r[valid]
    giv = np.where(valid)[0].astype(np.int32)
    cv = colors[valid]
    nv = len(dv)

    # Tile coverage per Gaussian
    tr = rv.astype(np.float32) / TILE_SIZE
    tminx = np.maximum(0, np.floor(mxv / TILE_SIZE - tr).astype(np.int32))
    tminy = np.maximum(0, np.floor(myv / TILE_SIZE - tr).astype(np.int32))
    tmaxx = np.minimum(tw, np.ceil(mxv / TILE_SIZE + tr).astype(np.int32))
    tmaxy = np.minimum(th, np.ceil(myv / TILE_SIZE + tr).astype(np.int32))

    w_tile = (tmaxx - tminx).astype(np.int32)
    h_tile = (tmaxy - tminy).astype(np.int32)
    ntp = (w_tile * h_tile).astype(np.int64)
    total = int(np.sum(ntp))

    t1 = time.time()
    print(f"  {n} Gaussians ({nv} valid) → {total:,} intersections ({t1-t0:.2f}s)", flush=True)

    # Fully vectorized expansion (same as Phase V2)
    starts = np.zeros(nv + 1, dtype=np.int64)
    starts[1:] = np.cumsum(ntp)
    nz = ntp > 0
    nz_starts = starts[:-1][nz]
    nz_lens = ntp[nz]
    nz_w = w_tile[nz]
    nz_h = h_tile[nz]
    nz_x0 = tminx[nz]
    nz_y0 = tminy[nz]
    nz_dv = dv[nz]
    nz_giv = giv[nz]
    nz_cv = cv[nz]

    pos = np.arange(total, dtype=np.int64)
    gauss_idx = np.searchsorted(nz_starts, pos, side='right') - 1
    offset = pos - nz_starts[gauss_idx]
    w_g = nz_w[gauss_idx]
    row = offset // w_g
    col = offset % w_g

    tile_ids = (nz_y0[gauss_idx] + row) * tw + (nz_x0[gauss_idx] + col)
    depth_out = nz_dv[gauss_idx].copy()
    gidx_out = nz_giv[gauss_idx].copy()
    color_out = nz_cv[gauss_idx].copy()  # RGBA, shape (total, 4)

    # Also compute per-pixel (x, y) for each intersection
    pix_x = (nz_x0[gauss_idx] + col) * TILE_SIZE + TILE_SIZE // 2
    pix_y = (nz_y0[gauss_idx] + row) * TILE_SIZE + TILE_SIZE // 2
    pix_x = np.clip(pix_x, 0, w - 1).astype(np.int32)
    pix_y = np.clip(pix_y, 0, h - 1).astype(np.int32)

    t2 = time.time()
    print(f"  Expansion: {t2-t1:.2f}s", flush=True)

    return {
        "depths": depth_out,
        "tile_ids": tile_ids,
        "gidx": gidx_out,
        "colors": color_out,
        "pix_x": pix_x,
        "pix_y": pix_y,
        "tnb": tnb,
        "imgb": imgb,
        "imgsz": (h, w),
        "n_tiles": tw * th,
        "tw": tw,
        "th": th,
    }


def merge_sort_inv(arr):
    """Count inversions in arr using merge sort."""
    n = len(arr)
    if n <= 1:
        return 0
    m = n // 2
    left, right = arr[:m].copy(), arr[m:].copy()
    inv = merge_sort_inv(left) + merge_sort_inv(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]; i += 1
        else:
            arr[k] = right[j]; j += 1; inv += len(left) - i
        k += 1
    while i < len(left):
        arr[k] = left[i]; i += 1; k += 1
    while j < len(right):
        arr[k] = right[j]; j += 1; k += 1
    return inv


def simulate_sorting(data):
    """
    Build baseline and C1 sort keys, simulate sorting, and compare.
    Returns per-intersection ordering analysis + per-tile rendering comparison.
    """
    depths = data["depths"]
    tile_ids = data["tile_ids"]
    gidx = data["gidx"]
    colors = data["colors"]
    tnb = data["tnb"]
    imgb = data["imgb"]
    n = len(depths)

    # ── Key construction ──────────────────────────────────────────
    # Note: image_id (iid) = 0 for single-image rendering.
    # We do NOT encode Gaussian index (gidx) into the sort key —
    # it is stored separately as flatten_ids in the actual pipeline.
    d32 = depths.view(np.uint32)
    dp_hi = (d32 >> 16).astype(np.int64)
    iid = 0  # single image → iid = 0

    # Baseline key: iid_enc | (tile_id << 32) | depth_id_enc
    baseline_key = (
        (iid.astype(np.int64) if isinstance(iid, np.ndarray) else np.int64(iid)) << (32 + tnb)
        | tile_ids.astype(np.int64) << 32
        | d32.astype(np.int64)
    )

    # C1 key: depth_upper | (tile_id << 16) | iid_enc
    c1_key = (
        dp_hi
        | tile_ids.astype(np.int64) << 16
        | (iid.astype(np.int64) if isinstance(iid, np.ndarray) else np.int64(iid)) << (16 + tnb)
    )

    # ── Sort ──────────────────────────────────────────────────────
    # Key insight: Neither baseline nor C1 uses gaussian_id as tiebreaker.
    # Both use CUB stable radix sort, which preserves input order for equal keys.
    # Use numpy stable sort on the key alone - this matches CUB behavior.
    
    # Baseline: sort by (tile_id, full_depth)
    # key: iid_enc | (tile_id << 32) | d32  → sorts tile-major, depth-minor
    bo = np.argsort(baseline_key, kind='stable')
    
    # C1: sort by (tile_id, depth_hi only)
    # key: depth_upper | (tile_id << 16) | iid_enc → sorts tile-major, depth_hi-minor
    co = np.argsort(c1_key, kind='stable')

    # ── Per-tile analysis ─────────────────────────────────────────
    # Group by tile_id in baseline-sorted order
    b_tiles = tile_ids[bo]
    unique_tiles, tile_starts = np.unique(b_tiles, return_index=True)
    tile_ends = np.append(tile_starts[1:], n)
    n_tiles = len(unique_tiles)

    changes_per_tile = np.zeros(n_tiles, dtype=np.int32)
    items_per_tile = np.zeros(n_tiles, dtype=np.int32)
    inv_per_tile = np.zeros(n_tiles, dtype=np.int32)

    # Build rank arrays for fast lookup
    # baseline_rank[idx] = position in baseline-sorted order
    baseline_rank = np.empty(n, dtype=np.int32)
    baseline_rank[bo] = np.arange(n)
    # c1_rank[idx] = position in C1-sorted order
    c1_rank = np.empty(n, dtype=np.int32)
    c1_rank[co] = np.arange(n)

    for ti in range(n_tiles):
        s, e = int(tile_starts[ti]), int(tile_ends[ti])
        n_items = e - s
        items_per_tile[ti] = n_items
        if n_items <= 1:
            continue

        # All intersection indices that belong to this tile
        tile_mask = tile_ids == unique_tiles[ti]
        b_positions = np.where(tile_mask)[0]  # indices in original array

        # Their ranks in each sort order (this IS the per-tile ordering)
        br = baseline_rank[tile_mask]
        cr = c1_rank[tile_mask]

        # Order within tile: argsort of the ranks gives position in sort
        # If baseline and C1 sort orders differ within this tile, these differ
        b_tile_order = np.argsort(br, kind='stable')
        c_tile_order = np.argsort(cr, kind='stable')
        reordered = int(np.sum(b_tile_order != c_tile_order))
        changes_per_tile[ti] = reordered

        # Inversions: count pairs where order differs between B and C
        # perm[baseline_position] = c1_position
        perm = np.empty(n_items, dtype=np.int32)
        perm[b_tile_order] = c_tile_order  
        inv_per_tile[ti] = merge_sort_inv(perm)

    total_changes = int(np.sum(changes_per_tile))
    total_inv = int(np.sum(inv_per_tile))
    total_items = int(np.sum(items_per_tile))

    # ── Rendering simulation ──────────────────────────────────────
    # Key idea: Within each tile, Gaussians are composited front-to-back
    # using the "over" operator. Since each Gaussian covers the entire tile
    # (it intersected it), we can compute a tile-level composite color
    # from the sorted sequence of RGBA values.
    #
    # tile_color = g1 over g2 over g3 ... over gN
    # This IS order-dependent: swapping two Gaussians with different
    # colors changes the final composite.
    h, w = data["imgsz"]
    tw, th = data["tw"], data["th"]

    def composite_tiles(sort_order):
        """Composite all Gaussians in each tile in sort order.
        Returns per-pixel image where each pixel of a tile gets
        that tile's composite color.
        """
        img = np.full((h, w, 4), [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Build per-tile sorted list from the global sort order
        # sort_order maps to original indices
        sorted_tids = tile_ids[sort_order]
        sorted_colors = colors[sort_order]
        
        # Find tile boundaries in sort order
        tile_changes = np.concatenate([[True], sorted_tids[1:] != sorted_tids[:-1]])
        tile_sorted_ids = sorted_tids[tile_changes]
        tile_ranges = np.where(tile_changes)[0]
        tile_ends = np.append(tile_ranges[1:], len(sort_order))
        
        for ti in range(len(tile_sorted_ids)):
            tid = tile_sorted_ids[ti]
            s = tile_ranges[ti]
            e = tile_ends[ti]
            
            # Alpha composite: (r1,a1) over ... over (rk,ak)
            # Result = (r_acc / a_acc, a_acc) after compositing all
            acc_r = 0.0; acc_g = 0.0; acc_b = 0.0; acc_a = 0.0
            
            for i in range(s, e):
                r, g, b, a = sorted_colors[i]
                # Over operator
                out_a = a + acc_a * (1.0 - a)
                if out_a > 0:
                    acc_r = (r * a + acc_r * acc_a * (1.0 - a)) / out_a
                    acc_g = (g * a + acc_g * acc_a * (1.0 - a)) / out_a
                    acc_b = (b * a + acc_b * acc_a * (1.0 - a)) / out_a
                    acc_a = out_a
                acc_a = min(acc_a, 1.0)
            
            # Fill this tile's pixels with the composite color
            tile_x = tid % tw
            tile_y = tid // tw
            x0 = tile_x * TILE_SIZE
            y0 = tile_y * TILE_SIZE
            x1 = min(x0 + TILE_SIZE, w)
            y1 = min(y0 + TILE_SIZE, h)
            img[y0:y1, x0:x1, 0] = acc_r
            img[y0:y1, x0:x1, 1] = acc_g
            img[y0:y1, x0:x1, 2] = acc_b
            img[y0:y1, x0:x1, 3] = acc_a
        
        return img

    t3 = time.time()
    img_base = composite_tiles(bo)
    t4 = time.time()
    print(f"  Baseline composite: {t4-t3:.1f}s", flush=True)
    
    img_c1 = composite_tiles(co)
    t5 = time.time()
    print(f"  C1 composite: {t5-t4:.1f}s", flush=True)

    # ── Compute metrics ───────────────────────────────────────────
    diff = img_base[:, :, :3] - img_c1[:, :, :3]
    mse = np.mean(diff ** 2)
    psnr = -10 * math.log10(max(mse, 1e-15)) if mse > 0 else 100.0
    max_pix_err = float(np.max(np.abs(diff)))
    ssim_val = compute_ssim(img_base[:, :, :3], img_c1[:, :, :3])

    return {
        "psnr": round(psnr, 4),
        "mse": round(float(mse), 10),
        "max_pixel_error": round(max_pix_err, 6),
        "ssim": round(float(ssim_val), 6),
        "n_intersections": n,
        "n_tiles": int(n_tiles),
        "total_ordering_changes": total_changes,
        "total_inversions": total_inv,
        "change_ratio": round(total_changes / total_items, 6) if total_items > 0 else 0,
        "inv_ratio": round(total_inv / (total_items * (total_items - 1) / 2), 8)
            if total_items > 1 else 0,
        "tile_changes_mean": round(float(np.mean(changes_per_tile[items_per_tile > 0])), 4),
        "tile_changes_max": int(np.max(changes_per_tile)),
        "tile_inv_mean": round(float(np.mean(inv_per_tile[items_per_tile > 1])), 4),
        "tile_inv_max": int(np.max(inv_per_tile)),
    }


def compute_ssim(img1, img2):
    """
    Simplified SSIM: luminance comparison only.
    Full SSIM requires Gaussian filter; this is a coarse estimate.
    """
    # Convert to grayscale
    g1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
    g2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]

    mu1 = np.mean(g1)
    mu2 = np.mean(g2)
    sig1 = np.var(g1)
    sig2 = np.var(g2)
    sig12 = np.mean((g1 - mu1) * (g2 - mu2))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2))
    return ssim


def key_bit_stats(data):
    """Print C1 vs baseline key bit width."""
    tw, th = data["tw"], data["th"]
    tnb = data["tnb"]
    imgb = data["imgb"]
    baseline_used = 32 + tnb + imgb
    c1_used = 16 + tnb + imgb
    print(f"\n  ── Key bits (tiles={tw*th}, tnb={tnb}) ──")
    print(f"  Baseline end_bit: {baseline_used}  (32 depth + {tnb} tile + {imgb} img)")
    print(f"  C1      end_bit: {c1_used}  (16 depth + {tnb} tile + {imgb} img)")
    print(f"  Reduction: {baseline_used - c1_used} bits")
    return {"baseline_end_bit": baseline_used, "c1_end_bit": c1_used}


def main():
    results = {}
    for sname, p in SCENES.items():
        print(f"\n{'='*60}\n  {sname} ({p['n']:,} Gaussians @ {p['w']}x{p['h']})\n{'='*60}", flush=True)

        ta = time.time()
        data = gen_gaussians(p)
        kbs = key_bit_stats(data)

        tb = time.time()
        print(f"  Simulating sorting & rendering...", flush=True)
        stats = simulate_sorting(data)
        tc = time.time()

        print(f"  Total time: {tc-ta:.1f}s", flush=True)
        print(f"  PSNR vs baseline:  {stats['psnr']:.4f} dB")
        print(f"  SSIM vs baseline:  {stats['ssim']:.6f}")
        print(f"  Max pixel error:   {stats['max_pixel_error']:.6f}")
        print(f"  MSE:               {stats['mse']:.10f}")
        print(f"  Ordering changes:  {stats['total_ordering_changes']:,} "
              f"({stats['change_ratio']*100:.4f}%)")
        print(f"  Inversions:        {stats['total_inversions']:,} "
              f"({stats['inv_ratio']*100:.6f}%)")
        print(f"  Tile changes (μ):  {stats['tile_changes_mean']:.4f} "
              f"(max: {stats['tile_changes_max']})")

        results[sname] = {
            "scene_params": p,
            "key_bits": kbs,
            "stats": stats,
        }

        with open(OUTPUT_DIR / "c1_verification_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Intermediate saved.", flush=True)

    # Summary table
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Scene':>10s} | {'PSNR':>8s} | {'SSIM':>8s} | {'MSE':>12s} | {'MaxErr':>8s} | "
          f"{'Changes':>8s} | {'Invs':>8s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for sn in results:
        r = results[sn]["stats"]
        print(f"  {sn:>10s} | {r['psnr']:>7.3f} | {r['ssim']:>7.5f} | "
              f"{r['mse']:>11.3e} | {r['max_pixel_error']:>7.5f} | "
              f"{r['change_ratio']*100:>7.4f}% | {r['inv_ratio']*100:>7.4f}%")

    print(f"\nResults saved to: {OUTPUT_DIR / 'c1_verification_results.json'}", flush=True)


if __name__ == "__main__":
    main()
