#!/usr/bin/env python3
"""
Phase V2 — C1 Ordering / Collision Verification (fully vectorized numpy)

No per-Gaussian loops. Uses searchsorted for position→Gaussian mapping.
"""

import numpy as np, json, math, time, sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "phase-c17-c2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)
TILE_SIZE = 16

SCENES = {
    "room":    {"n": 10000, "d_min": 0.2,  "d_max": 6.0,   "h": 1080, "w": 1920},
    "bicycle": {"n": 10000, "d_min": 0.5,  "d_max": 50.0,  "h": 1080, "w": 1920},
    "garden":  {"n": 10000, "d_min": 0.3,  "d_max": 30.0,  "h": 1080, "w": 1920},
}

def gen_and_intersect(p):
    n, h, w = p["n"], p["h"], p["w"]
    dmin, dmax = p["d_min"], p["d_max"]
    tw = math.ceil(w / TILE_SIZE); th = math.ceil(h / TILE_SIZE)
    tnb = int(math.floor(math.log2(tw * th))) + 1
    
    t0 = time.time()
    
    # Generate Gaussians
    depths = np.exp(np.random.uniform(np.log(dmin), np.log(dmax), n)).astype(np.float32)
    mx = (w * np.random.beta(1.5, 1.5, n)).astype(np.float32)
    my = (h * np.random.beta(1.5, 1.5, n)).astype(np.float32)
    r = np.clip((64.0 / (depths + 0.01)).astype(np.int32), 1, w // 4)
    
    valid = r > 0
    dv = depths[valid]; mxv = mx[valid]; myv = my[valid]; rv = r[valid]
    giv = np.where(valid)[0].astype(np.int32)
    nv = len(dv)
    
    # Tile coverage
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
    
    # ---- Fully vectorized expansion ----
    # Cumulative start indices
    starts = np.zeros(nv + 1, dtype=np.int64)
    starts[1:] = np.cumsum(ntp)
    
    # Filter non-zero Gaussians
    nz = ntp > 0
    nz_starts = starts[:-1][nz]
    nz_lens = ntp[nz]
    nz_w = w_tile[nz]; nz_h = h_tile[nz]
    nz_x0 = tminx[nz]; nz_y0 = tminy[nz]
    nz_dv = dv[nz]; nz_giv = giv[nz]
    
    # Position → Gaussian index mapping (fully vectorized)
    pos = np.arange(total, dtype=np.int64)
    gauss_idx = np.searchsorted(nz_starts, pos, side='right') - 1
    
    # Within-Gaussian offset
    offset = pos - nz_starts[gauss_idx]
    w_g = nz_w[gauss_idx]
    
    # Row and column within Gaussian's tile rectangle
    row = offset // w_g  # y within tile rect
    col = offset % w_g   # x within tile rect
    
    # Tile IDs
    tile_ids = (nz_y0[gauss_idx] + row) * tw + (nz_x0[gauss_idx] + col)
    
    # Depth and gidx (constant per Gaussian)
    depth_out = nz_dv[gauss_idx]
    gidx_out = nz_giv[gauss_idx]
    
    t2 = time.time()
    print(f"  Expansion: {t2-t1:.2f}s", flush=True)
    
    return depth_out, tile_ids, gidx_out, tnb, (h, w), nv

def analyze(depths, tile_ids, gidx, tnb, sname, imgsz):
    n = len(depths)
    d32 = depths.view(np.uint32)
    iids = np.zeros(n, dtype=np.int64)
    dp_hi = (d32 >> 16).astype(np.int64)
    
    # Keys
    bk = iids << (32 + tnb) | tile_ids.astype(np.int64) << 32 | d32.astype(np.int64)
    ck = dp_hi | tile_ids.astype(np.int64) << 16 | iids << (16 + tnb)
    
    # Collision groups
    ck_grp = np.column_stack([tile_ids, dp_hi])
    _, inv, cnts = np.unique(ck_grp, axis=0, return_inverse=True, return_counts=True)
    in_coll = cnts[inv] >= 2
    n_coll = int(np.sum(in_coll))
    coll_pairs = int(sum(c * (c - 1) // 2 for c in cnts[cnts >= 2]))
    total_pairs = n * (n - 1) // 2
    coll_rate = coll_pairs / total_pairs if total_pairs > 0 else 0
    
    # Sort simulations
    bo = np.lexsort((gidx, bk))
    b_st = tile_ids[bo]
    
    # Per-tile analysis
    unique_tiles, tile_starts = np.unique(b_st, return_index=True)
    tile_ends = np.append(tile_starts[1:], n)
    n_tiles = len(unique_tiles)
    
    # Collision group inversion analysis (efficient: process groups via sorted)
    # For collision groups (same tile, same depth_upper), compare baseline vs C1 ordering.
    large_gi_mask = cnts >= 2
    large_gi_indices = np.where(large_gi_mask)[0]
    
    if len(large_gi_indices) > 0:
        sort_idx = np.lexsort((dp_hi, tile_ids))
        sorted_gidx = gidx[sort_idx]
        sorted_d32 = d32[sort_idx]
        sorted_ck = ck_grp[sort_idx]
        
        group_changes = np.concatenate([[True], 
            np.any(sorted_ck[1:] != sorted_ck[:-1], axis=1)])
        group_ids = np.cumsum(group_changes) - 1
        _, grp_starts, grp_counts = np.unique(group_ids, return_index=True, return_counts=True)
        
        total_invs = 0
        total_cpairs = 0
        
        for gs, gc in zip(grp_starts, grp_counts):
            if gc < 2:
                continue
            ge = gs + gc
            gu = sorted_d32[gs:ge].copy()
            gg = sorted_gidx[gs:ge].copy()
            
            bo_g = np.lexsort((gg, gu))
            co_g = np.argsort(gg, kind='stable')
            
            rank = np.empty(gc, dtype=np.int32)
            for pos, item in enumerate(co_g):
                rank[pos] = int(np.flatnonzero(bo_g == item)[0])
            
            inv_cnt = _invc(rank)
            if inv_cnt > 0:
                total_invs += inv_cnt
                total_cpairs += gc * (gc - 1) // 2
    else:
        total_invs = 0
        total_cpairs = 0
    
    inv_rate = total_invs / total_cpairs if total_cpairs > 0 else 0
    coll_to_inv = total_invs / coll_pairs if coll_pairs > 0 else 0
    
    # Per-tile stats
    tile_items = []
    tci, tcg = [], []
    for ti in range(n_tiles):
        s, e = int(tile_starts[ti]), int(tile_ends[ti])
        cnt = e - s
        tile_items.append(cnt)
        if cnt < 2:
            continue
        tid = unique_tiles[ti]
        mask = tile_ids == tid
        tn = int(np.sum(mask))
        cn = int(np.sum(in_coll[mask]))
        tci.append(cn / tn)
        _, dc = np.unique(dp_hi[mask], return_counts=True)
        tcg.append(np.sum(dc >= 2) / len(dc))
    
    # Exponent breakdown
    exps = (d32 >> 23) & 0xFF
    eb = {}
    for e in sorted(np.unique(exps)):
        mask = exps == e; ne = int(np.sum(mask))
        if ne == 0: continue
        dp = dp_hi[mask]
        _, dc = np.unique(dp, return_counts=True)
        cp = int(sum(c * (c - 1) // 2 for c in dc[dc >= 2]))
        tp = ne * (ne - 1) // 2
        r0 = 2.0 ** (int(e) - 127); r1 = 2.0 ** (int(e) + 1 - 127) if e < 254 else float('inf')
        qstep_m = 2.0 ** (int(e) - 134)  # = 2^(e-127) / 2^7
        eb[int(e)] = {"range": f"[{r0:.4f},{r1:.4f})", "count": ne,
            "bins": int(len(dc)), "coll_pairs": cp,
            "coll_rate": cp / tp if tp > 0 else 0,
            "qstep_mm": round(qstep_m * 1000, 3)}
    
    return {"scene": sname, "img_size": list(imgsz), "tile_n_bits": tnb,
        "n_gaussians": int(np.max(gidx) + 1), "n_intersections": n, "n_tiles": n_tiles,
        "coll_pairs": coll_pairs, "coll_rate": round(coll_rate, 8),
        "collided_items": n_coll, "collided_item_frac": round(n_coll / n, 6),
        "inv_in_groups": total_invs, "inv_rate_in_groups": round(inv_rate, 8),
        "coll_to_inv_rate": round(coll_to_inv, 8),
        "tile_item_mean": float(np.mean(tile_items)) if tile_items else 0,
        "tile_item_max": int(np.max(tile_items)) if tile_items else 0,
        "tile_coll_item_mean": float(np.mean(tci)) if tci else 0,
        "tile_coll_item_max": float(np.max(tci)) if tci else 0,
        "tile_coll_group_mean": float(np.mean(tcg)) if tcg else 0,
        "tile_coll_group_max": float(np.max(tcg)) if tcg else 0,
        "exp_buckets": eb}

def _invc(arr):
    n = len(arr)
    if n <= 1: return 0
    m = n // 2; l, r = arr[:m].copy(), arr[m:].copy()
    inv = _invc(l) + _invc(r)
    i = j = k = 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]: arr[k] = l[i]; i += 1
        else: arr[k] = r[j]; j += 1; inv += len(l) - i
        k += 1
    while i < len(l): arr[k] = l[i]; i += 1; k += 1
    while j < len(r): arr[k] = r[j]; j += 1; k += 1
    return inv

def main():
    results = {}; timings = {}
    for sname, p in SCENES.items():
        print(f"\n{'='*60}\n  {sname} ({p['n']:,} Gaussians)\n{'='*60}", flush=True)
        ta = time.time()
        depths, tile_ids, gidx, tnb, imgsz, nv = gen_and_intersect(p)
        print(f"  Analyzing...", flush=True)
        res = analyze(depths, tile_ids, gidx, tnb, sname, imgsz)
        tb = time.time()
        results[sname] = res
        timings[sname] = {"gen_s": round(tb-ta, 1)}
        r = res
        print(f"  intersections: {r['n_intersections']:,}", flush=True)
        print(f"  coll_pairs:     {r['coll_pairs']:,}  ({r['coll_rate']*100:.4f}%)", flush=True)
        print(f"  collided_items: {r['collided_items']:,}  ({r['collided_item_frac']*100:.2f}%)", flush=True)
        print(f"  inv_in_groups:  {r['inv_in_groups']:,}  ({r['inv_rate_in_groups']*100:.4f}%)", flush=True)
        print(f"  coll→inv:       {r['coll_to_inv_rate']*100:.2f}%", flush=True)
        print(f"  tile coll item: μ={r['tile_coll_item_mean']*100:.2f}%  max={r['tile_coll_item_max']*100:.2f}%", flush=True)
        print(f"  tile coll grp:  μ={r['tile_coll_group_mean']*100:.2f}%  max={r['tile_coll_group_max']*100:.2f}%", flush=True)
        for e, b in sorted(r["exp_buckets"].items()):
            print(f"    E{e:3d} {b['range']:>20s}: cnt={b['count']:>6,} bins={b['bins']:>4d} "
                  f"coll={b['coll_rate']*100:>7.4f}% qstep={b['qstep_mm']:>6.2f}mm", flush=True)
    
    out = {"meta": {"tile_size": TILE_SIZE, "scenes": {k: v["n"] for k, v in SCENES.items()}},
           "results": results, "timings": timings}
    with open(OUTPUT_DIR / "c1_ordering_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved", flush=True)
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}\n"
          f"  {'Scene':>10s} | {'Isects':>9s} | {'Coll%':>8s} | {'Inv%':>8s} | {'CollItem%':>9s}\n"
          f"  {'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}", flush=True)
    for sn in results:
        r = results[sn]
        print(f"  {sn:>10s} | {r['n_intersections']:>9,} | {r['coll_rate']*100:>7.4f}% | "
              f"{r['inv_rate_in_groups']*100:>7.4f}% | {r['collided_item_frac']*100:>8.4f}%", flush=True)

if __name__ == "__main__":
    main()
