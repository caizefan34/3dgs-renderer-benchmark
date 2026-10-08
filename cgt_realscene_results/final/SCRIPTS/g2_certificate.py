#!/usr/bin/env python3
"""G2 continuous-domain certificate via DENSE COVERAGE (plan sec.5.5).

G1 fidelity was measured on RANDOM held-out odd pixels within each tile. The
plan (sec.5.5) notes random held-out tests cannot bound the error over the whole
continuous ray domain. This script strengthens that with a DETERMINISTIC dense
sub-pixel grid over each tile's 2D ray domain: fit the student on the tile's even
pixels (as G1), then evaluate teacher vs student on a dense (16*sub x 16*sub) grid
covering the tile extent, and report the max abs (E,T) error over that dense grid.

HONESTY: this is a dense-coverage EMPIRICAL certificate (max error over a dense
deterministic grid), NOT a rigorous interval-arithmetic bound. It is strictly
stronger evidence than random held-out (deterministic + dense + covers the tile
extent), but still a sample, not a proof over the continuum. Interval arithmetic
or certified subdivision would be required for a rigorous bound (NOT_REACHED).

Reuses g1_compressibility.py teacher/fit ports (numerically identical).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import gsplat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_compressibility import (read_cameras_bin, read_images_bin, q2R, tile_teacher,
                                fit_cgt, TILE, POWER_CLIP, GATE_MAX_ABS, RANKS, predict_and_vjp)


def extract_tiles(scene, ckpt, data, view_idx, max_tiles, seed):
    cams = read_cameras_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    means = ck["xyz"].cuda().float(); scales = torch.exp(ck["scaling"].cuda().float())
    quats = ck["rotation"].cuda().float(); opacities = torch.sigmoid(ck["opacity"].cuda().float()); shs = ck["shs"].cuda().float()
    deg = int(ck["active_sh_degree"]); rng = np.random.default_rng(seed)
    iid, name, cam, qv, tv = imgs[view_idx]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device="cuda")
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device="cuda")
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    campos = torch.as_tensor(t_c2w, dtype=torch.float32, device="cuda")[None]
    rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H,
                                        sh_degree=deg, render_mode="RGB", packed=True, tile_size=TILE)
    dirs = torch.nn.functional.normalize(means - campos, dim=-1)
    colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0, 1).detach().cpu().numpy()[meta["gaussian_ids"].detach().cpu().numpy()]
    def flat2(x):
        x = meta[x].detach().cpu().numpy(); return x[0] if x.ndim == 3 else x
    means2d = flat2("means2d"); conics = flat2("conics"); opac = flat2("opacities")
    io = meta["isect_offsets"].detach().cpu().numpy().reshape(-1); fi = meta["flatten_ids"].detach().cpu().numpy()
    th, tw = int(meta["tile_height"]), int(meta["tile_width"]); total = len(fi)
    starts = io.astype(np.int64)
    def end_of(ti, tj):
        if tj + 1 < tw: return int(starts[ti * tw + tj + 1])
        if ti + 1 < th: return int(starts[(ti + 1) * tw + 0])
        return total
    tiles = []
    for ti in range(th):
        for tj in range(tw):
            s = int(starts[ti * tw + tj]); e = end_of(ti, tj)
            if e - s > 0: tiles.append((ti, tj, s, e))
    counts = np.array([e - s for (_, _, s, e) in tiles])
    order = np.argsort(counts); ranks_sel = np.empty(len(tiles), dtype=np.int64); ranks_sel[order] = np.arange(len(tiles))
    decile = np.minimum((ranks_sel * 10) // max(len(tiles), 1), 9)
    sel = []
    for dec in range(10):
        pool = np.where(decile == dec)[0]
        if len(pool) == 0: continue
        sel.append(rng.choice(pool, size=max(1, max_tiles // 10), replace=False))
    sel = np.concatenate(sel) if sel else np.array([], dtype=np.int64)
    out = []
    for idx in sel:
        ti, tj, s, e = tiles[idx]; gids = fi[s:e]
        if len(gids) < 1: continue
        ox = tj * TILE; oy = ti * TILE
        xs = ox + np.arange(TILE) + 0.5; ys = oy + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        cx0, cy0 = px.mean(0); px[:, 0] -= cx0; px[:, 1] -= cy0
        out.append(dict(px=px, gids=gids, means2d=means2d, conics=conics, opac=opac, colors=colors, M=int(e - s)))
    return out


def dense_grid(px, sub):
    """Dense sub-sampled grid over the tile's 2D ray extent, centered like px."""
    x0, x1 = px[:, 0].min(), px[:, 0].max(); y0, y1 = px[:, 1].min(), px[:, 1].max()
    n = TILE * sub
    xs = np.linspace(x0, x1, n); ys = np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    g = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    g[:, 0] -= g[:, 0].mean(); g[:, 1] -= g[:, 1].mean()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--views", type=int, default=2); ap.add_argument("--max-tiles-per-view", type=int, default=12)
    ap.add_argument("--ranks", type=str, default="4,8,16"); ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    rng = np.random.default_rng(args.seed)
    tiles = []
    for vi in range(min(args.views, 6)):
        try:
            tiles += extract_tiles(args.scene, args.ckpt, args.data, vi, args.max_tiles_per_view, args.seed + vi)
        except Exception as ex:
            import traceback; print(f"[view {vi}] extract failed: {ex}", flush=True)
    if not tiles:
        print("NO_TILES", flush=True); return
    print(f"extracted {len(tiles)} tiles; M median={int(np.median([t['M'] for t in tiles]))}", flush=True)

    recs = []
    for ti, t in enumerate(tiles):
        px, gids = t["px"], t["gids"]
        m2d, con, op, col = t["means2d"], t["conics"], t["opac"], t["colors"]
        # teacher on training (even) pixels + held-out (odd) pixels, as G1
        E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2d, con, op, col)
        y = np.concatenate([E, T], axis=1); P = y.shape[0]
        tr = np.arange(0, P, 2); te = np.arange(1, P, 2)
        qt = np.column_stack([px[tr, 0], px[tr, 1]]); yt = y[tr]
        qte = np.column_stack([px[te, 0], px[te, 1]]); yte = y[te]
        # dense grid
        dg = dense_grid(px, args.sub)
        Ed, Td = tile_teacher(dg[:, 0:1], dg[:, 1:2], gids, m2d, con, op, col)
        yd = np.concatenate([Ed, Td], axis=1)
        for rank in ranks:
            seed = args.seed + ti * 1000 + rank
            p = fit_cgt(qt, yt, rank, seed, args.iterations)
            # held-out error (odd pixels), as G1
            pte = predict_and_vjp(p, qte)
            held_max = float(np.max(np.abs(pte - yte)))
            # dense-coverage error
            pd = predict_and_vjp(p, dg)
            dense_max = float(np.max(np.abs(pd - yd)))
            # also value-only dense (first 3 cols) vs teacher on black bg
            recs.append(dict(scene=args.scene, tile_idx=ti, M=int(t["M"]), rank=int(rank),
                             heldout_max_abs=held_max, dense_max_abs=dense_max,
                             dense_pass=dense_max <= GATE_MAX_ABS, heldout_pass=held_max <= GATE_MAX_ABS,
                             n_dense=int(len(dg))))
        if (ti + 1) % 5 == 0:
            print(f"tile {ti+1}/{len(tiles)} done", flush=True)

    # aggregate per rank
    agg = {}
    for rank in ranks:
        rs = [r for r in recs if r["rank"] == rank]
        dm = np.array([r["dense_max_abs"] for r in rs]); hm = np.array([r["heldout_max_abs"] for r in rs])
        agg[str(rank)] = dict(n=len(rs),
            heldout_max_abs_median=float(np.median(hm)), heldout_max_abs_p95=float(np.percentile(hm, 95)),
            dense_max_abs_median=float(np.median(dm)), dense_max_abs_p95=float(np.percentile(dm, 95)),
            dense_max_abs_max=float(dm.max()),
            dense_pass_rate=float(np.mean([r["dense_pass"] for r in rs])),
            heldout_pass_rate=float(np.mean([r["heldout_pass"] for r in rs])),
            dense_vs_heldout_ratio_median=float(np.median(dm / np.maximum(hm, 1e-12))))
    out = dict(gate="G2_continuous_domain_certificate_dense_coverage",
               scene=args.scene, n_tiles=len(tiles), sub=args.sub,
               n_dense_grid=int(TILE * args.sub) ** 2, value_gate_max_abs=GATE_MAX_ABS,
               scope=("dense-coverage EMPIRICAL certificate (max error over deterministic dense sub-pixel grid), "
                      "NOT a rigorous interval-arithmetic bound (that remains NOT_REACHED)."),
               per_rank=agg, records=recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("WROTE", args.out, flush=True)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
