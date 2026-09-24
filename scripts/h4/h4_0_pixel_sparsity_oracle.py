#!/usr/bin/env python3
"""H4-0 Exact pixel-support spatial-sparsity oracle (production fidelity).

Replicates the frozen B2 F5 forward gating loop for every tile t and every
inbound pixel p, walking the tile's depth-sorted gaussian list in order:

    sigma  := 0.5*(A.dx^2 + C.dy^2) + B.dx.dy     # conic = (A, B, C)
    alpha  := min(0.999, op * exp(-sigma))        # exp() call
    reject if sigma < 0
    reject if alpha < 1/255
    next_T := T*(1-alpha); if next_T <= 1e-4: terminate WITHOUT compositing
    else compose(sigma, alpha); T = next_T

It further derives, for each gaussian/tile pair and each block granularity,
the exact hierarchical truth sets: block b is LIVE for gaussian g iff at
least one inbound pixel in b lies in the exact support set S(g).

Work-count definitions (all exact, aggregated over all tiles of a scene):
  upper     U : sum_t n_t * P_t            (cache baseline, all entries)
  visited   V : sum_t sum_p visit_count(p) (sequential baseline)
  sigma_cnt   : visited entries that evaluate sigma (== V)
  sigma_neg   : visited entries rejected by sigma < 0
  exp_cnt     : visited entries that reach the exponent/alpha evaluation
  alpha_neg   : exp_cnt entries rejected by alpha < 1/255
  acc         : accepted entries (sigma >= 0 AND alpha >= 1/255)
  comp_cnt    : acc entries that are actually composited (not terminators)
  term_cnt    : entries where the pixel terminated (acc, but not composited)

Scheme S (hierarchical culling): every visited entry (p,g) is skipped if
g is not LIVE in p's block at the mask granularity of S; the remaining
entries are "iters_after_S", and of those the ones that would reach the
alpha evaluation are "exps_after_S".

The D scheme is the combined mask A32 AND B16 plus the sigma-range test
(sigma <= ln(op/1/255)); i.e. an entry survives only if its gaussian is
live in both the A32-block and the B16-block of the pixel AND the gaussian's
sigma-range condition passes.

Gate rule (conservative canonical):
  PASS     D removes >= 50% of V on >= 2/3 scenes
  MARGINAL D removes >= 25% of V on >= 2/3 scenes
  WEAK     otherwise

Usage:
  python h4_0_pixel_sparsity_oracle.py [--al <path>] [--scenes a,b,c]
      [--max-lo-side 2048] [--seed 4200] [--verify-scalar] [--selftest]
"""

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

TILE = 16
TAU = np.float32(1.0 / 255.0)
TRANS_LOG = float(np.log(1e-4))          # -9.210340371976182
SQRT_TAU = float(np.sqrt(TAU))

SCHEMES = [
    ("T_complete", 16, 16),   # whole-tile   (1  block  of 256)
    ("A32",         2, 16),   # 8  blocks of 32  (2 rows x 16 cols)
    ("B64",         8,  8),   # 4  blocks of 64
    ("B16",         4,  4),   # 16 blocks of 16
    ("B4",          2,  2),   # 64 blocks of 4
    ("W16",         1, 16),   # 16 blocks of 16 (row-block)
]

_GRID = {}


def _block_grid(bh, bw):
    """Return (256,) array: block id for each local pixel offset 0..255."""
    key = (bh, bw)
    g = _GRID.get(key)
    if g is None:
        nby = TILE // bh
        nbx = TILE // bw
        g = np.empty(TILE * TILE, dtype=np.int64)
        for r in range(TILE):
            for c in range(TILE):
                g[r * TILE + c] = (r // bh) * nbx + (c // bw)
        _GRID[key] = g
    return g


def _norm(x, shape):
    a = np.asarray(x.cpu() if hasattr(x, "cpu") else x)
    if a.size == 0:
        return np.zeros(shape, dtype=np.float32)
    return np.ascontiguousarray(a, dtype=np.float32).reshape(shape)


def _pick(f, *names):
    for n in names:
        if n in f:
            return f[n]
    raise KeyError(f"fixture keys {sorted(f.keys())} lack {names}")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _import_f2b2(path):
    """Import the frozen H3 fixture module; return its f2_b2 callable."""
    p = str(Path(path).resolve())
    import importlib.util
    spec = importlib.util.spec_from_file_location("h4_0_fixture_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(p).parent))
    spec.loader.exec_module(mod)
    return mod.f2_b2


def build_fixture(f2_b2, scene, cam, max_lo, device):
    # Frozen H3 fixture signature: f2_b2(scene, max_long_side, device) — no cam
    # argument (the fixture loader selects camera 0 internally).
    f = f2_b2(scene, max_lo, device)
    W = int(_pick(f, "width", "W"))
    H = int(_pick(f, "height", "H"))
    tw = int(_pick(f, "tw", "n_tiles_x"))
    th = int(_pick(f, "th", "n_tiles_y"))
    m2d = _norm(_pick(f, "m2d", "means2d", "uv"), (-1, 2))
    con = _norm(_pick(f, "con", "conics", "conic", "cov2d_inv"), (-1, 3))
    opa = _norm(_pick(f, "opacity", "opacities"), (-1,))
    dep = _norm(_pick(f, "depth", "z", "depths"), (-1,))
    flat_t = _pick(f, "flat", "tile_gaussians")
    offs_t = _pick(f, "off", "offs", "offset", "tile_offsets")
    try:
        flat_t = flat_t.cpu()
        offs_t = offs_t.cpu()
    except AttributeError:
        pass
    flat = np.asarray(flat_t, dtype=np.int64).reshape(-1)
    offs = np.asarray(offs_t, dtype=np.int64).reshape(-1)
    # offs has tw*th elements (start indices); append total len for last-tile end
    if offs.size == tw * th:
        offs = np.append(offs, flat.size)
    return dict(W=W, H=H, tw=tw, th=th, m2d=m2d, con=con, op=opa, dep=dep,
                flat=flat, offs=offs)


def gids_for_tile(f, tid):
    """Return the tile's sorted gaussian id slice from the flat list."""
    return f["flat"][f["offs"][tid]:f["offs"][tid + 1]]


def simulate_tile(tid, f, gids):
    """Run the exact B2 gating loop for one tile; return per-tile dict."""
    W, H = f["W"], f["H"]
    n = len(gids)
    if n == 0:
        return None

    tx, ty = tid % f["tw"], tid // f["tw"]
    x0, y0 = tx * TILE, ty * TILE

    # full 16x16 local grid, raster order (kernel local-id layout)
    cols_local = np.arange(16, dtype=np.int32)[None, :].repeat(16, axis=0).ravel()
    rows_local = np.arange(16, dtype=np.int32)[:, None].repeat(16, axis=1).ravel()
    px = (x0 + cols_local).astype(np.float32) + np.float32(0.5)
    py = (y0 + rows_local).astype(np.float32) + np.float32(0.5)
    inb = (px < np.float32(W)) & (py < np.float32(H))
    P = int(inb.sum())
    if P == 0:
        return None

    gix = np.asarray(gids, dtype=np.int64)
    gx = f["m2d"][gix, 0].astype(np.float32)
    gy = f["m2d"][gix, 1].astype(np.float32)
    A = f["con"][gix, 0].astype(np.float32)
    B = f["con"][gix, 1].astype(np.float32)
    C = f["con"][gix, 2].astype(np.float32)
    og = f["op"][gix].astype(np.float32)

    dx = gx[:, None] - px[None, :]
    dy = gy[:, None] - px[None, :]
    sigma = (np.float32(0.5) * (A[:, None] * (dx * dx) +
                                C[:, None] * (dy * dy)) +
             B[:, None] * (dx * dy)).astype(np.float32)

    # alpha = min(0.999, opacity * exp(-sigma))  -- numpy float64 exp,
    # cast to float32 to mirror __expf within quantization error
    alpha = np.minimum(np.float32(0.999),
                       og[:, None] * np.exp(-sigma.astype(np.float64)))
    alpha = alpha.astype(np.float32)

    sup = (sigma >= np.float32(0.0)) & (alpha >= TAU)   # (n_gaussians, 256)

    # ---- per-pixel sequential termination (exact kernel order) -----------
    # log-space cumulative product over accepted entries; find first j where
    # T(j) = prod_{i<=j, accepted} (1-alpha_{i,pixel}) <= 1e-4
    log_keep = np.where(sup, np.log1p(-alpha.astype(np.float64)), 0.0)
    cum_log = np.cumsum(log_keep, axis=0)                # (n, 256)
    termitable = cum_log <= TRANS_LOG
    has_term = termitable.any(axis=0)                    # pixel terminated?
    term_idx = np.where(has_term, termitable.argmax(axis=0), n).astype(np.int64)
    vis = np.arange(n)[:, None] < (term_idx[None, :] + 1)

    # inbound slice
    vis_i = vis[:, inb]        # (n_gaussians, P)
    sig_i = sigma[:, inb]
    alp_i = alpha[:, inb]
    sup_i = sup[:, inb]
    tmx_i = term_idx[inb]

    # per-gaussian support size (number of inbound pixels supporting g)
    occ = sup_i.sum(axis=1)

    # ---- hierarchical schemes ---------------------------------------------
    schemes = {}
    for name, bh, bw in SCHEMES:
        blk = _block_grid(bh, bw)
        nb = (TILE // bh) * (TILE // bw)
        binb = blk[inb]                                     # (P,)
        Bin = np.zeros((P, nb), dtype=np.float64)
        Bin[np.arange(P), binb] = 1.0
        supcount = sup_i.astype(np.float64) @ Bin          # (n, nb) counts
        live = supcount > 0                                 # (n, nb)
        live_map = live[:, binb]                            # (n, P)
        for dense in (True, False):
            lm = live_map if dense else (~live_map)
            iters_after = int((vis_i & lm).sum())
            # exp calls: entries that reach the alpha evaluation
            exp_after = int((vis_i & lm & sup_i).sum())
            schemes[(name, "dense" if dense else "sparse")] = {
                "iters_after": iters_after,
                "iters_saved": n * P - iters_after,
                "iters_saved_frac_of_visited":
                    (n * P - iters_after) / max(n * P, 1),
                "exp_after": exp_after,
                "exp_saved": n * P - exp_after,
                "alive_blocks": int(live.sum()),
                "block_cells": bh * bw,
            }

    return dict(
        tid=tid, n=n, P=P,
        upper=n * P,
        visited=int(vis_i.sum()),
        sigma_neg=int((vis_i & (sig_i < np.float32(0.0))).sum()),
        alpha_neg=int((vis_i & (sig_i >= np.float32(0.0)) &
                       (alp_i < TAU)).sum()),
        exp=int((vis_i & sup_i).sum()),
        comp=int((vis_i & sup_i & ~termitable[:, inb]).sum()),
        term=int(has_term[inb].sum()),
        occ=occ,
        schemes=schemes,
    )


def aggregate(results):
    agg = dict(tiles=0, upper=0, visited=0, sigma_neg=0, alpha_neg=0,
               exp=0, comp=0, term=0, occ=0)
    schemes = {}
    for r in results:
        if r is None:
            continue
        agg["tiles"] += 1
        agg["upper"] += r["upper"]
        agg["visited"] += r["visited"]
        agg["sigma_neg"] += r["sigma_neg"]
        agg["alpha_neg"] += r["alpha_neg"]
        agg["exp"] += r["exp"]
        agg["comp"] += r["comp"]
        agg["term"] += r["term"]
        agg["occ"] += int(r["occ"].sum())
        for k, s in r["schemes"].items():
            acc = schemes.setdefault(k, dict(
                iters_after=0, iters_saved=0, exp_after=0, exp_saved=0,
                alive_blocks=0, block_cells=0, total_cells=0))
            acc["iters_after"] += s["iters_after"]
            acc["iters_saved"] += s["iters_saved"]
            acc["exp_after"] += s["exp_after"]
            acc["exp_saved"] += s["exp_saved"]
            acc["alive_blocks"] += s["alive_blocks"]
            acc["total_cells"] += r["n"] * s["block_cells"]
    for s in schemes.values():
        s["iters_saved_frac_of_visited"] = \
            s["iters_saved"] / max(agg["visited"], 1)
        s["exp_saved_frac_of_visited"] = \
            s["exp_saved"] / max(agg["visited"], 1)
        s["alg_density"] = \
            (s["iters_after"] / max(s["total_cells"], 1))
        s["saved_vs_cache"] = \
            (s["iters_saved"] / max(agg["upper"], 1))
    return agg, schemes


def print_report(scene, agg, schemes, el):
    print(f"=== {scene} ===")
    print(f"tiles with data  : {agg['tiles']:,}")
    print(f"upper  U         : {agg['upper']:,}")
    print(f"visited V        : {agg['visited']:,}  "
          f"({agg['visited']/max(agg['upper'],1):.2%} of U)")
    print(f"sigma rejects    : {agg['sigma_neg']:,} "
          f"({agg['sigma_neg']/max(agg['visited'],1):.2%} of V)")
    print(f"alpha rejects    : {agg['alpha_neg']:,} "
          f"({agg['alpha_neg']/max(agg['visited'],1):.2%} of V)")
    print(f"accepted         : {agg['exp']:,} "
          f"({agg['exp']/max(agg['visited'],1):.2%} of V)")
    print(f"composited       : {agg['comp']:,}  (accepted - terminators)")
    print(f"terminated px    : {agg['term']:,}")
    print(f"support px       : {agg['occ']:,} "
          f"(mean/gauss {agg['occ']/max(agg['tiles'],1):.2f})")
    print()
    print(f"{'scheme':<14s} {'iters_after':>14s} {'saved_V%':>9s} "
          f"{'exp_after':>12s} {'exp_saved%':>10s} {'alg_dens':>8s} "
          f"{'live_blk':>9s} {'blk_memMB':>9s}")
    for k, s in schemes.items():
        name, kind = k
        print(f"{name+':'+kind:<14s} {s['iters_after']:>14,d} "
              f"{s['iters_saved_frac_of_visited']:>9.2%} "
              f"{s['exp_after']:>12,d} {s['exp_saved_frac_of_visited']:>10.2%} "
              f"{s['alg_density']:>8.4f} {s['alive_blocks']:>9,d} "
              f"{s['alive_blocks']/1e6:>9.2f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--al", required=False, help="path to h3_fwd_0_al.py")
    ap.add_argument("--scenes", default="room,bicycle,garden")
    ap.add_argument("--max-lo-side", type=int, default=2048, dest="max_lo")
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--gpu", type=int, default=0, help="cuda device id")
    ap.add_argument("--write-scene-json", action="store_true",
                    help="emit per-scene JSON files under out-dir")
    ap.add_argument("--verify-scalar", action="store_true",
                    help="cross-check the vectorised result with scalar code")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    if not args.out_dir or not args.al:
        ap.error("--out-dir and --al are required (or use --selftest)")

    import torch
    torch.cuda.set_device(args.gpu)

    f2_b2 = _import_f2b2(args.al)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": time.strftime("%Y%m%dT%H%M%S"),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "config": {"max_long_side": args.max_lo, "camera": 0, "tile": TILE},
        "scenes": {},
        "pooled": {},
    }

    pooled = dict(tiles=0, upper=0, visited=0, sigma_neg=0, alpha_neg=0,
                  exp=0, comp=0, term=0, occ=0)
    pooled_schemes = None

    t0 = time.time()
    device = f"cuda:{args.gpu}"
    for scene in scenes:
        f = build_fixture(f2_b2, scene, 0, args.max_lo, device)
        n_gauss = f["m2d"].shape[0]
        print(f"  {scene}: {n_gauss:,} gaussians  {f['W']}x{f['H']}  "
              f"tiles={f['tw']}x{f['th']}", flush=True)

        results = []
        for tid in range(f["tw"] * f["th"]):
            gids = gids_for_tile(f, tid)
            r = simulate_tile(tid, f, gids)
            if r is None:
                continue
            results.append(r)

        agg, schemes = aggregate(results)
        agg["n"] = n_gauss
        for k in ("tiles", "upper", "visited", "sigma_neg", "alpha_neg",
                  "exp", "comp", "term", "occ"):
            pooled[k] += agg[k]
        if pooled_schemes is None:
            pooled_schemes = schemes
        else:
            for k, s in schemes.items():
                p = pooled_schemes.setdefault(k, dict(
                    iters_after=0, iters_saved=0, exp_after=0, exp_saved=0,
                    alive_blocks=0, block_cells=0, total_cells=0))
                for kk in ("iters_after", "iters_saved", "exp_after",
                           "exp_saved", "alive_blocks", "total_cells"):
                    p[kk] += s[kk]

        # Convert tuple scheme keys to strings for JSON
        schemes_json = {}
        for k, s in schemes.items():
            key = f"{k[0]}_{k[1]}"
            schemes_json[key] = {kk: (int(vv) if isinstance(vv, (int, np.integer)) else
                                      float(vv) if isinstance(vv, (float, np.floating)) else vv)
                                 for kk, vv in s.items()}
        rec = {
            "resolution": [f["W"], f["H"]],
            "n_pixels_total": f["W"] * f["H"],
            "n_gaussians_visible": int(n_gauss),
            "tiles_total": f["tw"] * f["th"],
            "tiles_with_data": int(agg["tiles"]),
            "totals": {k: int(v) for k, v in agg.items()},
            "schemes": schemes_json,
            "elapsed_s": time.time() - t0,
        }
        report["scenes"][scene] = rec
        if args.write_scene_json:
            (out_dir / f"pixel_sparsity_{scene}.json").write_text(
                json.dumps(rec, indent=2))
        print_report(scene, agg, schemes, time.time() - t0)

    pooled["n"] = sum(report["scenes"][s]["n_gaussians_visible"] for s in scenes)
    pooled["pixels"] = sum(report["scenes"][s]["resolution"][0] *
                           report["scenes"][s]["resolution"][1] for s in scenes)
    # Convert pooled scheme keys to strings for JSON
    pooled_schemes_json = {}
    if pooled_schemes:
        for k, s in pooled_schemes.items():
            key = f"{k[0]}_{k[1]}"
            pooled_schemes_json[key] = {kk: (int(vv) if isinstance(vv, (int, np.integer)) else
                                             float(vv) if isinstance(vv, (float, np.floating)) else vv)
                                        for kk, vv in s.items()}
    report["pooled"] = {"totals": {k: (int(v) if isinstance(v, (int, np.integer)) else
                                       float(v) if isinstance(v, (float, np.floating)) else v)
                                   for k, v in pooled.items()},
                        "schemes": pooled_schemes_json}
    report["elapsed_s"] = time.time() - t0
    (out_dir / "pixel_sparsity_summary.json").write_text(
        json.dumps(report, indent=2))
    print("\n=== DONE ===", flush=True)
    print(f"summary written: {out_dir / 'pixel_sparsity_summary.json'}")


def _selftest():
    """Self-contained smoke test: 3x2 tile with 2 gaussians."""
    rng = np.random.default_rng(0)

    # --- 3x2 tile, 2 gaussians --------------------------------------------
    f0 = dict(
        W=3, H=2, tw=1, th=1,
        m2d=np.array([[0.0, 0.0], [2.0, 1.0]], np.float32),
        con=np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], np.float32),
        op=np.array([0.9, 0.3], np.float32),
        flat=np.array([0, 1], dtype=np.int64),
        offs=np.array([0, 2], dtype=np.int64),
    )
    r0 = simulate_tile(0, f0, gids_for_tile(f0, 0))
    assert r0["upper"] == 12, r0["upper"]
    assert r0["visited"] >= 2, r0
    assert r0["sigma_neg"] >= 0
    assert r0["schemes"][("A32", "dense")]["iters_saved"] <= r0["upper"]

    # --- full 16x16 tile, random gaussians --------------------------------
    n = 64
    f1 = dict(
        W=16, H=16, tw=1, th=1,
        m2d=rng.normal(7.5, 2.0, (n, 2)).astype(np.float32),
        con=rng.uniform(0.5, 2.0, (n, 3)).astype(np.float32),
        op=rng.uniform(0.1, 0.9, (n,)).astype(np.float32),
        flat=np.arange(n, dtype=np.int64),
        offs=np.array([0, n], dtype=np.int64))
    r1 = simulate_tile(0, f1, gids_for_tile(f1, 0))
    assert r1["n"] == n and r1["P"] == 256
    assert 0 <= r1["sigma_neg"] <= r1["visited"]
    assert 0 <= r1["alpha_neg"] <= r1["visited"]

    # --- empty tile --------------------------------------------------------
    f2 = dict(W=16, H=16, tw=1, th=1,
              m2d=np.zeros((0, 2), np.float32),
              con=np.zeros((0, 3), np.float32),
              op=np.zeros((0,), np.float32),
              flat=np.zeros(0, dtype=np.int64),
              offs=np.array([0, 0], dtype=np.int64))
    assert simulate_tile(2, f2, []) is None

    print("selftest: all OK")
    return True


if __name__ == "__main__":
    main()
