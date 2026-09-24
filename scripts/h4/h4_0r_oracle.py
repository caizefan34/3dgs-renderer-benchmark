#!/usr/bin/env python3
"""H4-0R: Pixel-Support Oracle Repair + HiGS 32-G Mask Feasibility.

Corrects H4-0 accounting errors:
  1. saved_vs_visited = (V - K) / V   [NOT (U - K) / V]
  2. baseline_exp_calls = V (since sigma_neg = 0)
  3. alpha_accepted (A) reported separately from exp calls
  4. Support condition: sigma <= log(255 * opacity), NOT sigma >= 0
  5. Mask storage per (tile, Gaussian-entry), not per tile
  6. B64/B16/B4 efficiency: K/A ratio
  7. 32-G transpose mask cost/benefit estimate
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
MAX_ALPHA = np.float32(0.999)
TRANS_LOG = float(np.log(1e-4))

# F4 exact intersection counts (tile-Gaussian pairs)
F4_INTERSECTIONS = {
    "room": 953144,
    "bicycle": 1412193,
    "garden": 533928,
}

SCHEMES = [
    ("B64", 8, 8),    # 4 blocks of 64 (8x8)
    ("B16", 4, 4),    # 16 blocks of 16 (4x4)
    ("B4", 2, 2),     # 64 blocks of 4 (2x2)
]

_GRID = {}


def _block_grid(bh, bw):
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


def _import_f2b2(path):
    p = str(Path(path).resolve())
    import importlib.util
    spec = importlib.util.spec_from_file_location("h4_0r_fixture_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(p).parent))
    spec.loader.exec_module(mod)
    return mod.f2_b2


def build_fixture(f2_b2, scene, max_lo, device):
    f = f2_b2(scene, max_lo, device)
    W = int(_pick(f, "width", "W"))
    H = int(_pick(f, "height", "H"))
    tw = int(_pick(f, "tw", "n_tiles_x"))
    th = int(_pick(f, "th", "n_tiles_y"))
    m2d = _norm(_pick(f, "m2d", "means2d", "uv"), (-1, 2))
    con = _norm(_pick(f, "con", "conics", "conic", "cov2d_inv"), (-1, 3))
    opa = _norm(_pick(f, "opacity", "opacities"), (-1,))
    flat_t = _pick(f, "flat", "tile_gaussians")
    offs_t = _pick(f, "off", "offs", "offset", "tile_offsets")
    try:
        flat_t = flat_t.cpu()
        offs_t = offs_t.cpu()
    except AttributeError:
        pass
    flat = np.asarray(flat_t, dtype=np.int64).reshape(-1)
    offs = np.asarray(offs_t, dtype=np.int64).reshape(-1)
    if offs.size == tw * th:
        offs = np.append(offs, flat.size)
    return dict(W=W, H=H, tw=tw, th=th, m2d=m2d, con=con, op=opa,
                flat=flat, offs=offs)


def gids_for_tile(f, tid):
    return f["flat"][f["offs"][tid]:f["offs"][tid + 1]]


def simulate_tile(tid, f, gids):
    """Exact B2 F5 gating loop + corrected support + block schemes."""
    W, H = f["W"], f["H"]
    n = len(gids)
    if n == 0:
        return None

    tx, ty = tid % f["tw"], tid // f["tw"]
    x0, y0 = tx * TILE, ty * TILE

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
    dy = gy[:, None] - py[None, :]
    sigma = (np.float32(0.5) * (A[:, None] * (dx * dx) +
                                C[:, None] * (dy * dy)) +
             B[:, None] * (dx * dy)).astype(np.float32)

    # alpha = min(0.999, op * exp(-sigma))
    alpha = np.minimum(MAX_ALPHA,
                       og[:, None] * np.exp(-sigma.astype(np.float64)))
    alpha = alpha.astype(np.float32)

    # --- Corrected support: sigma <= log(255 * op) when op >= 1/255 ---
    # Exact condition: op * exp(-sigma) >= 1/255
    #   => sigma <= log(255 * op)   (when op >= 1/255)
    #   => NO support               (when op < 1/255)
    log_thresh = np.where(og >= TAU,
                          np.log(255.0 * og.astype(np.float64)),
                          np.float64(-np.inf)).astype(np.float32)
    support = (sigma <= log_thresh[:, None]).astype(np.float32)  # (n, 256)

    # --- Per-pixel sequential termination ---
    log_keep = np.where(support > 0,
                        np.log1p(-alpha.astype(np.float64)), 0.0)
    cum_log = np.cumsum(log_keep, axis=0)
    termitable = cum_log <= TRANS_LOG
    has_term = termitable.any(axis=0)
    term_idx = np.where(has_term, termitable.argmax(axis=0), n).astype(np.int64)
    vis = np.arange(n)[:, None] < (term_idx[None, :] + 1)

    # Inbound slices
    vis_i = vis[:, inb]          # (n, P)
    sig_i = sigma[:, inb]
    alp_i = alpha[:, inb]
    sup_i = support[:, inb] > 0  # (n, P)
    tmx_i = term_idx[inb]

    # --- Corrected work accounting ---
    U = n * P
    V = int(vis_i.sum())

    # Baseline: sigma evaluations = V (every visited pair evaluates sigma)
    # Since sigma_neg = 0: baseline_exp_calls = V
    sigma_neg = int((vis_i & (sig_i < 0)).sum())
    baseline_sigma_evals = V
    baseline_exp_calls = V - sigma_neg  # = V when sigma_neg=0

    # Alpha rejects/accepts (among visited pairs)
    alpha_rej = int((vis_i & (alp_i < TAU)).sum())
    alpha_acc = int((vis_i & (alp_i >= TAU)).sum())
    # Composited = accepted minus terminators
    comp = int((vis_i & (alp_i >= TAU) & ~termitable[:, inb]).sum())
    term_px = int(has_term[inb].sum())

    # Total support set size (for reference)
    occ = int(sup_i.sum())

    # --- Block schemes: K = pairs retained (visited AND in live block) ---
    schemes = {}
    for name, bh, bw in SCHEMES:
        blk = _block_grid(bh, bw)
        nb = (TILE // bh) * (TILE // bw)
        binb = blk[inb]
        Bin = np.zeros((P, nb), dtype=np.float64)
        Bin[np.arange(P), binb] = 1.0
        # Block is LIVE for gaussian g iff at least one inbound pixel
        # in the block is in the support set S(g)
        supcount = sup_i.astype(np.float64) @ Bin   # (n, nb)
        live = supcount > 0                          # (n, nb)
        live_map = live[:, binb]                     # (n, P)

        # K = visited pairs in live blocks (dense scheme)
        K = int((vis_i & live_map).sum())
        # exp_after = visited pairs in live blocks that would reach exp
        # (all of them, since sigma_neg=0 for all)
        exp_after = K  # since every retained pair evaluates exp
        # A in live blocks (alpha-accepted among retained)
        A_in_live = int((vis_i & live_map & (alp_i >= TAU)).sum())

        alive_blocks = int(live.sum())
        schemes[name] = {
            "nb_blocks": nb,
            "K": K,
            "exp_after": exp_after,
            "A_in_live": A_in_live,
            "alive_blocks": alive_blocks,
            "bits_per_entry": nb,
        }

    return dict(
        tid=tid, n=n, P=P,
        U=U, V=V,
        sigma_neg=sigma_neg,
        baseline_sigma_evals=baseline_sigma_evals,
        baseline_exp_calls=baseline_exp_calls,
        alpha_rej=alpha_rej,
        alpha_acc=alpha_acc,
        comp=comp,
        term_px=term_px,
        occ=occ,
        schemes=schemes,
    )


def aggregate(results):
    agg = dict(tiles=0, U=0, V=0, sigma_neg=0,
               baseline_sigma_evals=0, baseline_exp_calls=0,
               alpha_rej=0, alpha_acc=0, comp=0, term_px=0, occ=0)
    schemes = {}
    for r in results:
        if r is None:
            continue
        agg["tiles"] += 1
        for k in ("U", "V", "sigma_neg", "baseline_sigma_evals",
                  "baseline_exp_calls", "alpha_rej", "alpha_acc",
                  "comp", "term_px", "occ"):
            agg[k] += r[k]
        for name, s in r["schemes"].items():
            acc = schemes.setdefault(name, dict(
                nb_blocks=0, K=0, exp_after=0, A_in_live=0,
                alive_blocks=0, bits_per_entry=0))
            acc["K"] += s["K"]
            acc["exp_after"] += s["exp_after"]
            acc["A_in_live"] += s["A_in_live"]
            acc["alive_blocks"] += s["alive_blocks"]
            acc["nb_blocks"] = s["nb_blocks"]
            acc["bits_per_entry"] = s["bits_per_entry"]
    # Compute derived metrics
    V = max(agg["V"], 1)
    A = max(agg["alpha_acc"], 1)
    for name, s in schemes.items():
        s["saved_vs_visited"] = (agg["V"] - s["K"]) / V
        s["K_over_A"] = s["K"] / A
        s["exp_saved_frac"] = (agg["baseline_exp_calls"] - s["exp_after"]) / max(agg["baseline_exp_calls"], 1)
    return agg, schemes


def compute_mask_storage(agg, f4_intersections, scene):
    """Mask storage per (tile, Gaussian-entry)."""
    n_entries = f4_intersections
    results = {}
    for name, bh, bw in SCHEMES:
        nb = (TILE // bh) * (TILE // bw)
        bits = nb
        bytes_packed = (n_entries * bits + 7) // 8
        bytes_aligned = n_entries * ((bits + 7) // 8)  # byte-aligned per entry
        results[name] = {
            "nb_blocks": nb,
            "bits_per_entry": bits,
            "n_entries": n_entries,
            "packed_bytes": int(bytes_packed),
            "packed_MB": bytes_packed / 1e6,
            "aligned_bytes": int(bytes_aligned),
            "aligned_MB": bytes_aligned / 1e6,
        }
    return results


def compute_32g_mask_oracle(agg, schemes, scene):
    """HiGS 32-G × 16 B16-block transpose mask cost/benefit."""
    # B16: 16 blocks, each Gaussian gets a 16-bit mask
    # Transpose: 16 x uint32 (each uint32 = which of 32 Gaussians are live in that block)
    #
    # Mask generation cost (per tile, per 32-G batch):
    #   - Compute 16-bit support mask for each of 32 Gaussians: 32 x 16 bit-sets
    #   - Transpose to 16 x uint32: 16 uint32 values, each built from 32 1-bit values
    #   - Cost: 32*16 bit-ops for mask construction + 16*32 bit-ops for transpose
    #     = ~512 bit operations per 32-G batch per tile
    #
    # Arithmetic avoided:
    #   - For each 4x4 block (16 blocks), skip Gaussians not in the mask
    #   - Each skipped Gaussian avoids: 1 sigma eval + 1 exp + 1 alpha compare
    #     + potential composite (alpha mult, T update)
    #   - Cost per avoided pixel-G pair: ~4-6 FLOPs (sigma) + ~3 FLOPs (exp)
    #     + ~2 FLOPs (alpha) = ~10 FLOPs conservative
    #
    # Using real counts:
    V = max(agg["V"], 1)
    A = max(agg["alpha_acc"], 1)
    b16 = schemes.get("B16", {})
    K_b16 = b16.get("K", V)

    # Number of 32-G batches per scene
    n_gauss = agg.get("n_gauss", 0)
    n_batches_32 = (n_gauss + 31) // 32

    # Mask generation: per tile, per 32-G batch
    # Each tile has n_t Gaussians; batches per tile = ceil(n_t/32)
    # Total mask-gen ops = sum_tiles ceil(n_t/32) * 512
    # Approximate: n_entries / 32 * 512 where n_entries = total tile-G pairs
    n_entries = agg.get("n_entries", 0)
    mask_gen_ops = (n_entries + 31) // 32 * 512  # bit-ops

    # Pixel-G arithmetic avoided = (V - K_b16) * ~10 FLOPs
    pairs_avoided = V - K_b16
    flops_avoided = pairs_avoided * 10  # conservative: 10 FLOPs per pair

    return {
        "n_gauss": n_gauss,
        "n_batches_32": n_batches_32,
        "n_entries": n_entries,
        "mask_gen_bitops": int(mask_gen_ops),
        "pairs_avoided": int(pairs_avoided),
        "flops_avoided_est": int(flops_avoided),
        "ratio_avoided_to_gen": flops_avoided / max(mask_gen_ops, 1),
        "K_b16": K_b16,
        "V": V,
        "A": A,
        "K_b16_over_A": K_b16 / A,
        "note": "mask_gen = ceil(n_entries/32)*512 bit-ops; "
                "flops_avoided = (V-K)*10 conservative FLOPs/pair",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--al", required=False, help="path to h3_fwd_0_al.py")
    ap.add_argument("--scenes", default="room,bicycle,garden")
    ap.add_argument("--max-lo-side", type=int, default=2048, dest="max_lo")
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--gpu", type=int, default=0)
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
    device = f"cuda:{args.gpu}"

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": time.strftime("%Y%m%dT%H%M%S"),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {"max_long_side": args.max_lo, "tile": TILE, "seed": args.seed},
        "scenes": {},
        "pooled": {},
    }

    pooled = dict(tiles=0, U=0, V=0, sigma_neg=0,
                  baseline_sigma_evals=0, baseline_exp_calls=0,
                  alpha_rej=0, alpha_acc=0, comp=0, term_px=0, occ=0,
                  n_gauss=0)
    pooled_schemes = None
    all_mask_storage = {}
    all_32g = {}

    t0 = time.time()
    for scene in scenes:
        f = build_fixture(f2_b2, scene, args.max_lo, device)
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
        agg["n_gauss"] = n_gauss
        agg["n_entries"] = int(f["flat"].size)

        for k in pooled:
            pooled[k] += agg.get(k, 0)
        if pooled_schemes is None:
            pooled_schemes = {name: dict(s) for name, s in schemes.items()}
        else:
            for name, s in schemes.items():
                p = pooled_schemes.setdefault(name, dict(
                    nb_blocks=0, K=0, exp_after=0, A_in_live=0,
                    alive_blocks=0, bits_per_entry=0))
                for kk in ("K", "exp_after", "A_in_live", "alive_blocks"):
                    p[kk] += s[kk]
                p["nb_blocks"] = s["nb_blocks"]
                p["bits_per_entry"] = s["bits_per_entry"]

        # Mask storage
        f4 = F4_INTERSECTIONS.get(scene, agg["n_entries"])
        mask_storage = compute_mask_storage(agg, f4, scene)

        # 32-G oracle
        oracle_32g = compute_32g_mask_oracle(agg, schemes, scene)

        rec = {
            "resolution": [f["W"], f["H"]],
            "n_gaussians": int(n_gauss),
            "n_entries_f4": f4,
            "n_entries_actual": int(f["flat"].size),
            "totals": {k: int(v) for k, v in agg.items()
                       if k not in ("n_gauss",)},
            "schemes": {name: {k: (int(v) if isinstance(v, (int, np.integer))
                                   else float(v) if isinstance(v, (float, np.floating))
                                   else v)
                               for k, v in s.items()}
                        for name, s in schemes.items()},
            "mask_storage": mask_storage,
            "higs32_mask_oracle": oracle_32g,
            "elapsed_s": time.time() - t0,
        }
        report["scenes"][scene] = rec

        print_report(scene, agg, schemes)
        print(f"  mask_storage: B64={mask_storage['B64']['aligned_MB']:.2f}MB "
              f"B16={mask_storage['B16']['aligned_MB']:.2f}MB "
              f"B4={mask_storage['B4']['aligned_MB']:.2f}MB")
        print(f"  32G: avoided={oracle_32g['pairs_avoided']:,} pairs, "
              f"gen_ops={oracle_32g['mask_gen_bitops']:,}, "
              f"ratio={oracle_32g['ratio_avoided_to_gen']:.0f}x", flush=True)

    # Pooled
    V = max(pooled["V"], 1)
    A = max(pooled["alpha_acc"], 1)
    for name, s in pooled_schemes.items():
        s["saved_vs_visited"] = (pooled["V"] - s["K"]) / V
        s["K_over_A"] = s["K"] / A
        s["exp_saved_frac"] = (pooled["baseline_exp_calls"] - s["exp_after"]) / max(pooled["baseline_exp_calls"], 1)

    # Pooled mask storage
    total_entries = sum(F4_INTERSECTIONS.get(s, 0) for s in scenes)
    pooled_mask = {}
    for name, bh, bw in SCHEMES:
        nb = (TILE // bh) * (TILE // bw)
        bits = nb
        pooled_mask[name] = {
            "nb_blocks": nb,
            "bits_per_entry": bits,
            "n_entries_total": total_entries,
            "packed_MB": (total_entries * bits + 7) // 8 / 1e6,
            "aligned_MB": total_entries * ((bits + 7) // 8) / 1e6,
        }

    report["pooled"] = {
        "totals": {k: int(v) for k, v in pooled.items()},
        "schemes": pooled_schemes,
        "mask_storage": pooled_mask,
    }
    report["elapsed_s"] = time.time() - t0

    (out_dir / "h4_0r_summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== DONE === {report['elapsed_s']:.1f}s")
    print(f"summary: {out_dir / 'h4_0r_summary.json'}")


def print_report(scene, agg, schemes):
    V = max(agg["V"], 1)
    A = max(agg["alpha_acc"], 1)
    print(f"\n=== {scene} ===")
    print(f"  U (upper)              : {agg['U']:>15,}")
    print(f"  V (visited)            : {agg['V']:>15,}  ({agg['V']/max(agg['U'],1):.2%} of U)")
    print(f"  sigma evaluations      : {agg['baseline_sigma_evals']:>15,}  (= V, sigma_neg={agg['sigma_neg']})")
    print(f"  exp evaluations        : {agg['baseline_exp_calls']:>15,}  (= V - sigma_neg)")
    print(f"  alpha rejects          : {agg['alpha_rej']:>15,}  ({agg['alpha_rej']/V:.2%} of V)")
    print(f"  alpha accepted (A)     : {agg['alpha_acc']:>15,}  ({agg['alpha_acc']/V:.2%} of V)")
    print(f"  composited             : {agg['comp']:>15,}")
    print(f"  terminated px          : {agg['term_px']:>15,}")
    print(f"  support size (occ)     : {agg['occ']:>15,}")
    print()
    print(f"  {'scheme':<8s} {'K (retained)':>14s} {'saved_V%':>10s} "
          f"{'exp_after':>12s} {'K/A':>8s} {'A_in_live':>10s} {'alive_blk':>10s}")
    for name, s in schemes.items():
        sv = (agg['V'] - s['K']) / V
        ka = s['K'] / A
        print(f"  {name:<8s} {s['K']:>14,d} {sv:>10.2%} "
              f"{s['exp_after']:>12,d} {ka:>8.3f} "
              f"{s['A_in_live']:>10,d} {s['alive_blocks']:>10,d}")


def _selftest():
    """Smoke test with synthetic data."""
    rng = np.random.default_rng(42)
    n = 32
    f = dict(
        W=16, H=16, tw=1, th=1,
        m2d=rng.normal(8.0, 2.0, (n, 2)).astype(np.float32),
        con=rng.uniform(0.5, 2.0, (n, 3)).astype(np.float32),
        op=rng.uniform(0.1, 0.9, (n,)).astype(np.float32),
        flat=np.arange(n, dtype=np.int64),
        offs=np.array([0, n], dtype=np.int64),
    )
    r = simulate_tile(0, f, gids_for_tile(f, 0))
    assert r is not None
    assert r["U"] == n * 256
    assert 0 <= r["V"] <= r["U"]
    assert r["baseline_exp_calls"] == r["V"] - r["sigma_neg"]
    assert r["alpha_rej"] + r["alpha_acc"] <= r["V"]
    assert r["alpha_acc"] <= r["V"]
    for name, s in r["schemes"].items():
        assert 0 <= s["K"] <= r["V"]
        assert s["K"] <= r["V"]
        assert s["bits_per_entry"] > 0
    print("selftest: all OK")
    return True


if __name__ == "__main__":
    main()
