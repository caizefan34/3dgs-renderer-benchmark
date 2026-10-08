#!/usr/bin/env python3
"""Synthetic CPU validation of g2_rigorous_certificate interval math (no GPU/gsplat).

Builds random tiles (means2d/conics/opac/colors) + random student params, then runs the
branch-and-bound interval certifier WITH runtime point-validation. If any interval box fails
to contain sampled point errors, it raises (fail-closed). Passing => interval enclosure valid.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_rigorous_certificate as G

def rand_tile(rng, M, TILE=16):
    xs = np.arange(TILE) + 0.5; ys = np.arange(TILE) + 0.5
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    c = px.mean(0); px[:, 0] -= c[0]; px[:, 1] -= c[1]
    means2d = rng.uniform(-TILE/2, TILE/2, size=(M, 2))
    conics = np.zeros((M, 3)); conics[:, 0] = rng.uniform(0.5, 2.0, M); conics[:, 2] = rng.uniform(0.5, 2.0, M); conics[:, 1] = rng.uniform(-0.5, 0.5, M)
    opac = rng.uniform(0.05, 0.99, M)
    colors = rng.uniform(0, 1, size=(M, 3))
    gids = np.arange(M)
    return dict(px=px, gids=gids, means2d=means2d, conics=conics, opac=opac, colors=colors, M=M)

def rand_student(rng, R):
    # within plausible BOUNDS_ONE ranges
    p = np.zeros((R, 10))
    p[:, 0] = rng.uniform(-8, 8, R)      # mean x
    p[:, 1] = rng.uniform(-8, 8, R)      # mean y
    p[:, 2] = rng.uniform(-2, 2, R)      # log l00
    p[:, 3] = rng.uniform(-2, 2, R)      # l10
    p[:, 4] = rng.uniform(-2, 2, R)      # log l11
    p[:, 5] = rng.uniform(-3, 3, R)      # logit weight prior
    p[:, 6] = rng.uniform(-3, 3, R)      # logit opacity
    p[:, 7:10] = rng.uniform(-3, 3, R*3).reshape(R, 3)  # logit color
    return p

def main():
    rng = np.random.default_rng(7)
    n_ok = 0; n_fail = 0; worst_ratios = []
    for trial in range(40):
        M = int(rng.integers(3, 120)); R = int(rng.choice([4, 8, 16]))
        t = rand_tile(rng, M); p = rand_student(rng, R)
        try:
            res = G.certify_tile(t, p, min_size=1.0, max_depth=9, rng=rng, validate=True)
            n_ok += 1
            # also: rigorous bound must be >= dense point max over a fine grid
            worst_pt = 0.0
            for _ in range(300):
                sx = rng.uniform(t["px"][:, 0].min(), t["px"][:, 0].max())
                sy = rng.uniform(t["px"][:, 1].min(), t["px"][:, 1].max())
                worst_pt = max(worst_pt, G.point_error(sx, sy, (t["gids"], t["means2d"], t["conics"], t["opac"], t["colors"], p)))
            worst_ratios.append(res["rigorous_max_abs"] / max(worst_pt, 1e-9))
        except Exception as e:
            n_fail += 1; print("FAIL trial", trial, "M", M, "R", R, "->", repr(e)[:160])
    print(f"VALIDATION: ok={n_ok} fail={n_fail}")
    if worst_ratios:
        wr = np.array(worst_ratios)
        print(f"bound/pointmax ratio: min={wr.min():.3f} median={np.median(wr):.3f} max={wr.max():.3f} (must be >=1.0)")
        print("INTERVAL_VALID" if n_fail == 0 and wr.min() >= 1.0 else "INTERVAL_SUSPECT")

if __name__ == "__main__":
    main()
