#!/usr/bin/env python3
"""Pre-fix defect forensics (CPU only): characterize which recomputed row each
stored mask16 actually belongs to, per rec_r slot class and for the failing
subset (stored != row 12+(rec_r&3)).
"""
import numpy as np

OUT = "/root/epic05_p18_r1/artifacts"
d = np.load(f"{OUT}/P18_R1_TRACE_room_prefix.npz")
d16 = np.load(f"{OUT}/P18_R1_D16_room_prefix_prefix_pre.npy")
rr = d["rec_r"].astype(np.int64)
mask16 = d["mask16"].astype(np.uint32)
n = d16.shape[0]
assert d16.shape == (n, 16)
Z = d16 == 0
pop = np.array([bin(int(m)).count("1") for m in mask16], dtype=np.int32)

print(f"n={n}")
print("per rec_r: frac(d16[i,row]==0) for row=0..15")
for k in range(16):
    sel = rr == k
    fr = Z[sel].mean(axis=0)
    print(f"r={k:2d} " + " ".join(f"{v:.3f}" for v in fr))

print()
print("columns as canonical rows: bal0 lower(ty0)=4pp .. bal1 upper(ty1)=4pp+3")
for ty in (0, 1):
    for rh in (0, 1):
        cls = (rr & 1) == ty
        cls &= ((rr >> 1) & 1) == rh
        sub = Z[cls]
        print(f"class ty={ty} rh={rh} n={sub.shape[0]}")
        for pp in range(4):
            row = 4 * pp + 2 * rh + ty
            print(f"   pp={pp} row={row:2d}: {sub[:, row].mean():.4f}")

print()
for k in (2, 3, 14, 15):
    sel = rr == k
    content = 12 + (k & 3)
    fails = ~Z[sel][:, content]
    sub = Z[sel][fails]
    nf = sub.shape[0]
    print(f"rec_r={k} content_row={content} fails={nf} ({nf / sel.sum():.4f})")
    if nf:
        cnt = sub.mean(axis=0)
        print("   among fails, zero frac per row: " + " ".join(f"{v:.3f}" for v in cnt))
        print(f"   pop: fails mean={pop[sel][fails].mean():.2f} median={np.median(pop[sel][fails]):.0f}"
              f" | succ mean={pop[sel][~fails].mean():.2f} median={np.median(pop[sel][~fails]):.0f}")
        # first zero row of fails
        first = Z[sel][fails].argmax(axis=1)
        print("   first-zero-row hist: " + str(np.bincount(first, minlength=16).tolist()))
