#!/usr/bin/env python3
"""R1-B mask-correctness check v2.

Restrict analysis to *informative* records (recomputed claimed-row mask is not
all-0 / all-1) and locate the row whose recomputed mask equals the stored mask.
Predictions:
  records correct                          -> shift 0 dominates
  last-pp-write overwrite bug              -> shift +4*(3-pp) spikes (25% each)
"""
import collections
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

H = np.float16
CHOL_SCALE = np.float32(np.sqrt(np.log2(np.e) / 2.0)) * np.float32(16.0)
LOG2_INV_AT = np.float32(7.9943534)
FUSED_W, FUSED_H = 8, 4

out = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_p18_r1/artifacts"
scene = sys.argv[2] if len(sys.argv) > 2 else "room"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

d = np.load(f"{out}/P18_R1_TRACE_{scene}.npz")
np.random.seed(0)
idx = np.random.choice(len(d["g"]), size=min(NS, len(d["g"])), replace=False)
x0 = np.int32(d["x0"][idx])
y0 = np.int32(d["y0"][idx])
mask_stored = d["mask16"][idx].astype(np.uint32)
r = d["rec_r"][idx].astype(np.int32)
n = len(idx)

mx = np.float32(d["pos_x"][idx])
my = np.float32(d["pos_y"][idx])
tx = np.arange(16, dtype=np.int32)
tl_col = ((x0 // 16) % FUSED_W).astype(np.float32)[:, None]
tl_row = ((y0 // 16) % FUSED_H).astype(np.float32)
mt_col = (x0 // (FUSED_W * 16)).astype(np.float32)
mt_row = (y0 // (FUSED_H * 16)).astype(np.float32)

ch4 = d["conic_h4"][idx].view(np.float16)          # raw halves: l0,l1,l2,opacity
opac = ch4[:, 3].astype(np.float32)
nlo = (-np.log2(np.maximum(opac, np.float32(1e-30)))).astype(np.float16)
l0 = H(ch4[:, 0].astype(np.float32) * CHOL_SCALE)
l1 = H(ch4[:, 1].astype(np.float32) * CHOL_SCALE)
l2 = H(ch4[:, 2].astype(np.float32) * CHOL_SCALE)

xy_x = H(mx / 16.0 - (mt_col * FUSED_W + FUSED_W * 0.5))
xy_y = H(my / 16.0 - (mt_row * FUSED_H + FUSED_H * 0.5))
dx_all = H(xy_x[:, None] - H(tl_col + tx[None, :] / 16.0 + 0.03125 - 4.0))

tl_row_s = tl_row
xy_y_s = xy_y
l0_s, l1_s, l2_s, nlo_s = l0, l1, l2, nlo
dx_all_s = dx_all


def mask_for_row(row_in_tile):
    """row_in_tile: [m] int32 -> mask [m] uint32; uses the currently selected
    per-record arrays (module-level *_s)."""
    ty = (row_in_tile & 1).astype(np.float32)
    base = H(tl_row_s + ty / 16.0 + 0.03125 - 2.0)
    dy = H(xy_y_s - H(base + (row_in_tile >> 1).astype(np.float32) * 0.125))
    u0_base = H(l0_s.astype(np.float32)[:, None] * dx_all_s.astype(np.float32))
    u0 = H(l1_s.astype(np.float32)[:, None] * dy.astype(np.float32)[:, None]
           + u0_base.astype(np.float32))
    u1 = H(l2_s * dy)
    sig = H(u1.astype(np.float32)[:, None] ** 2
            + H(u0.astype(np.float32) ** 2 + nlo_s.astype(np.float32)[:, None]))
    ok = (sig.astype(np.float32) < LOG2_INV_AT).astype(np.uint32)
    m = np.zeros(len(row_in_tile), dtype=np.uint32)
    for b in range(16):
        m |= ok[:, b] << b
    return m


def set_sel(sel):
    global tl_row_s, xy_y_s, l0_s, l1_s, l2_s, nlo_s, dx_all_s
    tl_row_s, xy_y_s = tl_row[sel], xy_y[sel]
    l0_s, l1_s, l2_s, nlo_s = l0[sel], l1[sel], l2[sel], nlo[sel]
    dx_all_s = dx_all[sel]


row_off = ((r & 1) + 4 * (r >> 2) + 2 * ((r >> 1) & 1)).astype(np.int32)
set_sel(np.arange(n))
claimed = mask_for_row(row_off)

info = (claimed != 0) & (claimed != 0xFFFF)
print(f"[{scene}] n={n} informative(claimed)={int(info.sum())} ({info.mean():.3f})")
print(f"   stored pop0={(mask_stored==0).mean():.3f} pop16={(mask_stored==0xFFFF).mean():.3f} | "
      f"recomputed pop0={(claimed==0).mean():.3f} pop16={(claimed==0xFFFF).mean():.3f}")
agree = (claimed == mask_stored)
print(f"   overall claimed-row agreement {agree.mean():.4f}; on informative {agree[info].mean():.4f}")

sub_pp = (r >> 2)[np.nonzero(info)[0]]
sub_h = ((r >> 1) & 1)[np.nonzero(info)[0]]
sub_ty = (r & 1)[np.nonzero(info)[0]]
sub_pp = sub_pp[:3000]; sub_h = sub_h[:3000]; sub_ty = sub_ty[:3000]
seli = np.nonzero(info)[0]
sub = seli[: min(3000, len(seli))]
print(f"   shift scan on {len(sub)} informative records")
set_sel(sub)
best = np.full(len(sub), -1, dtype=np.int32)
bestag = np.zeros(len(sub), dtype=np.int32)
mats = {}
for tl in range(16):
    m = mask_for_row(np.full(len(sub), tl, dtype=np.int32))
    mats[tl] = m
    eq = (m == mask_stored[sub]).astype(np.int32)
    upd = eq > bestag
    bestag[upd] = eq[upd]
    best[upd] = tl
shift = best - row_off[sub]
print("   shift histogram:", sorted(collections.Counter(shift.tolist()).items()))
print("   best agreement(whole-mask equal): mean %.3f" % bestag.mean())
tab = collections.Counter()
for j in range(len(sub)):
    tab[(int(sub_pp[j]), int(sub_h[j]), int(sub_ty[j]), int(best[j]))] += 1
agg = collections.Counter()
for (ppv, hh, tyv, brow), c in tab.items():
    agg[(ppv, brow - (tyv + 4 * ppv + 2 * hh))] += c
print("   (pp, best_row - claimed_row) counts:", sorted(agg.items()))
row_hist = collections.Counter()
for (ppv, hh, tyv, brow), c in tab.items():
    row_hist[(ppv, brow)] += c
print("   row distribution per pp:")
for ppv in range(4):
    items = [(brow, c) for (p2, brow), c in row_hist.items() if p2 == ppv]
    print(f"     pp={ppv}:", sorted(items)[:20])

bad = np.nonzero(bestag < 12)[0]
print(f"   records with best agreement<12: {len(bad)}")
for j in bad[:8]:
    rr = r[sub[j]]
    print(f"    rec r={rr} pp={rr>>2} h={(rr>>1)&1} ty={rr&1} row_off={row_off[sub[j]]} "
          f"stored={mask_stored[sub[j]]:016b} claimed={mats[row_off[sub[j]]][j]:016b} "
          f"best_row={best[j]} best={mats[best[j]][j]:016b} agreed={bestag[j]}")
