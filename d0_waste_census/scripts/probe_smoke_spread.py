#!/usr/bin/env python3
"""D0: smoke-run spread probe. Compares per-row loss/N traces across clean and traced runs."""
import json
import sys

import numpy as np

RUNS = "/root/epic05_d0/runs"
PAIRS = {
    "room": ("C3room1", "C3room2", "T3room"),
    "bicycle": ("C3bicycle", None, "T3bicycle"),
}


def load(rid):
    tr = json.load(open(f"{RUNS}/{rid}/results.json"))
    rows = tr.get("per_iter", [])
    return (np.array([r["iter"] for r in rows]),
            np.array([r["loss"] for r in rows]),
            np.array([r["n_gs"] for r in rows]))


for scene, (c1, c2, tr) in PAIRS.items():
    it, l1, n1 = load(c1)
    _, lt, nt = load(tr)
    l2 = n2 = None
    if c2:
        _, l2, n2 = load(c2)
    print(f"===== {scene}: rows={len(it)} iters={it[:3]}...{it[-3:]}")
    hdr = f"{'iter':>6} {'L_clean':>10} "
    if l2 is not None:
        hdr += f"{'L_clean2':>10} "
    hdr += f"{'L_trace':>10} "
    if l2 is not None:
        hdr += f"{'d(cc)':>9} "
    hdr += f"{'d(ct)':>9} | {'N_clean':>8} "
    if n2 is not None:
        hdr += f"{'N_clean2':>8} "
    hdr += f"{'N_trace':>8}"
    print(hdr)
    for i, t in enumerate(it):
        show = (t % 1000 == 0) or i == len(it) - 1 or abs(l1[i] - lt[i]) > 0.02
        if not show:
            continue
        line = f"{t:>6} {l1[i]:>10.5f} "
        if l2 is not None:
            line += f"{l2[i]:>10.5f} "
        line += f"{lt[i]:>10.5f} "
        if l2 is not None:
            line += f"{abs(l1[i] - l2[i]):>9.5f} "
        line += f"{abs(l1[i] - lt[i]):>9.5f} | {n1[i]:>8} "
        if n2 is not None:
            line += f"{n2[i]:>8} "
        line += f"{nt[i]:>8}"
        print(line)
    # where is the max loss diff
    d = np.abs(l1 - lt)
    j = int(d.argmax())
    print(f"  max d(clean,trace)={d[j]:.5f} at iter {it[j]} "
          f"({'row0' if j == 0 else 'row%d' % j})")
    if l2 is not None:
        dc = np.abs(l1 - l2)
        print(f"  max d(clean,clean2)={dc.max():.5f} at iter {it[int(dc.argmax())]}")
        print(f"  rows differing: clean-clean2={int((dc>0).sum())}/{len(it)} "
              f"clean-trace={int((d>0).sum())}/{len(it)}")
        nd = np.abs(n1 - n2)
        nt2 = np.abs(n1 - nt)
        print(f"  N: cc max diff={nd.max()} at {it[int(nd.argmax())]}; "
              f"ct max diff={nt2.max()} at {it[int(nt2.argmax())]}")
    dn = np.abs(n1 - nt)
    print(f"  N: first nonzero d(clean,trace) row={int((dn>0).argmax())} "
          f"iter={it[int((dn>0).argmax())]}")
