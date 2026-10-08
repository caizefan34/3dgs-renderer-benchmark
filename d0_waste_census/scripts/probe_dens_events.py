#!/usr/bin/env python3
"""D0: densify-event comparison at a fixed iteration window across runs."""
import json
import sys

import numpy as np

REF = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_{scene}/results.json"
RUNS = "/root/epic05_d0/runs"

scene = sys.argv[1] if len(sys.argv) > 1 else "room"
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 500
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 1200

sources = [("ref", REF.format(scene=scene))]
sources += [(rid, f"{RUNS}/{rid}/results.json") for rid in
            (("C3room1", "C3room2", "T3room") if scene == "room" else
             ("C3bicycle", "T3bicycle"))]

evs = {}
for name, path in sources:
    tr = json.load(open(path))
    evs[name] = {e["iter"]: e for e in tr.get("densification_events", [])
                 if lo <= e["iter"] <= hi}

iters = sorted({i for e in evs.values() for i in e})
hdr = f"{'iter':>5} | " + " | ".join(
    f"{n:>34}" for n, _ in sources)
print(hdr)
sub = f"{'':>5} | " + " | ".join(
    f"{'nb->na':>13} {'cl/spl/pr':>11} {'sel':>7}" for _ in sources)
print(sub)
for it in iters:
    row = f"{it:>5} | "
    cells = []
    for name, _ in sources:
        e = evs[name].get(it)
        if e is None:
            cells.append(f"{'-':>34}")
            continue
        g = e.get("grad_norm_stats") or {}
        cells.append(f"{e['n_before']:>6}->{e['n_after']:<6} "
                     f"{e['cloned']:>3}/{e['split']:>3}/{e['pruned_total']:>3} "
                     f"{e.get('selected', -1):>7}")
    print(row + " | ".join(cells))

# candidate-set divergence: for the widest event in the window, compare
# selected counts and grad stats
print()
for name, _ in sources:
    e = evs[name].get(800)
    if e:
        g = e.get("grad_norm_stats") or {}
        s = e.get("selected_grad_norm_stats") or {}
        print(f"{name} iter800: loss={e.get('loss'):.5f} sel={e.get('selected')} "
              f"grad_p99={g.get('p99')} sel_mean={s.get('mean')} "
              f"nan={e.get('grad_nan_count')} inf={e.get('grad_inf_count')}")
