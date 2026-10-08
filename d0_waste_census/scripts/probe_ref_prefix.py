#!/usr/bin/env python3
"""D0: compare smoke runs vs the 30K P12 reference prefix (iterations 1..3000, same recipe)."""
import json

import numpy as np

REF = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_{scene}/results.json"
RUNS = "/root/epic05_d0/runs"


def load(path):
    tr = json.load(open(path))
    rows = tr.get("per_iter", [])
    return (np.array([r["iter"] for r in rows]),
            np.array([r["loss"] for r in rows]),
            np.array([r["n_gs"] for r in rows]))


for scene, rids in (("room", ("C3room1", "C3room2", "T3room")),
                    ("bicycle", ("C3bicycle", "T3bicycle"))):
    it, lr, nr = load(REF.format(scene=scene))
    k = int((it <= 3000).sum())
    it, lr, nr = it[:k], lr[:k], nr[:k]
    print(f"===== {scene}: reference rows<=3000: {k} (last iter {it[-1]})")
    runs = {rid: load(f"{RUNS}/{rid}/results.json") for rid in rids}
    # first divergence row (loss and N) between ref and each run
    for rid, (it2, l2, n2) in runs.items():
        assert np.array_equal(it, it2), (rid, it[:3], it2[:3])
        dl = np.abs(lr - l2)
        dn = np.abs(nr - n2)
        jl, jn = int(dl.argmax()), int((dn > 0).argmax()) if (dn > 0).any() else -1
        print(f"  {rid}: max|dL|={dl.max():.5f}@iter{it[int(dl.argmax())]}  "
              f"first dN!=0 @row{jn} iter{it[jn] if jn>=0 else '-'}  "
              f"finalN ref={nr[-1]} run={n2[-1]} d={n2[-1]-nr[-1]} "
              f"({100.0*(n2[-1]-nr[-1])/nr[-1]:+.2f}%)")
    # columns next to the 800 spike for room
    if scene == "room":
        print("  iter | L_ref  L_c1  L_c2  L_tr | N_ref  N_c1  N_c2  N_tr")
        for i, t in enumerate(it):
            ltr = runs["T3room"][1][i]
            if abs(lr[i] - ltr) > 0.02 or (700 <= t <= 900 and t % 100 == 0):
                print(f"  {t:>4} | {lr[i]:.5f} {runs['C3room1'][1][i]:.5f} "
                      f"{runs['C3room2'][1][i]:.5f} {ltr:.5f} | {nr[i]} "
                      f"{runs['C3room1'][2][i]} {runs['C3room2'][2][i]} "
                      f"{runs['T3room'][2][i]}")
        # top-5 loss diffs ref vs trace
        dl = np.abs(lr - runs["T3room"][1])
        idx = np.argsort(-dl)[:5]
        print("  top-5 |dL| ref-vs-trace:", [(int(it[i]), round(float(dl[i]), 5))
                                            for i in idx])
