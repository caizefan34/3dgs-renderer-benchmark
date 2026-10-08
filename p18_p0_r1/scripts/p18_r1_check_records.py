#!/usr/bin/env python3
"""R1-B integrity check: compare the record-level live-pop histogram (from the
dumped primary records) against the kernel inline full-population histogram."""
import json
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_p18_r1/artifacts"
man = json.load(open(f"{out}/P18_R1_TRACE_MANIFEST.json"))

for scene in ["room", "bicycle"]:
    d = np.load(f"{out}/P18_R1_TRACE_{scene}.npz")
    print("==", scene, "keys:", sorted(d.keys()))
    per_cam = man["scenes"][scene]["per_cam"]
    all_pop = []
    cam_of = d["cam"]
    for ci in man["scenes"][scene]["cams"]:
        sel = cam_of == ci
        if not sel.any():
            continue
        m = d["mask16"][sel]
        pop = np.array([bin(int(x)).count("1") for x in m], dtype=np.int64)
        all_pop.append(pop)
        entry = per_cam[str(ci)]
        hist = np.array(entry["hist17"], dtype=np.float64)
        # inline hist counts each real row 4x per call (4 pp iterations), ungated;
        # records are 1/256-gated: compare normalized shapes
        frac_rec = np.bincount(pop, minlength=17)[:17] / max(len(pop), 1)
        frac_hist = hist / max(hist.sum(), 1)
        mean_rec = float((np.arange(17) * np.bincount(pop, minlength=17)[:17]).sum() / max(len(pop), 1))
        hp = hist[1:]
        mean_hist = float((hp * np.arange(1, 17)).sum() / max(hp.sum(), 1))
        print(f"cam {ci}: n_rec={len(pop)} mean_pop_rec={mean_rec:.3f} mean_pop_hist={mean_hist:.4f} "
              f"frac0_rec={frac_rec[0]:.4f} frac0_hist={frac_hist[0]:.4f} "
              f"frac16_rec={frac_rec[16]:.4f} frac16_hist={frac_hist[16]:.4f} "
              f"L1delta={np.abs(frac_rec - frac_hist).max():.5f}")
    if all_pop:
        pop = np.concatenate(all_pop)
        h = np.bincount(pop, minlength=17)[:17]
        print(f"scene {scene} TOTAL n_rec={len(pop)} pop_hist={h.tolist()}")
        print(f"  mean={pop.mean():.4f} median={int(np.median(pop))} "
              f"frac0={h[0]/len(pop):.5f} frac16={h[16]/len(pop):.5f} "
              f"frac_live_ge1={(len(pop)-h[0])/len(pop):.5f}")
