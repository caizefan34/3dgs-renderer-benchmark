#!/usr/bin/env python3
"""Triangle test for the cross-build inertness control.

Three dumps per scene: pristine process 1 (p1), pristine process 2 (p2), and the
patched binary with the switch off (off, same process as on). Both pristine
processes ran the identical call sequence (3 main runs, each followed by an
absgrad run), so a call-index effect is controlled.

The question the inertness control must answer is not "does off == pri to the
last bit" (nothing does, the backward is atomic-scattered and its rounding
pattern is a per-process property) but "is off displaced from the pristine
distribution". Two readings of a failing ratio:

  process-level   each process draws its own rounding pattern; off is a third
                  draw -> d(p1,off) ~ d(p2,off) ~ d(p1,p2), and the two
                  offsets that involve off point in *unrelated* directions
                  (small |cos| between d(p1,off) and d(p2,off), and off does not
                  sit consistently on one side of both).
  build-level     the patched build's arithmetic (register allocation, atomic
                  ordering, instruction scheduling) systematically shifts the
                  result -> d(p1,off) ~ d(p2,off) >> d(p1,p2) AND the two
                  offsets agree in direction (cos ~ +1): off is displaced from
                  the *pair* of pristine processes along one axis.

Reported per tensor: mean-vector offsets (absmax, L2, L2 relative to the mean
magnitude) for the three pairs, the direction agreement, and the
deterministic-element statistic of each pair.
"""
import argparse
import json
import sys

sys.path.insert(0, "/root/epic05_p10")
from p10_h2_equivalence import det_offset_stats, _np  # noqa: E402

KEYS = ["means3d", "quats", "scales", "opacity", "sh", "means2d_raw", "conics_raw"]


def loads(p1, p2, h8, key):
    import torch
    def runs(path, prefix):
        d = torch.load(path, weights_only=False)["dump"]
        return [d[f"{prefix}{i}"][key] for i in range(3)]
    return runs(p1, "pri"), runs(p2, "pri"), runs(h8, "off")


def mean_vec(ts):
    import numpy as np
    return np.mean(np.stack([_np(t) for t in ts]), axis=0)


def pair_stats(Ma, Mb):
    import numpy as np
    d = Mb - Ma
    na, nb = np.linalg.norm(Ma), np.linalg.norm(Mb)
    return {
        "offset_absmax": float(np.abs(d).max()),
        "offset_l2": float(np.linalg.norm(d)),
        "offset_rel_l2": float(np.linalg.norm(d) / max(na, 1e-30)),
        "value_absmax": float(max(np.abs(Ma).max(), np.abs(Mb).max())),
        "n_elements": int(d.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np

    out = {"scene": a.scene,
           "question": "is the patched-with-switch-off arm displaced from the "
                       "pristine process distribution, or is it a third draw from it?",
           "dumps": {"p1": a.p1, "p2": a.p2, "h8_off": a.h8},
           "keys": {}}
    for k in KEYS:
        try:
            P1, P2, O = loads(a.p1, a.p2, a.h8, k)
        except KeyError:
            continue
        M1, M2, Mo = mean_vec(P1), mean_vec(P2), mean_vec(O)
        d12, d1o, d2o = M2 - M1, Mo - M1, Mo - M2
        n1, n2 = np.linalg.norm(d1o), np.linalg.norm(d2o)
        cos = float(np.dot(d1o, d2o) / max(n1 * n2, 1e-300))
        dref = float(np.linalg.norm(d12))
        # off's perpendicular displacement from the pristine pair's line: the part
        # of the displacement that cannot be explained by moving along d(p1,p2)
        u = d12 / max(dref, 1e-300)
        proj = float(np.dot(d1o, u))
        perp = float(np.linalg.norm(d1o - proj * u))
        out["keys"][k] = {
            "p1_p2": pair_stats(M1, M2),
            "p1_off": pair_stats(M1, Mo),
            "p2_off": pair_stats(M2, Mo),
            "cross_build_over_cross_process_l2": float(np.linalg.norm(d1o) / max(dref, 1e-300)),
            "off_offsets_direction_cosine": cos,
            "off_displacement_along_p1p2": proj,
            "off_displacement_perpendicular": perp,
            "off_perp_over_p1p2_l2": float(perp / max(dref, 1e-300)),
            "det_offset_p1_p2": det_offset_stats(P1, P2),
            "det_offset_p1_off": det_offset_stats(P1, O),
            "det_offset_p2_off": det_offset_stats(P2, O),
        }
        r = out["keys"][k]
        print(f"{a.scene} {k:12s} "
              f"l2 p1p2={dref:.3e} p1off={np.linalg.norm(d1o):.3e} p2off={n2:.3e} "
              f"(cb/cp={r['cross_build_over_cross_process_l2']:.2f}) "
              f"cos={cos:+.3f} perp/p1p2={r['off_perp_over_p1p2_l2']:.2f} "
              f"absmax p1p2={r['p1_p2']['offset_absmax']:.3e} p1off={r['p1_off']['offset_absmax']:.3e}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"TRIANGLE {a.scene} -> {a.out}")


if __name__ == "__main__":
    main()
