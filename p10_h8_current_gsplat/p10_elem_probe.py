#!/usr/bin/env python3
"""Element-level probe: are the dominant cross-arm offsets deterministic?

For a tensor, prints the top-K elements by |mean(off) - mean(pri1)| together
with every run's value in all three dumps. Reading:
  p1 runs bit-identical AND p2 runs bit-identical AND both equal, off different
    -> a deterministic displacement introduced by the build at that element;
  p1 != p2 at that element while each is internally stable
    -> the element is not reproducible across processes either, so the offset is
       part of the same process-level spread;
  spread inside a single arm's runs
    -> ordinary atomic ordering noise.
"""
import argparse

KEYS = ["means3d", "quats", "scales", "opacity", "sh", "means2d_raw", "conics_raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--n-gaussians", type=int, required=True)
    a = ap.parse_args()

    import numpy as np
    import torch

    def runs(path, prefix):
        d = torch.load(path, weights_only=False)["dump"]
        return [np.asarray(d[f"{prefix}{i}"][a.key].detach().cpu(), dtype=np.float32).reshape(-1)
                for i in range(3)]

    P1, P2, O = runs(a.p1, "pri"), runs(a.p2, "pri"), runs(a.h8, "off")
    M1 = np.mean(np.stack(P1), axis=0)
    M2 = np.mean(np.stack(P2), axis=0)
    Mo = np.mean(np.stack(O), axis=0)

    n = M1.size
    K = n // a.n_gaussians
    print(f"scene={a.scene} key={a.key} n_elements={n} n_gaussians={a.n_gaussians} "
          f"elements_per_gaussian={K}")

    for name, D in (("p1->off", Mo - M1), ("p1->p2", M2 - M1), ("p2->off", Mo - M2)):
        idx = np.argsort(-np.abs(D))[:a.topk]
        print(f"-- top |D| for {name} (D_absmax={np.abs(D).max():.3e})")
        for e in idx:
            g, c = divmod(int(e), K)
            vals = [f"{v[e]:+.8e}" for v in P1] + [f"{v[e]:+.8e}" for v in P2] + [f"{v[e]:+.8e}" for v in O]
            same1 = len({float(v[e]) for v in P1}) == 1
            same2 = len({float(v[e]) for v in P2}) == 1
            sameO = len({float(v[e]) for v in O}) == 1
            print(f"   elem {e} (g{g}.{c}) D={D[e]:+.3e} "
                  f"p1={vals[0:3]} p2={vals[3:6]} off={vals[6:9]} "
                  f"stable(p1,p2,off)=({same1},{same2},{sameO})")


if __name__ == "__main__":
    main()
