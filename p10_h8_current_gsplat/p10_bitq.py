#!/usr/bin/env python3
"""Bit-level cross-build comparison of the shared tensors.

For each tensor, counts elements that differ bit-wise (d != 0 exactly) between
arm pairs.  Within-arm pairs are the reproducibility control, cross-build pairs
are the question.  Any cross-build difference on a tensor that is deterministic
within each arm localises where the two compilations diverge.
"""
import argparse

import numpy as np
import torch


def arr(t):
    return np.asarray(t.detach().cpu(), dtype=np.float64).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--pri-prefix", default="pri")
    ap.add_argument("--h8-prefix", default="off")
    ap.add_argument("--keys", default="render,alpha,means2d,conics,means2d_raw,conics_raw,"
                                      "means3d,quats,scales,opacity,sh")
    args = ap.parse_args()
    pri = torch.load(args.pristine, weights_only=False)
    h8 = torch.load(args.h8, weights_only=False)
    dp, dh = pri["dump"], h8["dump"]

    PP, HH = args.pri_prefix, args.h8_prefix

    def get(arm, idx):
        return dp[f"{arm}{idx}"] if arm in (PP,) else dh[f"{arm}{idx}"]

    pairs = [(PP, 0, PP, 1), (PP, 0, PP, 2), (HH, 0, HH, 1), (HH, 0, HH, 2),
             (PP, 0, HH, 0), (PP, 0, HH, 1), (PP, 0, HH, 2),
             (PP, 1, HH, 0), (PP, 2, HH, 2)]
    print(f"scene={pri['scene']} N={pri['n_gaussians']}")
    print(f"{'tensor':13s} " + " ".join(f"{a}{i}-{b}{j}" for a, i, b, j in pairs))
    for k in args.keys.split(","):
        row = []
        for a, i, b, j in pairs:
            d = arr(get(a, i)[k]) - arr(get(b, j)[k])
            neq = int(np.count_nonzero(d))
            row.append(f"{neq}({np.abs(d).max():.2g})")
        print(f"{k:13s} " + " ".join(f"{x:>17s}" for x in row))


if __name__ == "__main__":
    main()
