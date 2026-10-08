#!/usr/bin/env python3
"""Elementwise deep dive on the cross-build raw boundary buffers (pristine vs
patched switch-off).  Settles whether the bicycle conics_raw offset is a
deterministic build difference or run-to-run atomic sampling.

CPU only; operates on the existing H2 dumps.  Tensors are flattened, and the
per-gaussian stride K is recovered as size / n_gaussians (the dumps keep the
batch dim, e.g. conics is [1, N, 3]), so every element maps to
(gaussian, component).

Reported per tensor:
  - |value| scale, relative size of the cross-build mean offset ||D|| / ||M||
  - per-element worst same-build scatter W (max over the 3 within-arm pairs,
    both arms) and how the offset D = mean(off) - mean(pri) compares to it
  - ELEMENTS WHERE W == 0 BUT D != 0: same-build arithmetic is bit-reproducible
    there, so a nonzero offset can only come from a codegen difference
  - concentration: share of ||D||^2 in the top-1 / 10 / 100 elements
  - top-k elements with every run's value, the offset and its ratio to W
  - for the affected gaussians: conic conditioning from meta['conics']
  - the leaf gradients at the same gaussians (does the offset propagate?)
"""
import argparse
import json

import numpy as np
import torch

PAIRS = ((0, 1), (0, 2), (1, 2))


def arr(t):
    return np.asarray(t.detach().cpu(), dtype=np.float64).reshape(-1)


def rel(a, b):
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-300))


def unfold(i, K):
    return {"flat": int(i), "gaussian": int(i // K), "component": int(i % K)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--pri-prefix", default="pri")
    ap.add_argument("--h8-prefix", default="off")
    ap.add_argument("--keys", default="conics_raw,means2d_raw")
    ap.add_argument("--leaf", default="scales,quats,means3d,opacity,sh")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pri = torch.load(args.pristine, weights_only=False)
    h8 = torch.load(args.h8, weights_only=False)
    assert pri["scene"] == h8["scene"]
    scene = pri["scene"]
    n_gauss = pri["n_gaussians"]
    dp, dh = pri["dump"], h8["dump"]

    def stack(d, arm, key):
        return np.stack([arr(d[f"{arm}{i}"][key]) for i in range(3)])

    PREFIX_P, PREFIX_H = args.pri_prefix, args.h8_prefix
    res = {"scene": scene, "n_gaussians": n_gauss,
           "arms": {"pri": PREFIX_P, "off": PREFIX_H}, "keys": {}}
    for key in args.keys.split(","):
        P = stack(dp, PREFIX_P, key)
        H = stack(dh, PREFIX_H, key)
        F = P.shape[1]
        K = F // n_gauss
        Mp, Mh = P.mean(0), H.mean(0)
        D = Mh - Mp
        Wp = np.max(np.stack([np.abs(P[a] - P[b]) for a, b in PAIRS]), 0)
        Wh = np.max(np.stack([np.abs(H[a] - H[b]) for a, b in PAIRS]), 0)
        W = np.maximum(Wp, Wh)
        ad = np.abs(D)
        order = np.argsort(ad)[::-1]
        tot = float(np.sum(ad ** 2)) or 1e-300
        conc = {f"top{n}_share_l2sq": float(np.sum(ad[order[:n]] ** 2) / tot)
                for n in (1, 10, 100, 1000)}
        zero_w = W == 0
        blk = {
            "flat_size": int(F), "per_gaussian_stride": int(K),
            "value_absmax": {"pri": float(np.max(np.abs(Mp))), "off": float(np.max(np.abs(Mh)))},
            "offset_rel_l2": rel(D, Mp),
            "offset_absmax": float(ad.max()),
            "same_build_scatter_absmax": float(W.max()),
            "n_elements": int(ad.size),
            "n_offset_gt_scatter": int(np.sum(ad > W)),
            "n_offset_gt_5x_scatter": int(np.sum(ad > 5 * W)),
            "n_offset_gt_10x_scatter": int(np.sum(ad > 10 * W)),
            "n_deterministic_elements_with_offset": int(np.sum(zero_w & (ad > 0))),
            "concentration": conc,
            "top": [],
        }
        top_gauss = []
        for i in order[:args.topk]:
            i = int(i)
            sign_agree = int(sum(np.sign(H[j][i] - P[k][i]) == np.sign(D[i])
                                 for k in range(3) for j in range(3) if D[i] != 0))
            blk["top"].append({
                "index": unfold(i, K),
                "value": {"pri_mean": float(Mp[i]), "off_mean": float(Mh[i]),
                          "pri_runs": [float(v) for v in P[:, i]],
                          "off_runs": [float(v) for v in H[:, i]]},
                "offset": float(D[i]),
                "offset_rel": float(ad[i] / max(abs(Mp[i]), 1e-300)),
                "same_build_scatter": float(W[i]),
                "ratio_offset_over_scatter": float(ad[i] / max(W[i], 1e-300)),
                "sign_agreement_9pairs": f"{sign_agree}/9",
            })
            if i // K not in top_gauss:
                top_gauss.append(i // K)
        top_gauss = top_gauss[:5]
        if "conics" in dh[f"{PREFIX_H}0"] and key == "conics_raw":
            co = arr(dh[f"{PREFIX_H}0"]["conics"])
            cp = arr(dp[f"{PREFIX_P}0"]["conics"])
            blk["top_gaussians_conic_state"] = [{
                "g": g,
                "conics_off0": [float(x) for x in co[g * 3:(g + 1) * 3]],
                "conics_pri0": [float(x) for x in cp[g * 3:(g + 1) * 3]],
                "conic_absmax": float(np.max(np.abs(co[g * 3:(g + 1) * 3]))),
            } for g in top_gauss]
        if key == "conics_raw":
            blk["leaf_offsets_at_affected_gaussians"] = {}
            for lk in args.leaf.split(","):
                LP, LH = stack(dp, PREFIX_P, lk), stack(dh, PREFIX_H, lk)
                if LP.shape[1] % n_gauss:
                    continue
                lk_K = LP.shape[1] // n_gauss
                d0, m0 = LH.mean(0) - LP.mean(0), LP.mean(0)
                w0 = np.maximum(np.max(np.stack([np.abs(LP[a] - LP[b]) for a, b in PAIRS]), 0),
                                np.max(np.stack([np.abs(LH[a] - LH[b]) for a, b in PAIRS]), 0))
                gmask = np.zeros(LP.shape[1], dtype=bool)
                for g in top_gauss:
                    gmask[g * lk_K:(g + 1) * lk_K] = True
                blk["leaf_offsets_at_affected_gaussians"][lk] = {
                    "stride": int(lk_K),
                    "global_offset_rel_l2": rel(d0, m0),
                    "global_offset_absmax": float(np.abs(d0).max()),
                    "global_scatter_absmax": float(w0.max()),
                    "affected_gaussians_offset_absmax": float(np.abs(d0[gmask]).max()),
                    "affected_gaussians_scatter_absmax": float(w0[gmask].max()),
                }
        res["keys"][key] = blk

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"=== conics probe {scene} (pri vs off, 3 runs each, N={n_gauss}) ===")
    for key, blk in res["keys"].items():
        print(f"[{key}] flat={blk['flat_size']} K={blk['per_gaussian_stride']} "
              f"|v|max pri={blk['value_absmax']['pri']:.6g} off={blk['value_absmax']['off']:.6g} "
              f"||D||/||M||={blk['offset_rel_l2']:.3e}")
        print(f"  offset_absmax={blk['offset_absmax']:.6g} scatter_absmax={blk['same_build_scatter_absmax']:.6g} "
              f"n(D>W)={blk['n_offset_gt_scatter']} n(D>5W)={blk['n_offset_gt_5x_scatter']} "
              f"n(D>10W)={blk['n_offset_gt_10x_scatter']} deterministic_elements={blk['n_deterministic_elements_with_offset']}")
        print(f"  concentration: " + " ".join(f"{k}={v:.3f}" for k, v in blk["concentration"].items()))
        for t in blk["top"][:3]:
            v = t["value"]
            print(f"  top g={t['index']['gaussian']} c={t['index']['component']} D={t['offset']:.6g} "
                  f"({t['offset_rel']:.2e} rel) W={t['same_build_scatter']:.3g} "
                  f"D/W={t['ratio_offset_over_scatter']:.3g} sign={t['sign_agreement_9pairs']}")
            print(f"      pri={['%.6g' % x for x in v['pri_runs']]}")
            print(f"      off={['%.6g' % x for x in v['off_runs']]}")
        for c in blk.get("top_gaussians_conic_state", []):
            print(f"  gauss {c['g']}: conics off0={['%.4g' % x for x in c['conics_off0']]} "
                  f"|max|={c['conic_absmax']:.4g}")
        for lk, s in blk.get("leaf_offsets_at_affected_gaussians", {}).items():
            print(f"  leaf[{lk}] global rel_l2={s['global_offset_rel_l2']:.3e} "
                  f"absmax={s['global_offset_absmax']:.4g} scatter={s['global_scatter_absmax']:.4g} | "
                  f"affected gaussians D={s['affected_gaussians_offset_absmax']:.4g} "
                  f"W={s['affected_gaussians_scatter_absmax']:.4g}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
