#!/usr/bin/env python3
"""P18-P0 Phase 1 (synthetic only) — quick scan-math validation."""
import os, sys, time
import numpy as np
import torch

DEV = "cuda"
AT = 1.0 / 255.0


def build_ext(src):
    from torch.utils.cpp_extension import load
    return load(name="p18_qscan_ref", sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-use_fast_math", "-gencode=arch=compute_80,code=sm_80"])


def metrics(d, q):
    d = d.reshape(-1); q = q.reshape(-1)
    diff = (q - d).abs()
    rel_mask = d > 1e-8
    rel = (diff[rel_mask] / d[rel_mask]).max() if rel_mask.any() else None
    bd = (d < AT).reshape(-1); bq = (q < AT).reshape(-1)
    return dict(max_abs=float(diff.max()), max_rel=(float(rel) if rel is not None else None),
                mean_abs=float(diff.mean()), branch_mm=int((bd != bq).sum()),
                n_nan=int((~torch.isfinite(q)).sum()), n=int(d.numel()))


def main():
    ext = build_ext("/root/epic05_p18/p18_qscan_ref.cu")
    torch.manual_seed(42)
    rows = []
    for mx in (-4.0, -1.0, 0.0, 1.0, 4.0):
        for my in (-3.0, 0.0, 3.0):
            for s in (0.3, 1.0, 3.0, 8.0):
                for opac in (0.1, 0.5, 0.9, 0.99):
                    for tilt in (0.0, 0.3, 0.7):
                        c00 = 1.0 / (s * s); c22 = c00; c11 = tilt * c00
                        A = 0.5 * c00
                        B = c11 + c00 * mx
                        C = 0.5 * c22 * my * my + c11 * mx * my + 0.5 * c00 * (mx * mx + my * my) - c00 * mx
                        rows.append([A, B, C])
    a = torch.tensor([r[0] for r in rows], device=DEV); b = torch.tensor([r[1] for r in rows], device=DEV); c = torch.tensor([r[2] for r in rows], device=DEV)
    n = a.numel()
    print(f"rows={n}")
    for vname, v in (("QSCAN4", 0), ("QSCAN8", 1), ("QSCAN16", 2)):
        var = torch.full((n,), v, device=DEV, dtype=torch.int32)
        for exp2, tag in ((True, "exp2"), (False, "exp")):
            od, oq, om, on = ext.qscan_ref(a, b, c, var, exp2, 1)
            m = metrics(od, oq)
            print(f"{vname}/{tag}: max_abs={m['max_abs']:.3e} max_rel={m['max_rel']} mean_abs={m['mean_abs']:.3e} branch_mm={m['branch_mm']} nan={m['n_nan']}")


if __name__ == "__main__":
    main()
