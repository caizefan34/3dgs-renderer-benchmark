#!/usr/bin/env python3
"""P18-P0-R1 shared helpers: real-trace workload builder + evaluator build.

Workload: from P18_R1_TRACE_<scene>.npz, one item per (record, pixel t) =
one baseline inner-loop G(x) evaluation on a REAL contiguous pixel.
Record-major layout: item = 16*record + t, so SUB in {4,8,16} divides the 16-px
record and no subgroup straddles a record boundary.

sigma-domain quadratic per record (kernel-equivalent, float):
  s = 1/16; dy = (pos_y - y0)*s - 0.03125
  L0 = l0*CHOL_SCALE; L1 = l1*CHOL_SCALE; L2 = l2*CHOL_SCALE; nlo = -log2(opac)
  s2 = L0*s; M = L0*(s*pos_x - 0.03125) + L1*dy
  a = s2*s2; b = -2*s2*M; c = M*M + (L2*dy)^2 + nlo
  q(x) = a x^2 + b x + c  (x = global pixel coordinate, pixel center x+0.5 built in)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

ART = Path("/root/epic05_p18_r1/artifacts")
CU = Path("/root/epic05_p18_r1/cuda/p18_r1_eval.cu")
CHOL_SCALE = 0.84932180028 * 16.0
S = 1.0 / 16.0
AT = 1.0 / 255.0

FLAGS = ["-O3", "-DNDEBUG", "-use_fast_math", "--expt-relaxed-constexpr",
         "-gencode=arch=compute_80,code=sm_80", "-std=c++20",
         "-diag-suppress", "20012,186"]


def log(msg):
    print(f"[p18r1e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jwrite(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    log(f"wrote {path}")


def build_ext(name="p18_r1_eval", extra_flags=None, cu_path=None):
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/epic05_p18_r1/build_eval")
    from torch.utils.cpp_extension import load
    flags = list(FLAGS) + list(extra_flags or [])
    return load(name=name, sources=[str(cu_path or CU)], extra_cuda_cflags=flags, verbose=False)


def build_workload(scene, n_max_items=0):
    """Returns dict with flat float32 arrays a,b,c,x [n_items] (record-major),
    n_items multiple of 256, plus row-level stats."""
    d = np.load(ART / f"P18_R1_TRACE_{scene}.npz")
    x0 = d["x0"].astype(np.float64)
    y0 = d["y0"].astype(np.float64)
    pos_x = d["pos_x"].astype(np.float64)
    pos_y = d["pos_y"].astype(np.float64)
    l0 = d["c_l0"].astype(np.float64)
    l1 = d["c_l1"].astype(np.float64)
    l2 = d["c_l2"].astype(np.float64)
    opac = d["c_opacity"].astype(np.float64)
    xpix = d["x_pixels"].astype(np.int32)      # [nrec,16] real global pixels
    mask = d["mask16"].astype(np.uint32)
    nrec = xpix.shape[0]

    assert np.all(np.diff(xpix, axis=1) == 1), "trace x must be contiguous"
    assert np.all(xpix[:, 1:] - xpix[:, :-1] == 1), "trace x contiguity re-check"

    L0 = l0 * CHOL_SCALE
    L1 = l1 * CHOL_SCALE
    L2 = l2 * CHOL_SCALE
    nlo = -np.log2(opac)
    dy = (pos_y - y0) * S - 0.03125
    s2 = L0 * S
    M = L0 * (S * pos_x - 0.03125) + L1 * dy
    u1 = L2 * dy
    A = (s2 * s2).astype(np.float32)
    B = (-2.0 * s2 * M).astype(np.float32)
    C = (M * M + u1 * u1 + nlo).astype(np.float32)

    n_items_want = nrec * 16
    if n_max_items:
        n_items_want = min(n_items_want, n_max_items)
    n_items = (n_items_want // 256) * 256
    nrec_use = n_items // 16

    a = np.repeat(A[:nrec_use], 16)
    b = np.repeat(B[:nrec_use], 16)
    c = np.repeat(C[:nrec_use], 16)
    x = xpix[:nrec_use].reshape(-1).astype(np.float32)

    pop = np.array([bin(int(m)).count("1") for m in mask[:nrec_use]], dtype=np.int32)
    h = np.bincount(pop, minlength=17)[:17]
    stats = {
        "scene": scene, "n_records_total": int(nrec), "n_records_used": int(nrec_use),
        "n_items": int(n_items),
        "live_pop_hist17": h.tolist(),
        "frac_pop0": float(h[0] / h.sum()), "frac_pop16": float(h[16] / h.sum()),
        "mean_live_pop": float((pop.astype(np.float64) * 1.0).mean()),
    }
    return {"a": a, "b": b, "c": c, "x": x, "stats": stats}


def dist_stats(hist):
    h = np.asarray(hist, dtype=np.float64)
    n = h.sum()
    return {"frac_pop0": float(h[0] / n), "frac_pop16": float(h[16] / n),
            "mean": float((h * np.arange(17)).sum() / n)}


def pop_stats(hist):
    """Full live-population stats; mean/median conditional-on-nonempty included
    (the raw mean/median are dominated by empty records)."""
    h = np.asarray(hist, dtype=np.float64)
    n = h.sum()
    vals = np.arange(17, dtype=np.float64)
    mean = float((h * vals).sum() / n)
    c = np.cumsum(h)
    med = int(np.searchsorted(c, n / 2.0))
    hpos = h[1:]
    npos = hpos.sum()
    mean_pos = float((hpos * np.arange(1, 17)).sum() / npos) if npos else 0.0
    cpos = np.cumsum(hpos)
    med_pos = int(np.searchsorted(cpos, npos / 2.0) + 1) if npos else 0
    return {"hist17": [int(v) for v in h], "n_records": int(n),
            "mean": mean, "median": med,
            "mean_given_nonempty": mean_pos, "median_given_nonempty": med_pos,
            "frac_pop0": float(h[0] / n), "frac_pop16": float(h[16] / n)}
