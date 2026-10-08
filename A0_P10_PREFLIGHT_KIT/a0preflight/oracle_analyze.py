#!/usr/bin/env python3
"""A0-P10 oracle-SVD / DCT structural analyzer (frozen semantics).

Runs the EXACT same rank/retention math as the frozen bundle analyzer
(a0_structure_analyze_frozen.py), so P10 numbers are directly comparable to the
frozen prereg gates. Per bucket it reports, for q_alpha and the joint color
matrix, SVD (oracle) and 2D-DCT (diagnostic) retention:
  - energy, effective_rank = (sum s^2)^2 / sum s^4
  - oracle_retention(r) for r in {1,2,4,8,16,32}   (SVD, uncentered)
  - dct_retention(r)   (fixed 2D DCT-II basis, ordered by fx+fy,fx,fy)  [diagnostic]
  - r90, r95, r99, r995

PRIMARY color tensor (REVISED per lead review):
  q_color[p,i,k] = fac * v_render_colors[p,k]   (the kernel's v_rgb_local, exact)
The joint matrix is Q_color_joint = q_color reshaped to [256, m*3] -- a TRUE
3-channel tensor. w_color (fac broadcast) is a secondary export and is NOT used
as the primary color adjoint.

The P10 verdict (frozen mechanical rule) is computed from the ORACLE SVD r99 /
r995 medians ONLY; DCT is diagnostic and does not affect the verdict.

Input: a directory of per-bucket .npz files produced by a0p10_run.py, each
containing q_alpha [256, m] float32 and q_color [256, m, 3] float32 (optionally
w_color [256, m]).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import dct_basis_2d  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32]
B = dct_basis_2d()


def rank_stats(Q):
    Q = np.asarray(Q, dtype=np.float64)
    e = float(np.sum(Q * Q))
    if e == 0:
        return {
            "energy": 0.0, "effective_rank": 0.0,
            "oracle_retention": {str(r): 1.0 for r in RANKS},
            "dct_retention": {str(r): 1.0 for r in RANKS},
            "r90": 0, "r95": 0, "r99": 0, "r995": 0,
        }
    s = np.linalg.svd(Q, compute_uv=False)
    se = s * s
    tot = float(np.sum(se))
    cum = np.cumsum(se) / tot

    def req(t):
        return int(np.searchsorted(cum, t) + 1)

    eff = float((tot ** 2) / float(np.sum(se ** 2)))
    oracle = {str(r): float(cum[min(r, len(cum)) - 1]) for r in RANKS}
    dct = {}
    for r in RANKS:
        Br = B[:, :r]
        Qh = Br @ (Br.T @ Q)
        dct[str(r)] = float(np.sum(Qh * Qh) / e)
    return {
        "energy": e, "effective_rank": eff,
        "oracle_retention": oracle, "dct_retention": dct,
        "r90": req(0.90), "r95": req(0.95), "r99": req(0.99), "r995": req(0.995),
    }


def analyze_bucket(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    qa = d["q_alpha"]
    qc = d["q_color"]
    if qa.ndim != 2 or qa.shape[0] != 256:
        raise SystemExit(f"{npz_path}: q_alpha shape {qa.shape}")
    if qc.ndim != 3 or qc.shape[0] != 256 or qc.shape[2] != 3 or qc.shape[1] != qa.shape[1]:
        raise SystemExit(f"{npz_path}: q_color shape {qc.shape}")
    # TRUE 3-channel joint tensor (q_color already includes v_render_colors).
    qjoint = qc.transpose(0, 2, 1).reshape(256, -1)
    extra = {k: (int(v) if v.ndim == 0 else list(map(int, v.shape))) for k, v in d.items()
             if k not in ("q_alpha", "q_color")}
    has_w = "w_color" in d.files
    return {
        "file": os.path.basename(npz_path),
        "m": int(qa.shape[1]),
        "m_le_bucket_width": int(qa.shape[1] <= 128),
        "has_w_color": bool(has_w),
        "meta": extra,
        "alpha": rank_stats(qa),
        "color_joint": rank_stats(qjoint),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", default=None, help="per-bucket npz files")
    ap.add_argument("--dir", default=None, help="directory of per-bucket npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verdict", action="store_true", help="skip the P10 verdict")
    a = ap.parse_args()
    files = list(a.npz or [])
    if a.dir:
        for f in sorted(os.listdir(a.dir)):
            if f.endswith(".npz"):
                files.append(os.path.join(a.dir, f))
    if not files:
        raise SystemExit("no npz files given (use --npz or --dir)")
    rows = [analyze_bucket(f) for f in files]
    out = {
        "campaign": "A0-P10_PREFLIGHT",
        "analyzer": "oracle-SVD/DCT (frozen a0_structure_analyze_frozen.py semantics)",
        "primary_color_tensor": "q_color[p,i,k] = fac * v_render_colors[p,k] (true 3-channel)",
        "basis": "2D orthonormal DCT-II, ordered by fx+fy then fx then fy (diagnostic only)",
        "ranks": RANKS,
        "n_buckets": len(rows),
        "rows": rows,
    }
    if not a.no_verdict:
        from lib import p10_verdict
        out["verdict"] = p10_verdict(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    v = out.get("verdict", {}).get("verdict", "n/a")
    print(json.dumps({"n_buckets": len(rows), "verdict": v, "out": a.out}))


if __name__ == "__main__":
    main()
