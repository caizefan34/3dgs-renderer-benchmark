#!/usr/bin/env python3
"""R1-B record verifier: GPU recompute of ALL 16 tile-row masks per record (exact
half2 path, p18_r1_maskchk.cu) -> d16[n,16] hamming distances vs the stored mask16.
Host-side analysis localizes which row the stored mask actually belongs to.

Usage: python p18_r1_verify_records.py <scene> [n_max] [tag]
Writes P18_R1_RECORD_VERIFY[_<tag>].json + P18_R1_D16_<scene>[_<tag>].npy
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

OUT = Path(os.environ.get("P18_R1_OUT", "/root/epic05_p18_r1/artifacts"))
CU = "/root/epic05_p18_r1/cuda/p18_r1_maskchk.cu"

FLAGS = ["-O3", "-DNDEBUG", "-use_fast_math", "--expt-relaxed-constexpr",
         "-gencode=arch=compute_80,code=sm_80",
         "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
         "-D__CUDA_NO_HALF2_OPERATORS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
         "-diag-suppress", "20012,186", "-std=c++20"]


def log(msg):
    print(f"[vrf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build():
    return load(name="p18_r1_maskchk", sources=[CU], extra_cuda_cflags=FLAGS, verbose=True)


def hist16(a):
    return np.bincount(a, minlength=17)[:17].tolist()


def main():
    scene = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    suffix = f"_{tag}" if tag else ""
    npz_path = OUT / f"P18_R1_TRACE_{scene}.npz"
    log(f"scene={scene} npz={npz_path} nmax={nmax} tag={tag or '-'}")

    d = np.load(npz_path)
    n_total = d["x0"].shape[0]
    n = n_total if nmax <= 0 else min(nmax, n_total)
    log(f"records: total={n_total} verifying={n}")

    x0 = d["x0"][:n].astype(np.uint32)
    y0 = d["y0"][:n].astype(np.uint32)
    rec = np.empty((n, 8), np.uint32)
    rec[:, 0] = d["g"][:n].astype(np.uint32)
    rec[:, 1] = d["mt_batch"][:n].astype(np.uint32)
    rec[:, 2] = x0 | (y0 << 16)
    rec[:, 3] = d["qidx"][:n].astype(np.uint32) | (d["mask16"][:n].astype(np.uint32) << 16)
    rec[:, 4] = d["pos_x"][:n].view(np.uint32)
    rec[:, 5] = d["pos_y"][:n].view(np.uint32)
    rec[:, 6] = d["conic01_raw"][:n]
    rec[:, 7] = d["conic23_raw"][:n]

    t = torch.from_numpy(rec.view(np.int32)).cuda()
    mod = build()
    (d16_t,) = mod.maskchk(t, n)
    torch.cuda.synchronize()
    d16 = d16_t.cpu().numpy()  # [n,16] uint8
    np.save(OUT / f"P18_R1_D16_{scene}{suffix}.npy", d16)

    row = (y0 % 16).astype(np.int64)
    rr_ = d["rec_r"][:n].astype(np.int64)
    ty = row & 1
    rh = (row >> 1) & 1
    p_lab = row >> 2  # pp label of the claimed row

    claimed_d = d16[np.arange(n), row]
    best_row = d16.argmin(axis=1).astype(np.int64)
    best_dist = d16.min(axis=1).astype(np.int64)
    # pre-fix diagnosis probe: with the old p18_r=lane_id defect the pp=3 write wins,
    # so the stored content row should be 12 + (ty + 2h) = 12 + (rec_r & 3)
    content_pred = 12 + (rr_ & 3)

    p_idx = np.arange(4, dtype=np.int64)[None, :]
    cand_cols = ty[:, None] + 4 * p_idx + 2 * rh[:, None]
    cand = np.take_along_axis(d16, cand_cols, axis=1)  # [n,4] rows ty+4p+2rh
    zero_cnt = (cand == 0).sum(axis=1)

    # for mismatching records: which row matches (first zero column)
    mis = claimed_d != 0
    first_match = np.where((d16 == 0).any(axis=1), (d16 == 0).argmax(axis=1), -1)
    fm = first_match[mis]
    fm_ok = fm >= 0
    fm_ty = fm & 1
    fm_rh = (fm >> 1) & 1
    fm_p = fm >> 2

    def tab(a, b=16):
        return np.bincount(a, minlength=b)[:b].tolist()

    def tab_signed(a, lo, hi):
        return [int((a == v).sum()) for v in range(lo, hi + 1)]

    res = {
        "scene": scene, "npz": str(npz_path), "n_total": int(n_total), "n_verified": int(n),
        "exact_claimed_row_frac": float((claimed_d == 0).mean()),
        "some_row_equal_frac": float((best_dist == 0).mean()),
        "claimed_d_hist": hist16(claimed_d),
        "best_dist_hist": hist16(best_dist),
        "best_row_hist": tab(best_row),
        "rec_r_hist": tab(rr_),
        "cand_zero_frac_by_p": [float((cand[:, p] == 0).mean()) for p in range(4)],
        "cand_zero_count_hist": tab(zero_cnt, 5),
        "cand_zero_frac_by_rec_r": [[float((cand[rr_ == k, p] == 0).mean()) for p in range(4)]
                                    + [int((rr_ == k).sum())] for k in range(16)],
        "mismatch_frac": float(mis.mean()),
        "mismatch_first_match_row_hist": tab(fm[fm_ok]),
        "mismatch_first_match_delta_ty_hist": tab_signed(fm_ty[fm_ok] - ty[mis][fm_ok], -1, 1),
        "mismatch_first_match_delta_rh_hist": tab_signed(fm_rh[fm_ok] - rh[mis][fm_ok], -1, 1),
        "mismatch_first_match_p_hist": tab(fm_p[fm_ok], 4),
        "mismatch_no_zero_row_frac": float((~fm_ok).mean()) if mis.any() else 0.0,
        "prefix_content_row_frac": float((d16[np.arange(n), content_pred] == 0).mean()),
        "prefix_content_row_delta_hist": tab_signed(content_pred - row, -16, 16),
    }
    if "slot" in d.files:
        slot = d["slot"][:n].astype(np.int64)
        res["cand_zero_frac_by_slot"] = [[float((cand[slot == s, p] == 0).mean()) for p in range(4)]
                                         + [int((slot == s).sum())] for s in (0, 1)]
    if "cta" in d.files:
        cta = d["cta"][:n].astype(np.int64)
        res["cand_zero_frac_by_cta_bucket"] = [
            [float((cand[(cta % 4) == s, p] == 0).mean()) for p in range(4)]
            + [int(((cta % 4) == s).sum())] for s in range(4)]

    log(f"exact(claimed) frac={res['exact_claimed_row_frac']:.4f} "
        f"some-row-equal={res['some_row_equal_frac']:.4f} mismatch={res['mismatch_frac']:.4f}")
    log(f"prefix content-row(12+(r&3)) probe frac={res['prefix_content_row_frac']:.4f}")
    log(f"cand zero frac by p=0..3: {['%.4f' % f for f in res['cand_zero_frac_by_p']]}")
    log(f"cand zero count hist (0..4): {res['cand_zero_count_hist']}")
    log(f"mismatch deltas: dty={res['mismatch_first_match_delta_ty_hist']} "
        f"drh={res['mismatch_first_match_delta_rh_hist']} p={res['mismatch_first_match_p_hist']}")
    log(f"mismatch first-match row hist: {res['mismatch_first_match_row_hist']}")

    outp = OUT / (f"P18_R1_RECORD_VERIFY_{tag}.json" if tag else "P18_R1_RECORD_VERIFY.json")
    outp.write_text(json.dumps({
        "phase": "P18-P0-R1",
        "artifact": "P18_R1_RECORD_VERIFY_" + tag if tag else "P18_R1_RECORD_VERIFY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("GPU exact half2-path recompute (p18_r1_maskchk.cu; CHOL_SCALE=13.589, "
                   "LOG2_INV_AT=7.9943534, flags incl. -use_fast_math) of all 16 tile-row "
                   "masks per record -> d16[n,16] uint8 hamming vs stored mask16; claimed-row "
                   "and pp-candidate analysis host-side"),
        "cuda_flags": FLAGS, "d16_npy": str(OUT / f"P18_R1_D16_{scene}{suffix}.npy"),
        **res}, indent=2))
    log(f"wrote {outp}")
    print("VERIFY_DONE", scene, res["exact_claimed_row_frac"], res["some_row_equal_frac"])


if __name__ == "__main__":
    main()
