"""Diagnose non-positive-definite covariance from the renderer's own quat->covar kernel.

Read-only w.r.t. the campaign: loads a frozen snapshot, calls gsplat's
quat_scale_to_covar_preci on the same F.normalize'd quats the renderer receives,
and reports how many rows fail Cholesky plus the scale/quat statistics of those
rows. No thresholds, camera rules or selection rules are touched.
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F


def six_to_mat3(six):
    m = six.new_zeros(six.shape[:-1] + (3, 3))
    m[..., 0, 0] = six[..., 0]
    m[..., 0, 1] = six[..., 1]
    m[..., 0, 2] = six[..., 2]
    m[..., 1, 1] = six[..., 3]
    m[..., 1, 2] = six[..., 4]
    m[..., 2, 2] = six[..., 5]
    m[..., 1, 0] = m[..., 0, 1]
    m[..., 2, 0] = m[..., 0, 2]
    m[..., 2, 1] = m[..., 1, 2]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "/root/epic05_grc3d/scripts")
    import gx_common
    _, ident = gx_common.boot_gmain(args.gpu)
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    snap = torch.load(args.snapshot, map_location="cuda", weights_only=False)
    q = snap["rotation"].cuda().float()
    ls = snap["scaling"].cuda().float()
    n = q.shape[0]
    qn = F.normalize(q, dim=-1)
    s = ls.exp()

    covs = []
    for i in range(0, n, args.chunk):
        six, _ = quat_scale_to_covar_preci(qn[i:i + args.chunk], s[i:i + args.chunk],
                                           True, False, True)
        covs.append(six_to_mat3(six))
    cov = torch.cat(covs, 0)

    bad = torch.empty(0, dtype=torch.long, device="cuda")
    for i in range(0, n, args.chunk):
        _, inf = torch.linalg.cholesky_ex(cov[i:i + args.chunk])
        bad = torch.cat((bad, (inf != 0).nonzero().flatten() + i))

    rec = {
        "scene": args.scene, "state": args.state, "snapshot": args.snapshot,
        "n": n, "non_pd_rows": int(bad.numel()),
        "non_pd_fraction": float(bad.numel()) / n,
        "max_symmetry_error": float((cov - cov.transpose(-1, -2)).abs().max()),
        "quat_norm_raw_min": float(q.norm(dim=-1).min()),
        "quat_norm_raw_max": float(q.norm(dim=-1).max()),
        "log_scale_min": float(ls.min()), "log_scale_max": float(ls.max()),
        "scale_min": float(s.min()), "scale_max": float(s.max()),
        "identity": {"gsplat_file": ident.get("gsplat_file"),
                     "gsplat_so_sha256": ident.get("gsplat_so_sha256")},
    }
    if bad.numel():
        b = bad[:8192]
        cb = cov[b].double()
        ev = torch.linalg.eigvalsh(cb)
        sb = ls[b]
        rec["bad_rows"] = {
            "examined": int(b.numel()),
            "eig_min": float(ev[:, 0].min()),
            "eig_median": float(ev[:, 1].median()),
            "eig_max": float(ev[:, 2].max()),
            "n_with_min_eig_negative": int((ev[:, 0] < -1e-30).sum()),
            "n_exactly_singular": int((ev[:, 0] == 0).sum()),
            "log_scale_of_bad_min": float(sb.min()),
            "log_scale_of_bad_max": float(sb.max()),
            "n_bad_all_scales_log_below_-15": int((sb < -15).all(-1).sum()),
            "n_bad_max_scale_log_below_-10": int((sb.max(-1).values < -10).sum()),
            "quat_norm_raw_of_bad_min": float(q[b].norm(dim=-1).min()),
            "quat_norm_raw_of_bad_max": float(q[b].norm(dim=-1).max()),
            "row_index_min": int(bad.min()), "row_index_max": int(bad.max()),
        }
    rec["note"] = ("Cholesky failure classification only; distinguish indefinite "
                   "(negative eigenvalue) from PSD-singular (zero eigenvalue).")
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
