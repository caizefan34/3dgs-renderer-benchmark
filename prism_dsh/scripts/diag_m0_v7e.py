"""v7e: decisive E1 test of the input-conversion hypothesis for the covars gap.

Hypothesis: route C (covars=) with 6-vectors produced by the renderer's OWN
device function on the SAME input the quats route receives (F.normalize(q))
is bitwise-identical to route Q, gap == 0.0.  Raw-quat conversions (K0) and
the adapter's pure-torch conversion (V0) differ by ~ULP and amplify to ~6e-3.

Also E0: in-process reproduction of the recorded M0 cam0 numbers (zero/cov)
to prove the probe harness matches run_gpu m0 exactly.
Read-only; no bundle edits.
"""
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

BUNDLE = "/root/epic05_prism/prism_gs"
GX = "/root/epic05_grc3d/scripts"
SNAP = "/root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt"
SEQP = "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy"
OUT = "/root/epic05_prism/outputs/diag_m0_v7e_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance  # noqa: E402
from gsplat.cuda._wrapper import quat_scale_to_covar_preci as KS2C  # noqa: E402

out = {"gpu": GPU, "env": ident, "bundle": BUNDLE}
EPS2D = 0.3


def save():
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


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


def vis_stats(a, b, mask, name):
    """Row-restricted (mask) maxabs/bitwise stats for [N, C] tensors."""
    aa = a.reshape(-1, a.shape[-1]).float()
    bb = b.reshape(-1, b.shape[-1]).float()
    d = (aa - bb).abs()
    vis = mask
    dv = d[vis]
    return {name + "_maxabs_vis": float(dv.max()) if dv.numel() else None,
            name + "_bitwise_eq_rows_vis": int((aa[vis] == bb[vis]).all(-1).sum()),
            name + "_vis_rows": int(vis.sum())}


snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
cam_ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:5]
out["camera_ids"] = cam_ids
dataset = G.build_dataset("bicycle")
cameras = []
for c in cam_ids:
    cam_ds, _gt = dataset.get_item(c)
    cameras.append(G.camera_dict(cam_ds))

q_used = F.normalize(params["quats"], dim=-1)
sc_used = params["log_scales"].exp()
six_p3, _ = KS2C(q_used.contiguous(), sc_used.contiguous(), True, False, True)
six_raw, _ = KS2C(params["quats"].contiguous(), sc_used.contiguous(), True, False, True)
m3_p3_direct, _ = KS2C(q_used.contiguous(), sc_used.contiguous(), True, False, False)
m3_raw_direct, _ = KS2C(params["quats"].contiguous(), sc_used.contiguous(), True, False, False)
m3_p3 = six_to_mat3(six_p3)
m3_raw = six_to_mat3(six_raw)
out_pre = {"six_p3_shape": list(six_p3.shape),
           "six_to_mat3_vs_direct_bitwise": bool((m3_p3 == m3_p3_direct).all()),
           "six_to_mat3_raw_vs_direct_bitwise": bool((m3_raw == m3_raw_direct).all())}
V0 = covariance(params["quats"], params["log_scales"])
out["setup"] = {
    "wrap_tri_extract_p3_vs_six_p3_bitwise": bool((m3_p3[..., [0, 0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2, 2]] == six_p3).all()),
    "wrap_tri_extract_raw_vs_six_raw_bitwise": bool((m3_raw[..., [0, 0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2, 2]] == six_raw).all()),
    "six_p3_vs_six_raw_maxabs": float((six_p3 - six_raw).abs().max()),
    "six_p3_vs_six_raw_rows_gt1e-6rel": int(((six_p3 - six_raw).abs()
        / six_raw.abs().clamp_min(1e-30)).amax(-1).gt(1e-6).sum()),
    "m3_p3_vs_V0_maxabs": float((m3_p3.double() - V0.double()).abs().max()),
    "m3_p3_symmetric_bitwise": bool((m3_p3 == m3_p3.transpose(-1, -2)).all()),
    "six_p3_psd_min_eig_subsample": float(torch.linalg.eigvalsh(
        m3_p3[:65536].double().cpu()).min()),
    "camera_ids": cam_ids,
    "six_p3_shape": list(six_p3.shape),
    "six_to_mat3_vs_direct_bitwise": bool((m3_p3 == m3_p3_direct).all()),
    "six_to_mat3_raw_vs_direct_bitwise": bool((m3_raw == m3_raw_direct).all()),
}
save()

for ci in range(3):
    cam = cameras[ci]
    res = {"cam_id": cam_ids[ci]}
    try:
        with torch.no_grad():
            img_q, meta_q = B.baseline(params, cam, degree)
            vis = (meta_q["radii"].reshape(-1, 2) > 0).any(-1)
            res["visible_rows"] = int(vis.sum())
            res["N"] = int(params["means"].shape[0])
            for tag, cv in (("c_p3_3", m3_p3), ("c_p3_direct", m3_p3_direct),
                            ("c_raw_3", m3_raw), ("c_v0", V0)):
                img, meta = B.baseline(params, cam, degree, covars=cv)
                res[tag + "_gap"] = float((img - img_q).abs().max())
                res[tag + "_bitwise_equal"] = bool((img == img_q).all())
                res[tag + "_conics"] = vis_stats(meta["conics"], meta_q["conics"], vis, tag)
                res[tag + "_radii_diff_rows"] = int(((meta["radii"].reshape(-1, 2)
                    != meta_q["radii"].reshape(-1, 2)).any(-1) & vis).sum())
                res[tag + "_means2d_bitwise_eq_rows_vis"] = int(
                    (meta["means2d"].reshape(-1, 2)[vis] == meta_q["means2d"].reshape(-1, 2)[vis]).all(-1).sum())
                del img, meta
    except Exception:
        res["error"] = traceback.format_exc()
    out["E1_cam%d" % ci] = res
    save()

# ---------- E0: reproduce recorded M0 numbers in-process (cam0 and cam1/2) ----
try:
    res = {}
    for ci in range(3):
        cam = cameras[ci]
        cam_id = cam_ids[ci]
        with torch.no_grad():
            pp = {k: v.detach().clone() for k, v in params.items()}
            original, st = B.state(pp, cam, degree)
            st_d = B.detach_state(st)
            valid = st_d["mask"][0].nonzero().flatten()[:256]
            zero = B.draw(st_d, valid, B.features(st_d, valid))
            cov = covariance(pp["quats"], pp["log_scales"])
            cov_img = B.baseline(pp, cam, degree, covars=cov)[0]
            r = {"cam_id": cam_id, "zero_error": float((original - zero).abs().max()),
                 "cov_error": float((original - cov_img).abs().max())}
            del original, st, st_d, zero, cov_img, cov, pp
        res["cam%d" % ci] = r
        torch.cuda.empty_cache()
    out["E0_reproduce_m0"] = res
except Exception:
    out["E0_reproduce_m0"] = {"error": traceback.format_exc()}
save()
print("DONE", flush=True)
