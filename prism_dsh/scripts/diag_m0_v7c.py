"""v7c: decisive route-vs-value autopsy for the covars-vs-quats conic gap.

S1 synthetic identity case (exact float64 prediction of conics)
S2 real cam0: meta field-wise diff between quats / covars=K0 / covars=V0 routes
S3 real cam0: global float64 re-prediction of conics from K0 / V0 / manual64,
   compared to BOTH routes' conics -> attributes the gap to a route or a value
S4 worst-row table
S6 real cam0: covar scale sweep (K0*(1+delta)) -> gap sensitivity.
Read-only; no bundle edits."""
import json
import math
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
OUT = "/root/epic05_prism/outputs/diag_m0_v7c_bicycle_5000.json"

GPU = int(os.environ.get("PRISM_GPU", "2"))
sys.path.insert(0, GX)
sys.path.insert(0, BUNDLE)
import gx_common as G  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
trainer, ident = G.boot_gmain(GPU)
from prism import gsplat_backend as B  # noqa: E402
from prism.geometry import covariance  # noqa: E402
from gsplat import rasterization  # noqa: E402
from gsplat.cuda._wrapper import quat_scale_to_covar_preci as KS2C  # noqa: E402

out = {"gpu": GPU, "env": ident}
EPS2D = 0.3


def save():
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


# ---------------- float64 reference pipeline (mirrors Utils.cuh exactly) -------
def rotmat64(q):
    q = q.double()
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = torch.rsqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x * n, y * n, z * n, w * n
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack((1 - 2 * (y2 + z2), 2 * (xy + wz), 2 * (xz - wy),
                        2 * (xy - wz), 1 - 2 * (x2 + z2), 2 * (yz + wx),
                        2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (x2 + y2)),
                       -1).reshape(q.shape[:-1] + (3, 3))


def covar64_from_quat_scale(quats, log_scales):
    q = F.normalize(quats, dim=-1)
    R = rotmat64(q)
    s2 = log_scales.exp().double().square()
    return (R * s2[..., None, :]) @ R.transpose(-1, -2)


def conics64(means, cov3d, viewmat, K, width, height, eps2d=EPS2D):
    """means [N,3], cov3d [N,3,3] (float64 already), viewmat [4,4], K [3,3].
    Returns means2d [N,2], covar2d_blur [N,2,2], conics [N,3]."""
    R = viewmat[:3, :3].double()
    t = viewmat[:3, 3].double()
    mc = means.double() @ R.t() + t
    cov_c = R @ cov3d @ R.t()
    fx, fy = K[0, 0].double(), K[1, 1].double()
    cx, cy = K[0, 2].double(), K[1, 2].double()
    W, H = float(width), float(height)
    tan_fx = 0.5 * W / fx
    tan_fy = 0.5 * H / fy
    lim_xp = (W - cx) / fx + 0.3 * tan_fx
    lim_xn = cx / fx + 0.3 * tan_fx
    lim_yp = (H - cy) / fy + 0.3 * tan_fy
    lim_yn = cy / fy + 0.3 * tan_fy
    x, y, z = mc[..., 0], mc[..., 1], mc[..., 2]
    rz = 1.0 / z
    rz2 = rz * rz
    tx = z * torch.minimum(torch.full_like(x, lim_xp), torch.maximum(torch.full_like(x, -lim_xn), x * rz))
    ty = z * torch.minimum(torch.full_like(y, lim_yp), torch.maximum(torch.full_like(y, -lim_yn), y * rz))
    m2d = torch.stack((fx * x * rz + cx, fy * y * rz + cy), -1)
    N = means.shape[0]
    J = torch.zeros(N, 2, 3, dtype=torch.float64, device=means.device)
    J[:, 0, 0] = fx * rz
    J[:, 0, 2] = -fx * tx * rz2
    J[:, 1, 1] = fy * rz
    J[:, 1, 2] = -fy * ty * rz2
    c2d = J @ cov_c @ J.transpose(-1, -2)
    c2d[..., 0, 0] += eps2d
    c2d[..., 1, 1] += eps2d
    inv = torch.linalg.inv(c2d)
    conic = torch.stack((inv[..., 0, 0], inv[..., 0, 1], inv[..., 1, 1]), -1)
    return m2d, c2d, conic


def stats(a, b, tag):
    a = a.double()
    b = b.double()
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-30)
    n_rows = a.shape[0]
    row_rel = rel.reshape(n_rows, -1).amax(-1)
    return {tag + "_maxabs": float(d.max()),
            tag + "_rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
            tag + "_bitwise_eq_rows": int((a == b).reshape(n_rows, -1).all(-1).sum()),
            tag + "_rows_rel_gt1e-5": int((row_rel > 1e-5).sum()),
            tag + "_rows_rel_gt1e-3": int((row_rel > 1e-3).sum()),
            tag + "_rows": int(n_rows)}


# ---------------- S1: synthetic identity case ---------------------------------
try:
    dev = "cuda"
    sm = torch.tensor([[0.0, 0.0, 5.0], [0.5, 0.25, 5.0], [-1.0, 0.5, 8.0],
                       [0.2, -0.3, 3.0], [0.0, 0.0, 4.0]], device=dev)
    sq = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [math.cos(0.35), math.sin(0.35), 0.0, 0.0],
                       [math.cos(0.4), 0.0, math.sin(0.4), 0.0],
                       [1.0, 0.0, 0.0, 0.0]], device=dev)
    sl = torch.log(torch.tensor([[0.05, 0.02, 0.01], [0.04, 0.03, 0.02],
                                 [0.10, 0.02, 0.02], [0.03, 0.03, 0.01],
                                 [0.06, 0.05, 0.04]], device=dev))
    sop = torch.full((5,), 4.0, device=dev)
    ssh = torch.zeros(5, 1, 3, device=dev)
    sK = torch.tensor([[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]], device=dev)
    sview = torch.eye(4, device=dev)
    Kw = dict(means=sm, opacities=sop.sigmoid(), colors=ssh, sh_degree=0,
              viewmats=sview[None], Ks=sK[None], width=1920, height=1080, eps2d=EPS2D,
              radius_clip=0.0, tile_size=16, packed=False, absgrad=True,
              render_mode="RGB", rasterize_mode="classic")
    iq, _, mq = rasterization(quats=F.normalize(sq, dim=-1), scales=sl.exp(), **Kw)
    sk0, _ = KS2C(sq.contiguous(), sl.exp().contiguous(), True, False, False)
    sk6, _ = KS2C(sq.contiguous(), sl.exp().contiguous(), True, False, True)
    sv0 = covariance(sq, sl)
    ic, _, mc = rasterization(quats=F.normalize(sq, dim=-1), scales=sl.exp(), covars=sk0, **Kw)
    iv, _, mv = rasterization(quats=F.normalize(sq, dim=-1), scales=sl.exp(), covars=sv0, **Kw)
    i6, _, m6 = rasterization(quats=F.normalize(sq, dim=-1), scales=sl.exp(), covars=sk6, **Kw)
    cov_man = covar64_from_quat_scale(sq, sl)
    m2d_p, c2d_p, conic_p = conics64(sm, cov_man, sview, sK, 1920, 1080)

    def rows5(mat):
        return [[float(v) for v in row] for row in mat.reshape(mat.shape[0], -1).tolist()]

    res = {
        "img_q_vs_c_maxabs": float((ic - iq).abs().max()),
        "img_q_vs_v_maxabs": float((iv - iq).abs().max()),
        "img_q_vs_6vec_maxabs": float((i6 - iq).abs().max()),
        "conics_route_Q": rows5(mq["conics"][0]),
        "conics_route_C_K0": rows5(mc["conics"][0]),
        "conics_route_V0": rows5(mv["conics"][0]),
        "conics_route_C_6vec": rows5(m6["conics"][0]),
        "conics_pred64": rows5(conic_p),
        "means2d_route_Q": rows5(mq["means2d"][0]),
        "means2d_pred64": rows5(m2d_p),
        "K0_rows": rows5(sk0),
        "cov_man64_rows": rows5(cov_man.float()),
        "cov_K0_vs_man_maxabs": float((sk0.double() - cov_man).abs().max()),
        "quat_norms": [float(v) for v in sq.norm(dim=-1).tolist()],
    }
    # per-row attribution: which predicted conic matches which route
    q_c = mq["conics"][0].double()
    c_c = mc["conics"][0].double()
    p_c = conic_p
    res["row_attr"] = []
    for i in range(5):
        res["row_attr"].append({
            "row": i,
            "pred_vs_Q_rel": float((p_c[i] - q_c[i]).norm() / q_c[i].norm().clamp_min(1e-30)),
            "pred_vs_C_rel": float((p_c[i] - c_c[i]).norm() / c_c[i].norm().clamp_min(1e-30)),
            "Q_vs_C_rel": float((q_c[i] - c_c[i]).norm() / c_c[i].norm().clamp_min(1e-30)),
        })
    out["S1_synth"] = res
    del iq, ic, iv, i6, mq, mc, mv, m6
    save()
except Exception:
    out["S1_synth"] = {"error": traceback.format_exc()}
    save()

# ---------------- real snapshot setup -----------------------------------------
snap = torch.load(SNAP, map_location="cpu", weights_only=False)
params = {"means": snap["xyz"].cuda().float(), "quats": snap["rotation"].cuda().float(),
          "log_scales": snap["scaling"].cuda().float(),
          "opacity_logits": snap["opacity"].cuda().float().reshape(-1),
          "sh0": snap["shs"][:, :1].cuda().float(), "shN": snap["shs"][:, 1:].cuda().float()}
degree = int(snap["active_sh_degree"])
seq = np.load(SEQP)
cam_ids = list(dict.fromkeys(int(c) for c in seq.tolist()))[:1]
dataset = G.build_dataset("bicycle")
cam_ds, gt = dataset.get_item(cam_ids[0])
cam = G.camera_dict(cam_ds)
N = params["means"].shape[0]
out["setup"] = {"cam_id": cam_ids[0], "N": int(N), "degree": degree,
                "quat_norm_mean": float(params["quats"].norm(dim=-1).mean()),
                "quat_norm_min": float(params["quats"].norm(dim=-1).min()),
                "quat_norm_max": float(params["quats"].norm(dim=-1).max()),
                "quat_norm_std": float(params["quats"].norm(dim=-1).std()),
                "log_scales_min": float(params["log_scales"].min()),
                "log_scales_max": float(params["log_scales"].max())}
save()

scales = params["log_scales"].exp()
K0, _ = KS2C(params["quats"].contiguous(), scales.contiguous(), True, False, False)
V0 = covariance(params["quats"], params["log_scales"])
with torch.no_grad():
    img_q, meta_q = B.baseline(params, cam, degree)
    img_c, meta_c = B.baseline(params, cam, degree, covars=K0)
    img_v, meta_v = B.baseline(params, cam, degree, covars=V0)

# ---------------- S2: meta field-wise diff ------------------------------------
try:
    res = {"img_q_vs_c": float((img_c - img_q).abs().max()),
           "img_q_vs_v": float((img_v - img_q).abs().max()),
           "img_c_vs_v": float((img_c - img_v).abs().max())}
    for f in ("means2d", "depths", "radii", "conics", "opacities", "compensations"):
        a = meta_q.get(f)
        if a is None:
            res[f] = "absent"
            continue
        a = a.reshape(-1, *a.shape[2:]).float()
        b = meta_c.get(f).reshape(-1, *meta_c[f].shape[2:]).float()
        c = meta_v.get(f).reshape(-1, *meta_v[f].shape[2:]).float()
        res[f] = {**stats(a, b, "q_vs_c"), **stats(a, c, "q_vs_v"), **stats(b, c, "c_vs_v")}
    out["S2_meta_fields"] = res
    save()
except Exception:
    out["S2_meta_fields"] = {"error": traceback.format_exc()}
    save()

# ---------------- S3: float64 re-prediction of conics --------------------------
try:
    res = {}
    m2d_p, c2d_p, conic_p_k0 = conics64(params["means"], K0.double(), cam["viewmat"], cam["K"],
                                        cam["width"], cam["height"])
    _, _, conic_p_v0 = conics64(params["means"], V0.double(), cam["viewmat"], cam["K"],
                                cam["width"], cam["height"])
    cov_man = covar64_from_quat_scale(params["quats"], params["log_scales"])
    _, _, conic_p_man = conics64(params["means"], cov_man, cam["viewmat"], cam["K"],
                                 cam["width"], cam["height"])
    q_c = meta_q["conics"].reshape(-1, 3).double()
    c_c = meta_c["conics"].reshape(-1, 3).double()
    v_c = meta_v["conics"].reshape(-1, 3).double()
    q_m2d = meta_q["means2d"].reshape(-1, 2).double()
    res["means2d_pred_vs_q"] = stats(m2d_p, q_m2d, "pred")
    res["predK0_vs_routeQ"] = stats(conic_p_k0, q_c, "pk0q")
    res["predK0_vs_routeC"] = stats(conic_p_k0, c_c, "pk0c")
    res["predV0_vs_routeQ"] = stats(conic_p_v0, q_c, "pv0q")
    res["predV0_vs_routeC"] = stats(conic_p_v0, c_c, "pv0c")
    res["predMAN_vs_routeQ"] = stats(conic_p_man, q_c, "pmanq")
    res["predMAN_vs_routeC"] = stats(conic_p_man, c_c, "pmanc")
    res["routeQ_vs_routeC_conics"] = stats(q_c, c_c, "qc")
    res["routeQ_vs_routeV_conics"] = stats(q_c, v_c, "qv")
    res["K0_vs_V0_maxabs"] = float((K0.double() - V0.double()).abs().max())
    res["K0_vs_man64_maxabs"] = float((K0.double() - cov_man).abs().max())
    res["V0_vs_man64_maxabs"] = float((V0.double() - cov_man).abs().max())
    # implied covar2d from route conics vs predicted covar2d from K0
    def implied2d(conic):
        inv = torch.empty(conic.shape[0], 2, 2, dtype=torch.float64, device=conic.device)
        inv[:, 0, 0] = conic[:, 0]
        inv[:, 0, 1] = conic[:, 1]
        inv[:, 1, 0] = conic[:, 1]
        inv[:, 1, 1] = conic[:, 2]
        c2d = torch.linalg.inv(inv)
        c2d[:, 0, 0] -= EPS2D
        c2d[:, 1, 1] -= EPS2D
        return c2d
    imp_q = implied2d(q_c)
    imp_c = implied2d(c_c)
    res["implied_q_vs_predK0_c2d"] = stats(imp_q.reshape(-1, 4), c2d_p.reshape(-1, 4), "iq")
    res["implied_c_vs_predK0_c2d"] = stats(imp_c.reshape(-1, 4), c2d_p.reshape(-1, 4), "ic")
    res["implied_c_vs_implied_q_c2d"] = stats(imp_c.reshape(-1, 4), imp_q.reshape(-1, 4), "cq")
    out["S3_pred"] = res
    save()
except Exception:
    out["S3_pred"] = {"error": traceback.format_exc()}
    save()

# ---------------- S4: worst rows table -----------------------------------------
try:
    d = (q_c - c_c).abs().reshape(q_c.shape[0], -1).amax(-1)
    top = d.topk(24).indices.tolist()
    rows = []
    for r in top:
        rows.append({
            "row": int(r),
            "q_conic": [float(x) for x in q_c[r].tolist()],
            "c_conic": [float(x) for x in c_c[r].tolist()],
            "predK0_conic": [float(x) for x in conic_p_k0[r].tolist()],
            "predMAN_conic": [float(x) for x in conic_p_man[r].tolist()],
            "predK0_vs_q_rel": float((conic_p_k0[r] - q_c[r]).norm() / q_c[r].norm().clamp_min(1e-30)),
            "predK0_vs_c_rel": float((conic_p_k0[r] - c_c[r]).norm() / c_c[r].norm().clamp_min(1e-30)),
            "K0_row": [float(x) for x in K0[r].reshape(-1).tolist()],
            "man64_row": [float(x) for x in cov_man[r].reshape(-1).tolist()],
            "quat": [float(x) for x in params["quats"][r].tolist()],
            "log_scale": [float(x) for x in params["log_scales"][r].tolist()],
            "radii_q": [int(x) for x in meta_q["radii"].reshape(-1, 2)[r].tolist()],
            "radii_c": [int(x) for x in meta_c["radii"].reshape(-1, 2)[r].tolist()],
            "depth_q": float(meta_q["depths"].reshape(-1)[r]),
            "depth_c": float(meta_c["depths"].reshape(-1)[r]),
        })
    out["S4_worst"] = rows
    save()
except Exception:
    out["S4_worst"] = {"error": traceback.format_exc()}
    save()

# ---------------- S6: covar scale sweep ----------------------------------------
try:
    res = {}
    with torch.no_grad():
        for tag, delta in (("d0", 0.0), ("d1e-4", 1e-4), ("d1e-3", 1e-3), ("d1e-2", 1e-2)):
            cc = K0 * (1.0 + delta)
            im, _ = B.baseline(params, cam, degree, covars=cc)
            res[tag] = float((im - img_q).abs().max())
            del im, cc
            save()
    out["S6_sweep"] = res
except Exception:
    out["S6_sweep"] = {"error": traceback.format_exc()}
save()
print("DONE", flush=True)
