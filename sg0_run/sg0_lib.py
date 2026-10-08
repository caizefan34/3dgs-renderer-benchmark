"""SplatMG SG0 shared library (future-displacement coarse-space oracle).

Execution engine for the frozen SG0_PREREG (2026-09-30). Oracle / diagnostic only.

Everything that touches rendering, loss, cameras, checkpoints and the native
Adam optimizer is REUSED VERBATIM from the frozen MOONSHOT0 A0/Q0 campaign
(a0_lib.py), so SG0 shares the exact frozen C0 semantics. The hierarchy,
coordinate transforms, least-squares coarse projection and oracle
reconstruction are new, and live entirely outside the frozen renderer.

Frozen objective (unchanged): 0.8*L1 + 0.2*(1-SSIM), full resolution.
"""
import os
import sys
import json
import math
import hashlib
import importlib.util
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_moonshot/a0")
import a0_lib as L  # frozen renderer / loss / camera / model loader

SH_DEGREE = 3
CELLS = [("room", 20000), ("room", 25000), ("bicycle", 20000), ("bicycle", 25000)]
CONFIGS = [("L32", "SIM13"), ("L32", "AFF22"), ("L128", "SIM13"), ("L128", "AFF22")]
FAMILIES = ["xyz", "scale", "rot", "opac", "shdc"]  # represented coarse groups
# fine DoF per Gaussian across represented groups: 3+3+3+1+3 = 13
REP_DOF_PER_G = 13


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)


def sha256_file(p):
    return L.sha256_file(p)


def cfg_name(level, basis):
    return f"{level}/{basis}"


# ---------------------------------------------------------------------------
# coordinate helpers (native / tangent spaces per SG0_PREREG section 5)
# ---------------------------------------------------------------------------

def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def quat_conj(a):
    return torch.stack([a[..., 0], -a[..., 1], -a[..., 2], -a[..., 3]], dim=-1)


def quat_log(q):
    """unit quaternion [w,x,y,z] -> rotation vector (axis*angle); inverse of quat_exp."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    flip = q[..., 0:1] < 0
    q = torch.where(flip, -q, q)
    w = q[..., 0:1].clamp(-1.0, 1.0)
    v = q[..., 1:]
    vnorm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    angle = 2.0 * torch.atan2(vnorm, w.clamp_min(-1.0))
    axis = v / vnorm
    return angle * axis


def quat_exp(r):
    """rotation vector -> unit quaternion [w,x,y,z]."""
    angle = r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = angle / 2.0
    axis = r / angle
    w = torch.cos(half)
    v = torch.sin(half) * axis
    return torch.cat([w, v], dim=-1)


def future_displacement(raw0, rawK):
    """theta_K - theta_0 in native/tangent coordinates for the 5 groups.

    raw0/rawK: dict with xyz, shs[N,16,3], scaling[N,3], rotation[N,4], opacity[N].
    Returns dict of detached float cuda tensors:
      xyz[N,3] scale[N,3] rot[N,3] opac[N,1] shdc[N,3]
    """
    q0 = torch.nn.functional.normalize(raw0["rotation"], dim=-1)
    qK = torch.nn.functional.normalize(rawK["rotation"], dim=-1)
    qrel = quat_mul(quat_conj(q0), qK)
    return {
        "xyz": (rawK["xyz"] - raw0["xyz"]).detach(),
        "scale": (rawK["scaling"] - raw0["scaling"]).detach(),
        "rot": quat_log(qrel).detach(),
        "opac": (rawK["opacity"] - raw0["opacity"]).detach(),
        "shdc": (rawK["shs"][:, 0, :] - raw0["shs"][:, 0, :]).detach(),
    }


# ---------------------------------------------------------------------------
# deterministic kd-tree hierarchy (SG0_PREREG section 4)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("idx", "left", "right", "parent", "depth", "is_leaf")

    def __init__(self):
        self.idx = None
        self.left = None
        self.right = None
        self.parent = None
        self.depth = 0
        self.is_leaf = False


def build_kdtree(means, max_leaf=32):
    """Deterministic balanced kd-tree on 3D Gaussian means.
    split axis = largest coordinate range (tie -> lowest axis index);
    split at stable median; tie break by Gaussian ID; recurse to leaf<=max_leaf."""
    N = means.shape[0]
    ids = np.arange(N, dtype=np.int64)

    def rec(idx, depth, parent):
        node = _Node()
        node.depth = depth
        node.parent = parent
        if len(idx) <= max_leaf:
            node.idx = idx
            node.is_leaf = True
            return node
        pts = means[idx]
        rng = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(rng))  # first max -> lowest axis index on tie
        order = np.lexsort((idx, pts[:, axis]))
        s = idx[order]
        mid = len(s) // 2
        node.left = rec(s[:mid], depth + 1, node)
        node.right = rec(s[mid:], depth + 1, node)
        return node

    return rec(ids, 0, None)


def collect_leaves(root):
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            leaves.append(n)
        else:
            stack.append(n.left)
            stack.append(n.right)
    return leaves


def ancestor_up(node, levels):
    cur = node
    for _ in range(levels):
        if cur.parent is None:
            break
        cur = cur.parent
    return cur


def build_hierarchy(means, levels=(0, 2)):
    """Return dict level_name -> list of member index arrays (clusters).

    L32 : leaf nodes (size<=32).
    L128: ancestor two binary levels above each leaf; boundary nodes stay as-is.
    """
    root = build_kdtree(means, max_leaf=32)
    leaves = collect_leaves(root)
    out = {}
    out["L32"] = [lf.idx for lf in leaves]
    groups = {}
    for lf in leaves:
        anc = ancestor_up(lf, 2)
        key = id(anc)
        groups.setdefault(key, {"anc": anc, "members": []})
        groups[key]["members"].append(lf.idx)
    out["L128"] = [np.concatenate(g["members"]) for g in groups.values()]
    return out


def hierarchy_summary(clusters_by_level):
    summary = {}
    for lvl, clusters in clusters_by_level.items():
        sizes = np.array([len(c) for c in clusters], dtype=np.int64)
        hist = {}
        for s in np.unique(sizes):
            hist[int(s)] = int((sizes == s).sum())
        summary[lvl] = {
            "n_clusters": int(len(clusters)),
            "size_min": int(sizes.min()),
            "size_max": int(sizes.max()),
            "size_mean": float(sizes.mean()),
            "size_hist": hist,
        }
    return summary


def random_partition_like(clusters_L32, seed=20260930):
    """One deterministic random partition matched to the L32 cluster-size
    multiset (diagnostic control only)."""
    sizes = sorted(len(c) for c in clusters_L32)
    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(np.concatenate(clusters_L32))
    out = []
    pos = 0
    for s in sizes:
        out.append(all_idx[pos:pos + s])
        pos += s
    return out


# ---------------------------------------------------------------------------
# least-squares coarse projection (SG0_PREREG sections 5-6)
# ---------------------------------------------------------------------------

def cluster_projection(disp_cluster, basis, means0_cluster, family):
    """Project ONE family's future displacement for ONE cluster onto the
    coarse basis. Returns the projected displacement (same shape as input)."""
    n = disp_cluster.shape[0]
    if basis == "SIM13":
        mean = disp_cluster.mean(dim=0, keepdim=True)
        return mean.expand_as(disp_cluster).clone()
    if family == "xyz":
        center = means0_cluster.mean(dim=0, keepdim=True)
        R = torch.cat([torch.ones((n, 1), device=disp_cluster.device,
                                  dtype=disp_cluster.dtype),
                       means0_cluster - center], dim=1)
        M, *_ = torch.linalg.lstsq(R, disp_cluster)
        return R @ M
    mean = disp_cluster.mean(dim=0, keepdim=True)
    return mean.expand_as(disp_cluster).clone()


def coarse_dof(level, basis, n_clusters):
    per = 13 if basis == "SIM13" else 22
    return per * n_clusters


def energy_capture(disp, proj):
    out = {}
    for f in FAMILIES:
        num = float((proj[f] ** 2).sum())
        den = float((disp[f] ** 2).sum())
        out[f] = num / den if den > 1e-30 else float("nan")
    num = sum(float((proj[f] ** 2).sum()) for f in FAMILIES)
    den = sum(float((disp[f] ** 2).sum()) for f in FAMILIES)
    out["total"] = num / den if den > 1e-30 else float("nan")
    return out


# ---------------------------------------------------------------------------
# oracle reconstruction (SG0_PREREG section 6)
# ---------------------------------------------------------------------------

def reconstruct_oracle(raw0, proj):
    """Build the oracle-jump model (theta_0 + projected future displacement).
    Higher-order SH is left unchanged from theta_0."""
    q0 = torch.nn.functional.normalize(raw0["rotation"], dim=-1)
    qdelta = quat_exp(proj["rot"])
    shs = raw0["shs"].clone()
    shs[:, 0, :] = shs[:, 0, :] + proj["shdc"]
    m = {
        "xyz": raw0["xyz"] + proj["xyz"],
        "scaling": raw0["scaling"] + proj["scale"],
        "opacity": raw0["opacity"] + proj["opac"],
        "shs": shs,
        "rotation": quat_mul(q0, qdelta),
    }
    return m


# ---------------------------------------------------------------------------
# evaluation (original full-resolution loss, PSNR, SSIM) on a set of cameras
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(m, cam_list, ds):
    import torch.nn.functional as F
    import fused_ssim
    per = []
    for cam, gt in cam_list:
        frame, _, _, _ = L.render_c0(m, cam, SH_DEGREE)
        frame = frame.clamp(0, 1)
        b = frame.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        l1 = F.l1_loss(b, g)
        ssim = float(fused_ssim.fused_ssim(b, g, padding="same", train=True))
        loss = float(0.8 * l1 + 0.2 * (1.0 - ssim))
        mse = float(((frame - gt) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        per.append({"loss": loss, "psnr": psnr, "ssim": ssim})
    losses = np.array([p["loss"] for p in per])
    psnrs = np.array([p["psnr"] for p in per])
    ssims = np.array([p["ssim"] for p in per])
    return {
        "per_camera": per,
        "loss_mean": float(losses.mean()),
        "psnr_mean": float(psnrs.mean()),
        "ssim_mean": float(ssims.mean()),
    }


# ---------------------------------------------------------------------------
# native Adam rollout (SG0_PREREG section 3)

def load_raw_opt_state(scene, ckpt_iter):
    """Return the raw torch optimizer_state_dict (param_groups + state)."""
    ckpt_path = f"/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{ckpt_iter}.pt"
    o = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return o["optimizer_state_dict"]


def build_optimizer(params, param_groups):
    """Native Adam matching the frozen C0 trainer. lr/eps/betas are read from
    the FROZEN checkpoint optimizer_state_dict param_groups (exact)."""
    groups_spec = [
        ("xyz", [params["xyz"]]),
        ("shs", [params["shs"]]),
        ("opacity", [params["opacity"]]),
        ("scaling", [params["scaling"]]),
        ("rotation", [params["rotation"]]),
    ]
    name_to_pg = {g["name"]: g for g in param_groups}
    groups = []
    for name, ps in groups_spec:
        pg_in = name_to_pg[name]
        groups.append({
            "params": ps,
            "lr": float(pg_in["lr"]),
            "eps": float(pg_in.get("eps", 1e-15)),
            "betas": tuple(pg_in.get("betas", (0.9, 0.999))),
            "name": name,
        })
    return torch.optim.Adam(groups)


def native_adam_rollout(raw0, opt_raw, design_cams, ds, K, clip_norm=1.0):
    """Run K exact native Adam steps on the frozen design sequence.
    opt_raw is the raw torch optimizer_state_dict. Returns rawK + loss trace."""
    params = {
        "xyz": torch.nn.Parameter(raw0["xyz"].clone().detach().float().requires_grad_(True)),
        "shs": torch.nn.Parameter(raw0["shs"].clone().detach().float().requires_grad_(True)),
        "scaling": torch.nn.Parameter(raw0["scaling"].clone().detach().float().requires_grad_(True)),
        "rotation": torch.nn.Parameter(raw0["rotation"].clone().detach().float().requires_grad_(True)),
        "opacity": torch.nn.Parameter(raw0["opacity"].clone().detach().float().requires_grad_(True)),
    }
    param_groups = opt_raw["param_groups"]
    opt = build_optimizer(params, param_groups)
    by_name = {g["name"]: g for g in opt.param_groups}
    pg_by_name = {g["name"]: g for g in param_groups}
    state = opt_raw["state"]
    dev = next(v.device for v in params.values())
    for nm, g in by_name.items():
        pidx = pg_by_name[nm]["params"][0]
        st = state[str(pidx)] if str(pidx) in state else state[pidx]
        for prm in g["params"]:
            opt.state[prm] = {
                "step": st["step"].clone().detach(),
                "exp_avg": st["exp_avg"].clone().detach().to(device=dev).float(),
                "exp_avg_sq": st["exp_avg_sq"].clone().detach().to(device=dev).float(),
            }
    trace = []
    ci = 0
    for step in range(K):
        cam, gt = design_cams[ci % len(design_cams)]
        ci += 1
        opt.zero_grad(set_to_none=True)
        frame, _, _, _ = L.render_c0(params, cam, SH_DEGREE)
        loss = L.train_loss(frame, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), clip_norm)
        opt.step()
        trace.append(float(loss.detach()))
    rawK = {k: v.detach().clone().float() for k, v in params.items()}
    return rawK, trace
