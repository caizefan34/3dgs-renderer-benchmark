#!/usr/bin/env python3
"""MR0 shared infra. Reuses the FROZEN HM0 renderer bootstrap + capture patch.
Adds: isolated group rendering, macro moment-init, 12-perturbation collective
response, deterministic disjoint greedy-NN grouping, Amdahl speed predictor.
NO production file is modified. Runtime monkey-patches only.
"""
import sys, os, json, hashlib, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/hm0_pkg")
sys.path.insert(0, "/root/epic05_moonshot/a0")
import hm0_common as C
import a0_lib as A

# ---- bootstrap C0 renderer (pre-registers gsplat.csrc -> NO JIT rebuild) ----
_ENV = C.bootstrap_c0()
from gsplat.experimental.render.functional.gaussian_inference import (  # noqa
    rasterize_gaussian_higs_dynamic, _HigsDynamicScene)

PARAM_KEYS = ("means", "quats", "scales", "opac", "colors")
# 12 frozen shared perturbations (order fixed; matches analyzer response_cos)
PERTURBATIONS = ("trans_x", "trans_y", "trans_z",
                 "logscale_x", "logscale_y", "logscale_z",
                 "rot_x", "rot_y", "rot_z",
                 "opacity", "sh_intensity", "sh_chroma")


def sync():
    torch.cuda.synchronize()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def isolated_group_render(means, quats, scales, opac, colors, cams, sh):
    """Render a SET of gaussians (any count) on transparent bg, multi-cam.
    Returns (frame [C,H,W,3], alpha [C,H,W]) in [0,1]."""
    viewmats = torch.stack([c.viewmatrix for c in cams]).unsqueeze(0)   # [1,C,4,4]
    Ks = torch.stack([c.K for c in cams]).unsqueeze(0)                  # [1,C,3,3]
    H, W = cams[0].image_height, cams[0].image_width
    bg = torch.zeros(3, device=means.device)                           # [3] per-scene
    scene = _HigsDynamicScene()
    out = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opac, colors, sh_degree=sh,
        viewmats=viewmats, Ks=Ks, width=W, height=H, background=bg,
        render_mode="RGB", enable_culling=True, camera_model="pinhole",
        backward_mode="higs_native", eps2d=0.3, scene=scene)
    return out["frame"], out["alpha"]


def activated(leaves):
    return (leaves["means"], F.normalize(leaves["quats"], dim=-1),
            torch.exp(leaves["scales"]), torch.sigmoid(leaves["opac"]),
            leaves["colors"])


def summary(frame, alpha):
    return frame.sum() + alpha.sum()


def leaves_from_model(m, idx):
    """Extract raw-parameter leaf tensors for gaussian indices idx (LongTensor)."""
    return {
        "means": m["xyz"][idx].detach().clone().float().requires_grad_(True),
        "quats": m["rotation"][idx].detach().clone().float().requires_grad_(True),
        "scales": m["scaling"][idx].detach().clone().float().requires_grad_(True),
        "opac": m["opacity"][idx].detach().clone().float().requires_grad_(True),
        "colors": m["shs"][idx].detach().clone().float().requires_grad_(True),
    }


# ---------------------------------------------------------------------------
# 12-perturbation collective response
# ---------------------------------------------------------------------------
def tangents(leaves):
    """12 shared-perturbation tangent tensors matching leaves' shapes."""
    T = {}
    for ax in (0, 1, 2):
        t = {k: torch.zeros_like(leaves[k]) for k in leaves}
        t["means"][:, ax] = 1.0
        T[f"trans_{ax}"] = t
    for ax in (0, 1, 2):
        t = {k: torch.zeros_like(leaves[k]) for k in leaves}
        t["scales"][:, ax] = 1.0          # log-scale additive
        T[f"logscale_{ax}"] = t
    for ax in (0, 1, 2):
        t = {k: torch.zeros_like(leaves[k]) for k in leaves}
        t["quats"] = 0.5 * _quat_axis(leaves["quats"], ax)   # small-angle rotation tangent
        T[f"rot_{ax}"] = t
    t = {k: torch.zeros_like(leaves[k]) for k in leaves}
    t["opac"][:] = 1.0                   # additive in logit opacity
    T["opacity"] = t
    t = {k: torch.zeros_like(leaves[k]) for k in leaves}
    t["colors"][:, 0, :] = 1.0           # SH-DC intensity (all channels)
    T["sh_intensity"] = t
    t = {k: torch.zeros_like(leaves[k]) for k in leaves}
    t["colors"][:, 0, :] = torch.tensor([1.0, -0.5, -0.5],
                                         device=leaves["colors"].device)  # chroma
    T["sh_chroma"] = t
    return T


def _quat_axis(q, ax):
    """Tangent of q under small rotation about axis ax: 0.5*(e_ax (x) q)."""
    w = torch.zeros_like(q)
    w[:, ax + 1] = 1.0
    # quaternion product w * q  (w is pure imaginary, scalar 0)
    x1, y1, z1, w1 = w[:, 1], w[:, 2], w[:, 3], w[:, 0]
    x2, y2, z2, w2 = q[:, 1], q[:, 2], q[:, 3], q[:, 0]
    return torch.stack([
        w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
        w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
        w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
    ], dim=-1)


def _render_summary(raw, cams, sh):
    """Render summary S = sum(frame) + sum(alpha) for a raw-param dict."""
    me, qu, sc, op, co = activated(raw)
    fr, al = isolated_group_render(me, qu, sc, op, co, cams, sh)
    return fr.sum() + al.sum()


def response_exact(leaves, cams, sh, T):
    """Group collective response 12-vec via FIRST-ORDER autograd (exact).

    create_graph=False -> the HIGS native backward is not differentiated.
    Returns a plain (non-grad) 12-vec; used as the fixed fit target."""
    g = {k: v.detach().requires_grad_(True) for k, v in leaves.items()}
    S = _render_summary(g, cams, sh)
    grads = torch.autograd.grad(S, list(g.values()), create_graph=False,
                                retain_graph=True)
    gv = dict(zip(g.keys(), grads))
    dev = next(iter(g.values())).device
    return torch.tensor([float(sum((gv[k] * t[k]).sum() for k in g).detach())
                         for t in T.values()], device=dev)


def response_fd(raw, cams, sh, T, eps=1e-3):
    """Macro collective response 12-vec via central finite differences of the
    render summary along the frozen tangents.

    DIFFERENTIABLE w.r.t. raw with a SINGLE backward: the HIGS native backward
    does NOT support double/second-order backward (create_graph=True raises),
    so the response is estimated by central differences of the render summary
    rather than by autograd.grad(create_graph=True)."""
    Rs = []
    for name, t in T.items():
        Sp = _render_summary({k: raw[k] + eps * t[k] for k in raw}, cams, sh)
        Sm = _render_summary({k: raw[k] - eps * t[k] for k in raw}, cams, sh)
        Rs.append((Sp - Sm) / (2 * eps))
    return torch.stack(Rs)


def macro_moment_init(g):
    with torch.no_grad():
        me, qu, sc, op, co = activated(g)
        w = op.clamp(1e-4, 1 - 1e-4)                      # [K]
        wsum = w.sum().clamp(min=1e-12)
        M = (w[:, None] * me).sum(0) / wsum              # [3]
        dm = me - M
        cov = (w[:, None, None] * (dm[:, :, None] * dm[:, None, :])).sum(0) / wsum
        ev, evec = torch.linalg.eigh(cov)
        scale = torch.sqrt(ev.clamp(min=1e-12))
        Rm = evec.T                                      # columns = eigenvectors
        tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
        q0 = torch.sqrt(torch.clamp(1 + tr, min=1e-12)) / 2
        q = torch.stack([q0, (Rm[2, 1] - Rm[1, 2]) / (4 * q0 + 1e-12),
                         (Rm[0, 2] - Rm[2, 0]) / (4 * q0 + 1e-12),
                         (Rm[1, 0] - Rm[0, 1]) / (4 * q0 + 1e-12)])
        q = F.normalize(q, dim=-1)
        opm = float(torch.sigmoid(g["opac"]).mean())
        com = co.mean(0, keepdim=True)                   # [1,16,3]
        # single macro Gaussian -> leading batch dim [1, ...] for the renderer
        return {
            "means": M.unsqueeze(0).clone().float().requires_grad_(True),
            "quats": q.unsqueeze(0).clone().float().requires_grad_(True),
            "scales": torch.log(scale).unsqueeze(0).clone().float().requires_grad_(True),
            "opac": torch.tensor([float(torch.logit(torch.tensor(opm)))],
                                  device=me.device).requires_grad_(True),
            "colors": com.clone().float().requires_grad_(True),
        }


def macro_to_full(mp):
    """Differentiable dict of ACTIVATED macro params (1 gaussian)."""
    me, qu, sc, op, co = activated(mp)
    return {"means": me, "quats": qu, "scales": sc, "opac": op, "colors": co}


def fit_macro(g, cams_fit, sh, T_group, Rg, n_steps=20, lr=0.05, w_rgb=1.0,
              w_alpha=1.0, w_resp=1.0, eps=1e-3):
    """Fit ONE macro gaussian to the group g on cams_fit.

    Objective: w_rgb*||dRGB||^2 + w_alpha*||dalpha||^2 + w_resp*||dR_hat||^2,
    where dR_hat is the cosine-distance between the macro's finite-difference
    collective response (12-vec) and the group's exact response (normalized).

    Implementation notes (HIGS native backward limitations):
    - The HIGS backward does NOT support double/second-order backward, so the
      collective response is estimated by FORWARD finite differences of the
      render summary (single backward per step).
    - The rasterizer caches autograd state by tensor identity; to avoid
      stale-saved-tensor errors across steps, fresh leaf tensors are cloned
      each step and updated functionally (manual Adam).
    - Raw params are clamped to keep exp()/sigmoid() finite (no NaN/Inf).
    """
    mp = {k: v.detach().clone() for k, v in macro_moment_init(g).items()}
    with torch.no_grad():
        me, qu, sc, op, co = activated(g)
        fr_g, al_g = isolated_group_render(me, qu, sc, op, co, cams_fit, sh)
        fr_g = fr_g.detach(); al_g = al_g.detach()
    Rg = Rg.detach()
    Rg_n = Rg / Rg.norm().clamp(min=1e-12)
    m_state = {k: torch.zeros_like(v) for k, v in mp.items()}
    v_state = {k: torch.zeros_like(v) for k, v in mp.items()}
    b1, b2, ea = 0.9, 0.999, 1e-8
    CLAMP = 8.0
    for step in range(n_steps):
        leaves = {k: mp[k].clone().requires_grad_(True) for k in mp}
        Tm = tangents(leaves)
        me, qu, sc, op, co = activated(leaves)
        fr_m, al_m = isolated_group_render(me, qu, sc, op, co, cams_fit, sh)
        S0 = fr_m.sum() + al_m.sum()
        loss_rgb = (fr_m - fr_g).pow(2).sum()
        loss_alpha = (al_m - al_g).pow(2).sum()
        Rs = []
        for name, t in Tm.items():
            Sp = _render_summary({k: leaves[k] + eps * t[k] for k in leaves}, cams_fit, sh)
            Rs.append((Sp - S0) / eps)
        Rm = torch.stack(Rs)
        Rm_n = Rm / Rm.norm().clamp(min=1e-12)
        loss_resp = (Rm_n - Rg_n).pow(2).sum()
        loss = w_rgb * loss_rgb + w_alpha * loss_alpha + w_resp * loss_resp
        loss.backward()
        with torch.no_grad():
            for k in mp:
                gk = leaves[k].grad
                m_state[k] = b1 * m_state[k] + (1 - b1) * gk
                v_state[k] = b2 * v_state[k] + (1 - b2) * gk * gk
                mh = m_state[k] / (1 - b1 ** (step + 1))
                vh = v_state[k] / (1 - b2 ** (step + 1))
                mp[k] = (mp[k] - lr * mh / (vh.sqrt() + ea)).clamp(-CLAMP, CLAMP)
    with torch.no_grad():
        leaves = {k: mp[k].clone().requires_grad_(True) for k in mp}
        Tm = tangents(leaves)
        me, qu, sc, op, co = activated(leaves)
        fr_m, al_m = isolated_group_render(me, qu, sc, op, co, cams_fit, sh)
        Rm = response_fd(leaves, cams_fit, sh, Tm, eps=eps)
        cos = float((Rm * Rg).sum() / (Rm.norm() * Rg.norm()).clamp(min=1e-30))
        rgb_rmse = float((fr_m - fr_g).pow(2).mean().sqrt())
        alpha_rmse = float((al_m - al_g).pow(2).mean().sqrt())
    fit = {k: v.detach().clone() for k, v in mp.items()}
    info = {"response_cos": cos, "rgb_rmse": rgb_rmse,
            "alpha_rmse": alpha_rmse, "n_steps": n_steps}
    return fit, info


# ---------------------------------------------------------------------------
# grouping: deterministic disjoint greedy nearest-neighbour
# ---------------------------------------------------------------------------
def grid_knn(xyz, seeds, k, qs=0.04):
    """For each seed, k nearest (incl self) among a grid neighbourhood."""
    from collections import defaultdict
    mins = xyz.min(0).values
    keys = torch.floor((xyz - mins) / qs).long()
    d = defaultdict(list)
    kl = keys.tolist()
    for i, key in enumerate(kl):
        d[(key[0], key[1], key[2])].append(i)
    out = []
    with torch.no_grad():
        for s in seeds:
            base = keys[s].tolist()
            found = None
            for R in (1, 2, 3, 5, 8, 12):
                neigh = []
                for dx in range(-R, R + 1):
                    for dy in range(-R, R + 1):
                        for dz in range(-R, R + 1):
                            neigh.extend(d.get((base[0] + dx, base[1] + dy, base[2] + dz), []))
                u = list(dict.fromkeys(neigh))
                if len(u) >= k + 1:
                    found = u; break
            if found is None:
                found = list(range(xyz.shape[0]))
            nt = torch.tensor(found, device=xyz.device)
            dd = torch.cdist(xyz[s:s + 1], xyz[nt]).reshape(-1)
            kk = min(k + 1, dd.numel())
            ii = torch.topk(dd, kk, largest=False).indices[1:]
            out.append(nt[ii])
    return out


def disjoint_groups(xyz, weights, G, Kmax, seed=0, cand_cap=3000):
    """Greedy disjoint groups: process seeds in descending weight, grow each group
    with Kmax nearest UNUSED neighbours from a capped grid candidate pool.
    Returns list of LongTensor (Kmax members, indices into xyz)."""
    from collections import defaultdict
    N = xyz.shape[0]
    dev = xyz.device
    order = torch.argsort(weights, descending=True).tolist()
    used = torch.zeros(N, dtype=torch.bool, device=dev)
    groups = []
    qs = 0.04
    mins = xyz.min(0).values
    keys = torch.floor((xyz - mins) / qs).long()
    kl = keys.tolist()
    d = defaultdict(list)
    for i, key in enumerate(kl):
        d[(key[0], key[1], key[2])].append(i)
    Rs = (1, 2, 3, 5, 8, 12, 20)
    for s in order:
        if len(groups) >= G:
            break
        if used[s]:
            continue
        used[s] = True
        base = kl[s]
        found = None
        for R in Rs:
            neigh = []
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    for dz in range(-R, R + 1):
                        neigh.extend(d.get((base[0] + dx, base[1] + dy, base[2] + dz), []))
            u = [i for i in dict.fromkeys(neigh) if not used[i]]
            if len(u) >= Kmax:
                found = u
                break
        if found is None:
            # capped fallback: unused pool (deterministic stride), never full N
            unused = torch.nonzero(~used, as_tuple=False).reshape(-1)
            if unused.numel() == 0:
                continue
            if unused.numel() > cand_cap:
                unused = unused[torch.linspace(0, unused.numel() - 1, cand_cap).long()]
            found = unused.tolist()
        nt = torch.tensor(found, device=dev)
        dd = torch.cdist(xyz[s:s + 1], xyz[nt]).reshape(-1)
        kk = min(Kmax, dd.numel())
        ii = torch.topk(dd, kk, largest=False).indices
        members = nt[ii]
        used[members] = True
        groups.append(members)
    return groups


# ---------------------------------------------------------------------------
# speed model (Amdahl from HM0 profile)
# ---------------------------------------------------------------------------
def predict_speedup(phase_frac, coverage, K):
    f_fixed = phase_frac["camera"] + phase_frac["l1"] + phase_frac["ssim"]
    f_fb = phase_frac["forward"] + phase_frac["backward"]
    f_opt = phase_frac["optimizer"]
    primitive_factor = (1.0 - coverage) + coverage / float(K)
    denom = f_fixed + (f_fb + f_opt) * primitive_factor
    return 1.0 / denom


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")