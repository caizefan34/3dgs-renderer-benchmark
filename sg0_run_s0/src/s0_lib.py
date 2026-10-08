"""SecantGS S0 shared library (temporal update-subspace ORACLE).

Execution engine for the frozen S0_PREREG (2026-09-30). Oracle / diagnostic only.

Everything that touches rendering, loss, cameras, checkpoints and the native
Adam optimizer is REUSED VERBATIM from the frozen MOONSHOT0 A0 / SG0 campaign
(a0_lib.py), so S0 shares the exact frozen C0 semantics. The temporal-subspace
oracle (previous-m executed Adam deltas span the next executed Adam delta) and
the optimizer-unit inner product are new, and live entirely outside the
frozen renderer.

Frozen objective (unchanged): 0.8*L1 + 0.2*(1-SSIM), full resolution.
"""
import os
import sys
import math
import json
import hashlib
import importlib.util
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_moonshot/a0")
import a0_lib as L  # frozen renderer / loss / camera / model loader

SH_DEGREE = 3
CELLS = [("room", 10000), ("room", 15000), ("bicycle", 10000), ("bicycle", 15000)]
MEMORIES = ("2", "4")                      # frozen m in {2,4}
HOLDOUT_STEP = 8                           # corrected true-holdout: i % 8 == 0
GROUPS = ["xyz", "scaling", "rotation", "opacity", "shs"]   # raw param groups
# lr-group name in the frozen optimizer_state_dict for each raw group
LR_NAME = {"xyz": "xyz", "scaling": "scaling", "rotation": "rotation",
           "opacity": "opacity", "shs": "shs"}


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)


def sha256_file(p):
    return L.sha256_file(p)


# ---------------------------------------------------------------------------
# corrected train / true-holdout camera semantics (CVT0_PREREG section 3-4)
# ---------------------------------------------------------------------------

def corrected_panels(n_cams):
    """Return (train_pool, holdout_pool, train_panel, holdout_panel).

    TRUE HOLDOUT = { i : i % 8 == 0 }; TRAIN POOL = { i : i % 8 != 0 }.
    Each 8-panel is midpoint-stratified from its ordered pool:
        q_k = floor((k + 0.5) * N_pool / 8); advance on duplicate.
    """
    holdout_pool = [i for i in range(n_cams) if i % HOLDOUT_STEP == 0]
    train_pool = [i for i in range(n_cams) if i % HOLDOUT_STEP != 0]
    assert len(set(train_pool) & set(holdout_pool)) == 0

    def midpoint8(pool):
        Np = len(pool)
        out, used = [], set()
        for k in range(8):
            q = int(math.floor((k + 0.5) * Np / 8))
            while q in used or q >= Np:
                q += 1
            used.add(q)
            out.append(int(pool[q]))
        return out

    return train_pool, holdout_pool, midpoint8(train_pool), midpoint8(holdout_pool)


# ---------------------------------------------------------------------------
# model / optimizer (frozen native Adam, topology frozen)
# ---------------------------------------------------------------------------

def load_raw_opt_state(scene, ckpt_iter):
    ckpt_path = f"/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{ckpt_iter}.pt"
    o = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return o["optimizer_state_dict"]


def frozen_group_lrs(opt_raw):
    """Per-raw-group frozen lr at the starting checkpoint (prereg section 4)."""
    name_to_pg = {g["name"]: g for g in opt_raw["param_groups"]}
    lrs = {}
    for g in GROUPS:
        lrs[g] = float(name_to_pg[LR_NAME[g]]["lr"])
    return lrs


def build_optimizer(params, param_groups):
    """Native Adam matching the frozen C0 trainer (lr/eps/betas from the frozen
    checkpoint optimizer_state_dict, exact)."""
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


# ---------------------------------------------------------------------------
# evaluation (original full-resolution loss / PSNR / SSIM), forward-only
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
# native Adam rollout with per-step raw-delta capture (S0_PREREG section 3)
# ---------------------------------------------------------------------------

def native_adam_rollout_capture(raw0, opt_raw, design_items, ds, K=16, clip_norm=1.0):
    """Run K exact native Adam steps on the frozen cyclic design sequence.
    design_items is a list of (cam_idx, (cam, gt)) tuples; the cyclic order over
    the 8 train cameras is driven by position in this list.
    opt_raw is the raw torch optimizer_state_dict. Returns:
      states[t]    : raw param dict AFTER t steps (states[0] == clone of raw0)
      src_before[t]: source-camera train loss at states[t-1] (pre-step), t=1..K
      src_after[t] : source-camera train loss at states[t]   (post-step), t=1..K
      src_cam[t]   : source camera dataset index used at step t
      (deltas are computed on demand as states[t]-states[t-1]; not stored)
      lrs[t]       : per-group lr active at step t (from the built optimizer)
    Evaluation forwards are forward-only and do NOT touch optimizer state.
    """
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

    def clone_raw():
        return {k: v.detach().clone().float() for k, v in params.items()}

    states = [clone_raw()]
    src_before, src_after, src_cam, lrs = [], [], [], []
    ci = 0
    ndesign = len(design_items)
    for step in range(K):
        cam_idx, (cam, gt) = design_items[ci % ndesign]
        ci += 1
        # source loss BEFORE the step (forward-only; same as the training forward)
        opt.zero_grad(set_to_none=True)
        frame, _, _, _ = L.render_c0(params, cam, SH_DEGREE)
        loss = L.train_loss(frame, gt)
        loss.backward()
        lb = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(list(params.values()), clip_norm)
        opt.step()
        # per-group lr active at this step
        lrs.append({g: float(by_name[g]["lr"]) for g in GROUPS})
        la = None
        # source loss AFTER the step (forward-only re-eval on same source cam)
        with torch.no_grad():
            fr2, _, _, _ = L.render_c0(params, cam, SH_DEGREE)
            b2 = fr2.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            g2 = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            import torch.nn.functional as F
            import fused_ssim
            l1 = F.l1_loss(b2, g2)
            ss = float(fused_ssim.fused_ssim(b2, g2, padding="same", train=True))
            la = float(0.8 * l1 + 0.2 * (1.0 - ss))
        states.append(clone_raw())
        src_before.append(lb)
        src_after.append(la)
        src_cam.append(int(cam_idx))
    # deltas are NOT stored: delta_t = states[t] - states[t-1] (computed on demand)
    return states, src_before, src_after, src_cam, lrs


# ---------------------------------------------------------------------------
# optimizer-unit inner product + oracle temporal projection (S0_PREREG 4-7)
# ---------------------------------------------------------------------------

def w_norm2(disp, lrs):
    """||delta||_W^2 = sum_g <d_g/lr_g, d_g/lr_g>."""
    tot = 0.0
    for g in GROUPS:
        lr = lrs[g]
        tot += float((disp[g] / lr).pow(2).sum())
    return tot


def w_dot(a, b, lrs):
    tot = 0.0
    for g in GROUPS:
        lr = lrs[g]
        tot += float((a[g] / lr * (b[g] / lr)).sum())
    return tot


def gram(basis, lrs):
    """G_ab = <d_a, d_b>_W for a list of raw-delta dicts."""
    m = len(basis)
    G = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(a, m):
            v = w_dot(basis[a], basis[b], lrs)
            G[a][b] = v
            G[b][a] = v
    return G


def oracle_projection(basis, target, lrs):
    """Solve c* = argmin ||D c - target||_W^2 + lambda ||c||^2,
    lambda = 1e-8 * trace(G) / m. Returns dict with coefficients, delta_proj,
    energies, validity. All math in float64 numpy (no data whitening)."""
    m = len(basis)
    G = gram(basis, lrs)
    G = np.asarray(G, dtype=np.float64)
    traceG = float(np.trace(G))
    if not np.isfinite(traceG) or traceG <= 0.0:
        return {"valid": False, "reason": "trace_G_nonpos", "trace_G": traceG,
                "coefficients": [None] * m, "E_span": None, "residual_energy": None}
    lam = 1e-8 * traceG / m
    # rhs_b = <d_a, target>_W
    rhs = np.array([w_dot(basis[a], target, lrs) for a in range(m)], dtype=np.float64)
    A = G + lam * np.eye(m, dtype=np.float64)
    try:
        c = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return {"valid": False, "reason": "linsolve_failed", "trace_G": traceG,
                "coefficients": [None] * m, "E_span": None, "residual_energy": None}
    if not np.all(np.isfinite(c)):
        return {"valid": False, "reason": "coeff_nonfinite", "trace_G": traceG,
                "coefficients": [float(x) for x in c], "E_span": None, "residual_energy": None}
    # delta_proj = sum_a c_a * basis[a]  (raw tensors)
    proj = {g: torch.zeros_like(target[g]) for g in GROUPS}
    for a in range(m):
        for g in GROUPS:
            proj[g] = proj[g] + float(c[a]) * basis[a][g]
    num = w_norm2(proj, lrs)
    den = w_norm2(target, lrs)
    E_span = num / den if den > 1e-300 else float("nan")
    resid = {g: (target[g] - proj[g]) for g in GROUPS}
    residual_energy = w_norm2(resid, lrs) / den if den > 1e-300 else float("nan")
    return {
        "valid": True, "reason": "ok", "trace_G": traceG, "lambda": lam,
        "coefficients": [float(x) for x in c],
        "delta_proj": proj, "E_span": float(E_span),
        "residual_energy": float(residual_energy),
        "proj_norm_W2": num, "target_norm_W2": den,
    }


def apply_delta_raw(raw_prev, delta):
    """theta_proj = theta_{t-1} + delta_proj in the SAME native raw tensors.
    No optimizer moments changed; no backward. C0 applies exp/sigmoid only at
    render (activated()), so raw add is the exact candidate semantics."""
    m = {}
    for g in GROUPS:
        m[g] = raw_prev[g] + delta[g]
    m["shs"] = raw_prev["shs"].clone()
    m["shs"][:, 0, :] = raw_prev["shs"][:, 0, :] + delta["shs"][:, 0, :]
    return m

