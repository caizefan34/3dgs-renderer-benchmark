"""MSS-E2R0 solver library + parameterized macro (GPU, FP32).

E2R0 asks whether a low-cost solver can restrict the per-Gaussian GN update to
sufficiently observable directions so that ray-capped MSS produces nonzero,
accepted, loss-decreasing updates within the original throughput envelope.

All solvers share ONE autograd Jacobian J (the expensive part) and differ only
in the 58x58 linear solve. The accept pipeline (clip_trust + compositing
residual reduction check + native-loss/cert certificate) is identical to the
frozen MSS0 theta_step; only the linear solve is swapped. Acceptance semantics,
cert tolerance and the trust clip are FROZEN (MSS0 spec points 5/8).
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F

import mss_macro as X
from mss_macro import (clip_trust, apply_retraction, dual_update, image_step,
                       native_crop_raw, batch_loss, _img_from_flat)
from mss_solver import make_residual_fn, autograd_jacobian, quat_from_rotvec, quat_mul

try:
    from mss_macro import TRUST_CLIP
except Exception:
    TRUST_CLIP = X.TRUST_CLIP


# ----------------------------------------------------------------------------
# spectral / conditioning diagnostics (all cheap: H is ndelta x ndelta, ~58)
# ----------------------------------------------------------------------------
def spectral_diag(H, h, evals_desc=None):
    d = H.diagonal()
    if evals_desc is None:
        evals = torch.linalg.eigvalsh(H)
        evals_desc = evals.flip(0)
    evals_desc = evals_desc.clamp_min(0.0)
    lmax = float(evals_desc[0])
    lmin = float(evals_desc[-1])
    trace = float(evals_desc.sum())
    fro2 = float((evals_desc ** 2).sum())
    eff_rank = (trace ** 2) / fro2 if fro2 > 0 else 0.0   # participation ratio in [1,n]
    n = int(evals_desc.numel())
    pos = evals_desc[evals_desc > 0]
    return {
        "grad_norm": float(h.norm()),
        "H_diag_min": float(d.min()), "H_diag_max": float(d.max()),
        "H_diag_mean": float(d.mean()), "H_diag_cv": float(d.std() / (d.mean() + 1e-30)),
        "eig_max": lmax, "eig_min": lmin,
        "cond": lmax / max(lmin, 1e-30),
        "effective_rank": eff_rank, "n_dim": n,
        "spectral_energy_total": trace,
    }


def predicted_residual_reduction(r0, J, delta):
    """Linear-model prediction of ||r0|| reduction from step delta."""
    rlin = r0 + J @ delta
    return float(r0.norm() - rlin.norm())


def exact_gn_direction(H, h, ridge):
    n = H.shape[0]
    A = H + ridge * torch.eye(n, device=H.device, dtype=H.dtype)
    try:
        return -torch.linalg.solve(A, h)
    except Exception:
        return -torch.linalg.lstsq(A, h).solution


# ----------------------------------------------------------------------------
# S1 -- Levenberg-Marquardt with identity damping, lambda sweep
# ----------------------------------------------------------------------------
def solve_lm(J, r0, x0, func, mu, lambdas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), **kw):
    H = J.T @ J; h = J.T @ r0
    spec = spectral_diag(H, h)
    n = H.shape[0]
    I = torch.eye(n, device=J.device, dtype=J.dtype)
    best = None
    for lam in lambdas:
        A = H + lam * I
        try:
            delta = -torch.linalg.solve(A, h)
        except Exception:
            delta = -torch.linalg.lstsq(A, h).solution
        pred = predicted_residual_reduction(r0, J, delta)
        if best is None or pred > best[0]:
            best = (pred, delta, float(lam))
    pred, delta, lam = best
    return delta.detach(), {**spec, "solver": "S1_lm", "best_lambda": lam,
                            "predicted_reduction": pred, "solver_iters": len(lambdas)}


# ----------------------------------------------------------------------------
# S2 -- diagonal / Jacobi-preconditioned LM
# ----------------------------------------------------------------------------
def solve_precond_lm(J, r0, x0, func, mu, lambdas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), **kw):
    H = J.T @ J; h = J.T @ r0
    spec = spectral_diag(H, h)
    d = H.diagonal().clamp_min(1e-8)
    dinv_sqrt = 1.0 / torch.sqrt(d)
    Hs = dinv_sqrt[:, None] * H * dinv_sqrt[None, :]
    hs = dinv_sqrt * h
    n = H.shape[0]
    I = torch.eye(n, device=J.device, dtype=J.dtype)
    best = None
    for lam in lambdas:
        A = Hs + lam * I
        try:
            y = -torch.linalg.solve(A, hs)
        except Exception:
            y = -torch.linalg.lstsq(A, hs).solution
        delta = dinv_sqrt * y
        pred = predicted_residual_reduction(r0, J, delta)
        if best is None or pred > best[0]:
            best = (pred, delta, float(lam))
    pred, delta, lam = best
    return delta.detach(), {**spec, "solver": "S2_precond_lm", "best_lambda": lam,
                            "predicted_reduction": pred, "solver_iters": len(lambdas)}


# ----------------------------------------------------------------------------
# S3 -- matrix-free truncated PCG (normal equations), no exact convergence
# ----------------------------------------------------------------------------
def solve_pcg(J, r0, x0, func, mu, iters=(1, 2, 4, 8, 16), ridge_frac=1e-6, **kw):
    H = J.T @ J; h = J.T @ r0
    spec = spectral_diag(H, h)
    n = H.shape[0]
    lmax = spec["eig_max"]
    ridge = ridge_frac * max(lmax, 1e-30)
    b = -h
    x = torch.zeros(n, device=J.device, dtype=J.dtype)
    r = b.clone()
    p = r.clone()
    rs = float(r @ r)
    best = (predicted_residual_reduction(r0, J, x), x.clone(), 0)
    want = sorted(iters)
    maxit = want[-1]
    it_used = 0
    for it in range(1, maxit + 1):
        Jp = J @ p
        Ap = J.T @ Jp + ridge * p
        pAp = float(p @ Ap)
        if pAp <= 1e-30:
            break
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        pred = predicted_residual_reduction(r0, J, x)
        if pred > best[0]:
            best = (pred, x.clone(), it)
        if rs_new < 1e-30:
            break
        beta = rs_new / rs
        p = r + beta * p
        rs = rs_new
        it_used = it
    pred, delta, chosen = best
    return delta.detach(), {**spec, "solver": "S3_pcg", "chosen_iters": chosen,
                            "max_iters": maxit, "ridge": ridge,
                            "predicted_reduction": pred, "solver_iters": chosen}


# ----------------------------------------------------------------------------
# S4 -- observable-subspace GN/LM (primary research candidate)
# ----------------------------------------------------------------------------
def solve_obs_subspace(J, r0, x0, func, mu, ks=(4, 8, 16, 32),
                       thresholds=(1e-1, 1e-2, 1e-3, 1e-4), ridge_frac=1e-6, **kw):
    H = J.T @ J; h = J.T @ r0
    evals, evecs = torch.linalg.eigh(H)
    evals_desc = evals.flip(0).clamp_min(0.0)
    evecs_desc = evecs.flip(1)
    spec = spectral_diag(H, h, evals_desc=evals_desc)
    n = H.shape[0]
    lmax = spec["eig_max"]
    ridge = ridge_frac * max(lmax, 1e-30)
    etot = float(evals_desc.sum())
    best = None

    def _eval(mask, k, policy):
        nonlocal best
        if int(mask.sum()) < 1:
            return
        Qk = evecs_desc[:, mask]
        Lk = evals_desc[mask]
        hsub = Qk.T @ h
        deltasub = -hsub / (Lk + ridge)
        delta = Qk @ deltasub
        pred = predicted_residual_reduction(r0, J, delta)
        energy = float(Lk.sum()) / max(etot, 1e-30)
        cand = (pred, delta.detach(), {"policy": policy, "retained_rank": int(mask.sum()),
                                       "discarded_rank": n - int(mask.sum()),
                                       "energy_retained": energy,
                                       "ridge": ridge,
                                       "cond_kept": float(Lk[0]) / max(float(Lk[-1]), 1e-30)})
        if best is None or pred > best[0]:
            best = cand

    for k in ks:
        k_eff = int(min(k, n))
        mask = torch.zeros(n, dtype=torch.bool, device=J.device)
        mask[:k_eff] = True
        _eval(mask, k, f"topk_{k_eff}")
    for tau in thresholds:
        _eval(evals_desc >= tau * max(lmax, 1e-30), 0, f"tau_{tau:g}")

    pred, delta, info = best
    return delta, {**spec, "solver": "S4_obs_subspace", "predicted_reduction": pred,
                   "solver_iters": info["retained_rank"], **info}


SOLVERS = {
    "S1_lm": solve_lm,
    "S2_precond_lm": solve_precond_lm,
    "S3_pcg": solve_pcg,
    "S4_obs_subspace": solve_obs_subspace,
}
SOLVER_DEFAULTS = {
    "S1_lm": {},
    "S2_precond_lm": {},
    "S3_pcg": {},
    "S4_obs_subspace": {},
}


# ----------------------------------------------------------------------------
# parameterized theta_step: same selection/capture as frozen theta_step, but the
# linear solve is delegated to a SOLVERS entry. Accept pipeline identical to MSS0.
# ----------------------------------------------------------------------------
def theta_step_solver(state, cam, pixels, cap, u, lam, sh_degree, solver_name,
                      solver_kwargs, mu, max_gaussians, max_rays=48):
    t0 = time.monotonic()
    gids = cap["gids"]; K = cap["K"]; P = gids.shape[0]
    alpha = cap["alpha"]; color = cap["color"]; Kmax = alpha.shape[1]
    dev = pixels.device
    N = state.p["xyz"].shape[0]
    ar = torch.arange(Kmax, device=dev).unsqueeze(0)
    valid = (ar < K[:, None])
    u_cur = u[:, :Kmax]; u_next = u[:, 1:Kmax + 1]
    r = (u_cur - u_next) - alpha.unsqueeze(2) * (color - u_next) + lam
    r = r * valid.unsqueeze(2)
    pair_energy = r.norm(dim=2)
    p_idx, j_idx = torch.where(valid)
    flat_gids = gids[valid]
    flat_e = pair_energy[valid]
    energy = torch.zeros(N, device=dev)
    energy.index_add_(0, flat_gids, flat_e)
    k_top = int(min(max_gaussians, (energy > 0).sum().item()))
    if k_top < 1:
        return {}, time.monotonic() - t0, []
    top = torch.topk(energy, k_top).indices.tolist()
    deltas = {}
    diags = []
    for gid in top:
        mask = (flat_gids == gid)
        if not bool(mask.any()):
            continue
        ps = p_idx[mask]; js = j_idx[mask]
        if max_rays and ps.numel() > max_rays:
            a_g = alpha[ps, js]; c_g = color[ps, js]
            uc_g = u[ps, js]; un_g = u[ps, js + 1]; lm_g = lam[ps, js]
            rcur = (uc_g - un_g) - a_g.unsqueeze(1) * (c_g - un_g) + lm_g
            keep = torch.topk(rcur.norm(dim=1), max_rays).indices
            ps = ps[keep]; js = js[keep]
        pix = pixels[ps].float()
        ucur = u[ps, js]; unext = u[ps, js + 1]; lm = lam[ps, js]
        base = {"mean": state.p["xyz"][gid].detach(),
                "logscale": state.p["scaling"][gid].detach(),
                "rot": F.normalize(state.p["rotation"][gid].detach(), dim=-1),
                "opacity_logit": state.p["opacity"][gid].detach().reshape(1),
                "sh": state.p["shs"][gid].detach()}
        func = make_residual_fn(base, pix, ucur, unext, lm, cam, sh_degree, dev)
        x0 = torch.zeros(state.ndelta, device=dev)
        try:
            J = autograd_jacobian(func, x0)
            r0 = func(x0).detach()
            nrm0 = float(r0.norm())
            if not (torch.isfinite(J).all() and nrm0 > 0):
                continue
            delta, sdiag = SOLVERS[solver_name](J, r0, x0, func, mu, **solver_kwargs)
            # cheap oracle direction on the SAME capped system (full-rank GN)
            H = J.T @ J; h = J.T @ r0
            d_exact = exact_gn_direction(H, h, 1e-8 * max(sdiag["eig_max"], 1e-30))
            sdiag["ray_count"] = int(ps.numel())
            sdiag["n_jac_rows"] = int(J.shape[0])
            sdiag["residual_norm0"] = nrm0
            sdiag["oracle_gn_step_norm"] = float(d_exact.norm())
            if delta is not None:
                cos = float((delta @ d_exact) / (delta.norm() * d_exact.norm() + 1e-30))
                sdiag["cosine_to_exact_gn"] = cos
                d_clip = clip_trust(delta)
                r_try = func(x0 + d_clip).detach()
                nrm_try = float(r_try.norm())
                sdiag["raw_step_norm"] = float(delta.norm())
                sdiag["final_step_norm"] = float(d_clip.norm())
                sdiag["clip_changed_max"] = float((delta - d_clip).abs().max())
                sdiag["actual_reduction_r"] = nrm0 - nrm_try
                sdiag["r_try_finite"] = bool(torch.isfinite(r_try).all())
                if sdiag.get("predicted_reduction", 0.0) > 0:
                    sdiag["rho_r"] = (nrm0 - nrm_try) / sdiag["predicted_reduction"]
                else:
                    sdiag["rho_r"] = float("nan")
                if nrm_try < nrm0 and bool(torch.isfinite(r_try).all()):
                    deltas[gid] = d_clip.detach()
                    sdiag["reduction_check_pass"] = True
                else:
                    sdiag["reduction_check_pass"] = False
            diags.append(sdiag)
        except Exception:
            continue
    return deltas, time.monotonic() - t0, diags


# ----------------------------------------------------------------------------
# parameterized macro: mirrors mss_macro but uses theta_step_solver.
# ----------------------------------------------------------------------------
def mss_macro_solver(state, cam, pixels, Y_img, cfg, ssim, ch, cw,
                     solver_name, solver_kwargs=None):
    solver_kwargs = solver_kwargs or {}
    sh_degree = state.sh_degree
    beta = cfg["beta"]; rho = cfg["rho"]
    image_steps = cfg["image_steps"]; eta = cfg["eta"]
    max_g = cfg["max_gaussians"]
    max_rays = cfg.get("max_rays", 48)
    use_dual = cfg.get("use_dual", True)
    P = pixels.shape[0]
    M = 3.0 * ch * cw
    bg = torch.zeros(3, device=pixels.device)
    cy0 = cfg["cy0"]; cx0 = cfg["cx0"]
    _t = time.monotonic()
    snap = {k: v.detach().clone() for k, v in state.p.items()}
    z0 = native_crop_raw(state, cam, ch, cw, cy0, cx0, sh_degree)
    native_loss0 = float(batch_loss(_img_from_flat(z0, ch, cw), Y_img, ssim))
    cap = X.capture_rays_light(state.gparams(), cam, sh_degree, pixels, device=pixels.device)
    t_cap = time.monotonic() - _t
    alpha = cap["alpha"]; color = cap["color"]; K = cap["K"]
    Kmax = int(K.max())
    u1 = cap["u1"].clone()
    d = torch.zeros(P, 3, device=pixels.device)
    lam = torch.zeros(P, Kmax, 3, device=pixels.device)
    mu = float(getattr(state, "mu", 1e-2))
    inner_sweeps = int(cfg.get("inner_sweeps", 1))
    z_cur = z0
    z_new = z0.clone(); img_loss = float("nan")
    native_loss1 = native_loss0; cert = float("nan")
    n_changed = 0; accepted = False
    theta_s = 0.0; diags = []
    for _ in range(max(1, inner_sweeps)):
        z_new, img_loss = image_step(z_cur, Y_img, u1, d, beta, ssim, M, eta, image_steps, ch, cw)
        u, diag, off, rhs = X.solve_u(alpha, color, bg.expand(P, 3).contiguous(),
                                      z_new, d, lam, beta, rho, K)
        deltas, theta_t, step_diags = theta_step_solver(
            state, cam, pixels, cap, u, lam, sh_degree, solver_name, solver_kwargs,
            mu, max_g, max_rays)
        theta_s += theta_t
        diags = step_diags
        n_changed = len(deltas)
        apply_retraction(state, deltas)
        if use_dual:
            lam, d = dual_update(lam, d, cap, u, z_new, beta, rho)
        z2 = native_crop_raw(state, cam, ch, cw, cy0, cx0, sh_degree)
        native_loss1 = float(batch_loss(_img_from_flat(z2, ch, cw), Y_img, ssim))
        cert = float((z_new - z2).abs().max())
        if (native_loss1 <= native_loss0 + 1e-7 and cert <= (1.0 / 255.0 + 2e-5)
                and torch.isfinite(torch.tensor(native_loss1))):
            accepted = True
            break
        z_cur = z2
    t_r1 = time.monotonic() - _t - t_cap - theta_s
    if not accepted:
        for k in snap:
            state.p[k].copy_(snap[k])
        d = torch.zeros(P, 3, device=pixels.device)
        lam = torch.zeros(P, Kmax, 3, device=pixels.device)
        mu = min(mu * 10.0, 1e4)
        native_loss1 = native_loss0
        cert = float("nan")
        n_changed = 0
    state.mu = mu
    return {"native_loss0": native_loss0, "native_loss1": native_loss1,
            "img_loss": img_loss, "cert": cert, "accept": accepted,
            "n_changed": n_changed, "theta_seconds": theta_s,
            "t_cap": t_cap, "t_img": 0.0, "t_r1": t_r1, "mu": mu,
            "loss_delta": native_loss1 - native_loss0, "diags": diags}
