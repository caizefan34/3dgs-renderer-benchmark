"""MSS0 macro solver for E2/E3 (GPU, FP32). Builds on E1-validated mss_solver/mss_adapter."""
from __future__ import annotations
import os, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mss_adapter import (gaussian_params, native_rasterize, capture_rays,
                         _sh_colors, ALPHA_CAP, ALPHA_THRESHOLD, T_MIN, RESOLUTIONS)
import mss_solver as S
from mss_solver import (kappa_clamp, native_image_loss, solve_u,
                        quat_from_rotvec, quat_mul)

IMG_EXT = (".JPG", ".jpg", ".PNG", ".png", ".JPEG", ".jpeg")
GT_ROOT = "/home/liaoyuanjun/3dgs-renderer-benchmark/data/datasets/mipnerf360"


def sh_channels(active_sh_degree):
    return (int(active_sh_degree) + 1) ** 2


def ndelta_for(active_sh_degree):
    return 10 + sh_channels(active_sh_degree) * 3


def load_gt_crop(scene_key, img_name, tw, th):
    base = os.path.join(GT_ROOT, scene_key, "images")
    path = None
    for ext in IMG_EXT:
        cand = os.path.join(base, img_name + ext)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"GT not found for {scene_key}/{img_name}")
    img = Image.open(path).convert("RGB").resize((tw, th))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def crop_pixels(ch, cw, cy0, cx0, device):
    ys = torch.arange(cy0, cy0 + ch, device=device)
    xs = torch.arange(cx0, cx0 + cw, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _img_from_flat(z, ch, cw):
    return z.reshape(ch, cw, 3).permute(2, 0, 1).unsqueeze(0)


def capture_rays_light(gparams, cam, sh_degree, pixels, device="cuda"):
    """Fully vectorized accepted-sequence capture (no per-pixel Python loop).

    Replicates capture_rays' accepted set (alpha>=threshold, front-to-back,
    early-stop at T<=T_MIN) using batched tensor ops over all crop pixels.
    Returns the same dict schema as capture_rays(light=True). Used by the macro
    theta step, which recomputes per-Gaussian Jacobians independently.
    """
    dev = torch.device(device)
    tile_size = 16
    pixels = torch.as_tensor(pixels, dtype=torch.long, device=dev)
    if pixels.ndim == 1:
        pixels = pixels.unsqueeze(0)
    P = pixels.shape[0]
    with torch.no_grad():
        ref_img, meta = native_rasterize(gparams, cam, sh_degree=sh_degree)
        ref_img = ref_img.detach()
        m2d_m = meta["means2d"][0].detach()
        con_m = meta["conics"][0].detach()
        op_m = meta["opacities"][0].detach()
        flat = meta["flatten_ids"].detach()
        tile_key = meta["isect_ids"].detach() >> 32
        tw = int(meta["tile_width"])
        col_m = _sh_colors(gparams, cam, sh_degree, device,
                           radii_mask=(meta["radii"][0] > 0).all(dim=-1)).detach()
        px = pixels[:, 0]; py = pixels[:, 1]
        tid = (py // tile_size) * tw + (px // tile_size)
        lo = torch.searchsorted(tile_key, tid)
        hi = torch.searchsorted(tile_key, tid + 1)
        counts = (hi - lo).clamp_min(0)
        total = int(counts.sum())
        ar = torch.arange(P, device=dev)
        ref = ref_img[py, px]
        if total == 0:
            return {"gids": torch.full((P, 0), -1, dtype=torch.long, device=dev),
                    "K": torch.zeros(P, dtype=torch.long, device=dev),
                    "alpha": torch.zeros(P, 0, device=dev),
                    "color": torch.zeros(P, 0, 3, device=dev),
                    "u1": torch.zeros(P, 3, device=dev), "ref": ref, "Kmax": 0,
                    "branch": torch.zeros(P, dtype=torch.bool, device=dev)}
        pix_rep = torch.repeat_interleave(ar, counts)
        cum = torch.cumsum(counts, 0); cum_prev = cum - counts
        starts = torch.repeat_interleave(lo, counts)
        within = torch.arange(total, device=dev) - torch.repeat_interleave(cum_prev, counts)
        all_g = flat[starts + within].long()
        mm = m2d_m[all_g]; cc = con_m[all_g]; oo = op_m[all_g]; ccol = col_m[all_g]
        cx = px[pix_rep].float() + 0.5; cy = py[pix_rep].float() + 0.5
        ddx = cx - mm[:, 0]; ddy = cy - mm[:, 1]
        ss = 0.5 * (cc[:, 0] * ddx * ddx + cc[:, 2] * ddy * ddy) + cc[:, 1] * ddx * ddy
        aa = torch.clamp_max(oo * torch.exp(-ss), ALPHA_CAP)
        valid = (ss >= 0) & (aa >= ALPHA_THRESHOLD)
        one_minus = torch.where(valid, 1.0 - aa, torch.ones_like(aa))
        # Per-pixel transmittance via segmented log-cumsum. A single global cumprod
        # underflows to 0 across the crop, which made T_before NaN and corrupted the
        # accepted set; reset the running product at each pixel boundary instead.
        log_om = torch.log(one_minus.clamp_min(1e-30))
        cs = torch.cumsum(log_om, 0)
        cs_at_start = torch.zeros(P, device=dev)
        nzt = cum_prev > 0
        cs_at_start[nzt] = cs[cum_prev[nzt] - 1]
        T_incl = torch.exp((cs - cs_at_start[pix_rep]).clamp_max(0.0))
        T_before = (T_incl / one_minus.clamp_min(1e-30)).clamp_max(1.0)
        accepted = valid & (T_before * one_minus > T_MIN)
        contrib = (T_before * aa).unsqueeze(1) * ccol
        u1 = torch.zeros(P, 3, device=dev)
        u1.index_add_(0, pix_rep, contrib * accepted.unsqueeze(1))
        cap_hit = valid & (aa >= ALPHA_CAP - 1e-7)
        early_stop = valid & (T_before * (1.0 - aa) <= T_MIN)
        branch_any = cap_hit | early_stop
        branch = torch.zeros(P, dtype=torch.bool, device=dev)
        if bool(branch_any.any()):
            branch[pix_rep[branch_any]] = True
        acc_count = torch.zeros(P, dtype=torch.long, device=dev)
        acc_count.index_add_(0, pix_rep, accepted.long())
        Kmax = int(acc_count.max().item())
        acc_cum = torch.cumsum(accepted.long(), 0)
        seg_cum_before_pix = torch.zeros(P, dtype=acc_cum.dtype, device=dev)
        nzp = cum_prev > 0
        seg_cum_before_pix[nzp] = acc_cum[cum_prev[nzp] - 1]
        acc_within = acc_cum - seg_cum_before_pix[pix_rep] - 1
        gids_pad = torch.full((P, Kmax), -1, dtype=torch.long, device=dev)
        alpha_pad = torch.zeros(P, Kmax, device=dev)
        color_pad = torch.zeros(P, Kmax, 3, device=dev)
        if Kmax > 0:
            ai = pix_rep[accepted]; aj = acc_within[accepted]
            gids_pad[ai, aj] = all_g[accepted]
            alpha_pad[ai, aj] = aa[accepted]
            color_pad[ai, aj] = ccol[accepted]
    return {"gids": gids_pad, "K": acc_count, "alpha": alpha_pad, "color": color_pad,
            "u1": u1, "ref": ref, "Kmax": Kmax, "branch": branch}


class MSSState:
    def __init__(self, params, device, sh_degree):
        self.p = {k: v.to(device).clone().float().contiguous()
                  for k, v in params.items() if torch.is_tensor(v)}
        self.sh_degree = int(sh_degree)
        self.ndelta = ndelta_for(sh_degree)
        self.nsh = sh_channels(sh_degree)
        self.mu = 1e-2
        for k in ("xyz", "scaling", "rotation", "opacity", "shs"):
            self.p[k].requires_grad_(False)

    def gparams(self):
        return gaussian_params(self.p)


def native_crop_raw(state, cam, ch, cw, cy0, cx0, sh_degree, no_grad=True):
    gparams = state.gparams()
    if no_grad:
        with torch.no_grad():
            ref, _ = native_rasterize(gparams, cam, sh_degree=sh_degree)
    else:
        ref, _ = native_rasterize(gparams, cam, sh_degree=sh_degree)
    return ref[cy0:cy0 + ch, cx0:cx0 + cw].reshape(-1, 3)


def batch_loss(z_img, Y_img, ssim):
    return native_image_loss(z_img, Y_img, ssim)


def image_step(z, Y_img, u1, d, beta, ssim, M, eta, steps, ch, cw, halve=4):
    cur = z.detach().clone().requires_grad_(True)
    best = cur.detach().clone()
    best_loss = float(batch_loss(_img_from_flat(cur, ch, cw), Y_img, ssim))
    cur_eta = eta
    for _ in range(steps):
        if cur.grad is not None:
            cur.grad = None
        lp = batch_loss(_img_from_flat(cur, ch, cw), Y_img, ssim)
        quad = (beta / (2.0 * M)) * ((cur - u1 + d) ** 2).sum()
        (lp + quad).backward()
        g = cur.grad.detach()
        step = cur_eta * M / (beta + 1.0)
        cand = (cur.detach() - step * g).clamp(0.0, 1.0)
        cand_loss = float(batch_loss(_img_from_flat(cand, ch, cw), Y_img, ssim))
        if cand_loss < best_loss:
            best = cand
            best_loss = cand_loss
            cur = cand.detach().requires_grad_(True)
        else:
            cur_eta *= 0.5
            halve -= 1
            if halve <= 0:
                break
    return best, best_loss


def block_gn(J, r, mu):
    H = J.T @ J
    h = J.T @ r
    D = H.diagonal().clamp_min(1e-8)
    A = H + mu * torch.diag(D)
    try:
        delta = torch.linalg.solve(A, -h)
    except Exception:
        delta = torch.linalg.lstsq(A, -h).solution
    return delta


TRUST_CLIP = {'mean': 0.05, 'logscale': 0.2, 'rotvec': 0.2, 'opacity': 0.2, 'sh': 0.05}


def clip_trust(delta):
    '''Spec point 5: clip per-Gaussian GN delta to the frozen trust range.'''
    d = delta.clone()
    d[0:3] = d[0:3].clamp(-TRUST_CLIP['mean'], TRUST_CLIP['mean'])
    d[3:6] = d[3:6].clamp(-TRUST_CLIP['logscale'], TRUST_CLIP['logscale'])
    d[6:9] = d[6:9].clamp(-TRUST_CLIP['rotvec'], TRUST_CLIP['rotvec'])
    d[9:10] = d[9:10].clamp(-TRUST_CLIP['opacity'], TRUST_CLIP['opacity'])
    d[10:] = d[10:].clamp(-TRUST_CLIP['sh'], TRUST_CLIP['sh'])
    return d


def theta_step(state, cam, pixels, cap, u, lam, sh_degree, mu, max_gaussians, max_rays=48):
    t0 = time.monotonic()
    gids = cap["gids"]; K = cap["K"]; P = gids.shape[0]
    alpha = cap["alpha"]; color = cap["color"]; Kmax = alpha.shape[1]
    dev = pixels.device
    N = state.p["xyz"].shape[0]
    ar = torch.arange(Kmax, device=dev).unsqueeze(0)
    valid = (ar < K[:, None])                       # [P,Kmax]
    u_cur = u[:, :Kmax]; u_next = u[:, 1:Kmax + 1]
    r = (u_cur - u_next) - alpha.unsqueeze(2) * (color - u_next) + lam   # [P,Kmax,3]
    r = r * valid.unsqueeze(2)
    pair_energy = r.norm(dim=2)                     # [P,Kmax]
    p_idx, j_idx = torch.where(valid)               # [Npairs]
    flat_gids = gids[valid]
    flat_e = pair_energy[valid]
    energy = torch.zeros(N, device=dev)
    energy.index_add_(0, flat_gids, flat_e)
    k_top = int(min(max_gaussians, (energy > 0).sum().item()))
    if k_top < 1:
        return {}, time.monotonic() - t0
    top = torch.topk(energy, k_top).indices.tolist()
    deltas = {}
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
        func = S.make_residual_fn(base, pix, ucur, unext, lm, cam, sh_degree, dev)
        x0 = torch.zeros(state.ndelta, device=dev)
        try:
            J = S.autograd_jacobian(func, x0)
            r0 = func(x0).detach()
            nrm0 = float(r0.norm())
            mu_c = mu
            delta = None
            for _ in range(3):
                d_try = block_gn(J, r0, mu_c)
                if torch.isfinite(d_try).all():
                    d_try = clip_trust(d_try)
                    r_try = func(x0 + d_try).detach()
                    if float(r_try.norm()) < nrm0:
                        delta = d_try
                        break
                mu_c = min(mu_c * 10.0, 1e4)
            if delta is not None:
                deltas[gid] = delta.detach()
        except Exception:
            continue
    return deltas, time.monotonic() - t0


def apply_retraction(state, deltas):
    nsh = state.nsh
    with torch.no_grad():
        for gid, delta in deltas.items():
            state.p["xyz"][gid] += delta[0:3]
            state.p["scaling"][gid] += delta[3:6]
            q_new = F.normalize(quat_mul(quat_from_rotvec(delta[6:9]),
                              F.normalize(state.p["rotation"][gid], dim=-1)), dim=-1)
            state.p["rotation"][gid] = q_new
            state.p["opacity"][gid] += delta[9]
            state.p["shs"][gid] += delta[10:].reshape(nsh, 3)


def dual_update(lam, d, cap, u, z, beta, rho):
    alpha = cap["alpha"]; color = cap["color"]; K = cap["K"]
    Kmax = alpha.shape[1]
    r = (u[:, :Kmax] - u[:, 1:Kmax + 1]) - alpha.unsqueeze(2) * (color - u[:, 1:Kmax + 1]) + lam
    valid = (torch.arange(Kmax, device=alpha.device)[None, :] < K[:, None]).float()
    r = r * valid.unsqueeze(2)
    lam_new = lam + rho * r
    d_new = d + beta * (z - u[:, 0])
    return lam_new, d_new


def mss_macro(state, cam, pixels, Y_img, cfg, ssim, ch, cw):
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
    cap = capture_rays_light(state.gparams(), cam, sh_degree, pixels, device=pixels.device)
    t_cap = time.monotonic() - _t
    alpha = cap["alpha"]; color = cap["color"]; K = cap["K"]
    Kmax = int(K.max())
    u1 = cap["u1"].clone()
    d = torch.zeros(P, 3, device=pixels.device)
    lam = torch.zeros(P, Kmax, 3, device=pixels.device)
    mu = float(getattr(state, "mu", 1e-2))
    inner_sweeps = int(cfg.get("inner_sweeps", 3))
    theta_s = 0.0
    z_cur = z0
    z_new = z0.clone(); img_loss = float("nan")
    native_loss1 = native_loss0; cert = float("nan")
    n_changed = 0; accepted = False
    for _ in range(max(1, inner_sweeps)):
        z_new, img_loss = image_step(z_cur, Y_img, u1, d, beta, ssim, M, eta, image_steps, ch, cw)
        u, diag, off, rhs = solve_u(alpha, color, bg.expand(P, 3).contiguous(),
                                    z_new, d, lam, beta, rho, K)
        deltas, theta_t = theta_step(state, cam, pixels, cap, u, lam, sh_degree, mu, max_g, max_rays)
        theta_s += theta_t
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
            "t_cap": t_cap, "t_img": 0.0, "t_r1": t_r1,
            "loss_delta": native_loss1 - native_loss0, "mu": mu}


def _enable_grads(state):
    for k in ("xyz", "scaling", "rotation", "opacity", "shs"):
        state.p[k].requires_grad_(True)


def full_native_step(state, cam, pixels, Y_img, optimizer, ssim, ch, cw, cy0, cx0):
    _enable_grads(state)
    optimizer.zero_grad()
    ref, _ = native_rasterize(state.gparams(), cam, sh_degree=state.sh_degree)
    z = ref[cy0:cy0 + ch, cx0:cx0 + cw].reshape(-1, 3)
    loss = batch_loss(_img_from_flat(z, ch, cw), Y_img, ssim)
    loss.backward()
    optimizer.step()
    for k in ("xyz", "scaling", "rotation", "opacity", "shs"):
        state.p[k].requires_grad_(False)
    return float(loss.detach())


def local_lm_step(state, cam, pixels, Y_img, optimizer, ssim, ch, cw, cy0, cx0):
    _enable_grads(state)
    holder = {}

    def closure():
        optimizer.zero_grad()
        ref, _ = native_rasterize(state.gparams(), cam, sh_degree=state.sh_degree)
        z = ref[cy0:cy0 + ch, cx0:cx0 + cw].reshape(-1, 3)
        loss = batch_loss(_img_from_flat(z, ch, cw), Y_img, ssim)
        loss.backward()
        holder["loss"] = float(loss.detach())
        return loss

    optimizer.step(closure)
    for k in ("xyz", "scaling", "rotation", "opacity", "shs"):
        state.p[k].requires_grad_(False)
    return holder.get("loss", float("nan"))