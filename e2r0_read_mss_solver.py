"""MSS0 production solver components (GPU, FP32).

Reuses the frozen-native numerics from mss_adapter and the algebra validated by
scripts/mss_reference.py (E0). Production differences from the fixture:
  * u-solve uses a batched GPU Thomas (RGB shares one SPD tridiagonal);
  * the image objective uses the FROZEN native SepSSIM (0.8 L1 + 0.2 D-SSIM),
    NOT the uniform-window fixture in mss_reference;
  * the per-Gaussian Jacobian is computed by autograd through gsplat\'s real
    projection + SH (geometry 10-DOF tangent + SH 3*(degree+1)^2).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from mss_adapter import (ALPHA_CAP, ALPHA_THRESHOLD, T_MIN, _sh_slice,
                         _project_and_sh)


# ---- frozen native SepSSIM (verbatim from baseline/reference_v1/trainer.py) ----
class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        b = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(b, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        ssim_map = (2 * mu_p * mu_t + self.C1) * (2 * bpt + self.C2) / \
                   ((mu_p ** 2 + mu_t ** 2 + self.C1) * (bp2 + bt2 + self.C2))
        return 1.0 - ssim_map.mean()


def kappa_clamp(z):
    return z.clamp(0.0, 1.0)


def native_image_loss(z, Y, ssim):
    """Frozen native training loss 0.8*L1(kappa(z),Y) + 0.2*D-SSIM. z raw [1,3,H,W]."""
    zc = kappa_clamp(z)
    l1 = F.l1_loss(zc, Y)
    dssim = ssim(zc, Y)
    return 0.8 * l1 + 0.2 * dssim


# ---- suffix states (feasible u) ----
def suffix_states(alpha, color, bg):
    """alpha [K], color [K,3], bg [3] -> u [K+1,3] with u[K]=bg (front-to-back)."""
    K = alpha.numel()
    u = torch.empty(K + 1, 3, dtype=alpha.dtype, device=alpha.device)
    u[K] = bg
    T = 1.0
    for i in range(K - 1, -1, -1):
        u[i] = alpha[i] * color[i] + (1.0 - alpha[i]) * u[i + 1]
    return u


# ---- batched GPU Thomas u-solve ----
def normal_coefficients_batched(alpha, color, bg, z, d, lam, beta, rho, Klen):
    """alpha [B,Kmax], color [B,Kmax,3], bg [B,3], z,d [B,3], lam [B,Kmax,3], Klen [B].
    Returns diag [B,Kmax], off [B,Kmax-1], rhs [B,Kmax,3] (padding = identity nodes)."""
    B, Kmax = alpha.shape
    dev = alpha.device
    ar = torch.arange(Kmax, device=dev).unsqueeze(0)          # [1,Kmax]
    valid = (ar < Klen.unsqueeze(1)).to(alpha.dtype)          # [B,Kmax]
    tau = 1.0 - alpha
    b = alpha.unsqueeze(2) * color - lam                       # [B,Kmax,3]
    last_mask = (ar == (Klen - 1).unsqueeze(1)).to(alpha.dtype)  # [B,Kmax]
    tau_last = (tau * last_mask).sum(1)                         # [B]
    b = b + last_mask.unsqueeze(2) * tau_last.unsqueeze(1).unsqueeze(2) * bg.unsqueeze(1)
    diag = rho * torch.cat([torch.ones(B, 1, device=dev), 1.0 + tau[:, :-1] ** 2], dim=1)
    diag = diag.clone()
    diag[:, 0] = diag[:, 0] + beta
    off = (-rho * tau[:, :-1]).clone()                         # [B,Kmax-1]
    rhs = (rho * b).clone()
    rhs[:, 1:] = rhs[:, 1:] - rho * tau[:, :-1, None] * b[:, :-1]
    rhs[:, 0] = rhs[:, 0] + beta * (z + d)
    # decouple padding nodes (identity) so valid nodes are unaffected
    diag = diag * valid + (1.0 - valid)
    rhs = rhs * valid.unsqueeze(2)
    off_valid = (ar[:, :-1] < (Klen - 1).unsqueeze(1)).to(alpha.dtype)  # [B,Kmax-1]
    off = off * off_valid
    return diag, off, rhs


def thomas_batched(diag, off, rhs):
    """Batched Thomas for symmetric tridiagonal (off[i] couples node i,i+1).
    diag [B,K], off [B,K-1], rhs [B,K,3] -> x [B,K,3]."""
    B, K = diag.shape
    cp = torch.zeros_like(off)      # [B,K-1]
    dp = torch.zeros_like(rhs)      # [B,K,3]
    cp[:, 0] = off[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0:1]
    for i in range(1, K):
        denom = diag[:, i] - off[:, i - 1] * cp[:, i - 1]          # [B]
        if i < K - 1:
            cp[:, i] = off[:, i] / denom                           # [B]
        dp[:, i] = (rhs[:, i] - off[:, i - 1].unsqueeze(1) * dp[:, i - 1]) / denom.unsqueeze(1)
    x = torch.zeros_like(rhs)
    x[:, K - 1] = dp[:, K - 1]
    for i in range(K - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i].unsqueeze(1) * x[:, i + 1]
    return x


def solve_u(alpha, color, bg, z, d, lam, beta, rho, Klen):
    """Returns u [B,Kmax+1,3] (u[:,K]=bg placeholder overwritten by real bg per ray)."""
    B, Kmax = alpha.shape
    diag, off, rhs = normal_coefficients_batched(alpha, color, bg, z, d, lam, beta, rho, Klen)
    x = thomas_batched(diag, off, rhs)                 # [B,Kmax,3]
    u = torch.cat([x, bg.unsqueeze(1)], dim=1)         # [B,Kmax+1,3]
    return u, diag, off, rhs


def normal_residual(diag, off, rhs, x, Klen):
    """Relative normal-equation residual over valid nodes."""
    B, K = diag.shape
    dev = diag.device
    ar = torch.arange(K, device=dev).unsqueeze(0)
    Ax = diag.unsqueeze(2) * x
    Ax[:, 1:] = Ax[:, 1:] + off[:, :, None] * x[:, :-1]
    Ax[:, :-1] = Ax[:, :-1] + off[:, :, None] * x[:, 1:]
    res = Ax - rhs
    valid = (ar < Klen.unsqueeze(1))
    res_v = res * valid.unsqueeze(2)
    rhs_v = rhs * valid.unsqueeze(2)
    num = res_v.norm()
    den = rhs_v.norm() + 1e-30
    return float((num / den).item())


# ---- certificate (batched) ----
def certificate_batched(z, u, alpha, color, Klen):
    """z [B,3], u [B,Kmax+1,3], alpha [B,Kmax], color [B,Kmax,3], Klen [B].
    Returns bound [B], signed [B,3]."""
    B, Kmax = alpha.shape
    dev = alpha.device
    ar = torch.arange(Kmax, device=dev).unsqueeze(0)
    valid = (ar < Klen.unsqueeze(1))
    tau = 1.0 - alpha
    r = u[:, :Kmax] - tau.unsqueeze(2) * u[:, 1:Kmax + 1] - alpha.unsqueeze(2) * color
    T = torch.cat([torch.ones(B, 1, device=dev), torch.cumprod(tau, 1)[:, :-1]], dim=1)  # [B,Kmax]
    r_abs_max = (r.abs().amax(2) * valid)               # [B,Kmax]
    bound = (z - u[:, 0]).abs().amax(1, keepdim=True) + (T * r_abs_max).sum(1, keepdim=True)
    signed = (T.unsqueeze(2) * r * valid.unsqueeze(2)).sum(1)  # [B,3]
    return bound.squeeze(1), signed


# ---- rotation helpers (gsplat convention: wxyz) ----
def quat_from_rotvec(delta_rot):
    """delta_rot [*,3] tangent -> unit quaternion [*,4] (wxyz)."""
    angle = delta_rot.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle
    axis = delta_rot / angle
    w = torch.cos(half)
    xyz = torch.sin(half) * axis
    return torch.cat([w, xyz], dim=-1)


def quat_mul(q1, q2):
    """Hamilton product q1*q2, both [*,4] wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_rotate(q, v):
    """Rotate v [*,3] by quaternion q [*,4] (wxyz): v' = q v q*."""
    qvec = q[..., 1:].unsqueeze(-2)              # [*,1,3]
    t = 2.0 * torch.cross(qvec.squeeze(-2), v, dim=-1)  # [*,3]
    return v + q[..., 0:1] * t + torch.cross(qvec.squeeze(-2), t, dim=-1)


# ---- per-Gaussian local Jacobian (geometry 10-DOF + SH) ----
def gauss_core(delta, base, pixels, cam, sh_degree, device):
    """Differentiable single-gaussian projection + residual.

    delta [58] = [dmean3, dlogscale3, drot3, dopacity_logit1, dsh48].
    base: mean[3], logscale[3], rot[4] (wxyz normalized), opacity_logit[1], sh[K,3].
    pixels [J,2] float (px,py). Returns (alpha[J], color[3], residual[3J])."""
    dmean = delta[0:3]; dlogscale = delta[3:6]; drot = delta[6:9]
    dop = delta[9:10]; dsh = delta[10:]
    q_delta = quat_from_rotvec(drot)
    q_new = F.normalize(quat_mul(q_delta, base["rot"]), dim=-1)
    mean = base["mean"] + dmean
    scale = torch.exp(base["logscale"] + dlogscale)
    opac = torch.sigmoid(base["opacity_logit"] + dop)
    sh = base["sh"] + dsh.reshape(base["sh"].shape)
    gp = {"means": mean.reshape(1, 3), "quats": q_new.reshape(1, 4),
          "scales": scale.reshape(1, 3), "opacities": opac.reshape(1),
          "colors": sh.reshape(1, -1, 3)}
    proj = _project_and_sh(gp, cam, sh_degree, device)
    m2d = proj["means2d"][0]; con = proj["conics"][0]
    op = proj["opac"][0]; col = proj["colors"][0]  # col is [3]
    cx = pixels[:, 0] + 0.5; cy = pixels[:, 1] + 0.5
    dx = cx - m2d[0]; dy = cy - m2d[1]
    sigma = 0.5 * (con[0] * dx * dx + con[2] * dy * dy) + con[1] * dx * dy
    alpha = torch.clamp_max(op * torch.exp(-sigma), ALPHA_CAP)
    color = col  # [3]
    return alpha, color


def make_residual_fn(base, pixels, u_cur, u_next, lam, cam, sh_degree, device):
    def func(delta):
        alpha, color = gauss_core(delta, base, pixels, cam, sh_degree, device)
        r = (u_cur - u_next) - alpha.unsqueeze(1) * (color.unsqueeze(0) - u_next) + lam
        return r.reshape(-1)
    return func


def make_ac_fn(base, pixels, cam, sh_degree, device):
    def func(delta):
        alpha, color = gauss_core(delta, base, pixels, cam, sh_degree, device)
        return torch.cat([alpha, color])
    return func


def autograd_jacobian(func, x):
    """Full Jacobian [m,n] of func(x)->[m] via reverse-mode (batched grad_outputs)."""
    x = x.detach().clone().requires_grad_(True)
    y = func(x).reshape(-1)
    m = y.numel(); n = x.numel()
    eye = torch.eye(m, device=x.device, dtype=x.dtype)
    try:
        J = torch.autograd.grad(y, x, grad_outputs=eye, is_grads_batched=True,
                                retain_graph=True)[0]
        return J
    except Exception:
        rows = []
        for i in range(m):
            rows.append(torch.autograd.grad(y[i], x, retain_graph=True)[0])
        return torch.stack(rows, dim=0)


def local_gn_delta(J, r, mu=1e-2):
    """(J'J + mu*diag(D)) delta = -J'r, D_j=max((J'J)_jj,1e-8)."""
    H = J.T @ J
    h = J.T @ r
    D = H.diagonal().clamp_min(1e-8)
    A = H + mu * torch.diag(D)
    delta = torch.linalg.solve(A, -h)
    return delta, float(mu)
