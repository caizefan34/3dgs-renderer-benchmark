"""Vectorized (batched, GPU/torch) Fisher-Rao math for the IG0 GPU probe.

Mirrors ig0/ig0_fisher.py (numpy fp64 reference) but operates on batched
tensors [N, ...] so it scales to millions of Gaussians. Cross-validated against
the numpy reference in validate_against_numpy().
"""
from __future__ import annotations
import torch


def normalize_quat(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul_batch(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def skew_batch(v):
    N = v.shape[0]
    S = torch.zeros(N, 3, 3, dtype=v.dtype, device=v.device)
    S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2];  S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] = v[:, 0]
    return S


def quat_to_R_batch(q):
    q = normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def sigma_inv_batch(q, log_scale):
    R = quat_to_R_batch(q)
    lam = torch.exp(2.0 * log_scale)
    inv_lam = 1.0 / lam
    Sigma_inv = (R * inv_lam.unsqueeze(1)) @ R.transpose(1, 2)
    return Sigma_inv, lam, R


def G_rot_batch(lam):
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    gx = (l2 - l3) ** 2 / (l2 * l3)
    gy = (l1 - l3) ** 2 / (l1 * l3)
    gz = (l1 - l2) ** 2 / (l1 * l2)
    return torch.stack([gx, gy, gz], dim=-1)


def natural_direction_batch(g_mu, g_logscale, g_omega, q, log_scale, rot_tol=1e-6):
    Sigma_inv, lam, R = sigma_inv_batch(q, log_scale)
    d_mu = -torch.bmm(Sigma_inv, g_mu.unsqueeze(-1)).squeeze(-1)
    d_logscale = -0.5 * g_logscale
    G_rot = G_rot_batch(lam)
    gmax = G_rot.max(dim=-1).values
    ratio = G_rot / gmax.unsqueeze(-1).clamp_min(1e-30)
    mask = (gmax.unsqueeze(-1) > 0) & (ratio >= rot_tol)
    d_omega = torch.where(mask, -g_omega / G_rot.clamp_min(1e-30), torch.zeros_like(g_omega))
    return d_mu, d_logscale, d_omega


def radius_geom_batch(dmu, dls, dom, Sigma_inv, lam, G_rot):
    r_mu = torch.bmm(dmu.unsqueeze(1), torch.bmm(Sigma_inv, dmu.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    r_ls = 2.0 * (dls * dls).sum(-1)
    r_rot = (dom * G_rot * dom).sum(-1)
    return r_mu + r_ls + r_rot


def quat_grad_to_tangent_batch(g_q, q0):
    q0n = normalize_quat(q0)
    w0 = q0n[:, 0:1]
    v0 = q0n[:, 1:]
    N = q0n.shape[0]
    J_raw = torch.zeros(N, 4, 3, dtype=q0.dtype, device=q0.device)
    J_raw[:, 0, :] = -0.5 * v0
    J_raw[:, 1:4, :] = 0.5 * w0.unsqueeze(-1) * torch.eye(3, dtype=q0.dtype, device=q0.device) + 0.5 * skew_batch(v0)
    P = torch.eye(4, dtype=q0.dtype, device=q0.device) - q0n.unsqueeze(-1) * q0n.unsqueeze(-2)
    J_eff = torch.matmul(P, J_raw)
    return torch.einsum("nik,ni->nk", J_eff, g_q)


def quat_relative_log_batch(q0, q1):
    q0n = normalize_quat(q0)
    q1n = normalize_quat(q1)
    q0_inv = torch.stack([q0n[:, 0], -q0n[:, 1], -q0n[:, 2], -q0n[:, 3]], dim=-1)
    q_rel = quat_mul_batch(q0_inv, q1n)
    w = q_rel[:, 0].clamp(-1.0, 1.0)
    v = q_rel[:, 1:]
    vn = v.norm(dim=-1)
    angle = 2.0 * torch.atan2(vn, w)
    axis = v / vn.unsqueeze(-1).clamp_min(1e-12)
    domega = angle.unsqueeze(-1) * axis
    return torch.where(vn.unsqueeze(-1) < 1e-12, torch.zeros_like(domega), domega)


def exp_quat_batch(omega):
    theta = omega.norm(dim=-1)
    half = 0.5 * theta
    s = theta.clamp_min(1e-12)
    q = torch.zeros(omega.shape[0], 4, dtype=omega.dtype, device=omega.device)
    q[:, 0] = torch.cos(half)
    q[:, 1:] = (torch.sin(half).unsqueeze(-1) / s.unsqueeze(-1)) * omega
    small = theta < 1e-12
    q[:, 1:] = torch.where(small.unsqueeze(-1), 0.5 * omega, q[:, 1:])
    q[:, 0] = torch.where(small, torch.ones_like(q[:, 0]), q[:, 0])
    return q


def apply_geometry_batch(mu, log_scale, q, dmu, dls, dom):
    mu2 = mu + dmu
    ls2 = log_scale + dls
    q2 = normalize_quat(quat_mul_batch(q, exp_quat_batch(dom)))
    return mu2, ls2, q2
