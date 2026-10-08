"""InfoGeo-GS IG0 core math (CPU, fp64, renderer-agnostic).

Implements the frozen math of IG0_PREREG.md sections 3-8:
  * anisotropic Gaussian Fisher-Rao metric (closed-form blocks + direct differential)
  * SO(3) quaternion tangent log-map / exponential map
  * quaternion -> right-trivialized local tangent gradient Jacobian
  * per-Gaussian Fisher natural direction (with MP identifiability convention)
  * native Adam geometry displacement
  * FR-GLOBAL / FR-LOCAL magnitude matching
  * state construction (geometry displacement application)

Everything here is pure CPU/numpy float64 and never imports torch or a CUDA
renderer, so it is safe under the "no local GPU execution" constraint.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-30


def skew(v):
    """3x3 cross-product matrix [v]_x so that [v]_x @ w == v x w."""
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def quat_to_R(q):
    """Rotation matrix from a quaternion q = [w, x, y, z] (not assumed unit)."""
    w, x, y, z = (np.float64(c) for c in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize_q(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        out = np.zeros(4, dtype=np.float64)
        out[0] = 1.0
        return out
    return q / n


def quat_mul(a, b):
    """Hamilton product a (x) b for quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def exp_quat(omega):
    """Unit quaternion Exp(omega) for the SO(3) exponential map."""
    omega = np.asarray(omega, dtype=np.float64)
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        q = np.zeros(4, dtype=np.float64)
        q[0] = 1.0
        q[1:] = 0.5 * omega
        return normalize_q(q)
    half = 0.5 * theta
    q = np.empty(4, dtype=np.float64)
    q[0] = np.cos(half)
    q[1:] = np.sin(half) * (omega / theta)
    return q


def quat_log_map(q_rel):
    """SO(3) log-map: shortest-axis-angle omega with Exp(omega)=q_rel (unit q)."""
    q_rel = normalize_q(q_rel)
    w = np.clip(q_rel[0], -1.0, 1.0)
    v = q_rel[1:]
    vec_norm = np.linalg.norm(v)
    if vec_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vec_norm, w)
    return angle * (v / vec_norm)


def quat_relative_log(q0, q1):
    """Right-trivialized local tangent domega with q1 == q0 (x) Exp(domega/2)."""
    q0 = normalize_q(q0)
    q1 = normalize_q(q1)
    q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]], dtype=np.float64)
    q_rel = quat_mul(q0_inv, q1)
    return quat_log_map(q_rel)


def sigma_from_quat_logscale(q, log_scale):
    """Sigma = R diag(lambda) R^T with lambda_k = exp(2 log_scale_k)."""
    R = quat_to_R(q)
    lam = np.exp(2.0 * np.asarray(log_scale, dtype=np.float64))
    Sigma = (R * lam) @ R.T
    return Sigma, R, lam


def fisher_blocks(lam):
    """Closed-form Fisher metric blocks (section 3). Returns (None, 2 I, G_rot)."""
    lam = np.asarray(lam, dtype=np.float64)
    l1, l2, l3 = lam
    G_rot = np.array(
        [
            [(l2 - l3) ** 2 / (l2 * l3), 0.0, 0.0],
            [0.0, (l1 - l3) ** 2 / (l1 * l3), 0.0],
            [0.0, 0.0, (l1 - l2) ** 2 / (l1 * l2)],
        ],
        dtype=np.float64,
    )
    return None, 2.0 * np.eye(3, dtype=np.float64), G_rot



def _dSigma_dlogscale(R, lam, k):
    """dSigma/dlog_scale_k = R diag(dlambda) R^T with dlambda = 2 lambda e_k."""
    dlam = np.zeros(3, dtype=np.float64)
    dlam[k] = 2.0 * lam[k]
    return (R * dlam) @ R.T


def _dSigma_domega(R, lam, j):
    """dSigma/domega_j (right-trivialized) = R([e_j]_x L - L[e_j]_x)R^T."""
    L = np.diag(lam)
    ej = np.eye(3, dtype=np.float64)[j]
    M = skew(ej) @ L - L @ skew(ej)
    return R @ M @ R.T


def fisher_differential_norm2(q, log_scale, dmu, dlogscale, domega, R=None, lam=None):
    """Direct differential form (section 3): ||dxi||_F^2 from dSigma and Sigma^-1."""
    if R is None or lam is None:
        Sigma, R, lam = sigma_from_quat_logscale(q, log_scale)
    else:
        Sigma = (R * lam) @ R.T
    Sigma_inv = np.linalg.inv(Sigma)
    dSigma = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        dSigma += dlogscale[k] * _dSigma_dlogscale(R, lam, k)
    for j in range(3):
        dSigma += domega[j] * _dSigma_domega(R, lam, j)
    mean_term = float(dmu @ Sigma_inv @ dmu)
    quad_term = 0.5 * float(np.trace(Sigma_inv @ dSigma @ Sigma_inv @ dSigma))
    return mean_term + quad_term


def fisher_block_norm2_full(Sigma_inv, lam, dmu, dlogscale, domega):
    """||dxi||_F^2 from the closed-form blocks (section 3)."""
    mean_term = float(dmu @ Sigma_inv @ dmu)
    logscale_term = float(dlogscale @ (2.0 * np.eye(3)) @ dlogscale)
    G_rot = fisher_blocks(lam)[2]
    rot_term = float(domega @ G_rot @ domega)
    return mean_term + logscale_term + rot_term


def fisher_radius_geom(Sigma_inv, lam, dmu, dlogscale, domega):
    """Scalar Fisher radius r^2 = delta^T G delta for a geometry tangent vector."""
    return fisher_block_norm2_full(Sigma_inv, lam, dmu, dlogscale, domega)


def fisher_metric_matrix(Sigma_inv, lam):
    """Assemble the 9x9 Fisher metric G in tangent coords [dmu, dlogscale, domega]."""
    G = np.zeros((9, 9), dtype=np.float64)
    G[0:3, 0:3] = Sigma_inv
    G[3:6, 3:6] = 2.0 * np.eye(3, dtype=np.float64)
    G[6:9, 6:9] = fisher_blocks(lam)[2]
    return G


def quat_tangent_jacobian(q0):
    """Analytic J_eff = d q(omega)/d omega |_{omega=0}, 4x3 (right-trivialized)."""
    q0 = normalize_q(q0)
    w0 = q0[0]
    v0 = q0[1:]
    J_raw = np.zeros((4, 3), dtype=np.float64)
    J_raw[0, :] = -0.5 * v0
    J_raw[1:4, :] = 0.5 * w0 * np.eye(3) + 0.5 * skew(v0)
    P = np.eye(4, dtype=np.float64) - np.outer(q0, q0)
    return P @ J_raw


def quat_grad_to_tangent(g_q, q0):
    """Map native quaternion gradient to right-trivialized local tangent g_omega."""
    J = quat_tangent_jacobian(q0)
    return J.T @ np.asarray(g_q, dtype=np.float64)


def natural_direction(g_mu, g_logscale, g_omega, Sigma_inv, lam, rot_tol=1e-6):
    """Per-Gaussian Fisher natural direction (dmu, dlogscale, domega)."""
    d_mu = -Sigma_inv @ np.asarray(g_mu, dtype=np.float64)
    d_logscale = -0.5 * np.asarray(g_logscale, dtype=np.float64)
    G_rot = fisher_blocks(lam)[2]
    g_omega = np.asarray(g_omega, dtype=np.float64)
    d_omega = np.zeros(3, dtype=np.float64)
    gmax = float(np.max(G_rot))
    if gmax == 0.0:
        d_omega = np.zeros(3, dtype=np.float64)
    else:
        for k in range(3):
            if G_rot[k, k] / gmax < rot_tol:
                d_omega[k] = 0.0
            else:
                d_omega[k] = -g_omega[k] / G_rot[k, k]
    return d_mu, d_logscale, d_omega


def adam_step_displacement(grad, m_prev, v_prev, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """One exact Adam step. Returns (m_new, v_new, displacement)."""
    grad = np.asarray(grad, dtype=np.float64)
    m = beta1 * np.asarray(m_prev, dtype=np.float64) + (1.0 - beta1) * grad
    v = beta2 * np.asarray(v_prev, dtype=np.float64) + (1.0 - beta2) * grad * grad
    mhat = m / (1.0 - beta1 ** t)
    vhat = v / (1.0 - beta2 ** t)
    step = lr * mhat / (np.sqrt(vhat) + eps)
    return m, v, -step


def fr_global_alpha(adam_geo, nat_geo, Sigma_inv, lam):
    """alpha = sqrt(r_A^2 / max(r_N^2, 1e-30)) over all Gaussians."""
    r_a2 = 0.0
    r_n2 = 0.0
    for (dmu, dls, dom), (nmu, nls, nom), Sinv, lm in zip(adam_geo, nat_geo, Sigma_inv, lam):
        r_a2 += fisher_radius_geom(Sinv, lm, dmu, dls, dom)
        r_n2 += fisher_radius_geom(Sinv, lm, nmu, nls, nom)
    return float(np.sqrt(r_a2 / max(r_n2, EPS)))


def fr_local_alpha(adam_geo, nat_geo, Sigma_inv, lam):
    """Per-Gaussian alpha_i = sqrt(rAi^2 / rNi^2) if rNi>0 else 0."""
    alphas = []
    for (dmu, dls, dom), (nmu, nls, nom), Sinv, lm in zip(adam_geo, nat_geo, Sigma_inv, lam):
        r_a2 = fisher_radius_geom(Sinv, lm, dmu, dls, dom)
        r_n2 = fisher_radius_geom(Sinv, lm, nmu, nls, nom)
        alphas.append(0.0 if r_n2 <= 0.0 else float(np.sqrt(r_a2 / r_n2)))
    return alphas


def apply_geometry(mu, log_scale, q, d_mu, d_logscale, d_omega):
    """Return (mu', log_scale', q') after a geometry tangent displacement."""
    mu2 = np.asarray(mu, dtype=np.float64) + d_mu
    log_scale2 = np.asarray(log_scale, dtype=np.float64) + d_logscale
    q2 = normalize_q(quat_mul(normalize_q(q), exp_quat(np.asarray(d_omega, dtype=np.float64))))
    return mu2, log_scale2, q2


__all__ = [
    "skew", "quat_to_R", "normalize_q", "quat_mul", "exp_quat", "quat_log_map",
    "quat_relative_log", "sigma_from_quat_logscale", "fisher_blocks",
    "fisher_differential_norm2", "fisher_block_norm2_full", "fisher_radius_geom",
    "fisher_metric_matrix", "quat_tangent_jacobian", "quat_grad_to_tangent",
    "natural_direction", "adam_step_displacement", "fr_global_alpha",
    "fr_local_alpha", "apply_geometry",
]
