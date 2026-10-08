"""PGR0 faithful CPU reference for the gsplat fused 3DGS projection path.

This module reimplements, in pure PyTorch (CPU), the exact math of the
gsplat `projection_ewa_3dgs_fused` forward kernel and its
`projection_ewa_3dgs_fused_bwd_kernel` (H8_MR == False), transcribed from:

  public_baselines/gsplat/gsplat/cuda/include/Utils.cuh
  public_baselines/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu

It exposes THREE gradient evaluations of the same projection:

  * forward()            : canonical forward, also returning the intermediates.
  * backward_recompute() : replicates the shipped CUDA backward, which RECOMPUTES
                           R, covar, mean_c, covar_c and the perspective Jacobian J
                           from the raw inputs (this is the redundancy under test).
  * backward_reuse()     : the hypothetical caching variant that REUSES the
                           forward-saved intermediates instead of recomputing them.

`autograd_grad()` provides the numeric ground truth via torch.autograd.

NO CUDA IS USED. Everything runs on CPU in float64 (reference) or float32
(CUDA-emulation) so the redundancy/similarity measurements are portable.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch

EPS = 1e-12  # MIN_COMPENSATION squared floor region; kept tiny, matches assert-free path


# ----------------------------------------------------------------------------- 
# basic linear-algebra helpers (batched, last dims are matrices)
# ----------------------------------------------------------------------------- 
def mat2(a, b, c, d):
    """Build a batch of 2x2 from columns/rows matching glm mat2(a,b,c,d)."""
    row0 = torch.stack([a, b], dim=-1)
    row1 = torch.stack([c, d], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def sym_mat2_from_upper(A, B, C):
    """conics are stored as (A, B, C) == (inv00, inv01, inv11); build symmetric mat2."""
    z = torch.zeros_like(A)
    return mat2(A, B, B, C)


def inverse_mat2(M):
    a = M[..., 0, 0]; b = M[..., 0, 1]; c = M[..., 1, 0]; d = M[..., 1, 1]
    det = a * d - b * c
    inv = 1.0 / det
    out = torch.empty_like(M)
    out[..., 0, 0] = d * inv
    out[..., 0, 1] = -b * inv
    out[..., 1, 0] = -c * inv
    out[..., 1, 1] = a * inv
    return out, det


def quat_to_rotmat(quat):
    """quat: (...,4) as (w,x,y,z). Returns (...,3,3), column-major like glm mat3."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    inv_norm = torch.rsqrt(x * x + y * y + z * z + w * w + EPS)
    x = x * inv_norm; y = y * inv_norm; z = z * inv_norm; w = w * inv_norm
    x2 = x * x; y2 = y * y; z2 = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    # columns
    c0 = torch.stack([1 - 2 * (y2 + z2), 2 * (xy + wz), 2 * (xz - wy)], dim=-1)
    c1 = torch.stack([2 * (xy - wz), 1 - 2 * (x2 + z2), 2 * (yz + wx)], dim=-1)
    c2 = torch.stack([2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (x2 + y2)], dim=-1)
    # glm constructs mat3 column-major; stack columns then transpose to row-major
    # so the returned matrix is the true rotation R (covar = R S^2 R^T, matching gsplat).
    return torch.stack([c0, c1, c2], dim=-1).transpose(-1, -2)


def quat_scale_to_covar(quat, scale):
    """C = R S S^T with M = R S."""
    R = quat_to_rotmat(quat)
    sx, sy, sz = scale[..., 0], scale[..., 1], scale[..., 2]
    S = torch.zeros(scale.shape[:-1] + (3, 3), dtype=scale.dtype, device=scale.device)
    S[..., 0, 0] = sx; S[..., 1, 1] = sy; S[..., 2, 2] = sz
    M = R @ S
    covar = M @ M.transpose(-1, -2)
    return covar, R, M


def posW2C(R, t, pW):
    return (R @ pW.unsqueeze(-1)).squeeze(-1) + t


def covarW2C(R, covarW):
    return R @ covarW @ R.transpose(-1, -2)


def persp_proj(mean_c, covar_c, fx, fy, cx, cy, width, height, want_jacobian=False):
    x = mean_c[..., 0]; y = mean_c[..., 1]; z = mean_c[..., 2]
    tan_fovx = 0.5 * width / fx
    tany = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tany
    lim_y_neg = cy / fy + 0.3 * tany
    rz = 1.0 / z
    rz2 = rz * rz
    tx = z * torch.clamp(x * rz, -lim_x_neg, lim_x_pos)
    ty = z * torch.clamp(y * rz, -lim_y_neg, lim_y_pos)
    # J is (2x3): row0 = (fx*rz, 0, -fx*tx*rz2); row1 = (0, fy*rz, -fy*ty*rz2)
    J = torch.zeros(mean_c.shape[:-1] + (2, 3), dtype=mean_c.dtype, device=mean_c.device)
    J[..., 0, 0] = fx * rz
    J[..., 0, 2] = -fx * tx * rz2
    J[..., 1, 1] = fy * rz
    J[..., 1, 2] = -fy * ty * rz2
    cov2d = J @ covar_c @ J.transpose(-1, -2)
    mean2d = torch.stack([fx * x * rz + cx, fy * y * rz + cy], dim=-1)
    if want_jacobian:
        return cov2d, mean2d, J
    return cov2d, mean2d


def add_blur(eps2d, cov2d):
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    cov2d = cov2d.clone()
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + eps2d
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + eps2d
    det_blur = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    comp = torch.sqrt(torch.clamp(det_orig / det_blur, min=EPS))
    return cov2d, det_blur, comp, det_orig


# ----------------------------------------------------------------------------- 
# FORWARD
# ----------------------------------------------------------------------------- 
@dataclass
class ProjInputs:
    means: torch.Tensor     # (N,3) requires_grad
    quats: torch.Tensor     # (N,4)
    scales: torch.Tensor    # (N,3)
    viewmats: torch.Tensor  # (C,4,4)
    Ks: torch.Tensor        # (C,3,3)
    width: int
    height: int
    eps2d: float = 0.3
    near_plane: float = 0.01
    far_plane: float = 1e10


@dataclass
class FwdResult:
    radii: torch.Tensor      # (C,N,2)
    means2d: torch.Tensor    # (C,N,2)
    depths: torch.Tensor     # (C,N)
    conics: torch.Tensor     # (C,N,3)
    compensations: Optional[torch.Tensor]
    # intermediates for reuse path
    R: torch.Tensor          # (C,N,3,3)
    covar: torch.Tensor      # (C,N,3,3)
    mean_c: torch.Tensor     # (C,N,3)
    covar_c: torch.Tensor    # (C,N,3,3)
    covar2d: torch.Tensor    # (C,N,2,2) post-blur
    covar2d_inv: torch.Tensor  # (C,N,2,2)
    J: torch.Tensor          # (C,N,2,3)
    covar2d_preblur: torch.Tensor  # (C,N,2,2) pre-blur (forward intermediate)


def _R_from_viewmat(viewmats):
    # viewmats (C,4,4) row-major; R rows are upper-left 3x3
    return viewmats[..., :3, :3]  # (C,3,3)


def forward(inp: ProjInputs, calc_compensations: bool = False) -> FwdResult:
    C = inp.viewmats.shape[0]
    N = inp.means.shape[0]
    device = inp.means.device; dtype = inp.means.dtype
    Rcam = _R_from_viewmat(inp.viewmats)              # (C,3,3)
    t = inp.viewmats[..., :3, 3]                      # (C,3)

    # broadcast camera tensors over gaussians
    R = Rcam.unsqueeze(1).expand(C, N, 3, 3).clone()  # (C,N,3,3)
    t_b = t.unsqueeze(1).expand(C, N, 3).clone()      # (C,N,3)

    means = inp.means.unsqueeze(0).expand(C, N, 3)
    covar, Rg, M = quat_scale_to_covar(inp.quats.unsqueeze(0).expand(C, N, 4),
                                       inp.scales.unsqueeze(0).expand(C, N, 3))
    covar = covar.expand(C, N, 3, 3).clone()

    mean_c = posW2C(R, t_b, means)                    # (C,N,3)
    covar_c = covarW2C(R, covar)                      # (C,N,3,3)

    K = inp.Ks
    fx = K[:, 0, 0].view(C, 1); fy = K[:, 1, 1].view(C, 1)
    cx = K[:, 0, 2].view(C, 1); cy = K[:, 1, 2].view(C, 1)
    W = float(inp.width); H = float(inp.height)
    fx_b = fx.expand(C, N); fy_b = fy.expand(C, N)
    cx_b = cx.expand(C, N); cy_b = cy.expand(C, N)

    cov2d, mean2d, J = persp_proj(mean_c, covar_c, fx_b, fy_b, cx_b, cy_b, W, H, want_jacobian=True)
    cov2d_preblur = cov2d
    cov2d_blur, det_blur, comp, det_orig = add_blur(inp.eps2d, cov2d)
    cov2d_inv, _ = inverse_mat2(cov2d_blur)

    conics = torch.stack([cov2d_inv[..., 0, 0], cov2d_inv[..., 0, 1], cov2d_inv[..., 1, 1]], dim=-1)

    # radii / validity (matches CUDA culling)
    GAUSSIAN_EXTEND = 3.33
    radius_x = torch.ceil(GAUSSIAN_EXTEND * torch.sqrt(torch.clamp(cov2d_blur[..., 0, 0], min=0)))
    radius_y = torch.ceil(GAUSSIAN_EXTEND * torch.sqrt(torch.clamp(cov2d_blur[..., 1, 1], min=0)))
    depths = mean_c[..., 2]
    valid = (depths > inp.near_plane) & (depths < inp.far_plane)
    radius = torch.stack([radius_x, radius_y], dim=-1)
    radius[~valid] = 0.0
    inside = (mean2d[..., 0] + radius[..., 0] > 0) & (mean2d[..., 0] - radius[..., 0] < W) & \
             (mean2d[..., 1] + radius[..., 1] > 0) & (mean2d[..., 1] - radius[..., 1] < H)
    radius[~inside] = 0.0
    radii = radius.to(torch.int32)

    compensations = comp if calc_compensations else None
    return FwdResult(radii, mean2d, depths, conics, compensations,
                     R, covar, mean_c, covar_c, cov2d_blur, cov2d_inv, J,
                     cov2d_preblur)


# ----------------------------------------------------------------------------- 
# ANALYTIC BACKWARD  (recompute vs reuse)
# ----------------------------------------------------------------------------- 
def _conic_to_mat2(v_conics):
    A = v_conics[..., 0]; B = v_conics[..., 1]; C = v_conics[..., 2]
    return sym_mat2_from_upper(A, B, C)


def inverse_vjp(Minv, v_Minv):
    # v_M = -Minv * v_Minv * Minv
    return -Minv @ v_Minv @ Minv


def add_blur_vjp(eps2d, covar2d_blur, compensation, v_compensation):
    # covar2d_blur already has eps2d added; conic_blur = covar2d_blur
    conic_blur = covar2d_blur
    a = conic_blur[..., 0, 0]; b = conic_blur[..., 0, 1]
    c = conic_blur[..., 1, 0]; d = conic_blur[..., 1, 1]
    det_cb = a * d - b * c
    v_sqr = v_compensation * 0.5 / (compensation + 1e-6)
    one_minus = 1 - compensation * compensation
    v = torch.zeros_like(conic_blur)
    v[..., 0, 0] = v_sqr * (one_minus * a - eps2d * det_cb)
    v[..., 0, 1] = v_sqr * (one_minus * b)
    v[..., 1, 0] = v_sqr * (one_minus * c)
    v[..., 1, 1] = v_sqr * (one_minus * d - eps2d * det_cb)
    return v


def persp_proj_vjp(mean_c, covar_c, fx_b, fy_b, cx_b, cy_b, W, H,
                   v_cov2d, v_mean2d):
    x = mean_c[..., 0]; y = mean_c[..., 1]; z = mean_c[..., 2]
    tan_fovx = 0.5 * W / fx_b; tany = 0.5 * H / fy_b
    lim_x_pos = (W - cx_b) / fx_b + 0.3 * tan_fovx
    lim_x_neg = cx_b / fx_b + 0.3 * tan_fovx
    lim_y_pos = (H - cy_b) / fy_b + 0.3 * tany
    lim_y_neg = cy_b / fy_b + 0.3 * tany
    rz = 1.0 / z; rz2 = rz * rz; rz3 = rz2 * rz
    tx = z * torch.clamp(x * rz, -lim_x_neg, lim_x_pos)
    ty = z * torch.clamp(y * rz, -lim_y_neg, lim_y_pos)
    # J (recomputed here exactly as in forward)
    J = torch.zeros(mean_c.shape[:-1] + (2, 3), dtype=mean_c.dtype, device=mean_c.device)
    J[..., 0, 0] = fx_b * rz
    J[..., 0, 2] = -fx_b * tx * rz2
    J[..., 1, 1] = fy_b * rz
    J[..., 1, 2] = -fy_b * ty * rz2

    v_cov3d = J.transpose(-1, -2) @ v_cov2d @ J
    v_mean3d = torch.zeros_like(mean_c)
    v_mean3d[..., 0] = fx_b * rz * v_mean2d[..., 0]
    v_mean3d[..., 1] = fy_b * rz * v_mean2d[..., 1]
    v_mean3d[..., 2] = -(fx_b * x * v_mean2d[..., 0] + fy_b * y * v_mean2d[..., 1]) * rz2

    v_J = v_cov2d @ J @ covar_c.transpose(-1, -2) + v_cov2d.transpose(-1, -2) @ J @ covar_c
    in_x = (x * rz <= lim_x_pos) & (x * rz >= -lim_x_neg)
    in_y = (y * rz <= lim_y_pos) & (y * rz >= -lim_y_neg)
    dz = torch.zeros_like(z)
    dz = dz + torch.where(in_x, -fx_b * rz2 * v_J[..., 0, 2],
                          -fx_b * rz3 * v_J[..., 0, 2] * tx)
    dz = dz + torch.where(in_y, -fy_b * rz2 * v_J[..., 1, 2],
                          -fy_b * rz3 * v_J[..., 1, 2] * ty)
    dz = dz + (-fx_b * rz2 * v_J[..., 0, 0] - fy_b * rz2 * v_J[..., 1, 1]
                + 2 * fx_b * tx * rz3 * v_J[..., 0, 2] + 2 * fy_b * ty * rz3 * v_J[..., 1, 2])
    v_mean3d[..., 2] = v_mean3d[..., 2] + dz
    return v_mean3d, v_cov3d


def posW2C_VJP(R, t, pW, v_pC):
    v_R = v_pC.unsqueeze(-1) @ pW.unsqueeze(-2)     # outer(v_pC, pW)
    v_t = v_pC
    v_pW = (R.transpose(-1, -2) @ v_pC.unsqueeze(-1)).squeeze(-1)
    return v_R, v_t, v_pW


def covarW2C_VJP(R, covarW, v_covarC):
    v_R = v_covarC @ R @ covarW.transpose(-1, -2) + v_covarC.transpose(-1, -2) @ R @ covarW
    v_covarW = R.transpose(-1, -2) @ v_covarC @ R
    return v_R, v_covarW


def quat_to_rotmat_vjp(quat, v_R):
    # NOTE: gsplat passes the ALREADY-NORMALIZED quaternion here (normalization
    # happens once upstream in the trainer via _safe_normalize). The CUDA vjp
    # formula is expressed in the normalized components (w,x,y,z) and its final
    # projection step removes the radial (norm) component. We therefore feed the
    # normalized components directly, matching the source exactly.
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qn = torch.stack([w, x, y, z], dim=-1)
    v_qn = torch.stack([
        2 * (x * (v_R[..., 1, 2] - v_R[..., 2, 1]) + y * (v_R[..., 2, 0] - v_R[..., 0, 2]) + z * (v_R[..., 0, 1] - v_R[..., 1, 0])),
        2 * (-2 * x * (v_R[..., 1, 1] + v_R[..., 2, 2]) + y * (v_R[..., 0, 1] + v_R[..., 1, 0]) + z * (v_R[..., 0, 2] + v_R[..., 2, 0]) + w * (v_R[..., 1, 2] - v_R[..., 2, 1])),
        2 * (x * (v_R[..., 0, 1] + v_R[..., 1, 0]) - 2 * y * (v_R[..., 0, 0] + v_R[..., 2, 2]) + z * (v_R[..., 1, 2] + v_R[..., 2, 1]) + w * (v_R[..., 2, 0] - v_R[..., 0, 2])),
        2 * (x * (v_R[..., 0, 2] + v_R[..., 2, 0]) + y * (v_R[..., 1, 2] + v_R[..., 2, 1]) - 2 * z * (v_R[..., 0, 0] + v_R[..., 1, 1]) + w * (v_R[..., 0, 1] - v_R[..., 1, 0])),
    ], dim=-1)
    return v_qn - (v_qn * qn).sum(-1, keepdim=True) * qn


def quat_scale_to_covar_vjp(quat, scale, R, v_covar):
    sx, sy, sz = scale[..., 0], scale[..., 1], scale[..., 2]
    S = torch.zeros(scale.shape[:-1] + (3, 3), dtype=scale.dtype, device=scale.device)
    S[..., 0, 0] = sx; S[..., 1, 1] = sy; S[..., 2, 2] = sz
    M = R @ S
    v_M = (v_covar + v_covar.transpose(-1, -2)) @ M
    v_R = v_M @ S
    v_quat = quat_to_rotmat_vjp(quat, v_R)
    # v_scale[k] = sum_i R[i,k] * v_M[i,k]  (column-k dot product)
    v_scale = (R * v_M).sum(dim=-2)
    return v_quat, v_scale


def viewmat_vjp(v_R, v_t, C):
    # scatter v_R (C,N,3,3) summed over N, v_t (C,N,3) into a (C,4,4) viewmat grad
    v_Rsum = v_R.sum(dim=1)          # (C,3,3)
    v_tsum = v_t.sum(dim=1)          # (C,3)
    v_vm = torch.zeros(C, 4, 4, dtype=v_R.dtype, device=v_R.device)
    v_vm[..., :3, :3] = v_Rsum
    v_vm[..., :3, 3] = v_tsum
    return v_vm


def recompute_intermediates(inp: ProjInputs):
    """Recompute R, covar, mean_c, covar_c, J, covar2d from raw inputs exactly as the
    shipped CUDA backward does. Returns them so the oracle can compare against the
    forward-saved values (the redundancy under test)."""
    C = inp.viewmats.shape[0]; N = inp.means.shape[0]
    Rcam = _R_from_viewmat(inp.viewmats); t = inp.viewmats[..., :3, 3]
    R = Rcam.unsqueeze(1).expand(C, N, 3, 3).clone()
    t_b = t.unsqueeze(1).expand(C, N, 3).clone()
    means = inp.means.unsqueeze(0).expand(C, N, 3)
    covar, _, _ = quat_scale_to_covar(inp.quats.unsqueeze(0).expand(C, N, 4),
                                      inp.scales.unsqueeze(0).expand(C, N, 3))
    covar = covar.expand(C, N, 3, 3).clone()
    mean_c = posW2C(R, t_b, means)
    covar_c = covarW2C(R, covar)
    K = inp.Ks
    fx_b = K[:, 0, 0].view(C, 1).expand(C, N)
    fy_b = K[:, 1, 1].view(C, 1).expand(C, N)
    cx_b = K[:, 0, 2].view(C, 1).expand(C, N)
    cy_b = K[:, 1, 2].view(C, 1).expand(C, N)
    cov2d, mean2d, J = persp_proj(mean_c, covar_c, fx_b, fy_b, cx_b, cy_b,
                                  float(inp.width), float(inp.height), want_jacobian=True)
    return {"R": R, "covar": covar, "mean_c": mean_c, "covar_c": covar_c,
            "J": J, "covar2d": cov2d, "mean2d": mean2d}


def backward(inp: ProjInputs, fwd: FwdResult, v_means2d, v_depths, v_conics,
             v_compensations, mode: str, calc_compensations: bool = False):
    """mode='recompute' recomputes R,covar,mean_c,covar_c,J from inputs (shipped CUDA bwd).
       mode='reuse'    reuses fwd intermediates (the caching variant under test)."""
    C = inp.viewmats.shape[0]; N = inp.means.shape[0]
    device = inp.means.device; dtype = inp.means.dtype

    if mode == 'reuse':
        R = fwd.R; covar = fwd.covar; mean_c = fwd.mean_c; covar_c = fwd.covar_c; J_fwd = fwd.J
        # also reuse the post-blur covar2d used to form conics
        cov2d_blur = fwd.covar2d; cov2d_inv = fwd.covar2d_inv
    else:
        ri = recompute_intermediates(inp)
        R = ri["R"]; covar = ri["covar"]; mean_c = ri["mean_c"]; covar_c = ri["covar_c"]
        J_fwd = ri["J"]
        cov2d_blur, _, _, _ = add_blur(inp.eps2d, ri["covar2d"])
        cov2d_inv, _ = inverse_mat2(cov2d_blur)

    # conic inverse VJP. conics store (A,B,C)=(inv00,inv01,inv11) with the off-diagonal
    # already halved (B = -(cov01+cov10)/2/det), so the adjoint off-diagonal carries *.5,
    # matching CUDA: mat2(v_conic_x, v_conic_y*.5, v_conic_y*.5, v_conic_z).
    z = torch.zeros_like(v_conics[..., 0])
    covar2d_inv_mat = mat2(v_conics[..., 0], v_conics[..., 1] * 0.5,
                           v_conics[..., 1] * 0.5, v_conics[..., 2])
    v_cov2d = inverse_vjp(cov2d_inv, covar2d_inv_mat)

    if calc_compensations and v_compensations is not None:
        # CUDA passes covar2d_inv (== conics, the post-blur inverse) as conic_blur.
        v_cov2d = v_cov2d + add_blur_vjp(inp.eps2d, cov2d_inv, fwd.compensations, v_compensations)

    K = inp.Ks
    fx_b = K[:, 0, 0].view(C, 1).expand(C, N)
    fy_b = K[:, 1, 1].view(C, 1).expand(C, N)
    cx_b = K[:, 0, 2].view(C, 1).expand(C, N)
    cy_b = K[:, 1, 2].view(C, 1).expand(C, N)

    v_mean2d_val = v_means2d  # (C,N,2) H8_MR==False path uses v_means2d directly
    v_mean_c, v_covar_c = persp_proj_vjp(mean_c, covar_c, fx_b, fy_b, cx_b, cy_b,
                                         float(inp.width), float(inp.height),
                                         v_cov2d, v_mean2d_val)
    v_mean_c[..., 2] = v_mean_c[..., 2] + v_depths

    v_R = torch.zeros_like(R)
    v_mean = torch.zeros_like(inp.means.unsqueeze(0).expand(C, N, 3))
    v_covar = torch.zeros(N, 3, 3, dtype=dtype, device=device)

    # accumulate over cameras
    v_mean_c_sum = torch.zeros(N, 3, dtype=dtype, device=device)
    v_covar_c_sum = torch.zeros(N, 3, 3, dtype=dtype, device=device)
    v_R_total = torch.zeros(N, 3, 3, dtype=dtype, device=device)
    for ci in range(C):
        R_ci = R[ci]; t_ci = inp.viewmats[ci, :3, 3]
        means_ci = inp.means
        covar_ci, _, _ = quat_scale_to_covar(inp.quats, inp.scales)
        v_Rc, v_tc, v_pWc = posW2C_VJP(R_ci, t_ci, means_ci, v_mean_c[ci])
        v_Rcc, v_covarWc = covarW2C_VJP(R_ci, covar_ci, v_covar_c[ci])
        v_mean_c_sum = v_mean_c_sum + v_pWc
        v_R_total = v_R_total + v_Rc + v_Rcc
        v_covar = v_covar + v_covarWc

    rotmat = quat_to_rotmat(inp.quats)  # (N,3,3) quaternion-derived rotation, NOT camera R
    v_quat, v_scale = quat_scale_to_covar_vjp(inp.quats, inp.scales, rotmat, v_covar)
    # viewmat gradient: sum per-camera adjoint contributions over cameras.
    v_viewmats = _multi_cam_viewmat_grad(inp, fwd, v_mean_c, v_covar_c)

    return {
        'v_means': v_mean_c_sum,
        'v_quats': v_quat,
        'v_scales': v_scale,
        'v_viewmats': v_viewmats,
        'v_covar': v_covar,
    }


def _multi_cam_viewmat_grad(inp, fwd, v_mean_c, v_covar_c):
    C = inp.viewmats.shape[0]; N = inp.means.shape[0]
    v_viewmats = torch.zeros_like(inp.viewmats)
    for ci in range(C):
        R_ci = _R_from_viewmat(inp.viewmats)[ci]
        t_ci = inp.viewmats[ci, :3, 3]
        v_Rc, v_tc, _ = posW2C_VJP(R_ci, t_ci, inp.means, v_mean_c[ci])
        covar_ci, _, _ = quat_scale_to_covar(inp.quats, inp.scales)
        v_Rcc, _ = covarW2C_VJP(R_ci, covar_ci, v_covar_c[ci])
        v_viewmats[ci, :3, :3] = (v_Rc + v_Rcc).sum(dim=0)
        v_viewmats[ci, :3, 3] = v_tc.sum(dim=0)
    return v_viewmats


# ----------------------------------------------------------------------------- 
# AUTOGRAD GROUND TRUTH
# ----------------------------------------------------------------------------- 
def _ag_projection(inp: ProjInputs, calc_compensations: bool = False):
    """Differentiable projection mirroring forward(); returns (outputs_dict, leaves)."""
    C = inp.viewmats.shape[0]; N = inp.means.shape[0]
    Rcam = _R_from_viewmat(inp.viewmats)
    t = inp.viewmats[..., :3, 3]
    R = Rcam.unsqueeze(1).expand(C, N, 3, 3)
    t_b = t.unsqueeze(1).expand(C, N, 3)
    means = inp.means.unsqueeze(0).expand(C, N, 3)
    covar, _, _ = quat_scale_to_covar(inp.quats.unsqueeze(0).expand(C, N, 4),
                                      inp.scales.unsqueeze(0).expand(C, N, 3))
    covar = covar.expand(C, N, 3, 3)
    mean_c = posW2C(R, t_b, means)
    covar_c = covarW2C(R, covar)
    K = inp.Ks
    fx_b = K[:, 0, 0].view(C, 1).expand(C, N)
    fy_b = K[:, 1, 1].view(C, 1).expand(C, N)
    cx_b = K[:, 0, 2].view(C, 1).expand(C, N)
    cy_b = K[:, 1, 2].view(C, 1).expand(C, N)
    cov2d, mean2d = persp_proj(mean_c, covar_c, fx_b, fy_b, cx_b, cy_b,
                               float(inp.width), float(inp.height))
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    cov2d_b = cov2d.clone()
    cov2d_b[..., 0, 0] = cov2d_b[..., 0, 0] + inp.eps2d
    cov2d_b[..., 1, 1] = cov2d_b[..., 1, 1] + inp.eps2d
    det = cov2d_b[..., 0, 0] * cov2d_b[..., 1, 1] - cov2d_b[..., 0, 1] * cov2d_b[..., 1, 0]
    conics = torch.stack([cov2d_b[..., 1, 1] / det,
                          -(cov2d_b[..., 0, 1] + cov2d_b[..., 1, 0]) / 2 / det,
                          cov2d_b[..., 0, 0] / det], dim=-1)
    depths = mean_c[..., 2]
    comp = None
    if calc_compensations:
        comp = torch.sqrt(torch.clamp(det_orig / det, min=EPS))
    return {'means2d': mean2d, 'depths': depths, 'conics': conics, 'compensations': comp}


def autograd_grad(inp: ProjInputs, v_means2d, v_depths, v_conics, v_compensations,
                  calc_compensations: bool = False):
    leaves = {'means': inp.means.detach().clone().requires_grad_(True),
              'quats': inp.quats.detach().clone().requires_grad_(True),
              'scales': inp.scales.detach().clone().requires_grad_(True),
              'viewmats': inp.viewmats.detach().clone().requires_grad_(True)}
    ag_in = ProjInputs(leaves['means'], leaves['quats'], leaves['scales'],
                       leaves['viewmats'], inp.Ks, inp.width, inp.height,
                       inp.eps2d, inp.near_plane, inp.far_plane)
    out = _ag_projection(ag_in, calc_compensations)
    loss = (out['means2d'] * v_means2d.unsqueeze(0)).sum() + \
           (out['depths'] * v_depths.unsqueeze(0)).sum() + \
           (out['conics'] * v_conics.unsqueeze(0)).sum()
    if calc_compensations and v_compensations is not None and out['compensations'] is not None:
        loss = loss + (out['compensations'] * v_compensations.unsqueeze(0)).sum()
    grads = torch.autograd.grad(loss, [leaves['means'], leaves['quats'], leaves['scales'], leaves['viewmats']])
    return {'v_means': grads[0], 'v_quats': grads[1], 'v_scales': grads[2], 'v_viewmats': grads[3]}
