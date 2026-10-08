"""
MSB0-P0 Exact-Replay Oracle  (P0 THEORY / SOURCE / EXACT REPLAY ONLY - NO CUDA)

Reproduces the FasterGSFused per-pixel front-to-back compositing forward pass and
its bucket-seeded reverse-mode backward pass in pure PyTorch, on EPIC-05 (A100),
so we can test which saved bucket-boundary state components are mathematically
necessary vs. exactly reconstructible from other already-available quantities.

Source lock (see MSB0_SOURCE_LOCK.json):
  faster-gaussian-splatting subdir @ ae665af2f91a4973bb92c40fdc4cba8d12fe3c96
  rasterization/include/kernels_forward.cuh  (blend_cu)
  rasterization/include/kernels_backward.cuh (blend_backward_cu)

Forward compositing (per pixel), Gaussians sorted front-to-back:
  T_0 = 1, C_0 = 0
  for each Gaussian j (front-to-back):
      alpha_j = opacity_j * exp(-0.5 * (conic . delta_j^2))      (clamped >=0)
      if alpha_j < min_alpha_threshold (1/255): skip
      C_{j+1} = C_j + T_j * alpha_j * color_j
      T_{j+1} = T_j * (1 - alpha_j)
      every 32 Gaussians: save bucket state (C_b, T_b)  <-- checkpoint
  final pixel = C_N + T_N * bg
  save tile_final_transmittances = T_N

Backward (per bucket of 32, reverse within bucket), seeded from saved state:
  color_pixel_after = C_final_after_bg - T_N*bg - C_b   (accumulated color AFTER bucket b)
  transmittance     = T_b                                 (transmittance entering bucket b)
  grad_color_pixel  = dL/d(pixel)  (from grad_image)
  grad_alpha_common = T_N * -dot(grad_color_pixel, bg)
  walk Gaussians in reverse within the bucket:
      blending_weight = transmittance * alpha
      dL_dcolor      += blending_weight * grad_color_pixel * color_grad_factor
      color_pixel_after -= blending_weight * color
      one_minus_alpha = 1 - alpha; oma_rcp = 1/max(oma, 1e-6)
      dL_dalpha_from_color = dot(transmittance*color - color_pixel_after*oma_rcp, grad_color_pixel)
      dL_dalpha_from_alpha = grad_alpha_common * oma_rcp
      dL_dalpha = sum
      dL_dopacity = gaussian * dL_dalpha
      gaussian_grad_helper = -alpha * dL_dalpha
      dL_dconic  += 0.5 * ggh * (dx^2, dx*dy, dy^2)
      dL_dmean2d += ggh * (conic.x*dx + conic.y*dy, conic.y*dx + conic.z*dy)
      transmittance *= one_minus_alpha
"""
import math
import json
import torch

# ---- constants from rasterization_config.h ----
TILE_W = 16
TILE_H = 16
BLOCK_SIZE_BLEND = TILE_W * TILE_H          # 256
MIN_ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4
ONE_MINUS_ALPHA_EPS = 1e-6
DILATION = 0.3
BUCKET = 32

# SH constants (sh_utils.cuh)
C0 = 0.28209479177387814
C1 = 0.48860251190291987
C2a = 1.0925484305920792
C2b = 0.94617469575755997
C2c = 0.31539156525251999
C2d = 0.54627421529603959
C3a = 0.59004358992664352
C3b = 1.7701307697799304
C3c = 2.8906114426405538
C3d = 0.45704579946446572
C3e = 2.2852289973223288
C3f = 1.865881662950577
C3g = 1.1195289973203462
C3h = 1.4453057213202769


def sh_to_color(sh0, shrest, mean3d, cam_pos, active_bases):
    """sh0: (N,3), shrest: (N,15,3), mean3d: (N,3). Returns (N,3) unclamped color."""
    result = 0.5 + C0 * sh0
    if active_bases > 1:
        d = mean3d - cam_pos
        norm = torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(1e-12)
        xyz = d / norm
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        c = shrest  # (N,15,3)
        result = result - C1 * y * c[:, 0] + C1 * z * c[:, 1] - C1 * x * c[:, 2]
        if active_bases > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            result = (result
                      + C2a * xy * c[:, 3] - C2a * yz * c[:, 4]
                      + (C2b * zz - C2c) * c[:, 5]
                      - C2a * xz * c[:, 6] + C2d * (xx - yy) * c[:, 7])
            if active_bases > 9:
                result = (result
                          + y * (C3a * yy - C3b * xx) * c[:, 8]
                          + C3c * xy * z * c[:, 9]
                          + y * (C3d - C3e * zz) * c[:, 10]
                          + z * (C3f * zz - C3g) * c[:, 11]
                          + x * (C3d - C3e * zz) * c[:, 12]
                          + C3h * z * (xx - yy) * c[:, 13]
                          + x * (C3b * yy - C3a * xx) * c[:, 14])
    return result


def project_gaussians(ckpt, w2c, cam_pos, active_bases, width, height,
                      focal_x, focal_y, center_x, center_y,
                      near_plane, far_plane, device=None):
    """Replicates preprocess_cu geometry -> per-Gaussian mean2d, conic, opacity, color, depth, screen_bounds."""
    dev = device if device is not None else ckpt['gaussians._means'].device
    means = ckpt['gaussians._means'].to(torch.float64).to(dev)
    scales = torch.exp(2.0 * ckpt['gaussians._scales'].to(torch.float64)).to(dev)
    rots = ckpt['gaussians._rotations'].to(torch.float64).to(dev)
    opac_raw = ckpt['gaussians._opacities'].to(torch.float64).to(dev)
    sh0 = ckpt['gaussians._sh_coefficients_0'].to(torch.float64).to(dev)[:, 0, :]   # (N,3)
    shrest = ckpt['gaussians._sh_coefficients_rest'].to(torch.float64).to(dev)       # (N,15,3)
    opacity = 1.0 / (1.0 + torch.exp(-opac_raw[:, 0]))

    N = means.shape[0]
    # depth (row 3 of w2c)
    w2c_r1, w2c_r2, w2c_r3 = w2c[0], w2c[1], w2c[2]
    depth = w2c_r3[0]*means[:,0] + w2c_r3[1]*means[:,1] + w2c_r3[2]*means[:,2] + w2c_r3[3]
    x = (w2c_r1[0]*means[:,0] + w2c_r1[1]*means[:,1] + w2c_r1[2]*means[:,2] + w2c_r1[3]) / depth
    y = (w2c_r2[0]*means[:,0] + w2c_r2[1]*means[:,1] + w2c_r2[2]*means[:,2] + w2c_r2[3]) / depth

    visible = (depth > near_plane) & (depth < far_plane) & (opacity >= MIN_ALPHA_THRESHOLD)

    # rotation matrix R (row-major m11..m33)
    r, qx, qy, qz = rots[:,0], rots[:,1], rots[:,2], rots[:,3]
    norm_sq = r*r + qx*qx + qy*qy + qz*qz
    nrcp = 1.0 / norm_sq
    R = torch.stack([
        1 - 2*(qy*qy + qz*qz)*nrcp, 2*(qx*qy - r*qz)*nrcp, 2*(qx*qz + r*qy)*nrcp,
        2*(qx*qy + r*qz)*nrcp, 1 - 2*(qx*qx + qz*qz)*nrcp, 2*(qy*qz - r*qx)*nrcp,
        2*(qx*qz - r*qy)*nrcp, 2*(qy*qz + r*qx)*nrcp, 1 - 2*(qx*qx + qy*qy)*nrcp,
    ], dim=1).reshape(N, 3, 3)
    # RSS = R * diag(scales^2) * R^T   (variance = exp(2*raw_scale) = scales^2)
    S2 = scales * scales  # (N,3)
    # RSS[i] = sum_k R[i,:,k]*S2[i,k] outer R[i,:,k]
    RS = R * S2.unsqueeze(1)  # (N,3,3) columns scaled
    RSS = torch.bmm(RS, R.transpose(1, 2))  # (N,3,3) symmetric

    # EWA: jw rows (3x3) from Jacobian; cov2d = J W cov3d W^T J^T with W=I here (world=cam after w2c already applied via x,y,depth)
    # In the CUDA code the Jacobian maps (x_ndc,y_ndc) using focal/depth; cov2d computed from jwc_r1,jwc_r2 (2x3) and cov3d (3x3)
    j11 = focal_x / depth
    j13 = -j11 * x
    j22 = focal_y / depth
    j23 = -j22 * y
    # jw_r1 = (j11*w1x + j13*w3x, j11*w1y + j13*w3y, j11*w1z + j13*w3z); w rows are w2c rows 1..3
    jw_r1 = torch.stack([j11*w2c_r1[0] + j13*w2c_r3[0],
                         j11*w2c_r1[1] + j13*w2c_r3[1],
                         j11*w2c_r1[2] + j13*w2c_r3[2]], dim=1)
    jw_r2 = torch.stack([j22*w2c_r2[0] + j23*w2c_r3[0],
                         j22*w2c_r2[1] + j23*w2c_r3[1],
                         j22*w2c_r2[2] + j23*w2c_r3[2]], dim=1)
    # jwc_r1 = jw_r1 * cov3d ; jwc_r2 = jw_r2 * cov3d
    jwc_r1 = torch.bmm(jw_r1.unsqueeze(1), RSS).squeeze(1)
    jwc_r2 = torch.bmm(jw_r2.unsqueeze(1), RSS).squeeze(1)
    cov2d_xx = (jw_r1 * jwc_r1).sum(1) + DILATION
    cov2d_xy = (jw_r1 * jwc_r2).sum(1)
    cov2d_yy = (jw_r2 * jwc_r2).sum(1) + DILATION
    det = cov2d_xx * cov2d_yy - cov2d_xy * cov2d_xy
    good = det >= 1e-6
    conic_x = cov2d_yy / det
    conic_y = -cov2d_xy / det
    conic_z = cov2d_xx / det

    mean2d_x = x * focal_x + center_x
    mean2d_y = y * focal_y + center_y

    color = sh_to_color(sh0, shrest, means, cam_pos, active_bases)
    color = torch.clamp(color, min=0.0)

    # screen bounds (tile granularity, 16px) using cutoff_factor=4*log(opacity*255)
    power_threshold = torch.log(opacity.clamp_min(1e-30) * 255.0)
    cutoff = 2.0 * power_threshold
    extent_x = torch.clamp(torch.sqrt(torch.clamp(cov2d_xx * cutoff, min=0)) - 0.5, min=0.0)
    extent_y = torch.clamp(torch.sqrt(torch.clamp(cov2d_yy * cutoff, min=0)) - 0.5, min=0.0)
    gw = (width + TILE_W - 1) // TILE_W
    gh = (height + TILE_H - 1) // TILE_H
    xmin = torch.clamp(torch.floor((mean2d_x - extent_x) / TILE_W).long(), min=0, max=gw - 1)
    xmax = torch.clamp(torch.ceil((mean2d_x + extent_x) / TILE_W).long(), min=0, max=gw - 1)
    ymin = torch.clamp(torch.floor((mean2d_y - extent_y) / TILE_H).long(), min=0, max=gh - 1)
    ymax = torch.clamp(torch.ceil((mean2d_y + extent_y) / TILE_H).long(), min=0, max=gh - 1)

    keep = visible & good & (xmax > xmin) & (ymax > ymin)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    return {
        'idx': idx,
        'mean2d': torch.stack([mean2d_x[idx], mean2d_y[idx]], dim=1),
        'conic': torch.stack([conic_x[idx], conic_y[idx], conic_z[idx]], dim=1),
        'opacity': opacity[idx],
        'color': color[idx],
        'depth': depth[idx],
        'bounds': (xmin[idx], xmax[idx], ymin[idx], ymax[idx]),
        'gw': gw, 'gh': gh,
    }

# =====================================================================
# Exact-replay forward + backward (per tile), faithful to blend_cu /
# blend_backward_cu.  All math in float64 for oracle fidelity.
# =====================================================================

def composite_tile(alphas, colors, opacities, gaussians, bg, n_positions):
    """alphas: (P, n_positions) per-pixel per-gaussian alpha (0 if skipped).
    colors: (P, n_positions, 3). Returns per-pixel bucket state and final."""
    P = alphas.shape[0]
    dev = alphas.device
    C = torch.zeros(P, 3, dtype=torch.float64, device=dev)
    T = torch.ones(P, dtype=torch.float64, device=dev)
    n_buckets = (n_positions + BUCKET - 1) // BUCKET
    bucket_C = []
    bucket_T = []
    last_contributor = torch.zeros(P, dtype=torch.long, device=dev)
    for j in range(n_positions):
        if j % BUCKET == 0:
            bucket_C.append(C.clone())
            bucket_T.append(T.clone())
        a = alphas[:, j]
        valid = a >= MIN_ALPHA_THRESHOLD
        # match CUDA: sub-threshold gaussians do not blend and do not update T
        a_eff = torch.where(valid, a, torch.zeros_like(a))
        C = C + (T * a_eff).unsqueeze(1) * colors[:, j, :]
        T = T * (1.0 - a_eff)
        last_contributor = torch.where(valid, torch.full_like(last_contributor, j + 1), last_contributor)
    C_final = C + T.unsqueeze(1) * bg.unsqueeze(0)
    T_final = T
    return {
        'bucket_C': bucket_C, 'bucket_T': bucket_T,
        'n_buckets': n_buckets,
        'C_final': C_final, 'T_final': T_final,
        'last_contributor': last_contributor,
        'C_no_bg': C,
    }


def backward_tile(alphas, colors, gaussians, bg, saved, grad_image,
                  mode='M0', bucket_C_override=None, bucket_T_override=None):
    """Replay blend_backward_cu. Returns per-position grads."""
    P, n_positions = alphas.shape
    dev = alphas.device
    conic = gaussians['conic'].to(torch.float64)
    mean2d = gaussians['mean2d'].to(torch.float64)
    opacity = gaussians['opacity'].to(torch.float64)
    color = torch.clamp(colors, min=0.0)

    gC = grad_image.to(torch.float64)
    bg64 = bg.to(torch.float64)

    dL_dcolor = torch.zeros(P, n_positions, 3, dtype=torch.float64, device=dev)
    dL_dalpha = torch.zeros(P, n_positions, dtype=torch.float64, device=dev)
    dL_dconic = torch.zeros(P, n_positions, 3, dtype=torch.float64, device=dev)
    dL_dmean2d = torch.zeros(P, n_positions, 2, dtype=torch.float64, device=dev)

    n_buckets = saved['n_buckets']
    bucket_C = saved['bucket_C'] if bucket_C_override is None else bucket_C_override
    bucket_T = saved['bucket_T'] if bucket_T_override is None else bucket_T_override
    T_final = saved['T_final']
    C_no_bg = saved['C_no_bg']
    pixel_xy = gaussians['pixel_xy'].to(torch.float64)  # (P,2)

    # M3: forward carry from the known constant entering-state (C_0=0, T_0=1).
    # This is the exact reconstruction that eliminates bucket_color_transmittance,
    # but it forces sequential cross-bucket processing (Gate 5 violation).
    if mode == 'M3':
        carry_C = torch.zeros(P, 3, dtype=torch.float64, device=dev)
        carry_T = torch.ones(P, dtype=torch.float64, device=dev)

    for b in range(n_buckets):
        lo = b * BUCKET
        hi = min(lo + BUCKET, n_positions)
        seg = range(lo, hi)
        if mode == 'M0':
            C_b = bucket_C[b]
            T_b = bucket_T[b]
        elif mode == 'M1':
            T_b = bucket_T[b]
            C_b = torch.zeros_like(C_no_bg)
        elif mode == 'M2':
            C_b = bucket_C[b]
            T_b = torch.ones_like(T_final)
        elif mode == 'M3':
            C_b = carry_C
            T_b = carry_T
        color_pixel_after = C_no_bg - C_b
        transmittance = T_b.clone()
        grad_alpha_common = T_final * (-(gC * bg64.unsqueeze(0)).sum(dim=1))
        for j in seg:
            a = alphas[:, j]
            valid = (a >= MIN_ALPHA_THRESHOLD)
            # match CUDA 'continue': sub-threshold gaussians are a no-op for the
            # transmittance walk and contribute zero gradient (ggh=0 below).
            a_eff = torch.where(valid, a, torch.zeros_like(a))
            gaussian = a_eff / opacity[j].clamp_min(1e-30)
            blending_weight = transmittance * a_eff
            dL_dcolor[:, j, :] = blending_weight.unsqueeze(1) * gC
            color_pixel_after = color_pixel_after - blending_weight.unsqueeze(1) * color[:, j, :]
            oma = 1.0 - a_eff
            oma_rcp = 1.0 / torch.clamp(oma, min=ONE_MINUS_ALPHA_EPS)
            dL_dalpha_from_color = ((transmittance.unsqueeze(1) * color[:, j, :]
                                     - color_pixel_after * oma_rcp.unsqueeze(1)) * gC).sum(dim=1)
            dL_dalpha_from_alpha = grad_alpha_common * oma_rcp
            dL_dalpha[:, j] = dL_dalpha_from_color + dL_dalpha_from_alpha
            ggh = (-a_eff * dL_dalpha[:, j]).unsqueeze(1)   # (P,1)
            delta = mean2d[j].unsqueeze(0) - pixel_xy         # (P,2)
            cj = conic[j]                                      # (3,)
            dL_dconic[:, j, :] = 0.5 * ggh * torch.stack(
                [delta[:,0]*delta[:,0], delta[:,0]*delta[:,1], delta[:,1]*delta[:,1]], dim=1)
            dL_dmean2d[:, j, :] = ggh * torch.stack(
                [cj[0]*delta[:,0] + cj[1]*delta[:,1],
                 cj[1]*delta[:,0] + cj[2]*delta[:,1]], dim=1)
            transmittance = transmittance * oma
        if mode == 'M3':
            # exiting state of bucket b = entering state of bucket b+1
            carry_C = C_no_bg - color_pixel_after
            carry_T = transmittance.clone()
    return {
        'dL_dcolor': dL_dcolor, 'dL_dalpha': dL_dalpha,
        'dL_dconic': dL_dconic, 'dL_dmean2d': dL_dmean2d,
    }


def fidelity_metrics(ref, cand):
    out = {}
    for k in ref:
        r = ref[k].reshape(-1); c = cand[k].reshape(-1)
        nan_inf = int((~torch.isfinite(c)).sum().item())
        cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), c.unsqueeze(0), dim=1).item()
        rel = (torch.linalg.norm(r - c) / (torch.linalg.norm(r) + 1e-30)).item()
        mae = (r - c).abs().max().item()
        out[k] = {'cos': cos, 'rel_l2': rel, 'max_abs_err': mae, 'nan_inf': nan_inf}
    return out


def passes_gate4(groups):
    return all(v['cos'] >= 0.999999 and v['rel_l2'] <= 1e-5 and v.get('nan_inf', 0) == 0
               for v in groups.values())


# =====================================================================
# Independent autograd ground truth (validates the hand-derived backward).
# Fully differentiable forward per tile; gradients w.r.t. opacity, conic,
# mean2d, color are obtained from PyTorch autograd.
# =====================================================================

def autograd_ground_truth(alpha_in, color_in, opacity_in, conic_in, mean2d_in,
                          pixel_xy, bg, grad_image, skip_threshold=True):
    """alpha_in: (P,M) raw alpha (opacity*gaussian). We recompute alpha from
    opacity/conic/mean2d so conic & mean2d grads are well-defined."""
    P, M = alpha_in.shape
    dev = alpha_in.device
    opacity = opacity_in.detach().clone().to(torch.float64).requires_grad_(True)
    conic = conic_in.detach().clone().to(torch.float64).requires_grad_(True)
    mean2d = mean2d_in.detach().clone().to(torch.float64).requires_grad_(True)
    color = color_in.detach().clone().to(torch.float64).requires_grad_(True)
    bg64 = bg.to(torch.float64)

    delta = mean2d.unsqueeze(0) - pixel_xy.unsqueeze(1)      # (P,M,2)
    exponent = (-0.5 * (conic[:,0].unsqueeze(0) * delta[:,:,0]**2
                        + conic[:,2].unsqueeze(0) * delta[:,:,1]**2)
                - conic[:,1].unsqueeze(0) * delta[:,:,0] * delta[:,:,1])
    gaussian = torch.exp(torch.clamp(exponent, max=0.0))      # (P,M)
    alpha = opacity.unsqueeze(0) * gaussian                   # (P,M)

    C = torch.zeros(P, 3, dtype=torch.float64, device=dev)
    T = torch.ones(P, dtype=torch.float64, device=dev)
    for j in range(M):
        a = alpha[:, j]
        if skip_threshold:
            valid = a >= MIN_ALPHA_THRESHOLD
        else:
            valid = torch.ones_like(a, dtype=torch.bool)
        # blend (zero-out sub-threshold contributions)
        a_eff = torch.where(valid, a, torch.zeros_like(a))
        C = C + (T * a_eff).unsqueeze(1) * color[:, j, :]
        T = T * (1.0 - a_eff)
    image = C + T.unsqueeze(1) * bg64.unsqueeze(0)
    loss = (image * grad_image.to(torch.float64)).sum()
    grads = torch.autograd.grad(loss, [opacity, conic, mean2d, color],
                                retain_graph=False, allow_unused=True)
    g_op, g_conic, g_mean2d, g_color = grads
    # derive dL_dalpha from dL_dopacity = dL_dalpha * gaussian
    dL_dalpha = g_op / gaussian.clamp_min(1e-30)
    return {
        'dL_dcolor': g_color,                 # (P,M,3)
        'dL_dalpha': dL_dalpha,               # (P,M)
        'dL_dconic': g_conic,                 # (P,M,3)
        'dL_dmean2d': g_mean2d,               # (P,M,2)
        'dL_dopacity': g_op,
    }
