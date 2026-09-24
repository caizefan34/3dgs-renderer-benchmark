#!/usr/bin/env python3
"""Compute SIGMA_GATE classification fractions from captured forward state."""
import json, math, os, sys
import numpy as np
import torch

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    SH_DEGREE = 3; K_SH = (SH_DEGREE + 1) ** 2
    from plyfile import PlyData
    scene_configs = {
        "room": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
            "native_w": 3114, "native_h": 2075,
        },
    }
    cfg = scene_configs[args.scene]
    scale = 2048 / max(cfg["native_w"], cfg["native_h"])
    width = int(round(cfg["native_w"] * scale))
    height = int(round(cfg["native_h"] * scale))

    ply = PlyData.read(cfg["ply"]); v = ply["vertex"]; N = len(v)
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), dtype=torch.float32, device=device)
    quats = torch.tensor(np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]), dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]), dtype=torch.float32, device=device))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], dtype=torch.float32, device=device))
    f_dc = torch.tensor(np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]), dtype=torch.float32, device=device)
    n_rest = 3 * (K_SH - 1)
    f_rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], dtype=torch.float32, device=device) for i in range(n_rest)], dim=1)
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc; sh[:, 1:] = f_rest

    with open(cfg["cams"]) as f:
        cams = json.load(f)
    c = cams[0]
    R = np.asarray(c["rotation"], dtype=np.float64); p = np.asarray(c["position"], dtype=np.float64)
    Rw2c = R.T; vm = np.eye(4); vm[:3, :3] = Rw2c; vm[:3, 3] = -Rw2c @ p
    scale_f = width / float(c["width"])
    K = np.array([[float(c["fx"]) * scale_f, 0.0, (width - 1) / 2.0],
                  [0.0, float(c["fy"]) * scale_f, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    vm_t = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    tile_size = 16
    tile_width = math.ceil(width / tile_size); tile_height = math.ceil(height / tile_size)

    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native

    with torch.no_grad():
        visible_ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm_t, K_t, width, height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, camera_model="pinhole")
        v_means, v_quats, v_scales, v_opacities, v_colors = _gather_visible_native(
            means, quats, scales, opacities, sh, visible_ids)
        N_visible = visible_ids.numel()
        v_opacities_b = v_opacities.unsqueeze(0).contiguous()
        opacities_bc = torch.broadcast_to(v_opacities_b[..., None, :], (1, 1, N_visible)).contiguous()
        v_means_b = v_means.unsqueeze(0).contiguous(); v_quats_b = v_quats.unsqueeze(0).contiguous()
        v_scales_b = v_scales.unsqueeze(0).contiguous()
        v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors

        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=v_means_b, covars=None, quats=v_quats_b, scales=v_scales_b,
            viewmats=vm_t, Ks=K_t, width=width, height=height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
            calc_compensations=False, camera_model="pinhole")

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_images=1, image_ids=None, gaussian_ids=None,
            conics=conics, opacities=opacities_bc)
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height).reshape((1, 1, tile_height, tile_width))

    n_isects = isect_ids.numel()
    print(f"[data] N_visible={N_visible}, n_isects={n_isects}")

    # Get data on CPU
    means2d_np = means2d[0, 0].cpu().numpy()  # [N_v, 2]
    conics_np = conics[0, 0].cpu().numpy()    # [N_v, 3]
    opacities_np = v_opacities.cpu().numpy()  # [N_v]
    flatten_ids_np = flatten_ids.cpu().numpy().astype(np.int64)
    isect_offsets_np = isect_offsets[0, 0].cpu().numpy().astype(np.int32)

    ALPHA_THRESHOLD = 1.0 / 255.0
    MAX_ALPHA = 0.99

    # For each intersection, compute sigma at the tile center pixel
    # (representative sample — the actual kernel evaluates per-pixel)
    # Use tile center as representative pixel
    offsets_flat = isect_offsets_np.reshape(-1)
    offsets_ext = np.append(offsets_flat, n_isects)
    tile_ids = np.searchsorted(offsets_ext, np.arange(n_isects), side="right") - 1
    tile_ids = np.clip(tile_ids, 0, tile_width * tile_height - 1)

    # Tile center pixel
    tile_x = (tile_ids % tile_width) * tile_size + tile_size / 2.0
    tile_y = (tile_ids // tile_width) * tile_size + tile_size / 2.0

    gids = flatten_ids_np
    gx = means2d_np[gids, 0]
    gy = means2d_np[gids, 1]
    cx = conics_np[gids, 0]
    cy = conics_np[gids, 1]
    cz = conics_np[gids, 2]
    opac = opacities_np[gids]

    dx = gx - tile_x
    dy = gy - tile_y
    sigma = 0.5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy

    # Classify
    n_total = len(sigma)
    drop_sigma_neg = np.sum(sigma < 0)
    drop_opac_low = np.sum((sigma >= 0) & (opac < ALPHA_THRESHOLD))

    # sigma_drop = log(opac / ALPHA_THRESHOLD) for opac >= ALPHA_THRESHOLD
    valid_for_drop = (sigma >= 0) & (opac >= ALPHA_THRESHOLD)
    sigma_drop = np.full(n_total, np.inf)
    mask = valid_for_drop
    sigma_drop[mask] = np.log(opac[mask] / ALPHA_THRESHOLD)
    drop_sigma_high = np.sum(valid_for_drop & (sigma > sigma_drop))

    # Clamp: opac >= MAX_ALPHA and sigma < sigma_clamp = log(opac / MAX_ALPHA)
    valid_for_clamp = valid_for_drop & (sigma <= sigma_drop)
    sigma_clamp = np.full(n_total, -np.inf)
    mask2 = valid_for_clamp & (opac >= MAX_ALPHA)
    sigma_clamp[mask2] = np.log(opac[mask2] / MAX_ALPHA)
    clamp = np.sum(mask2 & (sigma < sigma_clamp))

    # Exp required
    exp_required = np.sum(valid_for_clamp & ~(mask2 & (sigma < sigma_clamp)))

    n_drop = drop_sigma_neg + drop_opac_low + drop_sigma_high
    n_clamp = clamp
    n_exp = exp_required

    # Also compute for corner pixels (worst case — farther from Gaussian center)
    # Use 4 corner pixels: (0,0), (15,0), (0,15), (15,15) relative to tile origin
    corners = [(0, 0), (15, 0), (0, 15), (15, 15)]
    corner_stats = {}
    for cx_off, cy_off in corners:
        px = (tile_ids % tile_width) * tile_size + cx_off + 0.5
        py = (tile_ids // tile_width) * tile_size + cy_off + 0.5
        dx_c = gx - px
        dy_c = gy - py
        sigma_c = 0.5 * (cx * dx_c * dx_c + cz * dy_c * dy_c) + cy * dx_c * dy_c
        # Classify
        d_sn = np.sum(sigma_c < 0)
        d_ol = np.sum((sigma_c >= 0) & (opac < ALPHA_THRESHOLD))
        sd = np.full(n_total, np.inf)
        m = (sigma_c >= 0) & (opac >= ALPHA_THRESHOLD)
        sd[m] = np.log(opac[m] / ALPHA_THRESHOLD)
        d_sh = np.sum(m & (sigma_c > sd))
        mc = m & (sigma_c <= sd) & (opac >= MAX_ALPHA)
        sc = np.full(n_total, -np.inf)
        mc2 = mc
        sc[mc2] = np.log(opac[mc2] / MAX_ALPHA)
        cl = np.sum(mc2 & (sigma_c < sc))
        ex = np.sum(m & (sigma_c <= sd) & ~mc2)
        corner_stats[f"corner_{cx_off}_{cy_off}"] = {
            "drop": int(d_sn + d_ol + d_sh), "clamp": int(cl), "exp": int(ex)
        }

    # Also sample random pixels within tiles
    rng = np.random.RandomState(42)
    n_samples = min(100000, n_isects)
    sample_idx = rng.choice(n_isects, n_samples, replace=False)
    rand_px = (tile_ids[sample_idx] % tile_width) * tile_size + rng.uniform(0, 16, n_samples)
    rand_py = (tile_ids[sample_idx] // tile_width) * tile_size + rng.uniform(0, 16, n_samples)
    dx_r = gx[sample_idx] - rand_px
    dy_r = gy[sample_idx] - rand_py
    sigma_r = 0.5 * (cx[sample_idx] * dx_r * dx_r + cz[sample_idx] * dy_r * dy_r) + cy[sample_idx] * dx_r * dy_r
    opac_r = opac[sample_idx]
    d_sn_r = np.sum(sigma_r < 0)
    d_ol_r = np.sum((sigma_r >= 0) & (opac_r < ALPHA_THRESHOLD))
    sd_r = np.full(n_samples, np.inf)
    m_r = (sigma_r >= 0) & (opac_r >= ALPHA_THRESHOLD)
    sd_r[m_r] = np.log(opac_r[m_r] / ALPHA_THRESHOLD)
    d_sh_r = np.sum(m_r & (sigma_r > sd_r))
    mc_r = m_r & (sigma_r <= sd_r) & (opac_r >= MAX_ALPHA)
    sc_r = np.full(n_samples, -np.inf)
    mc2_r = mc_r
    sc_r[mc2_r] = np.log(opac_r[mc2_r] / MAX_ALPHA)
    cl_r = np.sum(mc2_r & (sigma_r < sc_r))
    ex_r = np.sum(m_r & (sigma_r <= sd_r) & ~mc2_r)

    results = {
        "scene": args.scene, "n_isects": int(n_isects),
        "tile_center_classification": {
            "N_sample_candidates": int(n_total),
            "N_drop_before_exp": int(n_drop),
            "N_clamp_without_exp": int(n_clamp),
            "N_exp_required": int(n_exp),
            "fraction_drop": float(n_drop / n_total),
            "fraction_clamp": float(n_clamp / n_total),
            "fraction_exp_required": float(n_exp / n_total),
        },
        "random_pixel_sample": {
            "N_sample": int(n_samples),
            "N_drop": int(d_sn_r + d_ol_r + d_sh_r),
            "N_clamp": int(cl_r),
            "N_exp": int(ex_r),
            "fraction_drop": float((d_sn_r + d_ol_r + d_sh_r) / n_samples),
            "fraction_clamp": float(cl_r / n_samples),
            "fraction_exp": float(ex_r / n_samples),
        },
        "corner_classifications": corner_stats,
    }

    print(f"\n===== SIGMA_GATE Classification (tile center) =====")
    print(f"N_sample_candidates = {n_total}")
    print(f"N_drop_before_exp = {n_drop} ({n_drop/n_total*100:.1f}%)")
    print(f"N_clamp_without_exp = {n_clamp} ({n_clamp/n_total*100:.1f}%)")
    print(f"N_exp_required = {n_exp} ({n_exp/n_total*100:.1f}%)")

    print(f"\n===== SIGMA_GATE Classification (random pixel sample) =====")
    print(f"N_sample = {n_samples}")
    print(f"N_drop = {d_sn_r + d_ol_r + d_sh_r} ({(d_sn_r + d_ol_r + d_sh_r)/n_samples*100:.1f}%)")
    print(f"N_clamp = {cl_r} ({cl_r/n_samples*100:.1f}%)")
    print(f"N_exp = {ex_r} ({ex_r/n_samples*100:.1f}%)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
