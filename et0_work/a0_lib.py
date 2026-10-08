"""MOONSHOT0 A0 shared library (AdjointTensor-GS backward-coefficient census).

Execution engine for the frozen A0_PREREG (2026-09-29). Diagnostic only.

Instrumentation: runtime monkey-patch wrapper around
`_HigsAutogradFunction._native_backward` in an ISOLATED worktree copy of the
frozen C0 renderer (no production file or .so is modified). The wrapper records
the exact frozen captured state and image adjoint, then calls the ORIGINAL
staticmethod, so instrumented and uninstrumented paths are structurally
identical (additionally verified numerically).

Coefficient extraction: exact PyTorch replay of the frozen back-to-front blend
walk (higs_blend_bwd_kernel PX=2 + rasterize_to_pixels_3dgs_blend_bwd,
SCALAR_ADJOINT=1, H8_MR=1, constants from gsplat/cuda/include/Common.h):

  ALPHA_THRESHOLD = 1/255, MAX_ALPHA = 0.99,
  TRANSMITTANCE_THRESHOLD = 1e-4, MIN_ONE_MINUS_ALPHA = 1e-6

  sigma = 0.5*(conic.x*dx^2 + conic.z*dy^2) + conic.y*dx*dy
  vis   = exp(-sigma); alpha = min(MAX_ALPHA, opac*vis)
  valid = (sigma>=0) & (alpha>=ALPHA_THRESHOLD) & inside
          & (flatten_idx <= last_ids[pix])          # early-stop validity
  per pixel, FAR -> NEAR (flatten_ids reversed), T starts at
  1 - render_alphas[pix], buffer starts at 0:
     ra  = 1/max(MIN_ONE_MINUS_ALPHA, 1-alpha); T *= ra
     fac = alpha*T                                   # == q_color[p,i]
     v_alpha = sum_k (rgbs[k]*T - buffer[k]*ra)*v_c[k] + T_final*ra*v_a
     q_alpha[p,i] = v_alpha                          # v_a = 0 for RGB loss
     buffer[k] += rgbs[k]*fac
  (C0 renders RGB with black background: backgrounds=None, no bg term.)

Local derivative factors (frozen, reused verbatim on both the exact and the
reconstructed side of the VJP fidelity test):
  color path : contribution to v_rgb[g] at (p,i) = q_color[p,i] * v_c[p,:]
  opacity    : v_opac[g]  += vis * q_alpha
  H8-MR moments (v_sigma = -opac*vis*q_alpha):
     Sx += v_sigma*dx; Sy += v_sigma*dy
     Sxx += v_sigma*dx^2; Sxy += v_sigma*dx*dy; Syy += v_sigma*dy^2
  contraction (higs_projection_bwd_kernel, H8_MR branch):
     v_means2d = (conic.x*Sx + conic.y*Sy, conic.y*Sx + conic.z*Sy)
     v_conic   = (0.5*Sxx, Sxy, 0.5*Syy)
  then inverse VJP conic->covar2d, persp projection VJP, quat/scale VJP
  (replicated by autograd through a torch forward validated against the
  captured means2d/conics), SH VJP via the frozen gsplat core op, sigmoid/exp/
  normalize chains to raw parameters.
"""
import os, sys, math, json, hashlib, importlib.util
import numpy as np

R = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_moonshot/a0_worktree"       # isolated diagnostic worktree
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(R, "scripts", "epic05", "phase7")
SRC = os.path.join(R, "src")
REFV1 = os.path.join(R, "baseline", "reference_v1")

ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4
MIN_ONE_MINUS_ALPHA = 1e-6

A0_CAPTURE = {"enabled": False, "records": []}

# ----------------------------------------------------------------------------
# environment / model / render
# ----------------------------------------------------------------------------

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def load_ext_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

def bootstrap_c0():
    """EXACT replica of publication_trainer_p12_n1snap.bootstrap_c0(), with
    sys.path pointing at the ISOLATED A0 worktree (copy of the frozen C0
    worktree). No production file is modified."""
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7)
    sys.path.insert(0, SRC)
    sys.path.insert(0, REFV1)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: F401
    from gsplat.experimental.render.functional.gaussian_inference import (  # noqa: F401
        _HigsAutogradFunction, _HIGS_DYNAMIC_SCENE)
    return {
        "arm": "c0", "so_sha256": sha256_file(C0_SO), "so_path": C0_SO,
        "core_sha256": sha256_file(CORE_SO), "core_path": CORE_SO,
        "worktree": C0_WT,
        "worktree_note": "isolated diagnostic copy of "
                         "/root/epic05_frozen/higs_c0_final30k_worktree",
        "instrumentation": "runtime monkey-patch of "
                           "_HigsAutogradFunction._native_backward (records "
                           "saved state + image adjoint, then calls original)",
        "env": {k: os.environ[k] for k in
                ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
                 "HIGS_BWD_ABSGRAD")},
    }

def load_dataset(scene, device="cuda"):
    from dataset import GTDataset
    return GTDataset(scene=scene, repo_root=R, resolution="1080p",
                     device=device, background="black")

def frozen_camera_ids(n_cams, k=8):
    return [int(round(n_cams * (i + 0.5) / k)) for i in range(k)]

def load_model(scene, ckpt_iter, device="cuda"):
    """Load raw params + frozen optimizer state (identical to q0_lib)."""
    import torch
    ckpt_path = f"/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{ckpt_iter}.pt"
    o = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    def P(x): return torch.nn.Parameter(o[x].to(device).float().requires_grad_(True))
    m = dict(xyz=P("xyz"), shs=P("shs"), scaling=P("scaling"),
             rotation=P("rotation"), opacity=P("opacity"))
    osd = o["optimizer_state_dict"]
    opt_state = {}
    for gi, g in enumerate(osd["param_groups"]):
        st = osd["state"][gi]
        opt_state[g["name"]] = {"step": int(st["step"]),
                                "exp_avg": st["exp_avg"].to(device).float(),
                                "exp_avg_sq": st["exp_avg_sq"].to(device).float(),
                                "lr": float(g["lr"])}
    ident = {
        "ckpt_path": ckpt_path, "ckpt_sha256": sha256_file(ckpt_path),
        "scene": scene, "ckpt_iter": int(ckpt_iter),
        "iteration": int(o.get("iteration", ckpt_iter)),
        "num_points": int(o["num_points"]),
        "active_sh_degree": int(o["active_sh_degree"]),
        "spatial_lr_scale": float(o["spatial_lr_scale"]),
        "group_order": [g["name"] for g in osd["param_groups"]],
    }
    return m, opt_state, ident

def activated(m):
    """C0 render inputs, matching the trainer's model.get_* calls exactly."""
    import torch
    import torch.nn.functional as F
    return dict(
        means=m["xyz"],
        quats=F.normalize(m["rotation"], dim=-1),
        scales=torch.exp(m["scaling"]),
        opacities=torch.sigmoid(m["opacity"]),
        colors=m["shs"],
    )

def render_c0(m, cam, sh_degree):
    """EXACT trainer render call (arm c0)."""
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    a = activated(m)
    out = rasterize_gaussian_higs_dynamic(
        a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB",
        eps2d=0.3)
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], info["means2d"], out

def train_loss(pred_hwc, gt_hwc, lambda_dssim=0.2):
    """Exact P11 fused_total_loss semantics (identical to q0_loss.py):
    L = (1-lambda)*L1 + lambda*(1 - fused_ssim(padding='same', train=True))
    on BCHW contiguous tensors."""
    import torch
    import torch.nn.functional as F
    import fused_ssim
    b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    g = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    L1 = F.l1_loss(b, g)
    ssim = fused_ssim.fused_ssim(b, g, padding="same", train=True)
    return (1.0 - lambda_dssim) * L1 + lambda_dssim * (1.0 - ssim)

def install_capture_patch():
    """Runtime monkey-patch of the ISOLATED worktree's _native_backward.
    The frozen code defines it as a plain class-body function called as
    `_HigsAutogradFunction._native_backward(ctx, grad_frame, grad_alpha)`
    (gaussian_inference.py:2288), so the wrapper keeps that exact calling
    convention."""
    from gsplat.experimental.render.functional import gaussian_inference as GI
    orig = GI._HigsAutogradFunction._native_backward

    def wrapped(ctx, grad_frame, grad_alpha):
        if A0_CAPTURE["enabled"]:
            A0_CAPTURE["records"].append({
                "saved_tensors": ctx.saved_tensors,
                "grad_frame": grad_frame, "grad_alpha": grad_alpha,
                "C": ctx.C, "width": ctx.width, "height": ctx.height,
                "tile_size": ctx.tile_size, "sh_degree": ctx.sh_degree,
                "N_total": ctx.N_total, "N_visible": ctx.N_visible,
                "render_mode": ctx.render_mode, "eps2d": ctx.eps2d,
            })
        return orig(ctx, grad_frame, grad_alpha)

    GI._HigsAutogradFunction._native_backward = wrapped
    def uninstall():
        GI._HigsAutogradFunction._native_backward = orig
        A0_CAPTURE["records"].clear()
    return {"patched": "_HigsAutogradFunction._native_backward (runtime wrapper)",
            "production_files_modified": False}, uninstall

def parse_capture(rec):
    """Frozen save_for_backward layout (F9 path):
      0 means, 1 quats, 2 scales, 3 opacities, 4 colors, 5 means2d_proxy,
      6 viewmats, 7 Ks, 8 master_visible_ids,
      9..18 captured = (means2d_f, conics_f, colors_eval_f, opacities_f,
                        tile_offsets_f, flatten_ids_f, render_alphas_f,
                        last_ids_f, radii_f, depth_acc),
      19..23 compact_saved (None under F9), 24 backgrounds_b."""
    st = rec["saved_tensors"]
    (means, quats, scales, opacities, colors, means2d_proxy, viewmats, Ks,
     visible_ids) = st[0:9]
    (means2d_f, conics_f, colors_eval_f, opacities_f, tile_offsets_f,
     flatten_ids_f, render_alphas_f, last_ids_f, radii_f, depth_acc) = st[9:19]
    return {
        "means": means, "quats": quats, "scales": scales,
        "opacities": opacities, "colors": colors,
        "visible_ids": visible_ids,
        "means2d_f": means2d_f, "conics_f": conics_f,
        "colors_eval_f": colors_eval_f, "opacities_f": opacities_f,
        "tile_offsets_f": tile_offsets_f, "flatten_ids_f": flatten_ids_f,
        "render_alphas_f": render_alphas_f, "last_ids_f": last_ids_f,
        "radii_f": radii_f, "depth_acc": depth_acc,
        "grad_frame": rec["grad_frame"], "grad_alpha": rec["grad_alpha"],
        "C": rec["C"], "width": rec["width"], "height": rec["height"],
        "tile_size": rec["tile_size"], "eps2d": rec["eps2d"],
        "sh_degree": rec["sh_degree"], "N_visible": rec["N_visible"], "N_total": rec["N_total"],
        "viewmats": viewmats, "Ks": Ks,
    }

# ----------------------------------------------------------------------------
# exact blend-walk replay (vectorized far->near, chunked)
# ----------------------------------------------------------------------------

def _tile_pixels(cap, tile_x, tile_y):
    """Row-major 16x16 pixel grid of one tile, with clamped indexing for
    out-of-image pixels (the kernel clamps pix_id; `inside` zeroes them)."""
    import torch
    ts = cap["tile_size"]
    H, W = cap["height"], cap["width"]
    dev = cap["flatten_ids_f"].device
    ys = tile_y * ts + torch.arange(ts, device=dev)
    xs = tile_x * ts + torch.arange(ts, device=dev)
    pix_y = ys[:, None].expand(ts, ts).reshape(-1)
    pix_x = xs[None, :].expand(ts, ts).reshape(-1)
    inside = (pix_y < H) & (pix_x < W)
    iy = pix_y.clamp(max=H - 1)
    ix = pix_x.clamp(max=W - 1)
    return pix_y, pix_x, iy, ix, inside

def replay_tile_walk(cap, image_idx, tile_x, tile_y, chunk=4096):
    """Exact far->near replay of the frozen blend walk for ONE tile.
    Returns None for an empty tile, else a dict:
      gids [n] (compact row ids, FAR->NEAR order),
      ridx [n] (global isect positions, FAR->NEAR),
      valid [n,256] accepted mask, q_alpha [n,256], q_color [n,256]
      (k-constant fac field), vis [n,256], dx [n,256], dy [n,256],
      opac_r [n] (per-gaussian opacity), v_c [256,3], px/py [256],
      inside [256]
    All tensors on GPU. n = tile's gaussian count."""
    import torch
    I, H, W = cap["C"], cap["height"], cap["width"]
    ts = cap["tile_size"]
    th = (H + ts - 1) // ts
    tw = (W + ts - 1) // ts
    tile_id = tile_y * tw + tile_x
    to_flat = cap["tile_offsets_f"].reshape(I, -1)
    n_isects = int(cap["flatten_ids_f"].numel())
    r_start = int(to_flat[image_idx, tile_id].item())
    if tile_id + 1 < tw * th:
        r_end = int(to_flat[image_idx, tile_id + 1].item())
    else:
        r_end = n_isects
    n_tile = r_end - r_start
    if n_tile == 0:
        return None
    dev = cap["flatten_ids_f"].device

    pix_y, pix_x, iy, ix, inside = _tile_pixels(cap, tile_x, tile_y)
    px = (pix_x + 0.5).float()
    py = (pix_y + 0.5).float()

    # gsplat stores render_alphas (and the frozen capture, last_ids) with a
    # trailing singleton channel dim [C,H,W,1]; normalize to [H,W]
    ra_img = cap["render_alphas_f"][image_idx].reshape(H, W)
    li_img = cap["last_ids_f"][image_idx].reshape(H, W)
    vrc = cap["grad_frame"].reshape(I, H, W, -1)[image_idx]      # [H, W, 3]
    vra_t = cap["grad_alpha"]
    vra = vra_t.reshape(I, H, W)[image_idx] if vra_t is not None \
        else torch.zeros((H, W), device=dev)

    T_final = (1.0 - ra_img.clamp(0, 1))[iy, ix]                # [256]
    bin_final = li_img[iy, ix]                                  # [256]
    v_c = vrc[iy, ix]                                           # [256, 3]
    v_a = vra[iy, ix]                                           # [256]

    # far->near rows
    gids_all = cap["flatten_ids_f"][r_start:r_end].flip(0)      # [n]
    ridx_all = torch.arange(r_end - 1, r_start - 1, -1, device=dev)
    means2d = cap["means2d_f"][gids_all]
    conics = cap["conics_f"][gids_all]
    rgbs = cap["colors_eval_f"][gids_all]
    opac_r = cap["opacities_f"][gids_all]                       # [n]

    out = {k: [] for k in
           ("valid", "q_alpha", "q_color", "vis", "dx", "dy")}
    T_run = T_final.clone()                                     # [256]
    buf_run = torch.zeros((256, 3), device=dev)                 # [256,3]
    for c0 in range(0, n_tile, chunk):
        c1 = min(n_tile, c0 + chunk)
        dx = means2d[c0:c1, 0:1] - px[None, :]                  # [m,256]
        dy = means2d[c0:c1, 1:2] - py[None, :]
        con = conics[c0:c1]                                     # [m,3]
        sigma = 0.5 * (con[:, 0:1] * dx * dx + con[:, 2:3] * dy * dy) \
                + con[:, 1:2] * dx * dy
        vis = torch.exp(-sigma)
        alpha = torch.clamp(opac_r[c0:c1, None] * vis, max=MAX_ALPHA)
        valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD) & inside[None, :] \
                & (ridx_all[c0:c1, None] <= bin_final[None, :])  # [m,256]
        ra = 1.0 / torch.clamp(1.0 - alpha, min=MIN_ONE_MINUS_ALPHA)
        ra_m = torch.where(valid, ra, torch.ones_like(ra))
        T_after = T_run[None, :] * torch.cumprod(ra_m, dim=0)   # [m,256]
        fac = alpha * T_after
        contrib = rgbs[c0:c1, None, :] * fac[..., None] \
            * valid.float()[..., None]                          # [m,256,3]
        csum = torch.cumsum(contrib, dim=0)
        buffer_in = buf_run[None, :, :] + csum - contrib        # exclusive
        v_alpha = ((rgbs[c0:c1, None, :] * T_after[..., None]
                    - buffer_in * ra[..., None]) * v_c[None, :, :]).sum(-1)
        v_alpha = v_alpha + T_final[None, :] * ra * v_a[None, :]
        q_alpha = torch.where(valid, v_alpha, torch.zeros_like(v_alpha))
        q_color = torch.where(valid, fac, torch.zeros_like(fac))
        out["valid"].append(valid)
        out["q_alpha"].append(q_alpha)
        out["q_color"].append(q_color)
        out["vis"].append(vis)
        out["dx"].append(dx)
        out["dy"].append(dy)
        T_run = T_after[-1]
        buf_run = buf_run + csum[-1]
    res = {
        "gids": gids_all, "ridx": ridx_all, "opac_r": opac_r,
        "v_c": v_c, "px": px, "py": py, "inside": inside,
        "T_final": T_final, "bin_final": bin_final,
    }
    for k, v in out.items():
        res[k] = torch.cat(v, dim=0) if len(v) > 1 else v[0]
    return res

def bucket_bounds(n_tile, gauss_bucket=32):
    """Bucket index ranges in the FAR->NEAR row ordering."""
    nB = (n_tile + gauss_bucket - 1) // gauss_bucket
    return [(b * gauss_bucket, min(n_tile, (b + 1) * gauss_bucket))
            for b in range(nB)]

def accumulate_rows(walk, rows_mask, v_rgb_acc, v_opac_acc, S_acc,
                    q_alpha=None, q_color=None, row_index=None,
                    clamp_grad=False):
    """Accumulate per-compact-row intermediate adjoints over the walk rows
    selected by `rows_mask` [n] (bool, FAR->NEAR rows of ONE tile).

    Uses either the walk's exact fields (q_alpha=None -> walk['q_alpha']) or
    REPLACEMENT fields (the reconstructed Q_hat, already masked to accepted
    pixels). Adds into:
      v_rgb_acc  [N_vis, 3]  color-path adjoint (colors_eval)
      v_opac_acc [N_vis]     activated-opacity adjoint
      S_acc      [N_vis, 5]  H8-MR moments (Sx, Sy, Sxx, Sxy, Syy)
    `row_index` is an index_add-capable long tensor of compact row ids for the
    masked rows (walk['gids'][rows_mask]).
    clamp_grad=True zeroes the opacity/moment chains at pixels where the
    forward alpha was clamped (opac*vis > MAX_ALPHA), i.e. applies the
    min() derivative; the correct variant for the frozen .so is settled
    empirically by the driver's single-camera gradient probe."""
    import torch
    if q_alpha is None:
        q_alpha = walk["q_alpha"]
    if q_color is None:
        q_color = walk["q_color"]
    idx = row_index
    qa = q_alpha[rows_mask]                                   # [m,256]
    qc = q_color[rows_mask]                                   # [m,256]
    vis = walk["vis"][rows_mask]                              # [m,256]
    dx = walk["dx"][rows_mask]
    dy = walk["dy"][rows_mask]
    opac = walk["opac_r"][rows_mask][:, None]                 # [m,1]
    v_c = walk["v_c"]                                         # [256,3]
    # color path: contribution to v_rgb[g] = sum_p qc[m,p] * v_c[p,:]
    v_rgb_rows = qc @ v_c                                     # [m,3]
    v_rgb_acc.index_add_(0, idx, v_rgb_rows)
    # opacity path: vis * q_alpha (optionally gated by the alpha clamp)
    if clamp_grad:
        gate = (opac * vis <= MAX_ALPHA).to(qa.dtype)         # [m,256]
    else:
        gate = torch.ones_like(qa)
    v_opac_rows = (gate * vis * qa).sum(dim=1)                # [m]
    v_opac_acc.index_add_(0, idx, v_opac_rows)
    # H8-MR moments: v_sigma = -opac*vis*q_alpha (same clamp gate)
    v_sigma = -gate * opac * vis * qa                         # [m,256]
    S_rows = torch.stack([
        (v_sigma * dx).sum(dim=1),
        (v_sigma * dy).sum(dim=1),
        (v_sigma * dx * dx).sum(dim=1),
        (v_sigma * dx * dy).sum(dim=1),
        (v_sigma * dy * dy).sum(dim=1),
    ], dim=1)                                                 # [m,5]
    S_acc.index_add_(0, idx, S_rows)
    return None

# ----------------------------------------------------------------------------
# downstream chain: moments -> parameter-group gradients (torch replica,
# forward validated against the captured frozen state)
# ----------------------------------------------------------------------------

def projection_forward_torch(means_w, quats, scales, viewmat, K, eps2d,
                             quat_xyzw=False, width=None, height=None,
                             viewmat_w2c=True):
    """Exact replica of the frozen persp projection (pinhole), including the
    frozen persp_proj's tan-FOV clamp: the Jacobian's third column is built
    from the CLAMPED tx = z*clamp(x/z, -lim_neg, lim_pos), ty likewise,
    while mean2d uses the unclamped x/y (Utils.cuh persp_proj). means_w
    [M,3] world; quats [M,4] NORMALIZED (w-first if not quat_xyzw); scales
    [M,3] activated. viewmat [4,4]: world-to-camera if viewmat_w2c (the
    gsplat convention: p_c = R @ p_w + t), else camera-to-world.
    K [3,3]; width/height required for the clamp limits.
    Returns means2d [M,2], conic [M,3] (= inverse(covar2d + eps2d*I))."""
    means2d, conic, _ = _projection_graph(
        means_w, quats, scales, viewmat, K, eps2d, quat_xyzw=quat_xyzw,
        width=width, height=height, viewmat_w2c=viewmat_w2c)
    return means2d, conic


def _projection_graph(means_w, quats, scales, viewmat, K, eps2d,
                      quat_xyzw=False, width=None, height=None,
                      viewmat_w2c=True):
    """Core differentiable graph of the frozen projection. Also returns the
    PRE-blur covar2d (J covar_c J^T); the frozen backward's inverse VJP
    adjoint is injected at cov2d_pre (blurred = pre + eps2d*I, eps const),
    so autograd through this graph reproduces the frozen persp_proj_vjp
    exactly, including the full J dependence and the clamp branch routing."""
    import torch
    assert width is not None and height is not None, \
        "width/height required for the frozen tan-FOV clamp"
    if viewmat_w2c:
        R_w2c = viewmat[:3, :3]
        t_w2c = viewmat[:3, 3]
        p_c = means_w @ R_w2c.t() + t_w2c[None, :]
    else:
        R_c2w = viewmat[:3, :3]
        t_c2w = viewmat[:3, 3]
        R_w2c = R_c2w.t()
        p_c = (means_w - t_c2w[None, :]) @ R_w2c.t()
    x_c, y_c, z_c = p_c[:, 0], p_c[:, 1], p_c[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zc = z_c.clamp(min=1e-9)
    means2d = torch.stack([fx * x_c / zc + cx, fy * y_c / zc + cy], dim=1)
    # covariance from quaternion + scale
    if quat_xyzw:
        xq, yq, zq, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    else:
        w, xq, yq, zq = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    Rq = torch.stack([
        1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - w * zq), 2 * (xq * zq + w * yq),
        2 * (xq * yq + w * zq), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - w * xq),
        2 * (xq * zq - w * yq), 2 * (yq * zq + w * xq), 1 - 2 * (xq * xq + yq * yq),
    ], dim=1).reshape(-1, 3, 3)                                # [M,3,3] rows
    # gsplat: covar = R S S^T R^T with S = diag(scales) -> diag(s^2),
    # NOT the rank-1 outer product s s^T (Utils.cuh quat_scale_to_covar_preci)
    SS = torch.diag_embed(scales * scales)                     # [M,3,3] diag(s^2)
    covar = Rq @ SS @ Rq.transpose(1, 2)                       # [M,3,3]
    covar_c = R_w2c[None] @ covar @ R_w2c.t()[None]            # [M,3,3]
    # frozen persp Jacobian: built from the CLAMPED tx/ty (tan-FOV clamp)
    tan_fovx = 0.5 * float(width) / fx
    tan_fovy = 0.5 * float(height) / fy
    lim_x_pos = (float(width) - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (float(height) - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    rz = 1.0 / zc
    rz2 = rz * rz
    tx = zc * (x_c * rz).clamp(min=-lim_x_neg, max=lim_x_pos)
    ty = zc * (y_c * rz).clamp(min=-lim_y_neg, max=lim_y_pos)
    O = torch.zeros_like(rz)
    J = torch.stack([
        fx * rz, O, -fx * tx * rz2,
        O, fy * rz, -fy * ty * rz2,
    ], dim=1).reshape(-1, 2, 3)                                # [M,2,3]
    cov2d_pre = J @ covar_c @ J.transpose(1, 2)                # [M,2,2]
    cov2d = cov2d_pre + eps2d * torch.eye(2, device=cov2d_pre.device,
                                          dtype=cov2d_pre.dtype)[None]
    c00 = cov2d[:, 0, 0]
    c01 = cov2d[:, 0, 1]
    c11 = cov2d[:, 1, 1]
    det = c00 * c11 - c01 * c01
    det = det.clamp(min=1e-12)
    conic = torch.stack([c11 / det, -c01 / det, c00 / det], dim=1)  # [M,3]
    return means2d, conic, cov2d_pre

def sh_dirs(means_w, viewmat, sign=+1.0, viewmat_w2c=True):
    """UNNORMALIZED view direction for the SH evaluation, matching the frozen
    kernel's dir = means - cam_pos (GatherlessProjectedProducer.cu /
    higs_sh_vjp_grid_kernel). The frozen spherical_harmonics op normalizes
    internally, and its backward with compute_v_dirs=True returns the
    adjoint w.r.t. this RAW input (the .cu's v_dir chain, added to
    grad_means). Camera center in world coords: for w2c viewmats (gsplat
    convention) C = -R^T t; for c2w viewmats C = t."""
    import torch
    if viewmat_w2c:
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]
        cam_center = -R.t() @ t
    else:
        cam_center = viewmat[:3, 3]
    return sign * (cam_center[None, :] - means_w)

def determine_conventions(cap, image_idx, m, rows, sh_degree=3):
    """Settle the replica's convention flags EMPIRICALLY against the captured
    frozen state on `rows` (a sample of compact rows): viewmat orientation
    (w2c vs c2w), quaternion component order, SH direction sign. Returns a
    conventions dict; the choice is a property of the frozen renderer,
    recorded in the identity JSON."""
    import torch
    # `rows` are COMPACT visible-row indices; the captured master params are
    # master-indexed -> map through visible_ids.
    master = cap["visible_ids"][rows.to(torch.long)]
    means_w = cap["means"][master].detach()
    quats_n = cap["quats"][master].detach()
    scales_a = cap["scales"][master].detach()
    viewmat = cap["viewmats"][0, 0]
    K = cap["Ks"][0, 0]
    shs_vis = cap["colors"][master].detach()
    ce_cap = cap["colors_eval_f"].reshape(cap["C"], -1, 3)[image_idx, rows]
    m2d_cap = cap["means2d_f"].reshape(cap["C"], -1, 2)[image_idx, rows]
    con_cap = cap["conics_f"].reshape(cap["C"], -1, 3)[image_idx, rows]
    best = None
    for viewmat_w2c in (True, False):
        for quat_xyzw in (False, True):
            m2d, conic = projection_forward_torch(
                means_w, quats_n, scales_a, viewmat, K, cap["eps2d"],
                quat_xyzw=quat_xyzw, width=cap["width"],
                height=cap["height"], viewmat_w2c=viewmat_w2c)
            m2d_e = float((m2d - m2d_cap).abs().max())
            con_e = float((conic - con_cap).abs().max())
            e_proj = m2d_e + con_e
            for sign in (+1.0, -1.0):
                dirs = sh_dirs(means_w, viewmat, sign=sign,
                               viewmat_w2c=viewmat_w2c)
                # frozen build schema: spherical_harmonics(degrees, dirs,
                # coeffs, masks); masks=None evaluates all degrees
                sph = torch.ops.gsplat.spherical_harmonics(
                    sh_degree, dirs, shs_vis, None)
                ce = torch.clamp_min(sph + 0.5, 0.0)
                e_sh = float((ce - ce_cap).abs().max())
                score = e_proj + e_sh
                if not (score == score):       # NaN guard
                    score = float("inf")
                if best is None or score < best["score"]:
                    best = {"quat_xyzw": quat_xyzw, "dirs_sign": sign,
                            "viewmat_w2c": viewmat_w2c,
                            "score": score, "proj_max_abs_err": e_proj,
                            "sh_max_abs_err": e_sh,
                            "means2d_max_abs_diff": m2d_e,
                            "conic_max_abs_diff": con_e,
                            "colors_eval_max_abs_diff": e_sh}
    return best

def downstream_vjp(cap, image_idx, rows, v_rgb_rows, v_opac_rows, S_rows,
                   m, conventions, sh_degree=3):
    """Map per-compact-row intermediate adjoints of the touched `rows` through
    the frozen downstream chain to RAW parameter gradients for those rows.

    rows: LongTensor of compact visible rows (unique).
    v_rgb_rows [n,3], v_opac_rows [n], S_rows [n,5] (Sx,Sy,Sxx,Sxy,Syy).
    conventions: dict from determine_conventions (quat_xyzw, dirs_sign).
    Returns dict of per-row raw grads:
      xyz [n,3], rotation [n,4], scaling [n,3], opacity [n],
      shs [n,16,3] (bands above active degree zero),
    plus validation diagnostics (forward replica vs captured)."""
    import torch
    # `rows` are COMPACT visible-row indices; captured master params are
    # master-indexed -> map through visible_ids.
    master = cap["visible_ids"][rows.to(torch.long)]
    means_w = cap["means"][master].detach().clone().requires_grad_(True)
    quats_n = cap["quats"][master].detach().clone().requires_grad_(True)
    scales_a = cap["scales"][master].detach().clone().requires_grad_(True)
    viewmat = cap["viewmats"][0, 0]                            # [4,4] c2w
    K = cap["Ks"][0, 0]
    eps2d = cap["eps2d"]

    means2d_rep, conic_rep, cov2d_pre = _projection_graph(
        means_w, quats_n, scales_a, viewmat, K, eps2d,
        quat_xyzw=conventions["quat_xyzw"],
        width=cap["width"], height=cap["height"],
        viewmat_w2c=conventions.get("viewmat_w2c", True))
    m2d_cap = cap["means2d_f"].reshape(cap["C"], -1, 2)[image_idx, rows]
    con_cap = cap["conics_f"].reshape(cap["C"], -1, 3)[image_idx, rows]
    ce_cap = cap["colors_eval_f"].reshape(cap["C"], -1, 3)[image_idx, rows]
    diag = {
        "means2d_max_abs_diff": float((means2d_rep - m2d_cap).abs().max()),
        "conic_max_abs_diff": float((conic_rep - con_cap).abs().max()),
    }

    # H8-MR contraction with the CAPTURED conic (frozen formulas,
    # HigsNativeBackward.cu): v_means2d, v_conic=(.5Sxx, Sxy, .5Syy)
    Sx, Sy, Sxx, Sxy, Syy = S_rows.unbind(dim=1)
    ax, ay, az = con_cap[:, 0], con_cap[:, 1], con_cap[:, 2]
    v_m2d = torch.stack([ax * Sx + ay * Sy, ay * Sx + az * Sy], dim=1)
    # frozen inverse_vjp: v_cov2d(blurred) = -A v_A A with A = CAPTURED
    # conic matrix [[ax, ay], [ay, az]], v_A = [[.5Sxx, .5Sxy], [.5Sxy,
    # .5Syy]] (the .cu's symmetrized matrix adjoint of (A00, A01, A11)).
    # Injecting this adjoint at cov2d_pre (blurred = pre + eps2d*I, eps
    # const) makes autograd through the replica graph reproduce the frozen
    # persp_proj_vjp exactly, including the full J dependence on the mean
    # and the tan-FOV clamp branch routing (Utils.cuh persp_proj_vjp).
    A = torch.stack([ax, ay, ay, az], dim=1).reshape(-1, 2, 2)
    VA = torch.stack([0.5 * Sxx, 0.5 * Sxy,
                      0.5 * Sxy, 0.5 * Syy], dim=1).reshape(-1, 2, 2)
    v_cov2d = -(A @ VA @ A)                                    # [n,2,2]
    torch.autograd.backward([means2d_rep, cov2d_pre],
                            grad_tensors=[v_m2d, v_cov2d])
    v_means_proj = means_w.grad.detach().clone()
    v_quats_n = quats_n.grad.detach().clone()
    v_scales_a = scales_a.grad.detach().clone()

    # chains to raw parameters
    rot_raw = m["rotation"].detach()[master]                   # [n,4] raw
    rn = rot_raw.norm(dim=1, keepdim=True).clamp(min=1e-12)
    qhat = rot_raw / rn
    # normalize VJP: v_raw = (v_q - qhat*(qhat.v_q)) / rn
    v_rot = (v_quats_n - qhat * (qhat * v_quats_n).sum(
        dim=1, keepdim=True)) / rn
    sig = torch.sigmoid(m["opacity"].detach()[master])
    v_opacity = v_opac_rows * sig * (1.0 - sig)
    v_scaling = v_scales_a * torch.exp(m["scaling"].detach()[master])

    # SH path: v_rgb_rows is the colors_eval adjoint; chain the frozen
    # clamp_min(sph+0.5, 0) then the frozen gsplat SH VJP op. The op's
    # backward with compute_v_dirs=True returns the adjoint w.r.t. the
    # UNNORMALIZED input dir (it normalizes internally), which is exactly
    # the .cu higs_sh_coeffs_to_color_fast_vjp v_dir chain that the SH grid
    # kernel atomically adds into grad_means.
    # frozen build schema: spherical_harmonics_bwd(degrees, dirs, coeffs,
    # masks, v_colors, compute_v_dirs) -> (v_coeffs, v_dirs?)
    shs_vis = cap["colors"][master].detach()                   # [n,16,3]
    dirs_raw = sh_dirs(means_w.detach(), viewmat,
                       sign=conventions["dirs_sign"],
                       viewmat_w2c=conventions.get("viewmat_w2c", True))
    sph = torch.ops.gsplat.spherical_harmonics(
        sh_degree, dirs_raw, shs_vis, None)
    ce_rep = torch.clamp_min(sph + 0.5, 0.0)
    diag["colors_eval_max_abs_diff"] = float((ce_rep - ce_cap).abs().max())
    v_sph = v_rgb_rows * (ce_cap > 0).float()
    v_shs_out = torch.ops.gsplat.spherical_harmonics_bwd(
        sh_degree, dirs_raw, shs_vis, None, v_sph.contiguous(), True)
    v_shs_vis, v_dirs = v_shs_out[0], v_shs_out[1]
    v_shs_vis = v_shs_vis.reshape(rows.numel(), 16, 3)
    assert v_dirs is not None, "compute_v_dirs=True returned None"
    v_means = v_means_proj + v_dirs.detach().reshape(-1, 3)

    return {
        "xyz": v_means, "rotation": v_rot, "scaling": v_scaling,
        "opacity": v_opacity, "shs": v_shs_vis,
        "master_rows": master, "diag": diag,
    }

# ----------------------------------------------------------------------------
# fixed polynomial basis (16x16 grid, total-degree ordered, orthonormalized)
# ----------------------------------------------------------------------------

_BASIS_CACHE = {}

def poly_basis(r):
    """Fixed orthonormalized 2D polynomial basis on the 16x16 tile grid.
    Coordinates: u = ((x+0.5)/16)*2-1, v = ((y+0.5)/16)*2-1 for x,y in 0..15
    (x = tile column, y = tile row; pixel layout row-major y*16+x).
    Monomials u^a v^b ordered by total degree d=a+b, then lexicographic (a,b).
    Orthonormalized (numpy float64 QR) over the 256 grid points; truncated to
    the first r functions. Returns [r, 256] float32 (rows = basis functions).
    The SAME basis/order is used for all scenes, views and buckets."""
    if r in _BASIS_CACHE:
        return _BASIS_CACHE[r]
    xs = np.arange(16, dtype=np.float64)
    u = ((xs + 0.5) / 16.0) * 2.0 - 1.0
    # pixel index p = y*16 + x (row-major): u depends on x=p%16,
    # v depends on y=p//16
    ucol = np.tile(u, 16)        # p -> u(x)
    vcol = np.repeat(u, 16)      # p -> v(y)
    monos = []
    for d in range(0, 40):
        for a in range(d, -1, -1):
            b = d - a
            monos.append((a, b))
        if len(monos) >= 36:
            break
    Mx = np.stack([ucol ** a * vcol ** b for (a, b) in monos], axis=1)
    Q, _ = np.linalg.qr(Mx)                     # [256, K] orthonormal cols
    B = np.ascontiguousarray(Q[:, :r].T)        # [r, 256]
    _BASIS_CACHE[r] = B.astype(np.float32)
    return _BASIS_CACHE[r]

def basis_meta():
    monos = []
    for d in range(0, 40):
        for a in range(d, -1, -1):
            monos.append([a, d - a])
        if len(monos) >= 36:
            break
    return {
        "grid": "16x16 tile pixels, row-major p = y*16 + x",
        "coords": "u = ((x+0.5)/16)*2-1, v = ((y+0.5)/16)*2-1",
        "monomial_order": "total degree d, then lexicographic (a,b) with "
                          "u^a v^b, a descending within degree",
        "orthonormalization": "numpy float64 QR over the 256 grid points",
        "truncation": "first r rows, r in {4, 8, 16, 32}",
        "first_8_monomials": monos[:8],
        "n_monomials_built": len(monos),
        "note": "fixed (not learned); identical for all scenes/views/buckets",
    }

def project_field(Q, B):
    """Q [..., 256] dense field -> B^T (B Q) rank-r projection (same shape).
    B [r,256] orthonormal rows (numpy float32 or torch tensor)."""
    import torch
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B)
    B = B.to(device=Q.device, dtype=Q.dtype)
    shape = Q.shape
    Qf = Q.reshape(-1, 256)
    coeff = Qf @ B.t()                                          # [N, r]
    Qh = coeff @ B                                              # [N, 256]
    return Qh.reshape(shape)

# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def merge_rows(rows_list, vals_list):
    """Concatenate (rows, vals) parts and merge duplicate master rows by
    summation (a master gaussian touched in several cameras accumulates).
    rows: LongTensor [n]; vals: [n, ...] any trailing group shape.
    Returns (unique_sorted_rows [u], merged_vals [u, ...])."""
    import torch
    rows = torch.cat(rows_list)
    vals = torch.cat(vals_list)
    ru, inv = torch.unique(rows, return_inverse=True)
    merged = torch.zeros((ru.numel(),) + vals.shape[1:],
                         dtype=vals.dtype, device=vals.device)
    merged.view(ru.numel(), -1).index_add_(
        0, inv, vals.reshape(vals.shape[0], -1))
    return ru, merged

def group_metrics(g_exact, g_recon):
    """Per-group (flattened) cosine, relative L2, norm ratio, max abs err."""
    import torch
    ge = g_exact.reshape(-1).double()
    gr = g_recon.reshape(-1).double()
    dot = (ge * gr).sum()
    ne = ge.norm()
    nr = gr.norm()
    cos = float(dot / (ne * nr).clamp(min=1e-30))
    rel = float((ge - gr).norm() / ne.clamp(min=1e-30))
    return {
        "cosine": cos,
        "relative_l2": rel,
        "norm_ratio": float(nr / ne.clamp(min=1e-30)),
        "max_abs_err": float((ge - gr).abs().max()),
        "n_coords": int(ge.numel()),
    }

def probe_score(dg, native_g, groups):
    """Per-group cosine of per-row replica grads vs single-camera native
    grads gathered at the replica's master rows (chain-variant probe)."""
    import torch
    out = {}
    for g in groups:
        v = dg[g].reshape(-1).double()
        n = native_g[g][dg["master_rows"]].reshape(-1).double()
        out[g] = float((v * n).sum() / (v.norm() * n.norm()).clamp(min=1e-30))
    out["mean_cosine"] = float(sum(out[g] for g in groups) / len(groups))
    return out

def adam_precond(vals, rows, opt_state_group, eps=1e-15, beta2=0.999):
    """Adam-preconditioned direction vals / (sqrt(v_hat) + eps) at the FROZEN
    optimizer state, gathered per touched row. vals [n, ...], rows [n] master
    row ids; exp_avg_sq is gathered with the same indexing. Bias correction
    uses the frozen step+1 (a global scalar per group; cancels in cosines,
    kept for exactness)."""
    import torch
    step = opt_state_group["step"] + 1
    v = opt_state_group["exp_avg_sq"][rows]                  # [n, ...]
    v_hat = v / (1.0 - beta2 ** step)
    return vals / (v_hat.sqrt() + eps)

def json_default(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
