"""Repeatable HiGS training-path benchmark (CUDA).

Compares, on the same scene + cameras + optimizer schedule:

- ``std``            : standard gsplat fused rasterization (forward + backward),
                       no culling. Reference "standard gsplat training path".
- ``higs_recompute`` : HiGS forward with the explicit standard-gsplat
                       recomputation fallback (metadata
                       ``backward_backend="gsplat_recompute"``).
- ``higs_native``    : HiGS forward + HiGS native CUDA backward
                       (``backward_backend="higs_native"``), frozen topology.
- ``higs_dynamic``   : HiGS native path with densify/prune + Adam-state sync.

Per backend reports: forward latency, backward latency, total iteration
latency, full training-step latency (including the optimizer step), peak VRAM,
culling ratio, and final PSNR / SSIM / LPIPS on held-out cameras. A speedup is
only claimed when the measured total iteration time actually wins.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

_SH_DEGREE = 3
_K = (_SH_DEGREE + 1) ** 2  # 16
_TILE_SIZE = 16  # HiGS macro-tile edge (px); must match the render tile_size.


# --------------------------------------------------------------------------
# Scene loading
# --------------------------------------------------------------------------

def load_ply_scene(ply_path, device):
    """Load a 3DGS PLY -> (means, quats, scales, opacities, sh) FP32 masters."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    N = len(v)
    means = torch.tensor(
        np.column_stack([v["x"], v["y"], v["z"]]),
        dtype=torch.float32, device=device,
    )
    quats = torch.tensor(
        np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]),
        dtype=torch.float32, device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(
        np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]),
        dtype=torch.float32, device=device,
    ))
    opacities = torch.sigmoid(
        torch.tensor(v["opacity"], dtype=torch.float32, device=device)
    )
    f_dc = torch.tensor(
        np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]),
        dtype=torch.float32, device=device,
    )
    # PLY stores 3*(K-1) rest scalars in channel-major order (RGB blocks
    # of K-1 coefficients each); reshape to the [N, K-1, 3] gsplat layout.
    f_rest = torch.stack(
        [torch.tensor(v[f"f_rest_{i}"], dtype=torch.float32, device=device)
         for i in range(3 * (_K - 1))],
        dim=1,
    )  # [N, 3*(K-1)]
    f_rest = f_rest.reshape(N, 3, _K - 1).permute(0, 2, 1)  # [N, K-1, 3]
    sh = torch.zeros(N, _K, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh


def load_cameras(scene_dir, width, height, n_train, n_eval, device):
    import json

    with open(os.path.join(scene_dir, "eval_cameras.json")) as f:
        cams = json.load(f)
    viewmats, Ks = [], []
    for c in cams:
        R = np.asarray(c["rotation"], dtype=np.float64)  # c2w rotation
        p = np.asarray(c["position"], dtype=np.float64)
        Rw2c = R.T
        vm = np.eye(4)
        vm[:3, :3] = Rw2c
        vm[:3, 3] = -Rw2c @ p
        scale = width / float(c["width"])
        K = np.array(
            [[float(c["fx"]) * scale, 0.0, (width - 1) / 2.0],
             [0.0, float(c["fy"]) * scale, (height - 1) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        viewmats.append(torch.tensor(vm, dtype=torch.float32, device=device))
        Ks.append(torch.tensor(K, dtype=torch.float32, device=device))
    n_cams = len(cams)
    train_idx = list(range(min(n_train, n_cams)))
    eval_idx = list(range(min(n_train, n_cams), min(n_train + n_eval, n_cams)))
    return torch.stack(viewmats).unsqueeze(0), torch.stack(Ks).unsqueeze(0), train_idx, eval_idx


def load_reference(gt_dir, cams, width, height, device):
    """Load GT photos for the given cameras.

    Applies the canonical ``reference_crop`` (if present) and resizes to the
    benchmark resolution, mirroring the repo's official metric conversion.
    """
    imgs = []
    for c in cams:
        img_name = c["img_name"]
        p = next(
            (
                os.path.join(gt_dir, img_name + ext)
                for ext in (".JPG", ".jpg", ".png", ".jpeg", ".PNG")
                if os.path.exists(os.path.join(gt_dir, img_name + ext))
            ),
            None,
        )
        if p is None:
            raise FileNotFoundError(f"{img_name} not found in {gt_dir}")
        im = Image.open(p).convert("RGB")
        crop = c.get("reference_crop")
        if crop is not None:
            im = im.crop((crop[0], crop[1], crop[2], crop[3]))
        im = im.resize((width, height), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        imgs.append(torch.tensor(arr, dtype=torch.float32, device=device))
    return torch.stack(imgs)  # [C, H, W, 3]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _gauss_win(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y):
    """x, y: [B, C, H, W] in [0, 1]. Mean SSIM (gaussian window, per channel)."""
    win = _gauss_win(11, 1.5, x.device).to(x.dtype)
    win = win.expand(x.shape[1], 1, -1, -1)
    pad = 11 // 2
    groups = x.shape[1]
    mu_x = F.conv2d(x, win, padding=pad, groups=groups)
    mu_y = F.conv2d(y, win, padding=pad, groups=groups)
    sx2 = F.conv2d(x * x, win, padding=pad, groups=groups) - mu_x ** 2
    sy2 = F.conv2d(y * y, win, padding=pad, groups=groups) - mu_y ** 2
    sxy = F.conv2d(x * y, win, padding=pad, groups=groups) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx2 + sy2 + c2)
    return num.div(den).clamp(0, 1).mean().item()


def psnr(x, y):
    mse = (x - y).pow(2).mean().item()
    if mse < 1e-12:
        return 60.0
    return -10.0 * np.log10(mse)


def lpips_score(lpips_model, x, y):
    """x, y: [C, H, W, 3] in [0, 1] -> [C, 3, H, W] in [-1, 1]."""
    xa = (x.permute(0, 3, 1, 2) * 2 - 1).contiguous()
    ya = (y.permute(0, 3, 1, 2) * 2 - 1).contiguous()
    with torch.no_grad():
        return lpips_model(xa, ya).mean().item()


# --------------------------------------------------------------------------
# Forward helpers
# --------------------------------------------------------------------------

def make_optimizer(params, lr_scale=1.0, fused=True):
    """Build the 5-group Adam optimizer used by every backend.

    ``fused=True`` (default when every param is a CUDA tensor) runs each param
    group's whole update in a single kernel, about 2x faster than the foreach
    path on large scenes (6.8 vs 14.9 ms/step on the 6.13M-Gaussian bicycle
    scene); a non-fused fallback keeps CPU / unsupported platforms working.
    """
    means, quats, scales, opacities, sh = params
    groups = [
        {"params": [means], "lr": 1.6e-4 * lr_scale},
        {"params": [quats], "lr": 1e-3 * lr_scale},
        {"params": [scales], "lr": 5e-3 * lr_scale},
        {"params": [opacities], "lr": 5e-2 * lr_scale},
        {"params": [sh], "lr": 2.5e-3 * lr_scale},
    ]
    if fused and all(p.is_cuda for g in groups for p in g["params"]):
        try:
            return torch.optim.Adam(groups, fused=True)
        except (TypeError, ValueError, RuntimeError):
            pass
    return torch.optim.Adam(groups)


def _l1_loss(frame, ref):
    return (frame - ref).abs().mean()


def _masked_l1_loss(frame, ref, tile_mask, tile_size, width, height):
    """Unbiased sampled-tile L1: mean |diff| over the sampled tiles only.

    Uniform fixed-count tile sampling makes the sampled-tile mean an unbiased
    estimator of the full-frame mean, so no 1/r rescale is needed here.
    ``frame`` is [1, C, H, W, 3], ``ref`` is [C, H, W, 3], ``tile_mask`` is
    [C, th, tw] bool.
    """
    diff = (frame.squeeze(0) - ref).abs()  # [C, H, W, 3]
    pm = (
        tile_mask.repeat_interleave(tile_size, dim=1)
        .repeat_interleave(tile_size, dim=2)[:, :height, :width]
        .unsqueeze(-1)
    )  # [C, H, W, 1]
    return diff.masked_select(pm.expand_as(diff)).mean()


def _tile_mean_errors(frame, ref, tile_size):
    """Exact per-tile mean |diff| (border tiles counted by real pixels only).

    ``frame`` is [1, C, H, W, 3], ``ref`` is [C, H, W, 3]. Returns [C, th, tw].
    Zero padding only pads the sum (zeros add nothing), so per-tile sums are
    exact; counts are computed from the true tile extents.
    """
    diff = (frame.squeeze(0) - ref).abs()  # [C, H, W, 3]
    C, H, W, _ = diff.shape
    th = (H + tile_size - 1) // tile_size
    tw = (W + tile_size - 1) // tile_size
    Hp, Wp = th * tile_size, tw * tile_size
    pad = (0, 0, 0, Wp - W, 0, Hp - H)
    diff_pad = torch.nn.functional.pad(diff, pad)
    tile_sum = diff_pad.reshape(C, th, tile_size, tw, tile_size, 3).sum(dim=(2, 4, 5))
    rows = torch.full((th,), tile_size, dtype=torch.long, device=diff.device)
    cols = torch.full((tw,), tile_size, dtype=torch.long, device=diff.device)
    rows[-1] = H - (th - 1) * tile_size
    cols[-1] = W - (tw - 1) * tile_size
    count = (rows[:, None] * cols[None, :] * 3).to(diff.dtype)  # [th, tw]
    return tile_sum / count  # [C, th, tw]


def _importance_l1_loss(frame, ref, mask, weights, tile_size, width, height):
    """Exact unbiased tile-level importance estimate of the full-frame L1.

    Sampled tiles are drawn iid (with replacement) with probabilities ``p``;
    ``mask`` is the set of tiles hit and ``weights = m / (k * p)`` where ``m``
    is the per-tile draw count. The estimator
    ``(1/P) sum_t mask_t * w_t * S_t`` (S_t = per-tile |diff| pixel sum,
    P = total pixels) is unbiased for the full-frame mean regardless of ``p``.
    """
    diff = (frame.squeeze(0) - ref).abs()  # [C, H, W, 3]
    C, H, W, _ = diff.shape
    th = (H + tile_size - 1) // tile_size
    tw = (W + tile_size - 1) // tile_size
    Hp, Wp = th * tile_size, tw * tile_size
    diff_pad = torch.nn.functional.pad(diff, (0, 0, 0, Wp - W, 0, Hp - H))
    tile_sum = diff_pad.reshape(C, th, tile_size, tw, tile_size, 3).sum(dim=(2, 4, 5))
    p_total = C * width * height * 3
    m = mask.reshape(C, th, tw).to(diff.dtype)
    w = weights.reshape(C, th, tw)
    return (m * w * tile_sum).sum() / p_total


def _error_guided_mask(tile_err, ratio, alpha, device, lambda_mix=1.0):
    """Importance-sample ``k`` tiles per image with p proportional to error.

    ``tile_err`` is [C, th, tw] (per-tile mean |diff| from the last refresh).
    ``lambda_mix`` in [0, 1] blends the error distribution with the uniform
    distribution (``p = (1 - lambda_mix) / n + lambda_mix * p_err``); 0.0 is
    exactly uniform, 1.0 is pure error-guided (default).
    Returns ``(mask [C, n_tiles] bool, weights [C, n_tiles] float)`` with
    ``weights = m / (k * p)`` (with-replacement multinomial draws), which makes
    the sampled estimator unbiased for any p > 0.
    """
    C, th, tw = tile_err.shape
    n = th * tw
    k = max(1, int(round(n * ratio)))
    e = tile_err.reshape(C, n)
    floor = (1e-3 * e.mean(dim=1, keepdim=True)).clamp_min(1e-6)
    p = (e + floor) ** alpha
    p = p / p.sum(dim=1, keepdim=True)
    if 0.0 <= lambda_mix < 1.0:
        p = (1.0 - lambda_mix) / n + lambda_mix * p
    idx = torch.multinomial(p, k, replacement=True)  # [C, k]
    m = torch.zeros(C, n, dtype=torch.float32, device=e.device)
    m.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    mask = m > 0
    weights = m / (k * p)
    return mask, weights


def _std_ll_forward(means, quats, scales, opacities, colors, viewmats, Ks,
                   width, height, sh_degree, radius_clip=0.0):
    """Low-level standard gsplat forward (raw CUDA kernels, no culling).

    Mirrors the kernels that ``rasterize_gaussian_higs_*`` use for the capture
    path, so the ``std_ll`` backend is the apples-to-apples baseline: the
    high-level ``rasterization()`` wrapper used by ``std`` carries ~9 ms/step of
    Python/alloc overhead at 1920x1080 x 4 cameras, which would otherwise
    inflate the reported HiGS margin.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_tiles,
        isect_offset_encode,
        _make_lazy_cuda_func,
    )
    from gsplat.rendering import _maybe_evaluate_sh

    C = viewmats.shape[-3]
    N = means.shape[-2]
    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=means.contiguous(),
        covars=None,
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=radius_clip,
        packed=False,
        calc_compensations=False,
        camera_model="pinhole",
    )
    opacities_bc = torch.broadcast_to(
        opacities[..., None, :], (1, C, N)
    ).contiguous()
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height,
        packed=False, n_images=C, image_ids=None, gaussian_ids=None,
        conics=conics, opacities=opacities_bc,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, C, tile_width, tile_height
    ).reshape((1, C, tile_height, tile_width))
    colors_eval = _maybe_evaluate_sh(
        sh_degree, colors, means, radii, viewmats, (1,), C, N, True,
    ).contiguous()
    bg_kernel = torch.zeros((1, C, 3), device=means.device)
    render_colors, render_alphas, _absgrad, last_ids = (
        _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(),
            conics.contiguous(),
            colors_eval.contiguous(),
            opacities_bc.contiguous(),
            bg_kernel,
            None,
            width,
            height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            False,
            False,
        )
    )
    return render_colors, render_alphas


def make_forward_fn(backend, width, height, handle, viewmats, Ks,
                     radius_clip=0.0, tile_sampling_ratio=1.0,
                     sampling_mode="uniform"):
    from gsplat.rendering import rasterization

    # "error_guided" is a harness-level strategy: the harness computes an
    # explicit tile_mask (with importance weights) and passes it in; the
    # rasterizer itself only knows uniform/stratified internal sampling.
    raster_sampling_mode = "uniform" if sampling_mode == "error_guided" else sampling_mode

    def forward_fn(params_in, cam_ids, sampling_ratio=None, tile_mask=None):
        m, q, s, o, c = params_in
        vm = viewmats[:, cam_ids]
        K = Ks[:, cam_ids]
        ratio = tile_sampling_ratio if sampling_ratio is None else sampling_ratio
        if backend == "std":
            out = rasterization(
                means=m.unsqueeze(0), quats=q.unsqueeze(0),
                scales=s.unsqueeze(0), opacities=o.unsqueeze(0), colors=c,
                viewmats=vm, Ks=K, width=width, height=height,
                sh_degree=_SH_DEGREE, packed=True, radius_clip=radius_clip,
            )
            return out[0], out[1], {}
        if backend == "std_ll":
            rc, ra = _std_ll_forward(
                m.unsqueeze(0), q.unsqueeze(0), s.unsqueeze(0),
                o.unsqueeze(0), c, vm, K, width, height,
                _SH_DEGREE, radius_clip=radius_clip,
            )
            return rc, ra, {}
        kw = dict(
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=_SH_DEGREE, use_higs_culling=True, radius_clip=radius_clip,
        )
        if backend in ("higs_native", "higs_recompute", "higs_native_ts"):
            from gsplat.experimental import rasterize_gaussian_higs_frozen
            if backend == "higs_recompute":
                mode, ratio = "gsplat_recompute", 1.0
            else:
                mode = "higs_native"
                if backend == "higs_native":
                    ratio = 1.0  # full-resolution baseline backend
            res = rasterize_gaussian_higs_frozen(
                m, q, s, o, c, backward_mode=mode, scene=handle,
                freeze_topology=True, tile_sampling_ratio=ratio,
                sampling_mode=raster_sampling_mode, tile_mask=tile_mask, **kw,
            )
            return res["frame"], res["alpha"], res["metadata"]
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        dyn_ratio = ratio if backend == "higs_dynamic_ts" else 1.0
        res = rasterize_gaussian_higs_dynamic(
            m, q, s, o, c, backward_mode="higs_native",
            tile_sampling_ratio=dyn_ratio,
            sampling_mode=raster_sampling_mode, tile_mask=tile_mask, **kw,
        )
        return res["frame"], res["alpha"], res["metadata"]

    return forward_fn


def probe_native_vs_recompute(
    params0, viewmats, Ks, cam_idx, ref, width, height, device,
):
    """One forward+backward each with higs_native and gsplat_recompute on
    identical inputs; returns (grad cosine sim mean, forward parity PSNR)."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        _HIGS_FROZEN_TRACKER,
        create_higs_renderer,
    )

    torch.manual_seed(1234)
    _HIGS_FROZEN_TRACKER.reset()
    params = [t.detach().clone().requires_grad_(True) for t in params0]
    handle = create_higs_renderer(
        params[0], params[1], params[2], params[3], params[4],
        sh_degree=_SH_DEGREE,
    )
    try:
        grads = {}
        parity_psnr = None
        for mode in ("higs_native", "gsplat_recompute"):
            for t in params:
                t.grad = None
            res = rasterize_gaussian_higs_frozen(
                params[0], params[1], params[2], params[3], params[4],
                viewmats=viewmats[:, [cam_idx]], Ks=Ks[:, [cam_idx]],
                width=width, height=height, sh_degree=_SH_DEGREE,
                use_higs_culling=True, backward_mode=mode, scene=handle,
                freeze_topology=True,
            )
            loss = _l1_loss(res["frame"], ref[:1])
            loss.backward()
            torch.cuda.synchronize(device)
            grads[mode] = [t.grad.detach().clone().flatten() for t in params]
            if mode == "higs_native":
                parity_psnr = psnr(res["frame"].reshape(1, height, width, 3), ref[:1])
        sims = []
        for a, b in zip(grads["higs_native"], grads["gsplat_recompute"]):
            if a.numel() == 0 or b.numel() == 0:
                sims.append(1.0)
            else:
                sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return float(np.mean(sims)), float(parity_psnr)
    finally:
        handle.release()


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def _lr_at_step(base_lr, decay, step, steps):
    """Exponential LR schedule: ``base_lr`` at step 0, ``base_lr*decay`` at the
    final step (``step`` is 0-based, ``decay`` in (0, 1], 1.0 = constant)."""
    if decay >= 1.0:
        return base_lr
    return base_lr * (decay ** ((step + 1) / max(1, steps)))


def run_backend(
    backend, params0, viewmats, Ks, train_idx, refs_train,
    eval_idx, refs_eval, width, height, steps, seed, device,
    densify_every, densify_threshold, prune_threshold, lpips_model,
    radius_clip=0.0, fused_adam=True, tile_sampling_ratio=1.0,
    anchor_densify=False, sampling_mode="uniform",
    error_alpha=1.0, error_refresh_every=25, error_lambda=1.0,
    eval_every=0, lr_decay=1.0, densify_window=None,
):
    torch.manual_seed(seed)
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_FROZEN_TRACKER
    _HIGS_FROZEN_TRACKER.reset()
    means, quats, scales, opacities, sh = [t.detach().clone() for t in params0]
    for t in (means, quats, scales, opacities, sh):
        t.requires_grad_(True)
    params = (means, quats, scales, opacities, sh)
    opt = make_optimizer(params, fused=fused_adam)
    _base_lrs = [g["lr"] for g in opt.param_groups]
    _lr_gamma = lr_decay ** (1.0 / max(1, steps)) if lr_decay < 1.0 else 1.0

    handle = None
    dynamic_scene = None
    if backend in ("higs_native", "higs_recompute", "higs_native_ts"):
        from gsplat.experimental.render.functional.gaussian_inference import (
            create_higs_renderer,
        )
        handle = create_higs_renderer(
            means, quats, scales, opacities, sh, sh_degree=_SH_DEGREE,
        )
    elif backend in ("higs_dynamic", "higs_dynamic_ts"):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
        )
        dynamic_scene = _HIGS_DYNAMIC_SCENE
        dynamic_scene.reset()

    forward_fn = make_forward_fn(
        backend, width, height, handle, viewmats, Ks, radius_clip=radius_clip,
        tile_sampling_ratio=tile_sampling_ratio,
        sampling_mode=sampling_mode,
    )
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times, bwd_times, total_times, train_times = [], [], [], []
    culling_ratios, n_visibles, topo_rebuilt, isect_fracs = [], [], [], []
    refresh_times = []
    eval_curve = []
    tile_err_cache = None
    ref = refs_train

    try:
        for it in range(steps):
            cam_ids = train_idx
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            is_densify_step = (
                backend in ("higs_dynamic", "higs_dynamic_ts")
                and densify_every > 0
                and (it + 1) % densify_every == 0
                and (densify_window is None or it < densify_window)
            )
            step_ratio = 1.0 if (anchor_densify and is_densify_step) else tile_sampling_ratio
            eg_mask = eg_weights = None
            if sampling_mode == "error_guided" and step_ratio < 1.0:
                if tile_err_cache is None or (it + 1) % error_refresh_every == 0:
                    with torch.no_grad():
                        r0 = torch.cuda.Event(enable_timing=True)
                        r1 = torch.cuda.Event(enable_timing=True)
                        r0.record()
                        frame_full, _, _ = forward_fn(
                            params, cam_ids, sampling_ratio=1.0,
                        )
                        tile_err_cache = _tile_mean_errors(
                            frame_full, ref, _TILE_SIZE,
                        )
                        r1.record()
                        torch.cuda.synchronize(device)
                        refresh_times.append(r0.elapsed_time(r1))
                eg_mask, eg_weights = _error_guided_mask(
                    tile_err_cache, step_ratio, error_alpha, device,
                    lambda_mix=error_lambda,
                )
                eg_mask = eg_mask.reshape(
                    tile_err_cache.shape[0], tile_err_cache.shape[1],
                    tile_err_cache.shape[2],
                )
            frame, alpha, meta = forward_fn(
                params, cam_ids, sampling_ratio=step_ratio, tile_mask=eg_mask,
            )
            ev1.record()
            torch.cuda.synchronize(device)
            fwd_ms = ev0.elapsed_time(ev1)
            tile_mask = meta.get("tile_mask") if meta else None
            if eg_mask is not None and eg_weights is not None:
                loss = _importance_l1_loss(
                    frame, ref, eg_mask, eg_weights, _TILE_SIZE, width, height,
                )
            elif step_ratio < 1.0 and tile_mask is not None:
                loss = _masked_l1_loss(
                    frame, ref, tile_mask, _TILE_SIZE, width, height,
                )
            else:
                loss = _l1_loss(frame, ref)

            ev2 = torch.cuda.Event(enable_timing=True)
            ev3 = torch.cuda.Event(enable_timing=True)
            ev2.record()
            loss.backward()
            ev3.record()
            torch.cuda.synchronize(device)
            bwd_ms = ev2.elapsed_time(ev3)
            total_ms = (time.perf_counter() - t0) * 1e3

            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            total_times.append(total_ms)
            if meta:
                culling_ratios.append(meta.get("culling_ratio", 0.0))
                n_visibles.append(meta.get("n_visible", 0))
                topo_rebuilt.append(float(meta.get("topology_rebuilt", False)))
                n_isects_full = meta.get("n_isects_full", 0)
                isect_fracs.append(
                    meta.get("n_isects", 0) / n_isects_full
                    if n_isects_full else 1.0
                )
            else:
                culling_ratios.append(0.0)
                n_visibles.append(means.shape[0])
                topo_rebuilt.append(0.0)

            if lr_decay < 1.0:
                _t = float(it + 1)
                for _g, _b in zip(opt.param_groups, _base_lrs):
                    _g["lr"] = _b * (_lr_gamma ** _t)

            opt.step()

            if (
                backend in ("higs_dynamic", "higs_dynamic_ts")
                and (it + 1) % densify_every == 0
                and (densify_window is None or it < densify_window)
            ):
                from gsplat.experimental.render.functional.gaussian_inference import (
                    _densify_gaussians,
                    _prune_gaussians,
                    sync_optimizer_state_for_topology_change,
                )
                grads = means.grad
                n_old = means.shape[0]
                dup_idx = (
                    grads.norm(dim=-1) > densify_threshold
                ).nonzero().flatten() if grads is not None else torch.tensor([], device=device)
                old_m, old_q, old_s, old_o, old_c = means, quats, scales, opacities, sh
                new_m, new_q, new_s, new_o, new_c = _densify_gaussians(
                    means, quats, scales, opacities, sh,
                    grads, threshold=densify_threshold,
                )
                new_m, new_q, new_s, new_o, new_c = _prune_gaussians(
                    new_m, new_q, new_s, new_o, new_c,
                    opacity_threshold=prune_threshold,
                )
                n_new = new_m.shape[0]
                if n_new != n_old:
                    pre_map = torch.cat([torch.arange(n_old, device=device), dup_idx])
                    keep = (new_o > prune_threshold).nonzero().flatten()
                    old_to_new = pre_map[keep]
                    with torch.no_grad():
                        means, quats, scales, opacities, sh = (
                            new_m.detach(), new_q.detach(),
                            new_s.detach(), new_o.detach(),
                            new_c.detach(),
                        )
                    for _t in (means, quats, scales, opacities, sh):
                        _t.requires_grad_(True)
                    params = (means, quats, scales, opacities, sh)
                    sync_optimizer_state_for_topology_change(
                        opt, old_to_new,
                        means=(old_m, means), quats=(old_q, quats),
                        scales=(old_s, scales), opacities=(old_o, opacities),
                        colors=(old_c, sh),
                    )
                    dynamic_scene.mark_dirty()

            if eval_every > 0 and (it + 1) % eval_every == 0:
                with torch.no_grad():
                    ev_frame, _, _ = forward_fn(
                        params, eval_idx, sampling_ratio=1.0,
                    )
                    ev_frame = ev_frame.reshape(
                        len(eval_idx), height, width, 3,
                    )
                    eval_curve.append({
                        "step": int(it + 1),
                        "psnr": float(psnr(ev_frame, refs_eval)),
                        "ssim": float(ssim(
                            ev_frame.permute(0, 3, 1, 2),
                            refs_eval.permute(0, 3, 1, 2),
                        )),
                        "lpips": float(lpips_score(
                            lpips_model, ev_frame, refs_eval,
                        )),
                        "n_gaussians": int(means.shape[0]),
                    })

            opt.zero_grad(set_to_none=True)
            ev4 = torch.cuda.Event(enable_timing=True)
            ev4.record()
            torch.cuda.synchronize(device)
            train_times.append(ev0.elapsed_time(ev4))

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1e9

        with torch.no_grad():
            ev_frame, _, _ = forward_fn(params, eval_idx, sampling_ratio=1.0)
            ev_frame = ev_frame.reshape(len(eval_idx), height, width, 3)
            p = psnr(ev_frame, refs_eval)
            s = ssim(ev_frame.permute(0, 3, 1, 2), refs_eval.permute(0, 3, 1, 2))
            l = lpips_score(lpips_model, ev_frame, refs_eval)
    finally:
        if handle is not None:
            handle.release()
        if dynamic_scene is not None:
            dynamic_scene.reset()
        _HIGS_FROZEN_TRACKER.reset()

    return {
        "backend": backend,
        "fwd_ms": float(np.mean(fwd_times)),
        "bwd_ms": float(np.mean(bwd_times)),
        "total_ms": float(np.mean(total_times)),
        "train_ms": float(np.mean(train_times)) if train_times else 0.0,
        "peak_vram_gb": peak,
        "culling_ratio": float(np.mean(culling_ratios)) if culling_ratios else 0.0,
        "n_visible_avg": float(np.mean(n_visibles)) if n_visibles else 0.0,
        "psnr": p,
        "ssim": s,
        "lpips": l,
        "final_n": means.shape[0],
        "topology_rebuilt_frac": float(np.mean(topo_rebuilt)) if topo_rebuilt else 0.0,
        "sampled_tile_ratio": float(tile_sampling_ratio),
        "isect_frac": float(np.mean(isect_fracs)) if isect_fracs else 1.0,
        "refresh_ms": float(np.mean(refresh_times)) if refresh_times else 0.0,
        "sampling_mode": sampling_mode,
        "eval_curve": eval_curve,
    }


def main():
    ap = argparse.ArgumentParser(description="HiGS training-path benchmark")
    ap.add_argument(
        "--base-dir",
        default="datasets/processed",
        help="root of processed official datasets (family/scene subdirs)",
    )
    ap.add_argument(
        "--scene",
        nargs="+",
        default=["tanks_and_temples/train", "mipnerf360/bicycle"],
        help="family/scene pairs, e.g. mipnerf360/garden, tanks_and_temples/truck",
    )
    ap.add_argument("--backends", nargs="+", default=["std", "higs_recompute", "higs_native", "higs_dynamic"])
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--densify-every", type=int, default=5)
    ap.add_argument("--densify-threshold", type=float, default=5e-3)
    ap.add_argument("--prune-threshold", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius-clip", type=float, default=0.0)
    ap.add_argument(
        "--anchor-densify",
        action="store_true",
        help="dynamic HiGS: run densify steps at full resolution (r=1.0)",
    )
    ap.add_argument(
        "--tile-sampling-ratio", type=float, default=1.0,
        help="HiGS native tile sampling ratio in (0, 1] (1.0 = full frame)",
    )
    ap.add_argument(
        "--sampling-mode", choices=("uniform", "stratified", "error_guided"), default="uniform",
        help="tile sampling: uniform iid, stratified (one tile per round(1/r)-tile "
        "stratum), or error_guided (importance-sample tiles proportional to the "
        "cached per-tile error, with unbiased importance weights)",
    )
    ap.add_argument(
        "--error-alpha", type=float, default=1.0,
        help="error-guided sampling: p ~ (tile_err + floor)^alpha (1.0 = variance-optimal for L1)",
    )
    ap.add_argument(
        "--error-refresh-every", type=int, default=25,
        help="error-guided sampling: refresh the full-res per-tile error map every N steps",
    )
    ap.add_argument(
        "--error-lambda", type=float, default=1.0,
        help="error-guided sampling: blend error distribution with uniform, "
        "p = (1-lambda)/n + lambda*p_err (1.0 = pure error-guided, 0.0 = uniform)",
    )
    ap.add_argument(
        "--eval-every", type=int, default=0,
        help="record full-res eval PSNR/SSIM/LPIPS every N steps (0 = only final)",
    )
    ap.add_argument(
        "--lr-decay", type=float, default=1.0,
        help="exponential LR decay: final LR factor over `--steps` (1.0 = constant)",
    )
    ap.add_argument(
        "--densify-window", type=int, default=0,
        help="dynamic: run densify/prune only while step < window (0 = whole run)",
    )
    ap.add_argument(
        "--no-fused-adam",
        action="store_false",
        dest="fused_adam",
        help="disable fused Adam (fall back to the foreach optimizer)",
    )
    args = ap.parse_args()

    import lpips

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    all_results = {}
    for scene in args.scene:
        scene_dir = os.path.join(args.base_dir, scene)
        ply_path = os.path.join(scene_dir, "point_cloud.ply")
        if not os.path.exists(ply_path):
            print(f"[skip] no point_cloud.ply in {scene_dir}")
            continue
        gt_dir = os.path.join(scene_dir, "eval_images")

        params0 = load_ply_scene(ply_path, device)
        print(f"scene={scene} n_gaussians={params0[0].shape[0]}")
        viewmats, Ks, train_idx, eval_idx = load_cameras(
            scene_dir, args.width, args.height, args.n_train, args.n_eval, device,
        )
        print(f"train_cams={train_idx} eval_cams={eval_idx} res={args.width}x{args.height}")

        with open(os.path.join(scene_dir, "eval_cameras.json")) as f:
            cams = json.load(f)
        refs_train = load_reference(gt_dir, [cams[i] for i in train_idx], args.width, args.height, device)
        refs_eval = load_reference(gt_dir, [cams[i] for i in eval_idx], args.width, args.height, device)

        cos, parity = probe_native_vs_recompute(
            params0, viewmats, Ks, eval_idx[0], refs_eval, args.width, args.height, device,
        )
        print(f"probe: native-vs-recompute grad cosine={cos:.6f} init parity PSNR={parity:.2f} dB")

        results = []
        for backend in args.backends:
            print(f"[run] backend={backend} scene={scene}", flush=True)
            try:
                r = run_backend(
                    backend, params0, viewmats, Ks, train_idx, refs_train,
                    eval_idx, refs_eval, args.width, args.height, args.steps,
                    args.seed, device, args.densify_every,
                    args.densify_threshold, args.prune_threshold,
                    lpips_model, radius_clip=args.radius_clip,
                    fused_adam=args.fused_adam,
                    tile_sampling_ratio=args.tile_sampling_ratio,
                    anchor_densify=args.anchor_densify,
                    sampling_mode=args.sampling_mode,
                    error_alpha=args.error_alpha,
                    error_refresh_every=args.error_refresh_every,
                    error_lambda=args.error_lambda,
                    eval_every=args.eval_every,
                    lr_decay=args.lr_decay,
                    densify_window=(
                        None if args.densify_window == 0 else args.densify_window
                    ),
                )
                r["probe_grad_cosine"] = cos
                r["probe_init_psnr"] = parity
                results.append(r)
                print("  " + json.dumps(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({"backend": backend, "error": str(e)})
        all_results[scene] = results

    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": vars(args),
        "scenes": all_results,
    }
    out = args.out or os.path.join(
        "results", f"higs-train-benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {out}")

    print("\n=== SUMMARY (ms / GB / PSNR / SSIM / LPIPS) ===")
    for scene, results in all_results.items():
        print(f"\n-- {scene} --")
        for r in results:
            if "error" in r:
                print(f"  {r['backend']}: ERROR {r['error']}")
                continue
            print(
                f"  {r['backend']:<16} fwd={r['fwd_ms']:8.1f}ms "
                f"bwd={r['bwd_ms']:8.1f}ms tot={r['total_ms']:8.1f}ms "
                f"train={r['train_ms']:8.1f}ms "
                f"vram={r['peak_vram_gb']:5.2f}GB cull={r['culling_ratio']:6.1%} "
                f"sr={r['sampled_tile_ratio']:g} isect={r['isect_frac']:6.1%} "
                f"PSNR={r['psnr']:5.2f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} "
                f"N={r['final_n']}"
            )


if __name__ == "__main__":
    main()
