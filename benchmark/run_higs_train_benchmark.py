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
latency, peak VRAM, culling ratio, and final PSNR / SSIM / LPIPS on held-out
cameras. A speedup is only claimed when the measured total iteration time
actually wins.
"""

import argparse
import json
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

def make_optimizer(params, lr_scale=1.0):
    means, quats, scales, opacities, sh = params
    return torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4 * lr_scale},
        {"params": [quats], "lr": 1e-3 * lr_scale},
        {"params": [scales], "lr": 5e-3 * lr_scale},
        {"params": [opacities], "lr": 5e-2 * lr_scale},
        {"params": [sh], "lr": 2.5e-3 * lr_scale},
    ])


def _l1_loss(frame, ref):
    return (frame - ref).abs().mean()


def make_forward_fn(backend, width, height, handle, viewmats, Ks):
    from gsplat.rendering import rasterization

    def forward_fn(params_in, cam_ids):
        m, q, s, o, c = params_in
        vm = viewmats[:, cam_ids]
        K = Ks[:, cam_ids]
        if backend == "std":
            out = rasterization(
                means=m.unsqueeze(0), quats=q.unsqueeze(0),
                scales=s.unsqueeze(0), opacities=o.unsqueeze(0), colors=c,
                viewmats=vm, Ks=K, width=width, height=height,
                sh_degree=_SH_DEGREE, packed=True,
            )
            return out[0], out[1], {}
        kw = dict(
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=_SH_DEGREE, use_higs_culling=True,
        )
        if backend in ("higs_native", "higs_recompute"):
            from gsplat.experimental import rasterize_gaussian_higs_frozen
            mode = "higs_native" if backend == "higs_native" else "gsplat_recompute"
            res = rasterize_gaussian_higs_frozen(
                m, q, s, o, c, backward_mode=mode, scene=handle,
                freeze_topology=True, **kw,
            )
            return res["frame"], res["alpha"], res["metadata"]
        from gsplat.experimental import rasterize_gaussian_higs_dynamic
        res = rasterize_gaussian_higs_dynamic(
            m, q, s, o, c, backward_mode="higs_native", **kw,
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

def run_backend(
    backend, params0, viewmats, Ks, train_idx, refs_train,
    eval_idx, refs_eval, width, height, steps, seed, device,
    densify_every, densify_threshold, prune_threshold, lpips_model,
):
    torch.manual_seed(seed)
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_FROZEN_TRACKER
    _HIGS_FROZEN_TRACKER.reset()
    means, quats, scales, opacities, sh = [t.detach().clone() for t in params0]
    for t in (means, quats, scales, opacities, sh):
        t.requires_grad_(True)
    params = (means, quats, scales, opacities, sh)
    opt = make_optimizer(params)

    handle = None
    dynamic_scene = None
    if backend in ("higs_native", "higs_recompute"):
        from gsplat.experimental.render.functional.gaussian_inference import (
            create_higs_renderer,
        )
        handle = create_higs_renderer(
            means, quats, scales, opacities, sh, sh_degree=_SH_DEGREE,
        )
    elif backend == "higs_dynamic":
        from gsplat.experimental.render.functional.gaussian_inference import (
            _HIGS_DYNAMIC_SCENE,
        )
        dynamic_scene = _HIGS_DYNAMIC_SCENE
        dynamic_scene.reset()

    forward_fn = make_forward_fn(backend, width, height, handle, viewmats, Ks)
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times, bwd_times, total_times = [], [], []
    culling_ratios, n_visibles, topo_rebuilt = [], [], []
    ref = refs_train

    try:
        for it in range(steps):
            cam_ids = train_idx
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            frame, alpha, meta = forward_fn(params, cam_ids)
            ev1.record()
            torch.cuda.synchronize(device)
            fwd_ms = ev0.elapsed_time(ev1)
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
            else:
                culling_ratios.append(0.0)
                n_visibles.append(means.shape[0])
                topo_rebuilt.append(0.0)

            opt.step()

            if backend == "higs_dynamic" and (it + 1) % densify_every == 0:
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

            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1e9

        with torch.no_grad():
            ev_frame, _, _ = forward_fn(params, eval_idx)
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
        "peak_vram_gb": peak,
        "culling_ratio": float(np.mean(culling_ratios)) if culling_ratios else 0.0,
        "n_visible_avg": float(np.mean(n_visibles)) if n_visibles else 0.0,
        "psnr": p,
        "ssim": s,
        "lpips": l,
        "final_n": means.shape[0],
        "topology_rebuilt_frac": float(np.mean(topo_rebuilt)) if topo_rebuilt else 0.0,
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
                    lpips_model,
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
                f"vram={r['peak_vram_gb']:5.2f}GB cull={r['culling_ratio']:6.1%} "
                f"PSNR={r['psnr']:5.2f} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f} "
                f"N={r['final_n']}"
            )


if __name__ == "__main__":
    main()
