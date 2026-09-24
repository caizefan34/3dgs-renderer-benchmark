#!/usr/bin/env python3
"""
C42 Candidate D: Downsampled SSIM Validation.

Scales: 1.0, 0.75, 0.5, 0.25

For each scale:
  1. D-SSIM forward latency (isolated)
  2. D-SSIM backward latency (isolated)
  3. Total iteration latency (render + loss + bwd + opt)
  4. GPU memory
  5. Gradient cosine similarity vs baseline (scale=1.0) for xyz, opacity, scales, rotations, shs
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset

DEVICE = "cuda"
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2
N_WARMUP = 30
N_MEASURE = 50
SCALES = [1.0, 0.75, 0.5, 0.25]
PARAM_NAMES = ["xyz", "opacity", "scales", "rotations", "shs"]


def d_ssim_downsampled(pred, target, scale=1.0, window_size=11, sigma=1.5, data_range=1.0):
    """D-SSIM with optional downsampling before computation."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1,3,H,W]
        target = target.unsqueeze(0).permute(0, 3, 1, 2)

    if scale < 1.0:
        # Downsample using area pooling (best for anti-aliasing)
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)

    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])

    mu_pred = blur(pred)
    mu_target = blur(target)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = blur(pred ** 2) - mu_pred_sq
    sigma_target_sq = blur(target ** 2) - mu_target_sq
    sigma_pred_target = blur(pred * target) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return 1.0 - ssim_map.mean()


def combined_downsampled(pred, target, scale=1.0, lambda_dssim=LAMBDA_DSSIM):
    l1 = F.l1_loss(pred, target)  # L1 always at full resolution
    dsim = d_ssim_downsampled(pred, target, scale=scale)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim


def render_with_grad(model, cam, img_w, img_h):
    data = model.forward()
    rendered, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=img_w, height=img_h,
        tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
    )
    return rendered[0].clamp(0, 1)


def cuda_time(fn, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def time_backward_only(loss_fn, pred_base, target, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    for _ in range(n_warmup):
        pred = pred_base.clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        pred = pred_base.clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss.backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def get_grad_vectors(model):
    """Get flattened gradient vectors for each param group."""
    grads = {}
    for name in PARAM_NAMES:
        param = getattr(model, name)
        if param.grad is not None:
            grads[name] = param.grad.detach().clone().flatten()
        else:
            grads[name] = torch.zeros(1, device=DEVICE)
    return grads


def cosine_sim(a, b):
    """Cosine similarity between two flat tensors."""
    dot = float(torch.dot(a, b).item())
    na = float(a.norm().item())
    nb = float(b.norm().item())
    return dot / (na * nb + 1e-12)


def main():
    print("=" * 72)
    print("C42 Candidate D: Downsampled SSIM Validation")
    print("=" * 72)

    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parent.parent.parent

    print("\n  [Loading dataset...]")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    img_w, img_h = cam0.image_width, cam0.image_height

    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=DEVICE)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, Image: {img_w}x{img_h}")

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * 46.64, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    # ── Pre-render image for isolated tests ──
    print("\n  [Pre-rendering image...]")
    with torch.no_grad():
        pred_base = render_with_grad(model, cam0, img_w, img_h).detach()
    print(f"  Pre-rendered: shape={pred_base.shape}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Gradient Cosine Similarity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("Phase 1: Gradient Cosine Similarity vs Baseline (scale=1.0)")
    print(f"{'='*72}")

    # Compute baseline gradients (scale=1.0) for multiple cameras
    baseline_grads = {}
    test_cams = [0, 50, 100, 150, 200]
    for cam_idx in test_cams:
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)
        for name in PARAM_NAMES:
            p = getattr(model, name)
            if p.grad is not None:
                p.grad = None

        pred = render_with_grad(model, cam, cam.image_width, cam.image_height)
        loss = combined_downsampled(pred, gt, scale=1.0)
        loss.backward()
        baseline_grads[cam_idx] = get_grad_vectors(model)
        for name in PARAM_NAMES:
            p = getattr(model, name)
            if p.grad is not None:
                p.grad = None

    # Compute cosine similarity for each scale
    cosine_results = {}
    for scale in SCALES:
        print(f"\n  Scale={scale:.2f}:")

        # For gradient comparison, we must use SAME pred (re-render is not deterministic
        # due to potential non-determinism, so we save baseline pred and reuse)
        # But we need pred to require grad for each scale — so we re-render and compare
        # the gradient direction (cosine sim is scale-invariant)
        per_cam_cosine = {name: [] for name in PARAM_NAMES}

        for cam_idx in test_cams:
            cam = dataset.get_camera(cam_idx)
            gt = dataset.get_gt_image(cam_idx)

            # Zero grads
            for name in PARAM_NAMES:
                p = getattr(model, name)
                if p.grad is not None:
                    p.grad = None

            # Render + backward with this scale
            pred = render_with_grad(model, cam, cam.image_width, cam.image_height)
            loss = combined_downsampled(pred, gt, scale=scale)
            loss.backward()
            scale_grads = get_grad_vectors(model)

            # Compare to baseline
            for name in PARAM_NAMES:
                bg = baseline_grads[cam_idx][name]
                sg = scale_grads[name]
                if bg.numel() == sg.numel():
                    cs = cosine_sim(bg, sg)
                    per_cam_cosine[name].append(cs)
                else:
                    per_cam_cosine[name].append(float('nan'))

            # Zero grads
            for name in PARAM_NAMES:
                p = getattr(model, name)
                if p.grad is not None:
                    p.grad = None

        cosine_results[scale] = {}
        for name in PARAM_NAMES:
            vals = [v for v in per_cam_cosine[name] if not math.isnan(v)]
            avg_cs = float(np.mean(vals)) if vals else 0.0
            min_cs = float(np.min(vals)) if vals else 0.0
            cosine_results[scale][name] = {"mean": avg_cs, "min": min_cs, "all": vals}
            print(f"    {name:12s}: cosine_sim = {avg_cs:.6f}  (min={min_cs:.6f})")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Isolated D-SSIM Timing
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"Phase 2: Isolated D-SSIM Timing ({N_MEASURE} measurements)")
    print(f"{'='*72}")

    isolated_timing = {}
    for scale in SCALES:
        print(f"\n  --- Scale={scale:.2f} ---")
        loss_fn = lambda p, t, s=scale: combined_downsampled(p, t, scale=s)

        # Forward only
        mean_fwd, std_fwd = cuda_time(lambda: loss_fn(pred_base.clone(), gt0))
        print(f"  Forward:    {mean_fwd:.3f} +/- {std_fwd:.3f} ms")

        # Backward only
        mean_bwd, std_bwd = time_backward_only(loss_fn, pred_base, gt0)
        print(f"  Backward:   {mean_bwd:.3f} +/- {std_bwd:.3f} ms")

        # Full fwd+bwd
        def full_fn():
            pred = pred_base.clone().requires_grad_(True)
            loss = loss_fn(pred, gt0)
            loss.backward()
        mean_full, std_full = cuda_time(full_fn)
        print(f"  Fwd+Bwd:    {mean_full:.3f} +/- {std_full:.3f} ms")

        isolated_timing[scale] = {
            "forward_ms": mean_fwd, "forward_std": std_fwd,
            "backward_ms": mean_bwd, "backward_std": std_bwd,
            "full_ms": mean_full, "full_std": std_full,
            "image_size": f"{int(img_h*scale)}x{int(img_w*scale)}" if scale < 1.0 else f"{img_h}x{img_w}",
        }

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: End-to-End Training Iteration Timing
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"Phase 3: End-to-End Training Iteration ({N_MEASURE} measurements)")
    print(f"{'='*72}")

    def training_step(loss_fn):
        pred = render_with_grad(model, cam0, img_w, img_h)
        loss = loss_fn(pred, gt0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    e2e_timing = {}
    for scale in SCALES:
        print(f"\n  --- Scale={scale:.2f} ---")
        loss_fn = lambda p, t, s=scale: combined_downsampled(p, t, scale=s)

        # Warmup
        for _ in range(N_WARMUP):
            training_step(loss_fn)
        torch.cuda.synchronize()

        times = []
        for _ in range(N_MEASURE):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            training_step(loss_fn)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        arr = np.array(times)
        mean_e2e = float(arr.mean())
        std_e2e = float(arr.std())
        print(f"  End-to-end: {mean_e2e:.3f} +/- {std_e2e:.3f} ms")
        e2e_timing[scale] = {"mean_ms": mean_e2e, "std_ms": std_e2e,
                             "min_ms": float(arr.min()), "max_ms": float(arr.max())}

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Memory
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("Phase 4: GPU Memory Allocation")
    print(f"{'='*72}")

    mem_results = {}
    for scale in SCALES:
        loss_fn = lambda p, t, s=scale: combined_downsampled(p, t, scale=s)

        # E2E memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        training_step(loss_fn)
        torch.cuda.synchronize()
        peak_e2e = torch.cuda.max_memory_allocated() / 1e6

        # Isolated memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        pred = pred_base.clone().requires_grad_(True)
        loss = loss_fn(pred, gt0)
        loss.backward()
        torch.cuda.synchronize()
        peak_iso = torch.cuda.max_memory_allocated() / 1e6

        print(f"  Scale={scale:.2f}: isolated peak={peak_iso:.1f}MB  e2e peak={peak_e2e:.1f}MB")
        mem_results[scale] = {"isolated_peak_MB": peak_iso, "e2e_peak_MB": peak_e2e}

    # ═══════════════════════════════════════════════════════════════
    # Summary & Decision
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    base_full = isolated_timing[1.0]["full_ms"]
    base_e2e = e2e_timing[1.0]["mean_ms"]

    print(f"\n  {'Scale':>6s}  {'Img size':>12s}  {'D-SSIM ms':>10s}  {'Speedup':>8s}  "
          f"{'E2E ms':>10s}  {'E2E gain':>8s}  {'Mem MB':>8s}  {'cos_xyz':>8s}  {'cos_opa':>8s}  "
          f"{'cos_scl':>8s}  {'cos_rot':>8s}  {'cos_shs':>8s}")
    print("  " + "-" * 120)

    for scale in SCALES:
        t = isolated_timing[scale]
        e = e2e_timing[scale]
        m = mem_results[scale]
        dssim_speedup = base_full / t["full_ms"]
        e2e_gain = (1 - e["mean_ms"] / base_e2e) * 100
        c = cosine_results[scale]

        print(f"  {scale:>6.2f}  {t['image_size']:>12s}  {t['full_ms']:>10.3f}  {dssim_speedup:>7.2f}x  "
              f"{e['mean_ms']:>10.3f}  {e2e_gain:>+7.1f}%  {m['e2e_peak_MB']:>8.1f}  "
              f"{c['xyz']['mean']:>8.4f}  {c['opacity']['mean']:>8.4f}  "
              f"{c['scales']['mean']:>8.4f}  {c['rotations']['mean']:>8.4f}  {c['shs']['mean']:>8.4f}")

    # Decision per scale
    print(f"\n  Decision per scale:")
    decisions = {}
    for scale in SCALES:
        t = isolated_timing[scale]
        e = e2e_timing[scale]
        dssim_speedup = base_full / t["full_ms"]
        e2e_gain = (1 - e["mean_ms"] / base_e2e) * 100
        c = cosine_results[scale]
        min_cosine = min(c[name]["mean"] for name in PARAM_NAMES)

        dssim_pass = dssim_speedup > 3.0
        e2e_pass = e2e_gain > 15.0
        cosine_pass = min_cosine > 0.95

        if dssim_pass and e2e_pass and cosine_pass:
            d = "KEEP"
        elif dssim_pass and e2e_pass and not cosine_pass:
            d = "MAYBE (cosine below 0.95)"
        elif dssim_pass and not e2e_pass:
            d = "MAYBE (e2e below 15%)"
        else:
            d = "DROP"

        decisions[scale] = d
        print(f"    scale={scale:.2f}: D-SSIM={dssim_speedup:.2f}x  E2E={e2e_gain:+.1f}%  "
              f"min_cosine={min_cosine:.4f}  → {d}")

    # ── Save ──
    output = {
        "experiment": "C42 Candidate D: Downsampled SSIM",
        "config": {"scales": SCALES, "n_warmup": N_WARMUP, "n_measure": N_MEASURE,
                   "lambda_dssim": LAMBDA_DSSIM, "scene": "room", "n_gaussians": N,
                   "image": f"{img_w}x{img_h}"},
        "cosine_similarity": {str(s): {name: {"mean": c[name]["mean"], "min": c[name]["min"]}
                                       for name in PARAM_NAMES}
                              for s, c in cosine_results.items()},
        "isolated_timing": {str(s): t for s, t in isolated_timing.items()},
        "end_to_end_timing": {str(s): e for s, e in e2e_timing.items()},
        "memory": {str(s): m for s, m in mem_results.items()},
        "decisions": {str(s): d for s, d in decisions.items()},
    }
    save_path = Path("results/phase-c42/c42_downsampled_ssim_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
