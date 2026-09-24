#!/usr/bin/env python3
"""
C43 Adaptive Tile Size Screening (5K).

Hypothesis: tile size should depend on workload statistics rather than fixed value.
Compare: tile16 (baseline) vs adaptive tile16/32 based on visible Gaussian count.

Single-module: only tile_size changes. Renderer, optimizer, loss, training all unchanged.
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization, fully_fused_projection, isect_tiles
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
SEED = 42
LAMBDA_DSSIM = 0.2
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0

# Adaptive tile threshold: if visible Gaussians > threshold, use tile32
ADAPTIVE_THRESHOLD = 50000  # Gaussians

EVAL_CAMERAS = list(range(0, 311, 25))  # 13 cameras


def render_with_tile(model, cam, tile_size):
    data = model.forward()
    rendered, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=tile_size, packed=PACKED, sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
    )
    return rendered[0].clamp(0, 1)


def count_visible_gaussians(model, cam, tile_size=16):
    """Estimate visible Gaussian count for a camera using a quick render."""
    data = model.forward()
    # Project to 2D (use non-packed for simplicity)
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        radius_clip=RADIUS_CLIP, packed=False, eps2d=EPS2D,
    )
    tw = (cam.image_width + tile_size - 1) // tile_size
    th = (cam.image_height + tile_size - 1) // tile_size
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, sort=False, packed=False)
    n_visible = (tiles_per_gauss > 0).sum().item()
    return n_visible


def d_ssim_loss(pred, target, window_size=11, sigma=1.5):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * (blur(pred * target) - mu_p * mu_t) + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (blur(pred**2) - mu_p**2 + blur(target**2) - mu_t**2 + C2))
    return 1.0 - ssim_map.mean()


def benchmark_tile(model, dataset, cam_indices, tile_size, label):
    """Benchmark rendering with a fixed tile size."""
    print(f"\n  [{label}] tile_size={tile_size}")
    render_times = []
    psnrs, ssims = [], []

    # Warmup
    for ci in cam_indices[:3]:
        cam = dataset.get_camera(ci)
        with torch.no_grad():
            _ = render_with_tile(model, cam, tile_size)
    torch.cuda.synchronize()

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = render_with_tile(model, cam, tile_size)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        render_times.append(dt)

        mse = float(((pred - gt) ** 2).mean())
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        psnrs.append(psnr)
        ssims.append(1.0 - float(d_ssim_loss(pred, gt)))

    print(f"    mean render: {np.mean(render_times):.2f} ms  std: {np.std(render_times):.2f} ms")
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")
    return {
        "label": label, "tile_size": tile_size,
        "mean_render_ms": float(np.mean(render_times)),
        "std_render_ms": float(np.std(render_times)),
        "min_render_ms": float(np.min(render_times)),
        "max_render_ms": float(np.max(render_times)),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "per_camera": [{"cam": ci, "render_ms": rt, "psnr": p, "ssim": s}
                       for ci, rt, p, s in zip(cam_indices, render_times, psnrs, ssims)],
    }


def benchmark_adaptive(model, dataset, cam_indices, label):
    """Benchmark rendering with adaptive tile size (16 or 32 based on visible count)."""
    print(f"\n  [{label}] adaptive tile16/32 (threshold={ADAPTIVE_THRESHOLD})")
    render_times = []
    psnrs, ssims = [], []
    tile_choices = []

    # Warmup
    for ci in cam_indices[:3]:
        cam = dataset.get_camera(ci)
        with torch.no_grad():
            _ = render_with_tile(model, cam, 16)
    torch.cuda.synchronize()

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Decide tile size
        n_vis = count_visible_gaussians(model, cam)
        chosen_tile = 32 if n_vis > ADAPTIVE_THRESHOLD else 16
        tile_choices.append({"cam": ci, "n_visible": n_vis, "tile_size": chosen_tile})

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = render_with_tile(model, cam, chosen_tile)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        render_times.append(dt)

        mse = float(((pred - gt) ** 2).mean())
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        psnrs.append(psnr)
        ssims.append(1.0 - float(d_ssim_loss(pred, gt)))

    n_t16 = sum(1 for t in tile_choices if t["tile_size"] == 16)
    n_t32 = sum(1 for t in tile_choices if t["tile_size"] == 32)
    print(f"    tile16 chosen: {n_t16}/{len(cam_indices)}  tile32 chosen: {n_t32}/{len(cam_indices)}")
    print(f"    mean render: {np.mean(render_times):.2f} ms  std: {np.std(render_times):.2f} ms")
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")
    return {
        "label": label, "strategy": "adaptive_16_32",
        "threshold": ADAPTIVE_THRESHOLD,
        "n_tile16": n_t16, "n_tile32": n_t32,
        "mean_render_ms": float(np.mean(render_times)),
        "std_render_ms": float(np.std(render_times)),
        "min_render_ms": float(np.min(render_times)),
        "max_render_ms": float(np.max(render_times)),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "tile_choices": tile_choices,
        "per_camera": [{"cam": ci, "render_ms": rt, "psnr": p, "ssim": s, "tile": tc["tile_size"]}
                       for ci, rt, p, s, tc in zip(cam_indices, render_times, psnrs, ssims, tile_choices)],
    }


def main():
    print("=" * 72)
    print("C43 Adaptive Tile Size Screening")
    print("=" * 72)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  Seed: {SEED}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    print(f"  Cameras: {len(dataset)}")

    # Load SfM checkpoint (use the 5K trained checkpoint for realistic workload)
    ckpt_path = repo_root / "results" / "a100" / "phase-c42" / "p2_checkpoints"
    ckpt_files = list(ckpt_path.glob("*iter10000*.pt")) if ckpt_path.exists() else []
    if ckpt_files:
        ckpt = torch.load(ckpt_files[0], map_location=DEVICE)
        model = GaussianModel.from_checkpoint_state(ckpt, device=DEVICE)
        model.set_sh_degree(3)
        print(f"  Loaded checkpoint: {ckpt_files[0].name}  GS={model.xyz.shape[0]:,}")
    else:
        # Fallback: use SfM init
        sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
        model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
        model.init_from_sfm(
            xyz=sfm_data["xyz"],
            opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=DEVICE)),
            scales_log=sfm_data.get("scales"),
            rotations_raw=sfm_data.get("rotations"),
            shs=sfm_data.get("shs"),
        )
        model.set_sh_degree(3)
        print(f"  Using SfM init: GS={model.xyz.shape[0]:,}")

    cams = EVAL_CAMERAS

    # Benchmark 1: tile16 (baseline)
    r16 = benchmark_tile(model, dataset, cams, 16, "baseline_tile16")

    # Benchmark 2: tile32 (comparison)
    r32 = benchmark_tile(model, dataset, cams, 32, "comparison_tile32")

    # Benchmark 3: adaptive tile16/32
    r_adaptive = benchmark_adaptive(model, dataset, cams, "adaptive_tile16_32")

    # Also test with more cameras for workload statistics
    all_cams = list(range(0, 311, 10))  # 31 cameras
    r_adaptive_31 = benchmark_adaptive(model, dataset, all_cams, "adaptive_31cams")

    # Analysis
    print(f"\n{'='*72}")
    print("ANALYSIS")
    print(f"{'='*72}")

    speedup_32_vs_16 = (1 - r32["mean_render_ms"] / r16["mean_render_ms"]) * 100
    speedup_adapt_vs_16 = (1 - r_adaptive["mean_render_ms"] / r16["mean_render_ms"]) * 100
    psnr_diff_32 = r32["mean_psnr"] - r16["mean_psnr"]
    psnr_diff_adapt = r_adaptive["mean_psnr"] - r16["mean_psnr"]
    ssim_diff_32 = r32["mean_ssim"] - r16["mean_ssim"]
    ssim_diff_adapt = r_adaptive["mean_ssim"] - r16["mean_ssim"]

    print(f"\n  tile16 vs tile32:     speedup={speedup_32_vs_16:+.1f}%  dPSNR={psnr_diff_32:+.2f}  dSSIM={ssim_diff_32:+.4f}")
    print(f"  tile16 vs adaptive:   speedup={speedup_adapt_vs_16:+.1f}%  dPSNR={psnr_diff_adapt:+.2f}  dSSIM={ssim_diff_adapt:+.4f}")
    print(f"  adaptive tile16/32 split: {r_adaptive['n_tile16']}/{r_adaptive['n_tile16']+r_adaptive['n_tile32']} t16, "
          f"{r_adaptive['n_tile32']}/{r_adaptive['n_tile16']+r_adaptive['n_tile32']} t32")

    # Correctness: adaptive must produce same PSNR/SSIM as baseline tile16
    # (tile_size doesn't affect rendering quality in gsplat — only performance)
    correctness_pass = abs(psnr_diff_adapt) < 0.01 and abs(ssim_diff_adapt) < 0.001
    speedup_pass = speedup_adapt_vs_16 > 0

    print(f"\n  Correctness (dPSNR<0.01, dSSIM<0.001): {'PASS' if correctness_pass else 'FAIL'}")
    print(f"  Speedup > 0%: {'PASS' if speedup_pass else 'FAIL'} ({speedup_adapt_vs_16:+.1f}%)")

    if correctness_pass and speedup_pass:
        decision = "KEEP — adaptive tile provides speedup with no quality loss"
    elif correctness_pass and not speedup_pass:
        decision = "DROP — no speedup benefit from adaptive tile"
    else:
        decision = "DROP — quality degradation detected"

    print(f"\n  DECISION: {decision}")

    output = {
        "experiment": "C43 Adaptive Tile Size Screening",
        "hypothesis": "tile size should depend on workload statistics rather than fixed value",
        "hardware": {"gpu": gpu_name},
        "config": {"seed": SEED, "scene": "room", "n_gaussians": model.xyz.shape[0],
                   "adaptive_threshold": ADAPTIVE_THRESHOLD, "eval_cameras": cams},
        "results": {
            "baseline_tile16": r16,
            "comparison_tile32": r32,
            "adaptive_13cams": r_adaptive,
            "adaptive_31cams": r_adaptive_31,
        },
        "analysis": {
            "speedup_32_vs_16_pct": speedup_32_vs_16,
            "speedup_adaptive_vs_16_pct": speedup_adapt_vs_16,
            "dpsnr_32": psnr_diff_32, "dpsnr_adaptive": psnr_diff_adapt,
            "dssim_32": ssim_diff_32, "dssim_adaptive": ssim_diff_adapt,
            "correctness_pass": correctness_pass,
            "speedup_pass": speedup_pass,
            "decision": decision,
        },
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / "c43_adaptive_tile_screening.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
