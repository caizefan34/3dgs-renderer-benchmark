#!/usr/bin/env python3
"""Phase V3 — Apply C1 patch and benchmark forward pass.

Steps:
  1. Generate scene .ckpt files with real 3DGS data
  2. Run baseline rendering (vanilla gsplat)
  3. Patch IntersectTile.cu → C1 variant
  4. Rebuild gsplat (pip install -e . or python setup.py develop)
  5. Run C1 rendering
  6. Compare outputs (PSNR, SSIM, timing)
  
Note: This script must be run from an environment where gsplat is installed
from source (editable install) so we can rebuild after patching.
"""

import torch, math, time, json, os, sys, shutil, subprocess, tempfile
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "phase-c17-c2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "phase-c17-c2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Scene generation
# ============================================================

def make_viewmat(eye, lookat, up=(0,1,0)):
    """Create view matrix from look-at parameters."""
    eye = torch.tensor(eye, dtype=torch.float32, device=DEVICE)
    lookat = torch.tensor(lookat, dtype=torch.float32, device=DEVICE)
    up = torch.tensor(up, dtype=torch.float32, device=DEVICE)
    
    fwd = (lookat - eye) / torch.norm(lookat - eye)
    right = torch.cross(fwd, up)
    right = right / torch.norm(right)
    up = torch.cross(right, fwd)
    
    viewmat = torch.eye(4, device=DEVICE)
    viewmat[0, :3] = right
    viewmat[1, :3] = up
    viewmat[2, :3] = -fwd
    viewmat[:3, 3] = -viewmat[:3, :3] @ eye
    return viewmat.unsqueeze(0)


def make_intrinsics(h, w, fov_deg=50):
    """Create intrinsics matrix."""
    fov = math.radians(fov_deg)
    fx = w / (2 * math.tan(fov / 2))
    fy = h / (2 * math.tan(fov / 2))
    K = torch.tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], 
                     device=DEVICE, dtype=torch.float32).unsqueeze(0)
    return K


def create_scene(name, params):
    """Create synthetic 3DGS scene with realistic parameters."""
    n = params["n"]
    h, w = params["h"], params["w"]
    d_min, d_max = params["d_min"], params["d_max"]
    fov = params.get("fov", 50)
    
    # Random depths (log-uniform)
    depths = torch.exp(torch.empty(n, device=DEVICE).uniform_(
        math.log(d_min), math.log(d_max)))
    
    # Positions in camera space → world
    tan_hfov = math.tan(math.radians(fov) / 2)
    tan_hfov_y = tan_hfov * h / w
    
    uv_x = torch.rand(n, device=DEVICE) * 2 - 1
    uv_y = torch.rand(n, device=DEVICE) * 2 - 1
    
    means = torch.stack([
        uv_x * depths * tan_hfov,
        uv_y * depths * tan_hfov_y,
        -depths,  # looking down -z
    ], dim=1)
    
    # Rotations
    quats = torch.randn(n, 4, device=DEVICE)
    quats = quats / quats.norm(dim=1, keepdim=True)
    
    # Scales (proportional to depth for consistent screen size)
    base_scale = 0.01
    scales = torch.rand(n, 3, device=DEVICE) * base_scale * 2 + base_scale * 0.1
    
    # Opacities
    opacities = torch.sigmoid(torch.randn(n, device=DEVICE) * 0.5).unsqueeze(1)
    opacities = opacities * 0.9 + 0.05  # range [0.05, 0.95]
    
    # Colors (as SH0 coefficients)
    colors = torch.rand(n, 3, device=DEVICE) * 0.8 + 0.1
    
    # Camera setup
    viewmat = make_viewmat(
        eye=[0, 0, 2 * d_min],  # place camera at 2x min depth
        lookat=[0, 0, 0],
    )
    K = make_intrinsics(h, w, fov)
    img_size = torch.tensor([h, w], device=DEVICE)
    
    scene = {
        "means": means.unsqueeze(0),
        "quats": quats.unsqueeze(0),
        "scales": scales.unsqueeze(0),
        "opacities": opacities.unsqueeze(0),
        "colors": colors.unsqueeze(0),
        "viewmat": viewmat,
        "K": K,
        "img_size": img_size,
        "name": name,
        "depth_tensor": depths.unsqueeze(0),
    }
    return scene


# ============================================================
# Rendering helpers
# ============================================================

def render_scene(scene, tile_size=16):
    """Render a scene using gsplat's fully_fused_projection."""
    from gsplat import rasterization
    
    result = rasterization(
        means=scene["means"],
        quats=scene["quats"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        colors=scene["colors"],
        viewmat=scene["viewmat"],
        K=scene["K"],
        img_size=scene["img_size"],
        tile_size=tile_size,
        packed=True,
    )
    
    # result = (render_colors, render_alphas, *optional_meta)
    if isinstance(result, (list, tuple)):
        render_colors = result[0]
        render_alphas = result[1]
    else:
        render_colors = result  # newer API might return just the image
    
    return render_colors, render_alphas


# ============================================================
# Patching utilities
# ============================================================

def find_gsplat_cu_path():
    """Find the IntersectTile.cu file in the installed gsplat."""
    import gsplat
    gsplat_dir = Path(gsplat.__file__).parent
    cu_path = gsplat_dir / "cuda" / "csrc" / "IntersectTile.cu"
    if cu_path.exists():
        return cu_path
    # Maybe installed from source
    alt_paths = list(gsplat_dir.glob("**/IntersectTile.cu"))
    if alt_paths:
        return alt_paths[0]
    raise FileNotFoundError(f"Cannot find IntersectTile.cu in {gsplat_dir}")


def find_c1_patch():
    """Find the C1 patch file in the benchmark repo."""
    repo_root = Path(__file__).resolve().parent.parent
    patch_path = repo_root / "patches" / "IntersectTile.c1.cu"
    if patch_path.exists():
        return patch_path
    raise FileNotFoundError(f"Cannot find C1 patch at {patch_path}")


def apply_c1_patch(backup=True):
    """Replace IntersectTile.cu with C1 variant."""
    cu_path = find_gsplat_cu_path()
    patch_path = find_c1_patch()
    
    if backup:
        backup_path = cu_path.with_suffix(".cu.baseline")
        if not backup_path.exists():
            shutil.copy2(cu_path, backup_path)
            print(f"  Backup saved: {backup_path}")
    
    shutil.copy2(patch_path, cu_path)
    print(f"  C1 patch applied: {patch_path} → {cu_path}")
    return cu_path


def restore_baseline():
    """Restore IntersectTile.cu from backup."""
    cu_path = find_gsplat_cu_path()
    backup_path = cu_path.with_suffix(".cu.baseline")
    if backup_path.exists():
        shutil.copy2(backup_path, cu_path)
        print(f"  Baseline restored: {backup_path} → {cu_path}")
        return True
    else:
        print(f"  No backup found at {backup_path}")
        return False


def rebuild_gsplat():
    """Rebuild gsplat CUDA extension in-place."""
    import gsplat
    gsplat_dir = Path(gsplat.__file__).parent
    
    # Try pip install --no-build-isolation
    print(f"  Rebuilding gsplat...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         "--force-reinstall", "--no-deps", str(gsplat_dir.parent)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  Build stdout: {result.stdout[-500:]}")
        print(f"  Build stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"gsplat rebuild failed ({result.returncode})")
    print(f"  Rebuild successful")
    return result


# ============================================================
# Benchmarking
# ============================================================

def benchmark_rendering(scene, n_runs=4, tile_size=16, label=""):
    """Run rendering and measure timing + output."""
    from gsplat import rasterization
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = rasterization(
                means=scene["means"], quats=scene["quats"],
                scales=scene["scales"], opacities=scene["opacities"],
                colors=scene["colors"], viewmat=scene["viewmat"],
                K=scene["K"], img_size=scene["img_size"],
                tile_size=tile_size, packed=True,
            )
    
    torch.cuda.synchronize()
    
    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    render_results = []
    
    for i in range(n_runs):
        start_events[i].record()
        with torch.no_grad():
            result = rasterization(
                means=scene["means"], quats=scene["quats"],
                scales=scene["scales"], opacities=scene["opacities"],
                colors=scene["colors"], viewmat=scene["viewmat"],
                K=scene["K"], img_size=scene["img_size"],
                tile_size=tile_size, packed=True,
            )
        end_events[i].record()
        torch.cuda.synchronize()
        
        render_colors = result[0] if isinstance(result, (list, tuple)) else result
        
        render_results.append({
            "render_colors": render_colors.detach().cpu().numpy(),
            "time_ms": start_events[i].elapsed_time(end_events[i]),
        })
    
    timings = [r["time_ms"] for r in render_results]
    render_data = render_results[-1]["render_colors"]  # use last run
    
    stats = {
        "label": label,
        "n_runs": n_runs,
        "timing_ms_mean": float(np.mean(timings)),
        "timing_ms_std": float(np.std(timings)),
        "timing_ms_min": float(np.min(timings)),
        "timing_ms_max": float(np.max(timings)),
        "timings": timings,
    }
    
    return render_data, stats


def compute_metrics(baseline_img, c1_img):
    """Compute PSNR, SSIM, pixel diff between two images."""
    b = baseline_img.astype(np.float64)
    c = c1_img.astype(np.float64)
    
    # Clamp
    b = np.clip(b, 0, 1)
    c = np.clip(c, 0, 1)
    
    # MSE per pixel
    mse = np.mean((b - c) ** 2)
    psnr = -10 * math.log10(max(mse, 1e-20)) if mse > 1e-20 else 100.0
    
    # Max absolute diff
    max_abs_diff = float(np.max(np.abs(b - c)))
    mean_abs_diff = float(np.mean(np.abs(b - c)))
    
    # SSIM-like (simplified: luminance + contrast per channel)
    def ssim_simple(img1, img2):
        c1 = 0.01**2
        c2 = 0.03**2
        mu1 = np.mean(img1, axis=(0, 1), keepdims=True)
        mu2 = np.mean(img2, axis=(0, 1), keepdims=True)
        sigma1_sq = np.var(img1, axis=(0, 1))
        sigma2_sq = np.var(img2, axis=(0, 1))
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(0, 1))
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(np.mean(ssim))
    
    ssim_val = ssim_simple(b, c)
    
    return {
        "psnr": round(psnr, 4),
        "ssim": round(ssim_val, 6),
        "max_abs_diff": round(max_abs_diff, 8),
        "mean_abs_diff": round(mean_abs_diff, 8),
        "mse": round(float(mse), 10),
    }


# ============================================================
# Main benchmark pipeline
# ============================================================

SCENES = {
    "room":    {"n": 100000, "d_min": 0.2,  "d_max": 6.0,   "h": 540, "w": 960, "fov": 50},
    "bicycle": {"n": 100000, "d_min": 0.5,  "d_max": 50.0,  "h": 540, "w": 960, "fov": 50},
    "garden":  {"n": 100000, "d_min": 0.3,  "d_max": 30.0,  "h": 540, "w": 960, "fov": 50},
}
TILE_SIZE = 16
N_RUNS = 3
WARMUP = 2


def main():
    all_results = {}
    
    # First run baseline on all scenes
    print("=" * 60)
    print("  PHASE V3 — FORWARD A/B VERIFICATION")
    print("=" * 60)
    
    print("\n--- Baseline Rendering ---")
    baseline_data = {}
    for sname, sp in SCENES.items():
        print(f"\n  Scene: {sname}")
        scene = create_scene(sname, sp)
        img, stats = benchmark_rendering(scene, N_RUNS, TILE_SIZE, f"baseline/{sname}")
        baseline_data[sname] = {"image": img, "stats": stats, "scene": scene}
        print(f"    Mean time: {stats['timing_ms_mean']:.2f} ± {stats['timing_ms_std']:.2f} ms")
    
    # Apply C1 patch and rebuild
    print("\n--- Applying C1 Patch ---")
    cu_path = apply_c1_patch(backup=True)
    rebuild_gsplat()
    
    # Clear CUDA cache after rebuild
    torch.cuda.empty_cache()
    
    # Run C1 rendering
    print("\n--- C1 Rendering ---")
    c1_data = {}
    for sname, sp in SCENES.items():
        print(f"\n  Scene: {sname}")
        # Redownload from GPU
        scene = create_scene(sname, sp)
        img, stats = benchmark_rendering(scene, N_RUNS, TILE_SIZE, f"c1/{sname}")
        c1_data[sname] = {"image": img, "stats": stats}
        print(f"    Mean time: {stats['timing_ms_mean']:.2f} ± {stats['timing_ms_std']:.2f} ms")
    
    # Restore baseline
    print("\n--- Restoring Baseline ---")
    restore_baseline()
    
    # Compute metrics
    print("\n--- Results ---")
    for sname in SCENES:
        baseline_img = baseline_data[sname]["image"]
        c1_img = c1_data[sname]["image"]
        b_stats = baseline_data[sname]["stats"]
        c_stats = c1_data[sname]["stats"]
        
        metrics = compute_metrics(baseline_img, c1_img)
        
        print(f"\n  {sname}:")
        print(f"    PSNR:           {metrics['psnr']:.4f} dB")
        print(f"    SSIM:           {metrics['ssim']:.6f}")
        print(f"    Max abs diff:   {metrics['max_abs_diff']:.8f}")
        print(f"    Mean abs diff:  {metrics['mean_abs_diff']:.8f}")
        print(f"    Baseline time:  {b_stats['timing_ms_mean']:.2f} ± {b_stats['timing_ms_std']:.2f} ms")
        print(f"    C1 time:        {c_stats['timing_ms_mean']:.2f} ± {c_stats['timing_ms_std']:.2f} ms")
        print(f"    Time change:    {(c_stats['timing_ms_mean'] - b_stats['timing_ms_mean']):.2f} ms "
              f"({(c_stats['timing_ms_mean']/b_stats['timing_ms_mean'] - 1)*100:+.2f}%)")
        
        all_results[sname] = {
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "max_abs_diff": metrics["max_abs_diff"],
            "mean_abs_diff": metrics["mean_abs_diff"],
            "mse": metrics["mse"],
            "baseline_timing": b_stats,
            "c1_timing": c_stats,
        }
    
    # Save
    output_path = OUTPUT_DIR / "c1_forward_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scene':>10s} | {'PSNR':>8s} | {'SSIM':>8s} | {'MaxD':>10s} | "
          f"{'Base(ms)':>10s} | {'C1(ms)':>10s} | {'Δ%':>8s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for sname in all_results:
        r = all_results[sname]
        bt = r["baseline_timing"]
        ct = r["c1_timing"]
        pct = (ct["timing_ms_mean"] / bt["timing_ms_mean"] - 1) * 100
        print(f"  {sname:>10s} | {r['psnr']:>8.4f} | {r['ssim']:>8.6f} | "
              f"{r['max_abs_diff']:>10.8f} | {bt['timing_ms_mean']:>8.2f}±{bt['timing_ms_std']:.2f} | "
              f"{ct['timing_ms_mean']:>8.2f}±{ct['timing_ms_std']:.2f} | {pct:>+7.2f}%")
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
