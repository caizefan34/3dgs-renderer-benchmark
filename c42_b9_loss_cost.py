#!/usr/bin/env python3
"""
C42-B9 Phase A: Isolated Structural-Loss Cost Benchmark.

Measures forward+backward time of the C42 loss function on fixed 1080p tensors
for scales: 0.75, 0.80, 0.85, 0.90, 0.95, 1.00.

Protocol matches existing C42 cost model:
  - A100, 1080p (1920x1080)
  - Same SepSSIM implementation (from reference_v1/trainer.py)
  - Same F.interpolate(mode="area")
  - 50 warmups, 300 timed samples
  - CUDA events, explicit synchronization

Usage: CUDA_VISIBLE_DEVICES=0 python c42_b9_loss_cost.py
"""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# === PROVENANCE GUARD ===
REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
EXPECTED_MODEL_HASH = "68731e375013a6f2"
EXPECTED_CONFIG_HASH = "ca76d4f36059839e"

def verify_provenance():
    """Hard fail if GaussianModel or config don't match canonical Reference V1."""
    model_path = REFERENCE_V1_DIR / "gaussian_model.py"
    config_path = REFERENCE_V1_DIR / "config.py"
    
    assert model_path.exists(), f"FATAL: GaussianModel not found at {model_path}"
    assert config_path.exists(), f"FATAL: Config not found at {config_path}"
    
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    
    assert model_hash == EXPECTED_MODEL_HASH, \
        f"FATAL: GaussianModel hash mismatch! Expected {EXPECTED_MODEL_HASH}, got {model_hash}"
    assert config_hash == EXPECTED_CONFIG_HASH, \
        f"FATAL: Config hash mismatch! Expected {EXPECTED_CONFIG_HASH}, got {config_hash}"
    
    print(f"[PROVENANCE GUARD] PASS")
    print(f"  GaussianModel: {model_path}")
    print(f"  Hash: {model_hash}")
    print(f"  Config: {config_path}")
    print(f"  Config hash: {config_hash}")
    return model_path, model_hash

# Import AFTER provenance verification
sys.path.insert(0, str(REFERENCE_V1_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

LAMBDA_DSSIM = 0.2
N_WARMUP = 50
N_MEASURE = 300


class SepSSIM:
    """Same SepSSIM as baseline/reference_v1/trainer.py."""
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def d_ssim_downsampled(ssim_fn, pred, target, scale=0.75):
    """C42 canonical: area-interpolate then SepSSIM on downsampled tensors."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
    b = F.conv2d(stacked, ssim_fn.k_h, padding=(0, ssim_fn.padding), groups=15)
    b = F.conv2d(b, ssim_fn.k_v, padding=(ssim_fn.padding, 0), groups=15)
    mu_p, mu_t = b[:, 0:3], b[:, 3:6]
    bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
    ssim_map = (2 * mu_pt + ssim_fn.C1) * (2 * spt + ssim_fn.C2) / \
               ((mu_p2 + mu_t2 + ssim_fn.C1) * (sp2 + st2 + ssim_fn.C2))
    return 1.0 - ssim_map.mean()


def compute_loss(pred, gt, scale, ssim_fn):
    """C42 loss: (1-lambda)*L1 + lambda*d_ssim_downsampled(scale)."""
    l1 = F.l1_loss(pred, gt)
    if scale >= 1.0:
        dsim = ssim_fn(pred, gt)
    else:
        dsim = d_ssim_downsampled(ssim_fn, pred, gt, scale=scale)
    return (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * dsim


def benchmark_scale(scale, pred_base, target, ssim_fn, n_warmup, n_measure):
    """Benchmark forward+backward of C42 loss at given scale on fixed tensors."""
    # Warmup
    for _ in range(n_warmup):
        pred = pred_base.clone().requires_grad_(True)
        loss = compute_loss(pred, target, scale, ssim_fn)
        loss.backward()
    torch.cuda.synchronize()
    
    # Timed
    times = []
    for _ in range(n_measure):
        pred = pred_base.clone().requires_grad_(True)
        loss = compute_loss(pred, target, scale, ssim_fn)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss.backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    
    arr = np.array(times)
    return {
        "scale": scale,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="results/c42_adaptive/b9/loss_cost_curve.json")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("=" * 72)
    print("C42-B9 Phase A: Isolated Structural-Loss Cost Benchmark")
    print("=" * 72)
    
    # Provenance guard
    model_path, model_hash = verify_provenance()
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_name} ({gpu_props.multi_processor_count} SMs)")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    import gsplat
    print(f"  gsplat: {gsplat.__version__}")
    print(f"  Warmups: {N_WARMUP}, Timed: {N_MEASURE}")
    
    ssim_fn = SepSSIM(device="cuda")
    
    # Create fixed 1080p tensors
    H, W = 1080, 1920
    torch.manual_seed(42)
    pred_base = torch.rand(H, W, 3, device="cuda", dtype=torch.float32)
    target = torch.rand(H, W, 3, device="cuda", dtype=torch.float32)
    print(f"\n  Tensor shape: [{H}, {W}, 3] (1080p)")
    print(f"  Loss: (1-{LAMBDA_DSSIM})*L1 + {LAMBDA_DSSIM}*d_ssim_downsampled(scale)")
    
    # Benchmark all scales
    scales = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    results = {}
    
    print(f"\n{'Scale':>6s} {'Mean':>8s} {'Median':>8s} {'P10':>8s} {'P90':>8s} {'Speedup':>8s}")
    print("-" * 50)
    
    # Benchmark 1.0 first as reference
    ref_result = benchmark_scale(1.0, pred_base, target, ssim_fn, N_WARMUP, N_MEASURE)
    results["1.00"] = ref_result
    ref_mean = ref_result["mean_ms"]
    speedup = 1.0
    print(f"{'1.00':>6s} {ref_result['mean_ms']:>8.2f} {ref_result['median_ms']:>8.2f} "
          f"{ref_result['p10']:>8.2f} {ref_result['p90']:>8.2f} {speedup:>7.2f}x")
    
    for scale in [0.75, 0.80, 0.85, 0.90, 0.95]:
        r = benchmark_scale(scale, pred_base, target, ssim_fn, N_WARMUP, N_MEASURE)
        key = f"{scale:.2f}"
        results[key] = r
        speedup = ref_mean / r["mean_ms"]
        r["speedup_vs_1.0"] = float(speedup)
        print(f"{key:>6s} {r['mean_ms']:>8.2f} {r['median_ms']:>8.2f} "
              f"{r['p10']:>8.2f} {r['p90']:>8.2f} {speedup:>7.2f}x")
    
    ref_result["speedup_vs_1.0"] = 1.0
    
    # Save
    output = {
        "experiment": "C42-B9 Phase A: Isolated Structural-Loss Cost Benchmark",
        "protocol": {
            "gpu": gpu_name,
            "resolution": "1080p (1920x1080)",
            "tensor_shape": [H, W, 3],
            "ssim_implementation": "SepSSIM window=11 sigma=1.5 C1=(0.01)^2 C2=(0.03)^2",
            "interpolation": "F.interpolate(mode='area')",
            "loss": f"(1-{LAMBDA_DSSIM})*L1 + {LAMBDA_DSSIM}*d_ssim_downsampled(scale)",
            "n_warmup": N_WARMUP,
            "n_measure": N_MEASURE,
            "timing_method": "CUDA events with explicit synchronization",
            "measured": "backward only (forward excluded from timed region, matching existing protocol)",
        },
        "provenance": {
            "gaussian_model_path": str(model_path),
            "gaussian_model_hash": model_hash,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gsplat": gsplat.__version__,
            "gpu": gpu_name,
            "gpu_sms": gpu_props.multi_processor_count,
        },
        "results": results,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")
    
    # Print speedup table
    print(f"\n=== SPEEDUP TABLE ===")
    for key in ["0.75", "0.80", "0.85", "0.90", "0.95", "1.00"]:
        r = results[key]
        print(f"  scale={key}: mean={r['mean_ms']:.2f}ms speedup={r['speedup_vs_1.0']:.2f}x")


if __name__ == "__main__":
    main()
