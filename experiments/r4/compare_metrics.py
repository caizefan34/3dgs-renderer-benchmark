#!/usr/bin/env python3
"""Compare baseline vs candidate_c training metrics for room scene."""
import json, os, sys

baseline_path = os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/training_metrics.json")
candidate_metrics_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/room/candidate_c/training_metrics.json"
candidate_timing_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/room/candidate_c/timing.json"
baseline_timing_path = os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/timing.json")

print("=== BASELINE room_30k ===")
try:
    d = json.load(open(baseline_path))
    for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
        psnr = v["psnr"]
        ssim = v["ssim"]
        n = v["N_gaussians"]
        print(f"  iter {k}: PSNR={psnr:.2f} SSIM={ssim:.4f} N={n}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== BASELINE timing ===")
try:
    d = json.load(open(baseline_timing_path))
    for k, v in d.items():
        if v["total_ms_mean"] > 0:
            t = v["total_ms_mean"]
            f = v["fwd_ms_mean"]
            b = v["bwd_ms_mean"]
            print(f"  {k}: total={t:.1f}ms fwd={f:.1f}ms bwd={b:.1f}ms")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== CANDIDATE C room ===")
try:
    d = json.load(open(candidate_metrics_path))
    for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
        psnr = v["psnr"]
        ssim = v["ssim"]
        n = v["N_gaussians"]
        print(f"  iter {k}: PSNR={psnr:.2f} SSIM={ssim:.4f} N={n}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== CANDIDATE C timing ===")
try:
    d = json.load(open(candidate_timing_path))
    for k, v in d.items():
        if v["total_ms_mean"] > 0:
            t = v["total_ms_mean"]
            f = v["fwd_ms_mean"]
            b = v["bwd_ms_mean"]
            print(f"  {k}: total={t:.1f}ms fwd={f:.1f}ms bwd={b:.1f}ms")
except Exception as e:
    print(f"  Error: {e}")

# Also check bicycle and garden
for scene in ["bicycle", "garden"]:
    print(f"\n=== {scene.upper()} baseline (existing) ===")
    bp = os.path.expanduser(f"~/3dgs-renderer-benchmark/results/reference_v1/s22/{scene}/training_metrics.json")
    try:
        d = json.load(open(bp))
        for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
            psnr = v["psnr"]
            ssim = v["ssim"]
            n = v["N_gaussians"]
            print(f"  iter {k}: PSNR={psnr:.2f} SSIM={ssim:.4f} N={n}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n=== {scene.upper()} candidate_c ===")
    cp = f"/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/{scene}/candidate_c/training_metrics.json"
    try:
        d = json.load(open(cp))
        for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
            psnr = v["psnr"]
            ssim = v["ssim"]
            n = v["N_gaussians"]
            print(f"  iter {k}: PSNR={psnr:.2f} SSIM={ssim:.4f} N={n}")
    except Exception as e:
        print(f"  Error: {e}")
