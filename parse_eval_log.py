#!/usr/bin/env python3
"""Parse unified eval log and save results — fixed scale detection using checkpoint paths."""
import json, re, sys
from pathlib import Path

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/completion_batch/unified_eval.log")
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/completion_batch/unified_metrics.json")

log = log_path.read_text()

# Parse: "Evaluating: <filename> (scene=<scene>)" ... "PSNR=... SSIM=... LPIPS=... N=..."
# But we also need the checkpoint path to distinguish room 1.0 vs 0.5
# The log shows "Evaluating: iter_30000.pt (scene=room)" but we need context.
# Actually, the eval script prints "Evaluating: <filename>" where filename is just the basename.
# We need to match based on the order from the script: room 1.0, room 0.75, room 0.5, garden 1.0, ...

# Better approach: parse the "Loading: " or checkpoint path from the log
# The eval script prints "Evaluating: <ckpt_path.name>" so we lose the full path.
# But we know the order from the script's ckpt_dirs dict.

# Known order from c42_unified_eval.py:
expected_order = [
    ("room", "1.0", "iter_30000.pt"),
    ("room", "0.75", "room_s0.75_iter_30000.pt"),
    ("room", "0.5", "iter_30000.pt"),
    ("garden", "1.0", "baseline_iter_30000.pt"),
    ("garden", "0.75", "s0.750_iter_30000.pt"),
    ("garden", "0.625", "garden_s0.625_iter_30000.pt"),
    ("garden", "0.5", "c42_iter_30000.pt"),
    ("bicycle", "1.0", "baseline_iter_30000.pt"),
    ("bicycle", "0.75", "bicycle_s0.75_iter_30000.pt"),
    ("bicycle", "0.625", "s0.625_iter_30000.pt"),
    ("bicycle", "0.5", "c42_iter_30000.pt"),
]

# Extract all eval results in order
pattern = r"PSNR=([\d.]+)\s+SSIM=([\d.]+)\s+LPIPS=([\d.]+)\s+N=([\d,]+)"
results_matches = re.findall(pattern, log)

print(f"Found {len(results_matches)} eval results, expected {len(expected_order)}")

results = {"experiment": "C42 Unified Metric Evaluation", "checkpoints": {}}

for i, (psnr, ssim, lpips, n_gaussians) in enumerate(results_matches):
    if i >= len(expected_order):
        break
    scene, scale, filename = expected_order[i]
    key = f"{scene}_s{scale}"
    results["checkpoints"][key] = {
        "scene": scene,
        "scale": scale,
        "checkpoint": filename,
        "n_gaussians": int(n_gaussians.replace(",", "")),
        "n_cameras": "all",
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(lpips),
        "status": "EVALUATED",
    }
    print(f"  {key}: PSNR={float(psnr):.4f} SSIM={float(ssim):.4f} LPIPS={float(lpips):.4f} N={int(n_gaussians.replace(',','')):,}")

# Canonical values for discrepancy check (from training-time eval on 10 cameras)
canonical = {
    "room_s1.0": {"psnr": 32.30, "ssim": 0.9263},
    "room_s0.5": {"psnr": 32.54, "ssim": 0.9185},
    "garden_s1.0": {"psnr": 29.63, "ssim": 0.8994, "lpips": 0.1400},
    "garden_s0.75": {"psnr": 29.28, "ssim": 0.8811, "lpips": 0.1613},
    "garden_s0.5": {"psnr": 29.23, "ssim": 0.8770, "lpips": 0.1655},
    "bicycle_s1.0": {"psnr": 26.47, "ssim": 0.8383, "lpips": 0.2436},
    "bicycle_s0.625": {"psnr": 26.14, "ssim": 0.8084, "lpips": 0.2750},
    "bicycle_s0.5": {"psnr": 26.43, "ssim": 0.8170, "lpips": 0.2633},
}

results["discrepancies"] = {}
for key, canon in canonical.items():
    if key in results["checkpoints"]:
        recomputed = results["checkpoints"][key]
        disc = {}
        for metric in ["psnr", "ssim", "lpips"]:
            if metric in canon and metric in recomputed:
                d = recomputed[metric] - canon[metric]
                disc[metric] = {"canonical": canon[metric], "recomputed": recomputed[metric], "delta": round(d, 4)}
        results["discrepancies"][key] = disc

results["provenance"] = {
    "evaluation_code": "unified — same PSNR/SSIM/LPIPS pipeline for all checkpoints",
    "ssim_implementation": "SepSSIM window=11 sigma=1.5 C1=(0.01)^2 C2=(0.03)^2",
    "lpips_net": "vgg",
    "all_cameras": True,
    "gaussian_model": "baseline/reference_v1/gaussian_model.py",
    "note": "Parsed from eval log; canonical values are from 10-camera training-time eval, unified values from all-camera re-evaluation",
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_path}")

# Print discrepancy summary
print("\n=== DISCREPANCY SUMMARY (unified all-camera vs canonical 10-camera) ===")
for key, disc in results["discrepancies"].items():
    for metric, vals in disc.items():
        flag = " ***" if abs(vals["delta"]) > 0.01 else " (OK)"
        print(f"  {key} {metric}: canon={vals['canonical']} unified={vals['recomputed']} delta={vals['delta']:+.4f}{flag}")

# Print unified SSIM deltas
print("\n=== UNIFIED SSIM DELTAS (vs scale=1.0, all cameras) ===")
for scene in ["room", "garden", "bicycle"]:
    m1 = results["checkpoints"].get(f"{scene}_s1.0")
    if not m1:
        continue
    for scale in ["0.75", "0.625", "0.5"]:
        ms = results["checkpoints"].get(f"{scene}_s{scale}")
        if ms:
            d = ms["ssim"] - m1["ssim"]
            d_psnr = ms["psnr"] - m1["psnr"]
            d_lpips = ms.get("lpips", 0) - m1.get("lpips", 0) if m1.get("lpips") else None
            print(f"  {scene} s={scale}: dSSIM={d:+.4f} dPSNR={d_psnr:+.4f} dLPIPS={d_lpips:+.4f}" if d_lpips is not None else f"  {scene} s={scale}: dSSIM={d:+.4f} dPSNR={d_psnr:+.4f}")
