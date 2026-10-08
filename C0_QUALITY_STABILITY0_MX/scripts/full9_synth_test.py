#!/usr/bin/env python3
"""Full-9 synthetic validation. Monkeypatches roots, generates 9 synthetic
pairs (seed42 + seed43/44) matching the seed42 degradation pattern, runs the
complete pipeline, confirms a terminal verdict + report render. Does NOT touch
real run outputs."""
import os, json, importlib.util
import numpy as np

T = "/dev/shm/c0_qs_full9test"
RAW4344 = os.path.join(T, "runs4344")   # seed43/44 synthetic root
SEED42 = os.path.join(T, "runs42")     # seed42 synthetic root (overrides causal root)
os.makedirs(T, exist_ok=True)
for d in (RAW4344, SEED42):
    os.makedirs(d, exist_ok=True)

SCENES = ["room", "bicycle", "garden"]
rng = np.random.default_rng(7)

# synthetic base quality (30K) per scene, with C0 degrading like seed42 causal
base_psnr = {"room": 32.5, "bicycle": 26.3, "garden": 27.8}
base_ssim = {"room": 0.92, "bicycle": 0.83, "garden": 0.86}
base_lpips = {"room": 0.30, "bicycle": 0.25, "garden": 0.24}
# seed42 mean degradation was ~ -0.2056 dB; replicate that direction
delta = -0.20

def write(root, sc, sd, arm):
    d = os.path.join(root, f"{sc}_s{sd}_{arm}")
    os.makedirs(d, exist_ok=True)
    bp, bs, bl = base_psnr[sc], base_ssim[sc], base_lpips[sc]
    jitter = rng.normal(0, 0.03)
    if arm == "a0":
        psnr = bp + jitter
        ssim = bs + rng.normal(0, 0.002)
        lpips = bl + rng.normal(0, 0.005)
        wall = 1000.0 + rng.normal(0, 20)
    else:
        psnr = bp + delta + jitter
        ssim = bs - 0.005 + rng.normal(0, 0.002)
        lpips = bl + 0.008 + rng.normal(0, 0.005)
        wall = 985.0 + rng.normal(0, 20)
    q = {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips),
         "n_gaussians": int(800000 + rng.normal(0, 40000)), "wall_time_s": float(wall)}
    json.dump(q, open(os.path.join(d, "quality.json"), "w"))
    json.dump({"total_s": float(wall)}, open(os.path.join(d, "timing.json"), "w"))
    json.dump({"final_N": q["n_gaussians"]}, open(os.path.join(d, "results.json"), "w"))

for sd in (42, 43, 44):
    for sc in SCENES:
        for arm in ("a0", "c0"):
            # route seed42 to SEED42 root, seed43/44 to RAW4344 (mimic real paths)
            write(SEED42 if sd == 42 else RAW4344, sc, sd, arm)

# copy frozen gates/protocol/source into T so derived/report find them
import shutil
SRC = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX"
for f in ("QUALITY_FROZEN_GATES.json", "QUALITY_PROTOCOL_IDENTITY.json",
          "QUALITY_SOURCE_IDENTITY.json", "QUALITY_SEED42_REUSE.json"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(T, f))

spec = importlib.util.spec_from_file_location("agg", os.path.join(SRC, "scripts", "aggregate_quality.py"))
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)
agg.SEED42_ROOT = SEED42
agg.SEED4344_ROOT = RAW4344
agg.CAMPAIGN = T
# rebind path helpers to use patched roots
agg.quality_path = lambda sc, sd, arm: os.path.join(SEED42 if sd == 42 else RAW4344, f"{sc}_s{sd}_{arm}", "quality.json")
agg.results_path = lambda sc, sd, arm: os.path.join(SEED42 if sd == 42 else RAW4344, f"{sc}_s{sd}_{arm}", "results.json")
agg.main()

v = json.load(open(os.path.join(T, "QUALITY_VERDICT.json")))
print("FULL9_VERDICT:", v["verdict"])
print("FULL9_n_pairs:", json.load(open(os.path.join(T, "QUALITY_30K_METRICS.json")))["n_pairs"])
print("FULL9_mean_dPSNR:", v["criteria"]["mean_d_psnr"])
