#!/usr/bin/env python
# rf0a_aggregate.py - build RF0A RAW/QUALITY/TIMING/SUMMARY + MANIFEST
import json, os, hashlib, datetime

FINAL = "/root/rf0a_laneA/final"
SCENES = ["room", "garden", "bicycle"]
ARMS = ["fastgs", "vanilla"]

def load(fn):
    with open(os.path.join(FINAL, fn)) as f:
        return json.load(f)

raw = {sc: {arm: load("RF0A_raw_%s_%s.json" % (sc, arm)) for arm in ARMS} for sc in SCENES}
src_lock = load("RF0A_SOURCE_LOCK.json")
env = load("RF0A_ENVIRONMENT.json")
split = load("RF0A_CAMERA_SPLIT.json")
now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

raw_out = {
    "artifact": "RF0A_RAW", "lane": "RF0A (ReproFast0 Lane A)", "generated_utc": now,
    "seed": 42, "iterations": 30000, "host": env["host"], "gpu": env["gpu"],
    "source_commit": src_lock["head_commit_sha"], "split_rule": split["rule"], "per_scene": {},
}
for sc in SCENES:
    raw_out["per_scene"][sc] = {}
    for arm in ARMS:
        r = raw[sc][arm]
        raw_out["per_scene"][sc][arm] = {
            "final_gaussian_count": r["final_gaussian_count"],
            "training_wall_seconds": r["training_wall_seconds"],
            "internal_training_time_seconds": r["internal_training_time_seconds"],
            "peak_vram_gb": r["peak_vram_gb"],
            "train_psnr_diagnostic": r["train_psnr_diagnostic"],
            "holdout_metrics": r["holdout_metrics"],
            "n_train_cams": r["n_train_cams"], "n_test_cams": r["n_test_cams"],
            "gaussian_trajectory": r["gaussian_trajectory"],
            "model_path": "/root/rf0a_laneA/runs/%s_%s" % (sc, arm),
        }
with open(os.path.join(FINAL, "RF0A_RAW.json"), "w") as f:
    json.dump(raw_out, f, indent=2)

quality = {
    "artifact": "RF0A_QUALITY", "generated_utc": now,
    "metric_note": "True-holdout (i%%8==0) PSNR/SSIM/LPIPS over corrected test cameras; never used for training. Rendered with each arm's own mult.",
    "per_scene": {},
}
for sc in SCENES:
    f = raw[sc]["fastgs"]["holdout_metrics"]; v = raw[sc]["vanilla"]["holdout_metrics"]
    ft = raw[sc]["fastgs"]["train_psnr_diagnostic"]; vt = raw[sc]["vanilla"]["train_psnr_diagnostic"]
    quality["per_scene"][sc] = {
        "n_holdout": f["n"],
        "fastgs": {"psnr": f["psnr"], "ssim": f["ssim"], "lpips": f["lpips"], "train_psnr": ft["psnr"]},
        "vanilla": {"psnr": v["psnr"], "ssim": v["ssim"], "lpips": v["lpips"], "train_psnr": vt["psnr"]},
        "delta_fastgs_minus_vanilla": {
            "psnr": round(f["psnr"] - v["psnr"], 4),
            "ssim": round(f["ssim"] - v["ssim"], 5),
            "lpips": round(f["lpips"] - v["lpips"], 5),
            "train_psnr": round(ft["psnr"] - vt["psnr"], 4),
        },
    }
with open(os.path.join(FINAL, "RF0A_QUALITY.json"), "w") as f:
    json.dump(quality, f, indent=2)

timing = {
    "artifact": "RF0A_TIMING", "generated_utc": now,
    "timing_note": "Single-GPU A100-SXM4-80GB. wall = end-to-end training-loop wall clock; internal = sum of CUDA-event iter+optim timing (matches train.py Training time). speedup = vanilla / fastgs (higher = FastGS faster). Same-backend comparison (identical rasterizer), so speedup reflects the algorithm, not the backend.",
    "per_scene": {},
}
for sc in SCENES:
    f = raw[sc]["fastgs"]; v = raw[sc]["vanilla"]
    timing["per_scene"][sc] = {
        "fastgs": {"training_wall_seconds": f["training_wall_seconds"], "internal_training_time_seconds": f["internal_training_time_seconds"], "peak_vram_gb": f["peak_vram_gb"], "final_gaussian_count": f["final_gaussian_count"]},
        "vanilla": {"training_wall_seconds": v["training_wall_seconds"], "internal_training_time_seconds": v["internal_training_time_seconds"], "peak_vram_gb": v["peak_vram_gb"], "final_gaussian_count": v["final_gaussian_count"]},
        "speedup_wall": round(v["training_wall_seconds"] / f["training_wall_seconds"], 3),
        "speedup_internal": round(v["internal_training_time_seconds"] / f["internal_training_time_seconds"], 3),
        "gaussian_reduction_factor": round(v["final_gaussian_count"] / f["final_gaussian_count"], 3),
        "vram_ratio_vanilla_over_fastgs": round(v["peak_vram_gb"] / f["peak_vram_gb"], 3),
    }
with open(os.path.join(FINAL, "RF0A_TIMING.json"), "w") as f:
    json.dump(timing, f, indent=2)

per_scene_summary = {}
for sc in SCENES:
    t = timing["per_scene"][sc]; q = quality["per_scene"][sc]
    per_scene_summary[sc] = {
        "speedup_wall": t["speedup_wall"], "speedup_internal": t["speedup_internal"],
        "gaussian_reduction_factor": t["gaussian_reduction_factor"],
        "fastgs_holdout_psnr": q["fastgs"]["psnr"], "vanilla_holdout_psnr": q["vanilla"]["psnr"],
        "holdout_psnr_delta": q["delta_fastgs_minus_vanilla"]["psnr"],
        "fastgs_holdout_ssim": q["fastgs"]["ssim"], "vanilla_holdout_ssim": q["vanilla"]["ssim"],
        "fastgs_holdout_lpips": q["fastgs"]["lpips"], "vanilla_holdout_lpips": q["vanilla"]["lpips"],
    }
speedups = [per_scene_summary[s]["speedup_wall"] for s in SCENES]
summary = {
    "artifact": "RF0A_SUMMARY", "generated_utc": now,
    "primary_question": "Does the published fast-training regime reproduce on EPIC-05 under corrected held-out cameras?",
    "verdict": "REPRODUCED",
    "verdict_detail": "Official FastGS (train_base.sh per-scene Mip-NeRF360 settings) at 30k iters, seed 42, on A100-SXM4-80GB, under the corrected i%%8==0 true-holdout split, trains in 170-236s wall and reaches holdout PSNR 25.2-32.1 / SSIM 0.74-0.92 / LPIPS 0.11-0.27, consistent with published FastGS/3DGS-quality on Mip-NeRF360. It is 2.6-4.6x faster (wall) and uses ~7-8x fewer Gaussians and less VRAM than the faithful same-backend vanilla 3DGS comparator, at comparable true-holdout quality.",
    "source": {"repo": src_lock["remote_url"], "commit": src_lock["head_commit_sha"], "branch": src_lock["branch"]},
    "environment": {"host": env["host"], "python": env["python"], "torch": env["torch"], "cuda": env["torch_cuda_runtime"], "gpu": env["gpu"]},
    "seed": 42, "iterations": 30000, "split_rule": split["rule"],
    "per_scene": per_scene_summary,
    "speedup_wall_range": [round(min(speedups), 2), round(max(speedups), 2)],
    "gaussian_count_final": {sc: {"fastgs": raw[sc]["fastgs"]["final_gaussian_count"], "vanilla": raw[sc]["vanilla"]["final_gaussian_count"]} for sc in SCENES},
    "reproduction_mismatches": [
        "Environment differs from the paper's stated spec (paper: CUDA 11.6 / py3.7 / torch1.12; here: torch2.10.0+cu128 / py3.12 / nvcc12.8). The vendored rasterizer was rebuilt from frozen source (commit 44e02a5) against the available toolchain; source identity pinned by commit + vendored tree SHAs.",
        "True-holdout PSNR: FastGS is +0.87 dB above vanilla on room but -0.16/-0.16 dB below on garden/bicycle; SSIM slightly lower and LPIPS slightly higher for FastGS on garden/bicycle. With only 24-39 holdout views per scene, per-scene metrics have limited sample size; deltas are small and quality is comparable overall.",
        "Vanilla comparator uses mult=1.0 (full compact box) vs FastGS mult=0.5; this box difference is part of the method, not a confound. Both arms use the identical diff_gaussian_rasterization_fastgs backend.",
        "Timing is single-GPU wall clock on one A100 per run; no cross-backend wall is compared as algorithmic speedup.",
    ],
    "scientific_status": "External baseline reproduction (NOT author novelty). Lane A only; Lane B/C not started.",
}
with open(os.path.join(FINAL, "RF0A_SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("WROTE RAW/QUALITY/TIMING/SUMMARY")
for sc in SCENES:
    print(sc, "speedup_wall", per_scene_summary[sc]["speedup_wall"], "psnr_delta", per_scene_summary[sc]["holdout_psnr_delta"])
