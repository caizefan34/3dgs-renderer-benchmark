#!/usr/bin/env python
# rf0a_report.py - generate RF0A_REPORT.md + MANIFEST.sha256
import json, os, hashlib

FINAL = "/root/rf0a_laneA/final"
SCENES = ["room", "garden", "bicycle"]

def load(fn):
    with open(os.path.join(FINAL, fn)) as f:
        return json.load(f)

raw = load("RF0A_RAW.json"); quality = load("RF0A_QUALITY.json")
timing = load("RF0A_TIMING.json"); summary = load("RF0A_SUMMARY.json")
src = load("RF0A_SOURCE_LOCK.json"); env = load("RF0A_ENVIRONMENT.json")
split = load("RF0A_CAMERA_SPLIT.json"); config = load("RF0A_CONFIG.json")

def g(x, n=2):
    return ("%."+str(n)+"f") % x

L = []
L.append("# ReproFast0 Lane A - Official FastGS Reproduction Report")
L.append("")
L.append("**Lane:** RF0A (ReproFast0 Lane A)  ")
L.append("**Generated (UTC):** %s  " % summary["generated_utc"])
L.append("**Host:** %s  " % env["host"])
L.append("**GPU:** %s  " % env["gpu"])
L.append("**Verdict:** %s  " % summary["verdict"])
L.append("")
L.append("## 1. Primary Scientific Question")
L.append("")
L.append("> " + summary["primary_question"])
L.append("")
L.append("**Answer: %s.** " % summary["verdict"])
L.append(summary["verdict_detail"])
L.append("")
L.append("## 2. Source Identity (frozen)")
L.append("")
L.append("| Field | Value |")
L.append("|---|---|")
L.append("| Remote URL | %s |" % src["remote_url"])
L.append("| Branch | %s |" % src["branch"])
L.append("| Commit SHA | `%s` |" % src["head_commit_sha"])
L.append("| git describe | %s |" % src["git_describe"])
L.append("| Working tree (tracked source) clean | yes (only untracked build artifacts: __pycache__, input.ply, cameras.json) |")
L.append("| Git submodules | NONE - rasterizer/simple-knn/fused-ssim vendored into main tree |")
L.append("| Vendored tree SHAs | rasterizer `%s`, simple-knn `%s`, fused-ssim `%s` |" % (
    src["vendored_component_tree_shas"]["diff_gaussian_rasterization_fastgs"][:12],
    src["vendored_component_tree_shas"]["simple_knn"][:12],
    src["vendored_component_tree_shas"]["fused_ssim"][:12]))
L.append("")
L.append("Built CUDA extension (rasterizer) `_C.so` sha256: `%s`" % (
    src["built_extensions"]["diff_gaussian_rasterization_fastgs"]["built_so_sha256"][0]["sha256"]))
L.append("")
L.append("## 3. Environment")
L.append("")
L.append("| Component | Value |")
L.append("|---|---|")
L.append("| Python | %s |" % env["python"])
L.append("| PyTorch | %s (CUDA runtime %s) |" % (env["torch"], env["torch_cuda_runtime"]))
L.append("| nvcc | %s |" % env["nvcc"])
L.append("| gcc | %s |" % env["gcc"])
L.append("| GPU | %s |" % env["gpu"])
L.append("")
L.append("Note: the FastGS `environment.yml` pins CUDA 11.6 / py3.7 / torch1.12, which is unavailable on EPIC-05. The vendored rasterizer was rebuilt from the frozen source against torch %s / nvcc %s. Source identity is pinned by commit + vendored tree SHAs (section 2), so the algorithm is unchanged; only the build toolchain differs." % (env["torch"], env["torch_cuda_runtime"]))
L.append("")
L.append("## 4. Data & Corrected True-Holdout Split")
L.append("")
L.append("Rule: %s" % split["rule"])
L.append("")
L.append("| Scene | Total cams | Train | Holdout (test) | cameras_extent radius |")
L.append("|---|---|---|---|---|")
for sc in SCENES:
    s = split["scenes"][sc]
    L.append("| %s | %d | %d | %d | %s |" % (sc, s["total_cameras"], s["train_count"], s["test_count"], g(s["cameras_extent_radius"],4)))
L.append("")
L.append("Holdout cameras (i %% 8 == 0) are rendered for PSNR/SSIM/LPIPS only; training samples exclusively from `getTrainCameras()`. No holdout camera is ever used for optimization.")
L.append("")
L.append("## 5. Arms & Comparator Choice")
L.append("")
L.append("- **Arm A (fastgs):** official FastGS, `train_base.sh` per-scene Mip-NeRF360 settings, verbatim `train.py` loop, `mult=0.5`, `optimizer_type=default`, seed 42, 30k iters.")
L.append("- **Arm B (vanilla):** faithful original INRIA 3DGS on the **same** `diff_gaussian_rasterization_fastgs` backend (same `Scene`/`GaussianModel`), differing only in: original mean-gradient clone/split densification, per-iteration dense Adam, `mult=1.0` (full compact box), no `final_prune_fastgs`. Canonical 3DGS Mip-NeRF360 hyperparameters.")
L.append("")
L.append("**Comparator justification:** the official `fastgs/FastGS` repo exposes **no** in-repo vanilla baseline (grep for vanilla `render`/`densify_and_prune` is empty; `optimizer_type` only selects `default` vs `sparse_adam`, both FastGS densification). Per the task spec, the faithful vanilla/comparator is the original INRIA 3DGS the repo is forked from (`LICENSE_ORIGINAL.md`), run on the identical rasterizer backend so that wall-time differences reflect the algorithm, not the backend.")
L.append("")
L.append("## 6. Per-Scene Results")
L.append("")
L.append("### 6.1 Timing & model size")
L.append("")
L.append("| Scene | Arm | Wall (s) | Internal (s) | Peak VRAM (GB) | Final Gaussians |")
L.append("|---|---|---|---|---|---|")
for sc in SCENES:
    for arm in ["fastgs","vanilla"]:
        t = timing["per_scene"][sc][arm]
        L.append("| %s | %s | %s | %s | %s | %d |" % (sc, arm, g(t["training_wall_seconds"],1), g(t["internal_training_time_seconds"],1), g(t["peak_vram_gb"],2), t["final_gaussian_count"]))
L.append("")
L.append("### 6.2 Speedup (vanilla / fastgs, same backend)")
L.append("")
L.append("| Scene | Speedup (wall) | Speedup (internal) | Gaussian reduction | VRAM ratio (van/fast) |")
L.append("|---|---|---|---|---|")
for sc in SCENES:
    t = timing["per_scene"][sc]
    L.append("| %s | %sx | %sx | %sx fewer | %sx |" % (sc, g(t["speedup_wall"],2), g(t["speedup_internal"],2), g(t["gaussian_reduction_factor"],1), g(t["vram_ratio_vanilla_over_fastgs"],2)))
L.append("")
L.append("### 6.3 True-holdout quality (i %% 8 == 0)")
L.append("")
L.append("| Scene | n | Arm | PSNR | SSIM | LPIPS | Train PSNR (diag) |")
L.append("|---|---|---|---|---|---|---|")
for sc in SCENES:
    for arm in ["fastgs","vanilla"]:
        qq = quality["per_scene"][sc][arm]
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (sc, quality["per_scene"][sc]["n_holdout"], arm, g(qq["psnr"],3), g(qq["ssim"],4), g(qq["lpips"],4), g(qq["train_psnr"],3)))
L.append("")
L.append("### 6.4 Quality delta (FastGS - vanilla, holdout)")
L.append("")
L.append("| Scene | dPSNR | dSSIM | dLPIPS | dTrainPSNR |")
L.append("|---|---|---|---|---|")
for sc in SCENES:
    d = quality["per_scene"][sc]["delta_fastgs_minus_vanilla"]
    L.append("| %s | %s | %s | %s | %s |" % (sc, "%+.3f"%d["psnr"], "%+.4f"%d["ssim"], "%+.4f"%d["lpips"], "%+.3f"%d["train_psnr"]))
L.append("")
L.append("## 7. Interpretation")
L.append("")
L.append("- **Speed regime reproduces:** FastGS trains in %s-%s s wall (30k iters) vs vanilla %s-%s s, a %sx-%sx speedup on the same A100 backend." % (
    g(min(timing["per_scene"][s]["fastgs"]["training_wall_seconds"] for s in SCENES),0),
    g(max(timing["per_scene"][s]["fastgs"]["training_wall_seconds"] for s in SCENES),0),
    g(min(timing["per_scene"][s]["vanilla"]["training_wall_seconds"] for s in SCENES),0),
    g(max(timing["per_scene"][s]["vanilla"]["training_wall_seconds"] for s in SCENES),0),
    g(summary["speedup_wall_range"][0],1), g(summary["speedup_wall_range"][1],1)))
L.append("- **Model compression:** FastGS uses ~%sx-%sx fewer Gaussians than vanilla at convergence." % (
    g(min(timing["per_scene"][s]["gaussian_reduction_factor"] for s in SCENES),1),
    g(max(timing["per_scene"][s]["gaussian_reduction_factor"] for s in SCENES),1)))
L.append("- **Quality is comparable:** holdout PSNR is within ~0.9 dB of vanilla (above on room, marginally below on garden/bicycle); SSIM/LPIPS slightly lower for FastGS on garden/bicycle. With only 24-39 holdout views per scene, per-scene deltas have limited sample size.")
L.append("- **Faster == fewer Gaussians == less VRAM**, consistent with the FastGS paper's claims.")
L.append("")
L.append("## 8. Reproduction Mismatches / Caveats")
L.append("")
for m in summary["reproduction_mismatches"]:
    L.append("- %s" % m)
L.append("")
L.append("## 9. Exact Commands")
L.append("")
L.append("All runs used `rf0a_runner.py` (verbatim official FastGS loop for arm A; faithful vanilla 3DGS for arm B) on EPIC-05, seed 42, 30k iters, `--eval` (corrected i%%8==0 holdout). Example (room, fastgs):")
L.append("")
L.append("```")
L.append("cd /root/rf0a_laneA/FastGS && export PYTHONPATH=/root/rf0a_laneA/FastGS")
L.append("CUDA_VISIBLE_DEVICES=0 python /root/rf0a_laneA/rf0a_runner.py --method fastgs \\")
L.append("  -s /root/epic05_p16rb/common_data/mipnerf360/room -i images --eval \\")
L.append("  --mult 0.5 --eval_mult 0.5 --optimizer_type default --iterations 30000 \\")
L.append("  --densification_interval 500 --highfeature_lr 0.02 --grad_abs_thresh 0.0008 \\")
L.append("  --out /root/rf0a_laneA/final/RF0A_raw_room_fastgs.json")
L.append("```")
L.append("")
L.append("Vanilla (room): same command with `--method vanilla --mult 1.0 --eval_mult 1.0` and the canonical 3DGS densification hyperparams (`--densification_interval 100 --densify_grad_threshold 0.0002 --percent_dense 0.01 --opacity_lr 0.05 ...`).")
L.append("")
L.append("## 10. Deliverables")
L.append("")
L.append("`RF0A_ENVIRONMENT.json`, `RF0A_SOURCE_LOCK.json`, `RF0A_CAMERA_SPLIT.json`, `RF0A_CONFIG.json`, `RF0A_RAW.json`, `RF0A_QUALITY.json`, `RF0A_TIMING.json`, `RF0A_SUMMARY.json`, `RF0A_REPORT.md`, `MANIFEST.sha256`.")
L.append("")
L.append("Verify: `cd /root/rf0a_laneA/final && sha256sum -c MANIFEST.sha256`")
L.append("")
L.append("## 11. Scientific Status")
L.append("")
L.append(summary["scientific_status"])
L.append("")

with open(os.path.join(FINAL, "RF0A_REPORT.md"), "w") as f:
    f.write("\n".join(L))
print("WROTE RF0A_REPORT.md lines=%d" % len(L))

# ---------- MANIFEST ----------
deliverables = ["RF0A_ENVIRONMENT.json","RF0A_SOURCE_LOCK.json","RF0A_CAMERA_SPLIT.json",
    "RF0A_CONFIG.json","RF0A_RAW.json","RF0A_QUALITY.json","RF0A_TIMING.json",
    "RF0A_SUMMARY.json","RF0A_REPORT.md"]
lines = []
for fn in sorted(deliverables):
    p = os.path.join(FINAL, fn)
    h = hashlib.sha256(open(p,"rb").read()).hexdigest()
    lines.append("%s  %s" % (h, fn))
with open(os.path.join(FINAL, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE MANIFEST.sha256 with %d entries" % len(lines))
