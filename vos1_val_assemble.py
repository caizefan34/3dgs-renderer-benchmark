import os, json, glob, hashlib, math, time, platform, subprocess, statistics

RES = "/root/vos1_val_out/results"
OUT = "/root/vos1_val_out/pack"
os.makedirs(OUT, exist_ok=True)

SCENES = ["room", "garden", "bicycle"]
SEEDS = [0, 1, 42]
SCHEDS = [1, 2]  # 1=baseline, 2=VOS

def load(scene, k, seed):
    with open(os.path.join(RES, f"result_{scene}_k{k}_s{seed}.json")) as f:
        return json.load(f)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

def git_info(p):
    try: return subprocess.check_output(["git","-C",p,"rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return None

# ---------- load all ----------
R = {(s,k,sd): load(s,k,sd) for s in SCENES for k in SCHEDS for sd in SEEDS}

def ms(xs):  # mean
    return sum(xs)/len(xs) if xs else 0.0
def sds(xs):  # sample std (ddof=1)
    if len(xs) < 2: return 0.0
    m = ms(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

# ---------- 1. MULTI_SEED_RESULTS ----------
multi = {}
for s in SCENES:
    multi[s] = {}
    for k in SCHEDS:
        runs = [R[(s,k,sd)] for sd in SEEDS]
        psnr = [r["psnr"] for r in runs]
        ssim = [r["ssim"] for r in runs]
        lpips = [r["lpips"] for r in runs]
        wall = [r["wall_time_s"] for r in runs]
        opt = [r["optimizer_time_s"] for r in runs]
        upd = [r["update_count"] for r in runs]
        skip = [r["skipped_updates"] for r in runs]
        multi[s][str(k)] = {
            "seeds": SEEDS,
            "psnr_per_seed": {str(sd): R[(s,k,sd)]["psnr"] for sd in SEEDS},
            "ssim_per_seed": {str(sd): R[(s,k,sd)]["ssim"] for sd in SEEDS},
            "lpips_per_seed": {str(sd): R[(s,k,sd)]["lpips"] for sd in SEEDS},
            "psnr_mean": ms(psnr), "psnr_std": sds(psnr), "psnr_min": min(psnr), "psnr_max": max(psnr),
            "ssim_mean": ms(ssim), "ssim_std": sds(ssim),
            "lpips_mean": ms(lpips), "lpips_std": sds(lpips),
            "wall_mean": ms(wall), "wall_std": sds(wall),
            "opt_time_mean": ms(opt), "opt_time_std": sds(opt),
            "update_count_mean": ms(upd), "update_count_std": sds(upd),
            "skipped_mean": ms(skip),
            "opt_fraction_mean": ms([o/w for o,w in zip(opt,wall)]),
        }

# VOS (k=2) vs baseline (k=1) delta per scene
delta = {}
for s in SCENES:
    b = multi[s]["1"]; v = multi[s]["2"]
    dpsnr = [R[(s,2,sd)]["psnr"] - R[(s,1,sd)]["psnr"] for sd in SEEDS]
    delta[s] = {
        "psnr_delta_per_seed": {str(sd): R[(s,2,sd)]["psnr"] - R[(s,1,sd)]["psnr"] for sd in SEEDS},
        "psnr_delta_mean": ms(dpsnr), "psnr_delta_std": sds(dpsnr), "psnr_delta_min": min(dpsnr),
        "ssim_delta_mean": v["ssim_mean"] - b["ssim_mean"],
        "lpips_delta_mean": v["lpips_mean"] - b["lpips_mean"],
        "update_reduction_mean": 1.0 - v["update_count_mean"]/b["update_count_mean"],
        "wall_speedup_mean": b["wall_mean"]/v["wall_mean"],
        "opt_time_reduction_mean": 1.0 - v["opt_time_mean"]/b["opt_time_mean"],
    }

multi_seed_results = {
    "config": {"scenes": SCENES, "seeds": SEEDS, "schedulers": {"1": "k=1 baseline (dense)", "2": "k=2 VOS"}, "iterations": 30000},
    "per_scene": multi,
    "vos_vs_baseline_delta": delta,
    "stability_summary": {
        "note": "std is sample std (ddof=1) across the 3 seeds. Low std = seed-stable.",
        "k1_psnr_std": {s: multi[s]["1"]["psnr_std"] for s in SCENES},
        "k2_psnr_std": {s: multi[s]["2"]["psnr_std"] for s in SCENES},
        "k1_opt_fraction_std": {s: sds([R[(s,1,sd)]["optimizer_time_s"]/R[(s,1,sd)]["wall_time_s"] for sd in SEEDS]) for s in SCENES},
    },
}

# ---------- 2. VISIBILITY_ANALYSIS ----------
vis = {}
for s in SCENES:
    vis[s] = {}
    for k in SCHEDS:
        runs = [R[(s,k,sd)]["visibility"] for sd in SEEDS]
        parts = {}
        for p in ["hot","warm","cold"]:
            parts[p] = {
                "count_mean": ms([r["partitions"][p]["count"] for r in runs]),
                "frac_mean": ms([r["partitions"][p]["frac"] for r in runs]),
                "vis_freq_mean": ms([r["partitions"][p]["vis_freq_mean"] for r in runs]),
                "grad_rms_mean": ms([r["partitions"][p]["grad_rms_mean"] for r in runs]),
                "upd_freq_mean": ms([r["partitions"][p]["upd_freq_mean"] for r in runs]),
                "skip_rate_mean": ms([r["partitions"][p]["skip_rate"] for r in runs]),
            }
        # histogram (20 bins) averaged
        nbins = len(runs[0]["vis_freq_histogram"])
        hist = [ms([r["vis_freq_histogram"][i] for r in runs]) for i in range(nbins)]
        # correlation binned averaged
        nbn = len(runs[0]["correlation_binned"])
        binned = []
        for i in range(nbn):
            binned.append({
                "bin_center": runs[0]["correlation_binned"][i]["bin_center"],
                "count_mean": ms([r["correlation_binned"][i]["count"] for r in runs]),
                "grad_rms_mean": ms([r["correlation_binned"][i]["grad_rms_mean"] for r in runs]),
                "vis_freq_mean": ms([r["correlation_binned"][i]["vis_freq_mean"] for r in runs]),
            })
        vis[s][str(k)] = {
            "vis_freq_mean": ms([r["vis_freq_mean"] for r in runs]),
            "vis_freq_median": ms([r["vis_freq_median"] for r in runs]),
            "grad_rms_mean": ms([r["grad_rms_mean"] for r in runs]),
            "pearson_vis_grad_mean": ms([r["pearson_vis_grad"] for r in runs]),
            "pearson_vis_grad_std": sds([r["pearson_vis_grad"] for r in runs]),
            "spearman_vis_grad_mean": ms([r["spearman_vis_grad"] for r in runs]),
            "spearman_vis_grad_std": sds([r["spearman_vis_grad"] for r in runs]),
            "partitions": parts,
            "vis_freq_histogram": hist,
            "correlation_binned": binned,
        }

visibility_analysis = {
    "definitions": {
        "vis_freq": "fraction of a Gaussian's lifetime iterations it was visible (radii>0)",
        "grad_rms": "RMS of the 3D position-gradient magnitude over the Gaussian's lifetime",
        "hot": "vis_freq > 0.5", "warm": "0.1 <= vis_freq <= 0.5", "cold": "vis_freq < 0.1",
    },
    "per_scene": vis,
    "key_findings": [
        "Visibility frequency and gradient magnitude are positively correlated (Spearman ~0.50-0.77): rarely-visible (cold) Gaussians carry ~5x smaller gradients than hot Gaussians.",
        "The k=2 scheduler preferentially skips cold Gaussians (skip ~48%) over hot (skip ~21-31%), i.e. it skips the lowest-gradient updates first.",
        "Visibility distribution is scheduler-independent (k=1 vs k=2 nearly identical), confirming visibility is a scene property, not a scheduler artifact.",
    ],
}

# ---------- 3. UPDATE_REDUCTION ----------
upd_red = {}
for s in SCENES:
    b = multi[s]["1"]; v = multi[s]["2"]
    upd_red[s] = {
        "baseline_update_count_mean": b["update_count_mean"],
        "vos_update_count_mean": v["update_count_mean"],
        "vos_skipped_mean": v["skipped_mean"],
        "update_reduction_mean": 1.0 - v["update_count_mean"]/b["update_count_mean"],
        "update_reduction_per_seed": {str(sd): 1.0 - R[(s,2,sd)]["update_count"]/R[(s,1,sd)]["update_count"] for sd in SEEDS},
        "update_fraction_vos": v["update_count_mean"]/(v["update_count_mean"]+v["skipped_mean"]),
    }
update_reduction = {
    "per_scene": upd_red,
    "update_reduction_geomean": math.exp(ms([math.log(upd_red[s]["update_reduction_mean"]) for s in SCENES])),
    "note": "Update reduction = 1 - (k=2 update count)/(k=1 update count), averaged over seeds. Logical optimizer-workload reduction; wall-clock is NOT reduced (kernel O(N), see AMDAHL).",
}

# ---------- 4. AMDAHL ----------
amdahl_per_scene = {}
for s in SCENES:
    fracs = [R[(s,1,sd)]["optimizer_time_s"]/R[(s,1,sd)]["wall_time_s"] for sd in SEEDS]
    amdahl_per_scene[s] = {
        "opt_fraction_mean": ms(fracs), "opt_fraction_std": sds(fracs),
        "max_wall_speedup_if_opt_eliminated": 1.0/(1.0-ms(fracs)),
    }
opt_frac_mean = ms([amdahl_per_scene[s]["opt_fraction_mean"] for s in SCENES])
amdahl = {
    "optimizer_fraction_of_wall_per_scene": amdahl_per_scene,
    "optimizer_fraction_mean": opt_frac_mean,
    "max_wall_speedup_if_optimizer_fully_eliminated": 1.0/(1.0-opt_frac_mean),
    "target_speedup": 1.2,
    "speedup_gate_reachable_by_optimizer_only": (1.0/(1.0-opt_frac_mean)) >= 1.2,
    "measured_wall_speedup_k2_geomean": math.exp(ms([math.log(delta[s]["wall_speedup_mean"]) for s in SCENES])),
    "measured_opt_time_reduction_k2_mean": ms([delta[s]["opt_time_reduction_mean"] for s in SCENES]),
    "note": "SparseGaussianAdam.step is O(N) regardless of the visibility mask (persistent asystep accumulator + full-N pass), so cutting update count does not cut optimizer time. Render+backward dominate wall time, capping any optimizer-only speedup below 1.2x.",
}

# ---------- 5. FIGURE_DATA ----------
fig = {
    "multi_seed_psnr": {
        "scenes": SCENES, "seeds": SEEDS,
        "k1": {s: [R[(s,1,sd)]["psnr"] for sd in SEEDS] for s in SCENES},
        "k2": {s: [R[(s,2,sd)]["psnr"] for sd in SEEDS] for s in SCENES},
        "k1_mean": {s: multi[s]["1"]["psnr_mean"] for s in SCENES},
        "k1_std": {s: multi[s]["1"]["psnr_std"] for s in SCENES},
        "k2_mean": {s: multi[s]["2"]["psnr_mean"] for s in SCENES},
        "k2_std": {s: multi[s]["2"]["psnr_std"] for s in SCENES},
    },
    "update_reduction": {
        "scenes": SCENES,
        "reduction_mean": {s: upd_red[s]["update_reduction_mean"] for s in SCENES},
        "reduction_per_seed": {s: [upd_red[s]["update_reduction_per_seed"][str(sd)] for sd in SEEDS] for s in SCENES},
    },
    "visibility_histogram": {
        "scenes": SCENES, "schedulers": [str(k) for k in SCHEDS],
        "bin_edges": [i/20.0 for i in range(21)],
        "counts": {s: {str(k): vis[s][str(k)]["vis_freq_histogram"] for k in SCHEDS} for s in SCENES},
    },
    "visibility_gradient_correlation": {
        "scenes": SCENES,
        "pearson": {s: {str(k): vis[s][str(k)]["pearson_vis_grad_mean"] for k in SCHEDS} for s in SCENES},
        "spearman": {s: {str(k): vis[s][str(k)]["spearman_vis_grad_mean"] for k in SCHEDS} for s in SCENES},
        "binned": {s: {str(k): vis[s][str(k)]["correlation_binned"] for k in SCHEDS} for s in SCENES},
    },
    "hot_warm_cold": {
        "scenes": SCENES,
        "partitions": ["hot","warm","cold"],
        "frac": {s: {str(k): {p: vis[s][str(k)]["partitions"][p]["frac_mean"] for p in ["hot","warm","cold"]} for k in SCHEDS} for s in SCENES},
        "grad_rms": {s: {str(k): {p: vis[s][str(k)]["partitions"][p]["grad_rms_mean"] for p in ["hot","warm","cold"]} for k in SCHEDS} for s in SCENES},
        "skip_rate": {s: {str(k): {p: vis[s][str(k)]["partitions"][p]["skip_rate_mean"] for p in ["hot","warm","cold"]} for k in SCHEDS} for s in SCENES},
    },
    "amdahl": {
        "scenes": SCENES,
        "opt_fraction": {s: amdahl_per_scene[s]["opt_fraction_mean"] for s in SCENES},
        "max_speedup_bound": {s: amdahl_per_scene[s]["max_wall_speedup_if_opt_eliminated"] for s in SCENES},
    },
}

# ---------- 6. SOURCE_LOCK ----------
src_files = [
    "/root/3dgs-renderer-benchmark/vos1_validate_train.py",
    "/root/3dgs-renderer-benchmark/vos1_validate_run.py",
    "/root/epic05_frozen/graphdeco-b0-pub/train.py",
    "/root/epic05_frozen/graphdeco-b0-pub/scene/gaussian_model.py",
    "/root/epic05_frozen/graphdeco-b0-pub/gaussian_renderer/__init__.py",
    "/root/epic05_frozen/graphdeco-b0-pub/submodules/diff-gaussian-rasterization/__init__.py",
]
import torch
source_lock = {
    "env": {"python": "/root/miniforge3/envs/p16_fastergs/bin/python", "torch": torch.__version__,
            "cuda": "available", "device": "NVIDIA A100-SXM4-80GB", "platform": platform.platform(),
            "graphdeco_repo": "/root/epic05_frozen/graphdeco-b0-pub", "graphdeco_commit": git_info("/root/epic05_frozen/graphdeco-b0-pub"),
            "rasterizer_install": "/root/miniforge3/envs/p16_fastergs/lib/python3.11/site-packages/diff_gaussian_rasterization"},
    "config": {"iterations": 30000, "seeds": SEEDS, "sh_degree": 3, "optimizer_type": "sparse_adam",
               "resolution": -1, "scenes": SCENES, "schedulers": {"1": "k=1 baseline", "2": "k=2 VOS"},
               "update_rule": "visible(radii>0): age=0, always update; invisible: age+=1, update when age%k==0",
               "unchanged": ["FasterGSFused", "rasterizer", "optimizer implementation", "densification", "pruning", "scheduler logic"],
               "only_varied": "training seed"},
    "files": [{"path": p, "sha256": sha256_file(p)} for p in src_files if os.path.exists(p)],
}

# ---------- write JSON deliverables ----------
def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, indent=2)

dump("VOS1_SOURCE_LOCK.json", source_lock)
dump("VOS1_MULTI_SEED_RESULTS.json", multi_seed_results)
dump("VOS1_VISIBILITY_ANALYSIS.json", visibility_analysis)
dump("VOS1_UPDATE_REDUCTION.json", update_reduction)
dump("VOS1_AMDAHL.json", amdahl)
dump("VOS1_FIGURE_DATA.json", fig)

# ---------- 7. REPORT ----------
L = []
L.append("# VOS1 Paper-Validation Pack — Visibility-Aware Optimizer Scheduling\n")
L.append("**Environment**: EPIC-05 A100-SXM4-80GB, torch 2.8.0+cu128, graphdeco pipeline.  ")
L.append("**Design**: 3 scenes x 2 schedulers (k=1 baseline, k=2 VOS) x 3 seeds {0,1,42} = 18 runs, 30000 iters, seed fixed per run. Only the training seed varies; FasterGSFused, rasterizer, optimizer implementation, densification, pruning, and scheduler logic are unchanged. No CUDA changes.\n")
L.append("## 1. Multi-Seed Stability\n")
L.append("| scene | sched | PSNR mean | PSNR std | SSIM mean | LPIPS mean | wall mean (s) | opt mean (s) | upd mean |")
L.append("|---|---|---|---|---|---|---|---|---|")
for s in SCENES:
    for k in SCHEDS:
        m = multi[s][str(k)]
        L.append(f"| {s} | k={k} | {m['psnr_mean']:.3f} | {m['psnr_std']:.4f} | {m['ssim_mean']:.4f} | {m['lpips_mean']:.4f} | {m['wall_mean']:.0f} | {m['opt_time_mean']:.1f} | {m['update_count_mean']/1e9:.1f}B |")
L.append("")
L.append("**VOS (k=2) minus baseline (k=1) PSNR delta, per seed:**\n")
L.append("| scene | seed 0 | seed 1 | seed 42 | mean | std |")
L.append("|---|---|---|---|---|---|")
for s in SCENES:
    d = delta[s]
    L.append(f"| {s} | {d['psnr_delta_per_seed']['0']:+.3f} | {d['psnr_delta_per_seed']['1']:+.3f} | {d['psnr_delta_per_seed']['42']:+.3f} | {d['psnr_delta_mean']:+.4f} | {d['psnr_delta_std']:.4f} |")
L.append("")
L.append("**Stability read**: k=1 is very seed-stable (PSNR std 0.005-0.040 dB). k=2 is stable on garden/bicycle (std ~0.019 dB) but **room k=2 is unstable** — seed 0 drops -0.271 dB while seeds 1/42 are ~neutral, giving room k=2 std 0.173 dB. The VOS skip pattern depends on camera visit order, so it amplifies seed sensitivity on the most fragile scene.\n")
L.append("## 2. Visibility Distribution\n")
L.append("Mean per-iteration visibility frequency (fraction of Gaussians with radii>0):\n")
L.append("| scene | k=1 vis_freq | k=2 vis_freq |")
L.append("|---|---|---|")
for s in SCENES:
    L.append(f"| {s} | {vis[s]['1']['vis_freq_mean']:.3f} | {vis[s]['2']['vis_freq_mean']:.3f} |")
L.append("")
L.append("Visibility is scheduler-independent (k=1 ~ k=2), confirming it is a scene property. The population is dominated by warm (vis_freq 0.1-0.5) Gaussians; ~9-19% are hot (>0.5) and ~27-34% are cold (<0.1).\n")
L.append("## 3. Visibility-Gradient Correlation\n")
L.append("| scene | sched | Pearson | Spearman |")
L.append("|---|---|---|---|")
for s in SCENES:
    for k in SCHEDS:
        L.append(f"| {s} | k={k} | {vis[s][str(k)]['pearson_vis_grad_mean']:.3f} | {vis[s][str(k)]['spearman_vis_grad_mean']:.3f} |")
L.append("")
L.append("Positive correlation (Spearman ~0.50-0.77): rarely-visible Gaussians carry much smaller position-gradients. Binned means show grad_rms rising from ~5e-6 (vis_freq~0.1) to ~1e-4 (vis_freq~0.9) — a ~20x increase. This is the core justification for visibility-aware skipping: invisible Gaussians contribute near-zero gradient, so their updates are near-no-ops.\n")
L.append("## 4. Hot/Warm/Cold Breakdown (k=2 VOS, seed-averaged)\n")
L.append("| scene | part | frac | vis_freq | grad_rms | upd_freq | skip_rate |")
L.append("|---|---|---|---|---|---|---|")
for s in SCENES:
    for p in ["hot","warm","cold"]:
        d = vis[s]["2"]["partitions"][p]
        L.append(f"| {s} | {p} | {d['frac_mean']*100:.1f}% | {d['vis_freq_mean']:.3f} | {d['grad_rms_mean']:.2e} | {d['upd_freq_mean']:.3f} | {d['skip_rate_mean']*100:.1f}% |")
L.append("")
L.append("The scheduler skips cold Gaussians most (~48%) and hot least (~21-31%): it removes the lowest-gradient updates first, exactly where the wasted work is.\n")
L.append("## 5. Update Reduction\n")
L.append("| scene | k=1 upd | k=2 upd | k=2 skipped | reduction |")
L.append("|---|---|---|---|---|")
for s in SCENES:
    u = upd_red[s]
    L.append(f"| {s} | {u['baseline_update_count_mean']/1e9:.1f}B | {u['vos_update_count_mean']/1e9:.1f}B | {u['vos_skipped_mean']/1e9:.1f}B | {u['update_reduction_mean']*100:.1f}% |")
L.append("")
L.append(f"Geometric-mean update reduction: **{update_reduction['update_reduction_geomean']*100:.1f}%**. This is a logical optimizer-workload reduction; it does NOT translate to wall-clock savings (see Amdahl).\n")
L.append("## 6. Amdahl Analysis\n")
L.append("| scene | opt fraction (mean) | max speedup if opt eliminated |")
L.append("|---|---|---|")
for s in SCENES:
    L.append(f"| {s} | {amdahl_per_scene[s]['opt_fraction_mean']*100:.1f}% | {amdahl_per_scene[s]['max_wall_speedup_if_opt_eliminated']:.3f}x |")
L.append("")
L.append(f"Optimizer is {opt_frac_mean*100:.1f}% of wall time on average; even fully eliminating it caps speedup at **{amdahl['max_wall_speedup_if_optimizer_fully_eliminated']:.3f}x** (< 1.2x). The SparseGaussianAdam kernel is O(N) regardless of the visibility mask, so the ~40% update reduction yields ~0% optimizer-time reduction. The 1.2x speedup gate is structurally unreachable by optimizer-only changes.\n")
L.append("## Verdict\n")
L.append("**Mechanism VALIDATED, with a stability caveat and no wall-clock benefit.** ")
L.append("- Visibility-aware skipping removes ~40% of optimizer updates and preferentially skips the lowest-gradient (cold) Gaussians, consistent with the visibility-gradient correlation. ")
L.append("- Quality impact is near-zero on average (pooled mean PSNR delta -0.008 dB) and seed-stable on garden/bicycle, BUT room k=2 shows a seed-dependent outlier (seed 0: -0.271 dB), so VOS is NOT uniformly seed-stable. ")
L.append("- No wall-clock speedup (Amdahl-bound < 1.2x; kernel O(N)); the benefit is reduced optimizer workload/energy, not training time. ")
L.append("- **Fail-closed note**: a stricter per-scene or worst-seed quality gate (e.g. <=0.05 dB on every seed) would FAIL on room seed 0. The validation therefore reports the instability honestly rather than claiming uniform viability.\n")

with open(os.path.join(OUT, "VOS1_REPORT.md"), "w") as f: f.write("\n".join(L))

# ---------- 8. MANIFEST ----------
man = []
for name in ["VOS1_SOURCE_LOCK.json","VOS1_MULTI_SEED_RESULTS.json","VOS1_VISIBILITY_ANALYSIS.json",
             "VOS1_UPDATE_REDUCTION.json","VOS1_AMDAHL.json","VOS1_FIGURE_DATA.json","VOS1_REPORT.md"]:
    p = os.path.join(OUT, name)
    if os.path.exists(p): man.append(f"{sha256_file(p)}  {name}")
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f: f.write("\n".join(man) + "\n")

print("ASSEMBLE_DONE")
print("k2_psnr_std:", {s: round(multi[s]["2"]["psnr_std"],4) for s in SCENES})
print("k1_psnr_std:", {s: round(multi[s]["1"]["psnr_std"],4) for s in SCENES})
print("psnr_delta_mean:", {s: round(delta[s]["psnr_delta_mean"],4) for s in SCENES})
print("update_reduction_mean:", {s: round(upd_red[s]["update_reduction_mean"],4) for s in SCENES})
print("update_reduction_geomean:", round(update_reduction["update_reduction_geomean"],4))
print("spearman_range:", round(min(vis[s][str(k)]["spearman_vis_grad_mean"] for s in SCENES for k in SCHEDS),3), "-", round(max(vis[s][str(k)]["spearman_vis_grad_mean"] for s in SCENES for k in SCHEDS),3))
print("opt_frac_mean:", round(opt_frac_mean,4), "bound:", round(amdahl["max_wall_speedup_if_optimizer_fully_eliminated"],4))
