import os, sys, json, glob, hashlib, math, time, platform, subprocess

RES = "/root/vos1_out/results"
OUT = "/root/vos1_out/vos1_pack"
os.makedirs(OUT, exist_ok=True)

SCENES = ["room", "garden", "bicycle"]
KS = [1, 2, 4, 8]
BASE_K = 1

def load(scene, k):
    p = os.path.join(RES, f"result_{scene}_k{k}.json")
    with open(p) as f:
        return json.load(f)

def load_trace(scene, k):
    p = os.path.join(RES, f"trace_{scene}_k{k}.json")
    with open(p) as f:
        return json.load(f)

# ---- gather ----
data = {}
for s in SCENES:
    for k in KS:
        data[(s, k)] = load(s, k)

# ---- per-scene derived ----
per_scene = {}
for s in SCENES:
    base = data[(s, BASE_K)]
    per_scene[s] = {"baseline_psnr": base["psnr"], "baseline_wall": base["wall_time_s"],
                    "baseline_opt": base["optimizer_time_s"], "baseline_updates": base["update_count"]}
    for k in KS:
        r = data[(s, k)]
        psnr_loss = base["psnr"] - r["psnr"]
        ssim_loss = base["ssim"] - r["ssim"]
        lpips_delta = r["lpips"] - base["lpips"]
        speedup = base["wall_time_s"] / r["wall_time_s"] if r["wall_time_s"] > 0 else 0.0
        opt_speedup = base["optimizer_time_s"] / r["optimizer_time_s"] if r["optimizer_time_s"] > 0 else 0.0
        upd_red = 1.0 - (r["update_count"] / base["update_count"]) if base["update_count"] > 0 else 0.0
        opt_red = 1.0 - (r["optimizer_time_s"] / base["optimizer_time_s"]) if base["optimizer_time_s"] > 0 else 0.0
        per_scene[s][k] = {
            "psnr": r["psnr"], "ssim": r["ssim"], "lpips": r["lpips"],
            "psnr_loss": psnr_loss, "ssim_loss": ssim_loss, "lpips_delta": lpips_delta,
            "wall_time_s": r["wall_time_s"], "optimizer_time_s": r["optimizer_time_s"],
            "iter_time_ms_mean": r["iter_time_ms_mean"],
            "update_count": r["update_count"], "skipped_updates": r["skipped_updates"],
            "update_fraction": r["update_fraction"],
            "speedup": speedup, "opt_speedup": opt_speedup,
            "update_workload_reduction": upd_red, "opt_time_reduction": opt_red,
            "final_N": r["final_N"], "n_test": r["n_test"],
        }

# ---- pooled (geomean for ratios, mean for losses) ----
def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

pooled = {}
for k in KS:
    psnr_losses = [per_scene[s][k]["psnr_loss"] for s in SCENES]
    ssim_losses = [per_scene[s][k]["ssim_loss"] for s in SCENES]
    upd_reds = [per_scene[s][k]["update_workload_reduction"] for s in SCENES]
    opt_reds = [per_scene[s][k]["opt_time_reduction"] for s in SCENES]
    speedups = [per_scene[s][k]["speedup"] for s in SCENES]
    def _geomean_pos(xs):
        xs = [x for x in xs if x > 0]
        if not xs:
            return 0.0
        return math.exp(sum(math.log(x) for x in xs) / len(xs))
    pooled[k] = {
        "psnr_loss_mean": sum(psnr_losses) / len(psnr_losses),
        "psnr_loss_max": max(psnr_losses),
        "psnr_loss_min": min(psnr_losses),
        "psnr_loss_median": sorted(psnr_losses)[len(psnr_losses)//2] if len(psnr_losses) % 2 == 1 else (sorted(psnr_losses)[len(psnr_losses)//2-1]+sorted(psnr_losses)[len(psnr_losses)//2])/2,
        "ssim_loss_mean": sum(ssim_losses) / len(ssim_losses),
        "update_workload_reduction_mean": sum(upd_reds) / len(upd_reds),
        "update_workload_reduction_geomean": _geomean_pos(upd_reds),
        "opt_time_reduction_mean": sum(opt_reds) / len(opt_reds),
        "speedup_geomean": _geomean_pos(speedups),
    }

# ---- gate ----
PSNR_LOSS_MAX = 0.05
SPEEDUP_MIN = 1.2
WORKLOAD_RED_MIN = 0.25

recommended_k = None
recommended_reason = ""
for k in [2, 4, 8]:
    q = pooled[k]["psnr_loss_mean"] <= PSNR_LOSS_MAX
    e = (pooled[k]["speedup_geomean"] >= SPEEDUP_MIN) or (pooled[k]["update_workload_reduction_mean"] >= WORKLOAD_RED_MIN)
    if q and e:
        if recommended_k is None or pooled[k]["psnr_loss_mean"] < pooled[recommended_k]["psnr_loss_mean"]:
            recommended_k = k
            recommended_reason = f"k={k}: pooled PSNR loss {pooled[k]['psnr_loss_mean']:.4f} dB <= {PSNR_LOSS_MAX}; workload reduction {pooled[k]['update_workload_reduction_mean']*100:.1f}%"

verdict_pass = recommended_k is not None
if verdict_pass:
    verdict = "VIABLE"
    failure_code = None
else:
    verdict = "DROP_VISIBILITY_HEADROOM"
    # diagnose failure
    qok = any(pooled[k]["psnr_loss_mean"] <= PSNR_LOSS_MAX for k in [2,4,8])
    eok = any((pooled[k]["speedup_geomean"] >= SPEEDUP_MIN) or (pooled[k]["update_workload_reduction_mean"] >= WORKLOAD_RED_MIN) for k in [2,4,8])
    if not qok and eok:
        failure_code = "PSNR_LOSS_EXCEEDS_0.05dB"
    elif qok and not eok:
        failure_code = "NO_SPEEDUP_AND_NO_WORKLOAD_REDUCTION"
    else:
        failure_code = "QUALITY_AND_ECONOMIC_BOTH_FAIL"

# ---- Amdahl ----
# optimizer fraction of wall time (baseline k=1, pooled)
opt_fracs = [per_scene[s][BASE_K]["optimizer_time_s"] / per_scene[s][BASE_K]["wall_time_s"] for s in SCENES]
opt_frac_mean = sum(opt_fracs) / len(opt_fracs)
amdahl = {
    "optimizer_fraction_of_wall_mean": opt_frac_mean,
    "optimizer_fraction_per_scene": {s: per_scene[s][BASE_K]["optimizer_time_s"]/per_scene[s][BASE_K]["wall_time_s"] for s in SCENES},
    "max_wall_speedup_if_optimizer_fully_eliminated": 1.0 / (1.0 - opt_frac_mean),
    "note": "Optimizer is only a small fraction of wall time (render+backward dominate). Even full elimination yields <1.2x. SparseGaussianAdam kernel cost is O(N) regardless of visibility mask, so reducing update count does not reduce wall time.",
    "measured_speedup_geomean_k8": pooled[8]["speedup_geomean"],
    "measured_opt_time_reduction_mean_k8": pooled[8]["opt_time_reduction_mean"],
}

# ---- TRACE (aggregate visibility/update trace) ----
trace_out = {}
for s in SCENES:
    trace_out[s] = {}
    for k in KS:
        tr = load_trace(s, k)
        trace_out[s][k] = tr

trace_summary = {}
for s in SCENES:
    trace_summary[s] = {}
    for k in KS:
        tr = trace_out[s][k]
        if not tr:
            continue
        invs = [t["invisible_fraction"] for t in tr]
        upds = [t["update_fraction"] for t in tr]
        trace_summary[s][k] = {
            "n_trace_points": len(tr),
            "invisible_fraction_first": invs[0], "invisible_fraction_last": invs[-1],
            "invisible_fraction_mean": sum(invs)/len(invs),
            "update_fraction_first": upds[0], "update_fraction_last": upds[-1],
            "update_fraction_mean": sum(upds)/len(upds),
        }

# ---- SOURCE_LOCK ----
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_info(path):
    try:
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return None

src_files = [
    "/root/3dgs-renderer-benchmark/vos1_train.py",
    "/root/3dgs-renderer-benchmark/vos1_run.py",
    "/root/epic05_frozen/graphdeco-b0-pub/train.py",
    "/root/epic05_frozen/graphdeco-b0-pub/scene/gaussian_model.py",
    "/root/epic05_frozen/graphdeco-b0-pub/scene/__init__.py",
    "/root/epic05_frozen/graphdeco-b0-pub/gaussian_renderer/__init__.py",
    "/root/epic05_frozen/graphdeco-b0-pub/submodules/diff-gaussian-rasterization/__init__.py",
]
source_lock = {
    "env": {
        "python": "/root/miniforge3/envs/p16_fastergs/bin/python",
        "torch": __import__("torch").__version__,
        "cuda": "available",
        "device": "NVIDIA A100-SXM4-80GB",
        "platform": platform.platform(),
        "graphdeco_repo": "/root/epic05_frozen/graphdeco-b0-pub",
        "graphdeco_commit": git_info("/root/epic05_frozen/graphdeco-b0-pub"),
        "rasterizer_install": "/root/miniforge3/envs/p16_fastergs/lib/python3.11/site-packages/diff_gaussian_rasterization",
    },
    "config": {
        "iterations": 30000, "seed": 42, "sh_degree": 3,
        "optimizer_type": "sparse_adam", "resolution": -1,
        "scenes": SCENES, "k_values": KS,
        "update_rule": "visible(radii>0): age=0, always update; invisible: age+=1, update when age%k==0",
        "unmodified": ["rasterizer", "gradients", "loss", "densification", "pruning"],
    },
    "files": [],
}
for fp in src_files:
    if os.path.exists(fp):
        source_lock["files"].append({"path": fp, "sha256": sha256_file(fp)})

# ---- SCHEDULER_RESULT ----
scheduler_result = {
    "recommended_k": recommended_k,
    "verdict": verdict,
    "per_scene": {s: {str(k): per_scene[s][k] for k in KS} for s in SCENES},
    "pooled": {str(k): pooled[k] for k in KS},
    "gate": {
        "psnr_loss_max_db": PSNR_LOSS_MAX,
        "speedup_min": SPEEDUP_MIN,
        "workload_reduction_min": WORKLOAD_RED_MIN,
        "quality_pass_any_k": any(pooled[k]["psnr_loss_mean"] <= PSNR_LOSS_MAX for k in [2,4,8]),
        "economic_pass_any_k": any((pooled[k]["speedup_geomean"] >= SPEEDUP_MIN) or (pooled[k]["update_workload_reduction_mean"] >= WORKLOAD_RED_MIN) for k in [2,4,8]),
    },
    "failure_code": failure_code,
}

# ---- TRAINING_METRICS ----
training_metrics = {
    "per_scene": {s: {str(k): per_scene[s][k] for k in KS} for s in SCENES},
    "pooled": {str(k): pooled[k] for k in KS},
}

# ---- write core JSONs ----
def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)

dump("VOS1_VERDICT.json", {
    "verdict": verdict,
    "pass": verdict_pass,
    "recommended_k": recommended_k,
    "failure_code": failure_code,
    "reason": recommended_reason if verdict_pass else f"FAIL: quality_pass={scheduler_result['gate']['quality_pass_any_k']} economic_pass={scheduler_result['gate']['economic_pass_any_k']}",
    "gate": scheduler_result["gate"],
    "pooled_psnr_loss_mean": {str(k): pooled[k]["psnr_loss_mean"] for k in KS},
    "pooled_psnr_loss_median": {str(k): pooled[k]["psnr_loss_median"] for k in KS},
    "per_scene_psnr_loss_k2": {s: per_scene[s][2]["psnr_loss"] for s in SCENES},
    "pooled_update_workload_reduction_mean": {str(k): pooled[k]["update_workload_reduction_mean"] for k in KS},
    "pooled_opt_time_reduction_mean": {str(k): pooled[k]["opt_time_reduction_mean"] for k in KS},
    "pooled_speedup_geomean": {str(k): pooled[k]["speedup_geomean"] for k in KS},
    "caveats": {
        "no_wall_speedup": True,
        "amdahl_max_speedup": amdahl["max_wall_speedup_if_optimizer_fully_eliminated"],
        "optimizer_fraction_wall": opt_frac_mean,
        "verdict_basis": "pooled_mean_psnr_loss",
        "per_scene_room_garden_exceed_0.05_at_k2": (per_scene["room"][2]["psnr_loss"] > 0.05) and (per_scene["garden"][2]["psnr_loss"] > 0.05),
    },
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
dump("VOS1_SOURCE_LOCK.json", source_lock)
dump("VOS1_TRACE.json", {"per_scene": trace_out, "summary": trace_summary})
dump("VOS1_SCHEDULER_RESULT.json", scheduler_result)
dump("VOS1_TRAINING_METRICS.json", training_metrics)
dump("VOS1_AMDAHL.json", amdahl)

# ---- REPORT ----
def fmt(x, n=4):
    return f"{x:.{n}f}"

lines = []
lines.append("# VOS1 Visibility-Aware Optimizer Scheduler — Report\n")
lines.append(f"**Verdict: {verdict}**  ")
lines.append(f"Recommended k: {recommended_k}  ")
lines.append(f"Failure code: {failure_code}")
lines.append("")
lines.append("## Objective")
lines.append("Test whether reducing the optimizer update frequency for invisible Gaussians (visibility-aware Adam scheduling) can cut optimizer workload while preserving quality.")
lines.append("")
lines.append("## Method")
lines.append("- Faithful 30K-iteration 3DGS training from COLMAP init (graphdeco pipeline, unmodified rasterizer/gradients/loss/densification/pruning).")
lines.append("- Per-Gaussian `invisible_age`: visible (radii>0) → age=0, always update; invisible → age+=1, update only when `age % k == 0`.")
lines.append("- Optimizer: `SparseGaussianAdam.step(should_update, N, iteration)`; `should_update = visible | (invisible_age % k == 0)`. k=1 ⇒ dense (all updated).")
lines.append("- Scenes: room, garden, bicycle @ 1920×1080, seed 42, 30000 iters.")
lines.append("- Gate: pooled PSNR loss ≤ 0.05 dB AND (speedup ≥ 1.2× OR update-workload reduction ≥ 25%).")
lines.append("")
lines.append("## Pooled Results (geomean/mean over 3 scenes)")
lines.append("")
lines.append("| k | PSNR loss mean (dB) | PSNR loss median (dB) | SSIM loss | upd-workload red. | opt-time red. | wall speedup |")
lines.append("|---|---|---|---|---|---|---|")
for k in KS:
    lines.append("| {} | {} | {} | {} | {}% | {}% | {}× |".format(
        k, fmt(pooled[k]['psnr_loss_mean']), fmt(pooled[k]['psnr_loss_median']),
        fmt(pooled[k]['ssim_loss_mean']),
        fmt(pooled[k]['update_workload_reduction_mean']*100,1),
        fmt(pooled[k]['opt_time_reduction_mean']*100,1),
        fmt(pooled[k]['speedup_geomean'],3)))
lines.append("")
lines.append("## Per-Scene PSNR (k=1 baseline → k)")
lines.append("")
lines.append("| scene | k=1 | k=2 | k=4 | k=8 |")
lines.append("|---|---|---|---|---|")
for s in SCENES:
    row = []
    for k in [1,2,4,8]:
        pv = per_scene[s][k]['psnr']
        dl = per_scene[s][k]['psnr_loss']
        row.append("{:.3f} (Δ{:+.3f})".format(pv, dl))
    lines.append("| " + s + " | " + " | ".join(row) + " |")
lines.append("")
lines.append("## Per-Scene Update-Workload Reduction (1 − updates_k / updates_k1)")
lines.append("")
lines.append("| scene | k=2 | k=4 | k=8 |")
lines.append("|---|---|---|---|")
for s in SCENES:
    row = [f"{per_scene[s][k]['update_workload_reduction']*100:.1f}%" for k in [2,4,8]]
    lines.append(f"| {s} | " + " | ".join(row) + " |")
lines.append("")
lines.append("## Per-Scene Wall Time (s) / Optimizer Time (s)")
lines.append("")
lines.append("| scene | k=1 wall | k=2 wall | k=4 wall | k=8 wall | k=1 opt | k=8 opt |")
lines.append("|---|---|---|---|---|---|---|")
for s in SCENES:
    lines.append(f"| {s} | {per_scene[s][1]['wall_time_s']:.0f} | {per_scene[s][2]['wall_time_s']:.0f} | {per_scene[s][4]['wall_time_s']:.0f} | {per_scene[s][8]['wall_time_s']:.0f} | {per_scene[s][1]['optimizer_time_s']:.1f} | {per_scene[s][8]['optimizer_time_s']:.1f} |")
lines.append("")
lines.append("## Amdahl Analysis")
lines.append(f"- Optimizer fraction of wall time (mean, k=1): **{opt_frac_mean*100:.1f}%**.")
lines.append(f"- Max wall speedup if optimizer fully eliminated: **{amdahl['max_wall_speedup_if_optimizer_fully_eliminated']:.3f}×** (< 1.2× gate).")
lines.append(f"- Measured wall speedup geomean (k=8): **{pooled[8]['speedup_geomean']:.3f}×**; opt-time reduction (k=8): **{pooled[8]['opt_time_reduction_mean']*100:.1f}%**.")
lines.append("- The `SparseGaussianAdam` kernel is O(N) regardless of the visibility mask (persistent `asystep` accumulator + full-N pass), so cutting the update count does not cut wall time.")
lines.append("")
lines.append("## Caveats & Limitations")
lines.append("- **No wall-clock speedup.** Measured wall-speedup geomean is ~1.00× for all k (k=2: {}×). The economic gate is met only via *logical* update-workload reduction, not via time savings.".format(fmt(pooled[2]['speedup_geomean'],3)))
lines.append("- **Amdahl-bound.** The optimizer is only {:.1f}% of wall time (render+backward dominate); even fully eliminating it caps speedup at {:.3f}× < 1.2×. The speedup gate is structurally unreachable by optimizer-only changes.".format(opt_frac_mean*100, amdahl['max_wall_speedup_if_optimizer_fully_eliminated']))
lines.append("- **Kernel is O(N).** `SparseGaussianAdam.step` runs a full-N pass (persistent `asystep` accumulator + `asystep.add_(mask)`), so cutting the update count does not cut optimizer time (opt-time reduction ~0-1.2%).")
lines.append("- **Per-scene quality is borderline.** At k=2, pooled *mean* PSNR loss is {:.4f} dB (≤ 0.05), but room ({:.3f}) and garden ({:.3f}) individually exceed 0.05 dB, and the pooled *median* is {:.4f} dB (> 0.05). bicycle is quality-neutral ({:+.3f}).".format(pooled[2]['psnr_loss_mean'], per_scene['room'][2]['psnr_loss'], per_scene['garden'][2]['psnr_loss'], pooled[2]['psnr_loss_median'], per_scene['bicycle'][2]['psnr_loss']))
lines.append("- The VIABLE verdict rests on the pooled-mean aggregation; a stricter per-scene or median reading would fail the 0.05 dB quality gate.")
lines.append("")
lines.append("## Conclusion")
if verdict_pass:
    lines.append("**VIABLE (k={}) by the gate's letter.** Pooled-mean PSNR loss {:.4f} dB ≤ 0.05 dB AND update-workload reduction {:.1f}% ≥ 25%. The technique genuinely removes ~{:.0f}% of optimizer updates at k=2 with no measurable wall-clock cost. HOWEVER, it delivers NO wall-clock speedup (Amdahl-bound < 1.2×; kernel O(N)), and per-scene quality on room/garden exceeds 0.05 dB. Practical value is limited to reducing optimizer *workload* (energy/ FLOPs of the update step), not training time.".format(recommended_k, pooled[recommended_k]['psnr_loss_mean'], pooled[recommended_k]['update_workload_reduction_mean']*100, pooled[recommended_k]['update_workload_reduction_mean']*100))
else:
    lines.append("**NOT VIABLE ({})**: visibility-aware scheduling reduces logical update workload but (a) does not reduce wall-clock/optimizer time (kernel is O(N); Amdahl-bound < 1.2×) and (b) incurs pooled PSNR loss > 0.05 dB. Recommend DROP_VISIBILITY_HEADROOM.".format(failure_code))
lines.append("")

with open(os.path.join(OUT, "VOS1_REPORT.md"), "w") as f:
    f.write("\n".join(lines))

# ---- MANIFEST ----
manifest_lines = []
for name in ["VOS1_VERDICT.json", "VOS1_REPORT.md", "VOS1_SOURCE_LOCK.json",
             "VOS1_TRACE.json", "VOS1_SCHEDULER_RESULT.json", "VOS1_TRAINING_METRICS.json",
             "VOS1_AMDAHL.json"]:
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        manifest_lines.append(f"{sha256_file(p)}  {name}")
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(manifest_lines) + "\n")

print("ASSEMBLE_DONE verdict=", verdict, "recommended_k=", recommended_k, "failure=", failure_code)
print("pooled psnr_loss_mean:", {k: round(pooled[k]['psnr_loss_mean'],4) for k in KS})
print("pooled workload_red_mean:", {k: round(pooled[k]['update_workload_reduction_mean'],4) for k in KS})
print("pooled speedup_geomean:", {k: round(pooled[k]['speedup_geomean'],4) for k in KS})
print("opt_frac_mean:", round(opt_frac_mean,4))
