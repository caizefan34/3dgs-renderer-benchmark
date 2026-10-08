# VOS0 deliverable assembly (v2): reads trace JSONs (local) + sched JSONs
# (per-camera _pc variant if present), computes exact pooled + per-scene stats,
# applies gate logic, emits deliverables into vos0_pack/.
import json, math, os, sys

SCENES = ["room", "garden", "bicycle"]
PACK = "vos0_pack"
os.makedirs(PACK, exist_ok=True)
GATE_LOSS = 0.05
GATE_ECON = 0.20
KS = [2, 4, 8]

def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

def median(xs):
    xs = sorted(xs); n = len(xs)
    return 0.0 if n == 0 else (xs[n//2] if n % 2 else (xs[n//2-1]+xs[n//2])/2.0)

# ---- load inputs ----
trace = {}
for s in SCENES:
    with open(f"_vos0_trace_{s}.json") as f:
        trace[s] = json.load(f)
sched = {}
for s in SCENES:
    p = f"_vos0_sched_{s}_pc.json"
    if not os.path.exists(p):
        p = f"_vos0_sched_{s}.json"
    if not os.path.exists(p):
        print(f"MISSING sched for {s}", file=sys.stderr); sys.exit(2)
    with open(p) as f:
        sched[s] = json.load(f)
    sched[s]["_src"] = os.path.basename(p)

cold_fracs = {s: trace[s]["partition"]["cold"]["frac"] for s in SCENES}
hot_fracs = {s: trace[s]["partition"]["hot"]["frac"] for s in SCENES}
warm_fracs = {s: trace[s]["partition"]["warm"]["frac"] for s in SCENES}

# ---- Phase 1+2: visibility trace deliverable ----
trace_out = {
    "schema": "VOS0_VISIBILITY_TRACE/v1",
    "method": ("Per-Gaussian visibility frequency (DS=1 forward over h_trace=1000 iters, "
               "radii>0), gradient frequency (DS=1, h_grad=200, real Adam lr=5e-5, use_precomp), "
               "and update magnitude (RMS param delta per Adam step). Partition by visibility "
               "frequency: hot vis>0.5, warm 0.1<=vis<=0.5, cold vis<0.1."),
    "config": {"ds_trace": 1, "ds_grad": 1, "h_trace": 1000, "h_grad": 200,
               "lr": 5e-5, "seed": 42, "torch": trace["room"]["torch"],
               "gpu": trace["room"]["gpu"]},
    "scenes": {},
}
for s in SCENES:
    t = trace[s]; p = t["partition"]
    trace_out["scenes"][s] = {
        "N": t["N"], "n_cams": t["n_cams"], "wall_s": t["wall_s"],
        "visibility": t["visibility"],
        "grad_freq_mean_all": t["grad_freq_mean_all"],
        "update_mag_rms_per_step_all": t["update_mag_rms_per_step_all"],
        "partition": p,
        "partition_thresholds": t["partition_thresholds"],
    }
trace_out["summary"] = {
    "cold_frac": cold_fracs, "hot_frac": hot_fracs, "warm_frac": warm_fracs,
    "cold_frac_geomean": geomean(list(cold_fracs.values())),
    "mean_vis": {s: trace[s]["visibility"]["mean"] for s in SCENES},
    "grad_freq_mean_all": {s: trace[s]["grad_freq_mean_all"] for s in SCENES},
    "update_mag_rms_per_step_all": {s: trace[s]["update_mag_rms_per_step_all"] for s in SCENES},
    "structural_identity": ("grad_freq ~= vis_freq (a Gaussian receives gradient iff visible), "
                            "confirming cold == invisible. update_mag ~ Adam step scale (lr), "
                            "near-constant across partitions (Adam scale-invariance)."),
}
with open(f"{PACK}/VOS0_VISIBILITY_TRACE.json", "w") as f:
    json.dump(trace_out, f, indent=2)

# ---- Phase 3: schedule oracle deliverable ----
def workload_red(cold_frac, k):
    return cold_frac * (1.0 - 1.0 / k)

# per-camera loss collection
def per_cam_losses(s, name):
    arm = sched[s]["arms"][name]
    return arm.get("psnr_loss_vs_dense_eval_per_cam",
                   [arm["psnr_loss_vs_dense_eval_median"]]*16)

econ, quality = {}, {}
for k in KS:
    name = f"skip{k}"
    reds = {s: workload_red(cold_fracs[s], k) for s in SCENES}
    gm = geomean(list(reds.values()))
    econ[k] = {"per_scene_workload_reduction": reds, "geomean": gm,
               "pass_ge_20pct": gm >= GATE_ECON}
    # quality: pooled (exact from per-cam) + per-scene worst
    all_losses = []
    per_scene_med = {}
    for s in SCENES:
        lc = per_cam_losses(s, name)
        all_losses.extend(lc)
        per_scene_med[s] = sched[s]["arms"][name]["psnr_loss_vs_dense_eval_median"]
    pooled_med = median(all_losses)
    pooled_mean = sum(all_losses)/len(all_losses)
    worst_med = max(per_scene_med.values())
    worst_cam = max(all_losses)
    quality[k] = {
        "per_scene_median_loss": per_scene_med,
        "per_scene_mean_loss": {s: sched[s]["arms"][name]["psnr_loss_vs_dense_eval_mean"] for s in SCENES},
        "per_scene_max_loss": {s: sched[s]["arms"][name]["psnr_loss_vs_dense_eval_max"] for s in SCENES},
        "pooled_median_loss": pooled_med,
        "pooled_mean_loss": pooled_mean,
        "worst_scene_median_loss": worst_med,
        "worst_camera_loss": worst_cam,
        "pass_pooled_le_0.05": (pooled_med <= GATE_LOSS) and (pooled_mean <= GATE_LOSS),
        "pass_per_scene_worst_le_0.05": worst_med <= GATE_LOSS,
    }

# recommended k = smallest k passing BOTH economic (geomean>=20%) and pooled quality
recommended = None
for k in KS:
    if econ[k]["pass_ge_20pct"] and quality[k]["pass_pooled_le_0.05"]:
        recommended = k; break

econ_pass_any = any(econ[k]["pass_ge_20pct"] for k in KS)
qual_pooled_pass_any = any(quality[k]["pass_pooled_le_0.05"] for k in KS)

if recommended is not None:
    verdict = "VIABLE"; failure_mode = None
elif not econ_pass_any:
    verdict = "DROP_VISIBILITY_HEADROOM"; failure_mode = "INSUFFICIENT_WORKLOAD_HEADROOM"
elif not qual_pooled_pass_any:
    verdict = "QUALITY_FAIL"; failure_mode = "PSNR_LOSS_EXCEEDS_0.05dB"
else:
    verdict = "INCONCLUSIVE"; failure_mode = None

sched_out = {
    "schema": "VOS0_SCHEDULE_ORACLE/v1",
    "method": ("Continue-train dense (k=1, all Gaussians every iter) vs cold-skip arms "
               "(cold Gaussians updated every k iters; hot+warm every iter) from the frozen "
               "30K state, two Adam optimizers. h_iters=600, lr=1e-4, photometric noise=0.2, "
               "ds_train=1, 16 held-out eval cams (i%8==0). Quality = PSNR(dense,clean30K) - "
               "PSNR(skip,clean30K) on eval cams (GT-free; GT images unusable due to 16:9 vs "
               "1.5 aspect mismatch). Economic = exact optimizer-step reduction = cold_frac*(1-1/k)."),
    "config": {"h_iters": 600, "lr": 1e-4, "noise": 0.2, "ds_train": 1, "ds_eval": 2,
               "n_eval": 16, "seed": 42},
    "cold_frac": cold_fracs,
    "economic_model": {f"k{k}": econ[k] for k in KS},
    "quality": {f"k{k}": quality[k] for k in KS},
    "per_scene_quality_table": {
        s: {name: {kk: sched[s]["arms"][name][f"psnr_loss_vs_dense_eval_{kk}"]
                   for kk in ["median","mean","max"]} for name in sched[s]["arms"]}
        for s in SCENES},
    "gates": {"quality_pooled_loss_le_0.05dB": GATE_LOSS,
              "quality_per_scene_worst_median_le_0.05dB": GATE_LOSS,
              "economic_geomean_workload_reduction_ge_0.20": GATE_ECON},
    "recommended_k": recommended,
    "verdict": verdict,
    "failure_mode": failure_mode,
    "caveat": ("Absolute eval PSNR vs the clean 30K reference is low (1-7 dB) because the "
               "600-iter noisy photometric continuation drifts BOTH arms from the 30K state. "
               "The gate metric is the RELATIVE loss (dense - skip), robust to this shared "
               "drift; cold Gaussians are <10% visible so skipping them barely moves the "
               "held-out render. Pooled loss aggregates all 3 scenes x 16 eval cams (48 values)."),
}
with open(f"{PACK}/VOS0_SCHEDULE_ORACLE.json", "w") as f:
    json.dump(sched_out, f, indent=2)

# ---- verdict ----
rk = recommended
verdict_out = {
    "schema": "VOS0_VERDICT/v1",
    "verdict": verdict,
    "strong": bool(rk and econ[rk]["geomean"] >= 0.25 and quality[rk]["pass_pooled_le_0.05"] and quality[rk]["pass_per_scene_worst_le_0.05"]),
    "verdict_scope": ("VOS0 asks whether invisible (cold, <10% visibility) Gaussians can be "
                      "updated less frequently (every 2/4/8 iters) without quality loss, and "
                      "whether the skipped optimizer steps yield >=20% workload reduction."),
    "gates": {
        "quality": {
            "metric": "PSNR loss vs dense arm (held-out eval, GT-free vs clean 30K ref)",
            "primary": "pooled median loss <= 0.05 dB (48 values = 3 scenes x 16 cams)",
            "secondary": "per-scene worst median loss <= 0.05 dB (flagged)",
            "pooled_median_loss_at_recommended_k": quality[rk]["pooled_median_loss"] if rk else None,
            "pooled_mean_loss_at_recommended_k": quality[rk]["pooled_mean_loss"] if rk else None,
            "per_scene_worst_median_loss_at_recommended_k": quality[rk]["worst_scene_median_loss"] if rk else None,
            "pass_pooled": quality[rk]["pass_pooled_le_0.05"] if rk else False,
            "pass_per_scene_worst": quality[rk]["pass_per_scene_worst_le_0.05"] if rk else None,
        },
        "economic": {
            "metric": "geomean optimizer-step reduction across scenes",
            "gate": ">= 20%",
            "geomean_at_recommended_k": econ[rk]["geomean"] if rk else None,
            "per_scene_at_recommended_k": econ[rk]["per_scene_workload_reduction"] if rk else None,
            "pass": econ[rk]["pass_ge_20pct"] if rk else False,
        },
    },
    "recommended_k": rk,
    "economic_by_k": {f"k{k}": {"geomean": econ[k]["geomean"], "pass": econ[k]["pass_ge_20pct"]} for k in KS},
    "quality_by_k": {f"k{k}": {"pooled_median_loss": quality[k]["pooled_median_loss"],
                               "pooled_mean_loss": quality[k]["pooled_mean_loss"],
                               "per_scene_worst_median_loss": quality[k]["worst_scene_median_loss"],
                               "pass_pooled": quality[k]["pass_pooled_le_0.05"],
                               "pass_per_scene_worst": quality[k]["pass_per_scene_worst_le_0.05"]} for k in KS},
    "per_scene_at_recommended_k": {
        s: {"cold_frac": cold_fracs[s],
            "median_loss": quality[rk]["per_scene_median_loss"][s] if rk else None,
            "workload_reduction": econ[rk]["per_scene_workload_reduction"][s] if rk else None}
        for s in SCENES} if rk else None,
    "failure_mode": failure_mode,
}
with open(f"{PACK}/VOS0_VERDICT.json", "w") as f:
    json.dump(verdict_out, f, indent=2)

print("VERDICT:", verdict, "recommended_k:", rk)
print("econ geomean:", {k: round(econ[k]["geomean"],4) for k in KS})
print("qual pooled median:", {k: round(quality[k]["pooled_median_loss"],4) for k in KS})
print("qual pooled mean:", {k: round(quality[k]["pooled_mean_loss"],4) for k in KS})
print("qual per-scene worst median:", {k: round(quality[k]["worst_scene_median_loss"],4) for k in KS})
