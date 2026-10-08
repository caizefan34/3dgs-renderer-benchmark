#!/usr/bin/env python3
"""C0-QUALITY-STABILITY0-MX aggregation.
Reads 9 scene x seed A0/C0 run pairs (seed42 reused from causal ladder, seed43/44
new), computes paired deltas, multi-seed stats, paired bootstrap CI, frozen-gate
pass/fail, speed, population trajectory, and the terminal quality verdict.

Run on MX. Writes into the campaign dir.
"""
import argparse, json, os, glob
import numpy as np

SCENES = ["room", "bicycle", "garden"]
SEEDS = [42, 43, 44]
ARMS = ["a0", "c0"]
SEED42_ROOT = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/C0_SAME_PARENT_CAUSAL_LADDER_MX/raw"
SEED4344_ROOT = "/dev/shm/c0_quality_stability/runs"
CAMPAIGN = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 12345


def load(path, default=None):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def quality_path(scene, seed, arm):
    if seed == 42:
        return os.path.join(SEED42_ROOT, f"{scene}_s42_{arm}", "quality.json")
    return os.path.join(SEED4344_ROOT, f"{scene}_s{seed}_{arm}", "quality.json")


def results_path(scene, seed, arm):
    if seed == 42:
        return os.path.join(SEED42_ROOT, f"{scene}_s42_{arm}", "results.json")
    return os.path.join(SEED4344_ROOT, f"{scene}_s{seed}_{arm}", "results.json")


def gm(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


def geomean(xs):
    return gm(xs)


def paired_bootstrap_ci(deltas, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Paired bootstrap 95% CI over the per-pair delta vector."""
    d = np.asarray([x for x in deltas if x is not None], dtype=float)
    if len(d) == 0:
        return {"mean": None, "median": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n, size=n)
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "lo": float(lo), "hi": float(hi), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=CAMPAIGN)
    args = ap.parse_args()
    O = args.outdir
    os.makedirs(O, exist_ok=True)

    gates = load(os.path.join(O, "QUALITY_FROZEN_GATES.json"), {})
    dpsnr_floor = gates.get("bands_fixed_before_runs", {}).get("dPSNR_floor_dB", -0.05)
    dssim_floor = gates.get("bands_fixed_before_runs", {}).get("dSSIM_floor", -0.0005)
    dlpips_ceil = gates.get("bands_fixed_before_runs", {}).get("dLPIPS_ceil", 0.001)
    catastrophic = gates.get("bands_fixed_before_runs", {}).get("catastrophic_pair_dPSNR", -0.5)

    # ---- collect per-pair ----
    pairs = {}          # (scene,seed) -> metrics
    missing = []
    for sc in SCENES:
        for sd in SEEDS:
            qa = load(quality_path(sc, sd, "a0"))
            qc = load(quality_path(sc, sd, "c0"))
            if not qa or not qc:
                missing.append({"scene": sc, "seed": sd, "a0": bool(qa), "c0": bool(qc)})
                continue
            pairs[(sc, sd)] = {
                "psnr_a0": qa["psnr"], "psnr_c0": qc["psnr"],
                "ssim_a0": qa["ssim"], "ssim_c0": qc["ssim"],
                "lpips_a0": qa["lpips"], "lpips_c0": qc["lpips"],
                "d_psnr": qc["psnr"] - qa["psnr"],
                "d_ssim": qc["ssim"] - qa["ssim"],
                "d_lpips": qc["lpips"] - qa["lpips"],
                "n_a0": qa.get("n_gaussians"), "n_c0": qc.get("n_gaussians"),
                "wall_a0": qa.get("wall_time_s"), "wall_c0": qc.get("wall_time_s"),
            }

    # ---- QUALITY_30K_METRICS.json ----
    per_pair_list = []
    for (sc, sd), p in sorted(pairs.items()):
        r = (p["wall_a0"] / p["wall_c0"]) if (p["wall_a0"] and p["wall_c0"]) else None
        per_pair_list.append({"scene": sc, "seed": sd, **p, "speedup": r})
    metrics = {"artifact": "QUALITY_30K_METRICS", "campaign": "C0_QUALITY_STABILITY0_MX",
               "definition": "delta = C0 - A0; positive dPSNR/dSSIM and negative dLPIPS = C0 better",
               "n_pairs": len(pairs), "missing": missing, "per_pair": per_pair_list}
    json.dump(metrics, open(os.path.join(O, "QUALITY_30K_METRICS.json"), "w"), indent=2)

    if missing:
        print("INCOMPLETE: missing pairs:", missing)
        # still write partial stats but flag incomplete

    # ---- deltas ----
    dpsnr = [p["d_psnr"] for p in pairs.values()]
    dssim = [p["d_ssim"] for p in pairs.values()]
    dlpips = [p["d_lpips"] for p in pairs.values()]
    speeds = [p["speedup"] for p in pairs.values() if p.get("speedup")]

    stats = {"artifact": "QUALITY_MULTI_SEED_STATS", "campaign": "C0_QUALITY_STABILITY0_MX",
             "n_pairs": len(pairs), "missing": missing,
             "dPSNR": paired_bootstrap_ci(dpsnr),
             "dSSIM": paired_bootstrap_ci(dssim),
             "dLPIPS": paired_bootstrap_ci(dlpips),
             "speedup": paired_bootstrap_ci(speeds),
             "per_seed": {}, "per_scene": {}, "direction_count": {}}
    # per-seed / per-scene
    for sd in SEEDS:
        dp = [pairs[(sc, sd)]["d_psnr"] for sc in SCENES if (sc, sd) in pairs]
        ds = [pairs[(sc, sd)]["d_ssim"] for sc in SCENES if (sc, sd) in pairs]
        dl = [pairs[(sc, sd)]["d_lpips"] for sc in SCENES if (sc, sd) in pairs]
        sp = [pairs[(sc, sd)]["speedup"] for sc in SCENES if (sc, sd) in pairs and pairs[(sc, sd)].get("speedup")]
        stats["per_seed"][str(sd)] = {"mean_d_psnr": float(np.mean(dp)) if dp else None,
                                       "mean_d_ssim": float(np.mean(ds)) if ds else None,
                                       "mean_d_lpips": float(np.mean(dl)) if dl else None,
                                       "speedup_geomean": geomean(sp)}
    for sc in SCENES:
        dp = [pairs[(sc, sd)]["d_psnr"] for sd in SEEDS if (sc, sd) in pairs]
        ds = [pairs[(sc, sd)]["d_ssim"] for sd in SEEDS if (sc, sd) in pairs]
        dl = [pairs[(sc, sd)]["d_lpips"] for sd in SEEDS if (sc, sd) in pairs]
        sp = [pairs[(sc, sd)]["speedup"] for sd in SEEDS if (sc, sd) in pairs and pairs[(sc, sd)].get("speedup")]
        stats["per_scene"][sc] = {"mean_d_psnr": float(np.mean(dp)) if dp else None,
                                   "mean_d_ssim": float(np.mean(ds)) if ds else None,
                                   "mean_d_lpips": float(np.mean(dl)) if dl else None,
                                   "speedup_geomean": geomean(sp)}
    stats["direction_count"] = {"dPSNR_negative": sum(1 for x in dpsnr if x < 0),
                                 "dPSNR_positive": sum(1 for x in dpsnr if x > 0),
                                 "dLPIPS_positive(worse)": sum(1 for x in dlpips if x > 0)}
    json.dump(stats, open(os.path.join(O, "QUALITY_MULTI_SEED_STATS.json"), "w"), indent=2)

    # ---- frozen gate pass/fail per pair ----
    pair_gates = []
    for pp in per_pair_list:
        neutral = (pp["d_psnr"] >= dpsnr_floor and pp["d_ssim"] >= dssim_floor
                   and pp["d_lpips"] <= dlpips_ceil)
        cat = pp["d_psnr"] < catastrophic
        pair_gates.append({"scene": pp["scene"], "seed": pp["seed"], "d_psnr": pp["d_psnr"],
                           "d_ssim": pp["d_ssim"], "d_lpips": pp["d_lpips"],
                           "within_neutrality": bool(neutral), "catastrophic": bool(cat)})
    n_pass = sum(1 for g in pair_gates if g["within_neutrality"])
    n_fail = len(pair_gates) - n_pass
    n_cat = sum(1 for g in pair_gates if g["catastrophic"])
    n_dpsnr_below_floor = sum(1 for g in pair_gates if g["d_psnr"] < dpsnr_floor)

    # ---- terminal verdict ----
    mean_dpsnr = stats["dPSNR"]["mean"]; ci = stats["dPSNR"]
    ci_excludes_zero_up = (ci["lo"] is not None and ci["lo"] > 0)      # mean significantly positive
    ci_excludes_zero_down = (ci["hi"] is not None and ci["hi"] < 0)     # mean significantly negative
    all_pairs_present = (len(missing) == 0)

    # scene/seed GM of speedup
    seed_gms = {sd: stats["per_seed"][str(sd)]["speedup_geomean"] for sd in SEEDS}
    scene_gms = {sc: stats["per_scene"][sc]["speedup_geomean"] for sc in SCENES}
    overall_speed_gm = geomean(speeds)
    n_pos_speed = sum(1 for x in speeds if x > 1)

    # verdict logic
    if not all_pairs_present:
        verdict = "QUALITY_INCONCLUSIVE"
    elif n_dpsnr_below_floor == 0 and n_cat == 0 and (ci["lo"] is None or ci["lo"] > dpsnr_floor):
        # no pair materially worse than gate; overall mean at/above floor
        verdict = "QUALITY_STABLE_NEUTRAL"
    elif ci_excludes_zero_down and n_dpsnr_below_floor >= max(1, int(np.ceil(len(pair_gates) * 0.5))):
        # majority of pairs negative AND overall mean CI excludes neutrality downward
        verdict = "QUALITY_SYSTEMATIC_DEGRADATION"
    elif ci_excludes_zero_up and n_dpsnr_below_floor == 0:
        verdict = "QUALITY_SYSTEMATIC_IMPROVEMENT"
    else:
        # mixed / seed-dependent: negative direction does not reproduce consistently
        verdict = "QUALITY_SEED_SENSITIVE"

    gate_summary = {"artifact": "QUALITY_GATE_SUMMARY", "n_pairs": len(pair_gates),
                    "n_passing": n_pass, "n_failing": n_fail, "n_catastrophic": n_cat,
                    "n_dpsnr_below_floor": n_dpsnr_below_floor,
                    "pairs": pair_gates}
    json.dump(gate_summary, open(os.path.join(O, "QUALITY_GATE_SUMMARY.json"), "w"), indent=2)

    # ---- SPEED ----
    speed = {"artifact": "QUALITY_SPEED_STABILITY", "definition": "R = T_A0 / T_C0 ; R>1 => C0 faster",
             "n_ratios": len(speeds), "per_pair": [{"scene": pp["scene"], "seed": pp["seed"], "speedup": pp["speedup"]} for pp in per_pair_list],
             "per_seed_geomean": seed_gms, "per_scene_geomean": scene_gms,
             "overall_geomean": overall_speed_gm, "n_positive": n_pos_speed,
             "n_negative": len(speeds) - n_pos_speed, "bootstrap": stats["speedup"]}
    json.dump(speed, open(os.path.join(O, "QUALITY_SPEED_STABILITY.json"), "w"), indent=2)

    # ---- POPULATION ----
    ratios = []
    pop = {"artifact": "QUALITY_POPULATION_TRAJECTORY", "per_pair": []}
    for pp in per_pair_list:
        if pp["n_a0"] and pp["n_c0"]:
            r = pp["n_c0"] / pp["n_a0"]
            ratios.append(r)
            pop["per_pair"].append({"scene": pp["scene"], "seed": pp["seed"],
                                     "n_a0": pp["n_a0"], "n_c0": pp["n_c0"], "n_c0_over_n_a0": r})
    pop["final_N_c0_over_a0_geomean"] = geomean(ratios)
    json.dump(pop, open(os.path.join(O, "QUALITY_POPULATION_TRAJECTORY.json"), "w"), indent=2)

    # ---- VERDICT ----
    verdict_obj = {"artifact": "QUALITY_VERDICT", "campaign": "C0_QUALITY_STABILITY0_MX",
                   "verdict": verdict, "all_pairs_present": all_pairs_present, "missing": missing,
                   "criteria": {
                       "n_pairs": len(pair_gates),
                       "mean_d_psnr": mean_dpsnr,
                       "dPSNR_ci95": [ci["lo"], ci["hi"]],
                       "n_dpsnr_below_floor": n_dpsnr_below_floor,
                       "n_catastrophic": n_cat,
                       "ci_excludes_zero_down": ci_excludes_zero_down,
                       "ci_excludes_zero_up": ci_excludes_zero_up,
                       "per_seed_mean_d_psnr": {sd: stats["per_seed"][str(sd)]["mean_d_psnr"] for sd in SEEDS},
                   },
                   "note": "verdict computed from frozen gates + paired bootstrap; no rescue/tuning applied"}
    json.dump(verdict_obj, open(os.path.join(O, "QUALITY_VERDICT.json"), "w"), indent=2)

    def _r(x, nd=4):
        return round(x, nd) if x is not None else None
    print("VERDICT:", verdict)
    print("mean dPSNR:", mean_dpsnr, "CI95:", [ci["lo"], ci["hi"]])
    print("n_dpsnr_below_floor:", n_dpsnr_below_floor, "/", len(pair_gates))
    print("per-seed mean dPSNR:", {sd: _r(stats["per_seed"][str(sd)]["mean_d_psnr"]) for sd in SEEDS})
    print("overall speed GM:", overall_speed_gm)
    return verdict_obj


if __name__ == "__main__":
    main()
