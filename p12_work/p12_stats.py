#!/usr/bin/env python3
"""P12 campaign statistics + verdicts.

Usage: p12_stats.py p12a|p12b

Preregistered thresholds (fixed BEFORE any 30K P12 run was observed):
- P12-A verdict rules (spec):
    SYSTEM_REVALIDATED_STRONG : GM>1 AND >=11/13 R>1 AND 95% CI entirely >1
                                AND quality-neutral AND clean campaign
    SYSTEM_REVALIDATED_POSITIVE: GM>1 AND majority R>1 AND quality-neutral
    SYSTEM_NEUTRAL            : approx parity (|log GM| small, mixed signs)
    SYSTEM_REVERSED           : current gsplat consistently faster
- P12-B verdict rules (spec):
    F9_REVALIDATED_STRONG      : GM>1 AND >=10/13 R>1 AND 95% CI entirely >1
                                 AND quality-neutral
    F9_REVALIDATED_POSITIVE_SMALL: GM>1 AND >=8/13 R>1 AND quality-neutral
    F9_NEUTRAL                 : approx parity / mixed directions
    F9_NEGATIVE                : consistently slower
- quality-neutral (both campaigns, preregistered): |mean dPSNR| <= 0.15 dB
  AND |mean dSSIM| <= 0.005 AND |mean dLPIPS| <= 0.005  (P8 seed/trajectory
  noise band). Evaluated as delta = (arm_expected_slower - arm_expected_faster)
  so positive dPSNR means the FASTER arm is worse.
- parity band for NEUTRAL: GM within [0.995, 1.005] OR signs mixed with
  |log GM| < 0.005.

Outputs (ART dir):
  P12A_SYSTEM_RUNS.json / P12B_F9_RUNS.json
  P12A_SYSTEM_QUALITY.json / P12B_F9_QUALITY.json
  P12A_SYSTEM_TRAJECTORY.json / P12B_F9_TRAJECTORY.json
  P12A_SYSTEM_STATS.json / P12B_F9_STATS.json
  P12A_SYSTEM_VERDICT.json / P12B_F9_VERDICT.json
  P12A_SYSTEM_REPORT.md / P12B_F9_REPORT.md
"""
import glob
import json
import math
import os
import sys
import time

ART = "/root/epic05_p12/artifacts"
RUNS = {"p12a": "/root/epic05_p12/runs/p12a", "p12b": "/root/epic05_p12/runs/p12b"}
SCENES = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
          "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
T975_DF12 = 2.17881  # two-sided 97.5% t quantile, df = 13-1 = 12

SPEC = {
    "p12a": {
        "arm_num": "gmain", "arm_den": "c0",
        "label_num": "GMAIN_FUSED", "label_den": "C0_FUSED",
        "ratio_name": "R_sys", "prefix": "P12A_SYSTEM",
        "delta_direction": "GMAIN_FUSED - C0_FUSED",
        "question": "Under the P11-validated fused-loss baseline, is the frozen "
                    "C0 system still faster than current gsplat main under the "
                    "strict matched 13-scene 30K protocol?",
        "old_ref": {"campaign": "P8 strict matched (SepSSIM common loss)",
                    "gm": 1.0979, "positive": "13/13"},
        "strong": {"min_positive": 11}, "positive": {"min_positive": 7},
        "verdicts": ["SYSTEM_REVALIDATED_STRONG", "SYSTEM_REVALIDATED_POSITIVE",
                     "SYSTEM_NEUTRAL", "SYSTEM_REVERSED"],
        "reversed_label": "current gsplat consistently faster",
    },
    "p12b": {
        "arm_num": "a0", "arm_den": "a1",
        "label_num": "A0_FUSED", "label_den": "A1_FUSED",
        "ratio_name": "R_F9", "prefix": "P12B_F9",
        "delta_direction": "A1_FUSED - A0_FUSED",
        "question": "Does the original F9 materialization-free producer retain "
                    "a reproducible full-training benefit after the common "
                    "SepSSIM bottleneck is removed?",
        "old_ref": {"campaign": "P2 historical SXM (SepSSIM common loss)",
                    "gm": 1.0111, "positive": "13/13"},
        "strong": {"min_positive": 10}, "positive": {"min_positive": 8},
        "verdicts": ["F9_REVALIDATED_STRONG", "F9_REVALIDATED_POSITIVE_SMALL",
                     "F9_NEUTRAL", "F9_NEGATIVE"],
        "reversed_label": "F9 consistently slower (A1 faster than A0 is the "
                          "claim; REVERSED means A1 slower)",
    },
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def mean(v):
    return sum(v) / len(v) if v else None


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def load_run(run_id, root):
    d = os.path.join(root, run_id)
    R = json.load(open(os.path.join(d, "results.json")))
    M = json.load(open(os.path.join(d, "epic05_launcher_meta.json")))
    MEM = json.load(open(os.path.join(d, "memory.json")))
    LM = json.load(open(os.path.join(d, "final_status.json")))
    return R, M, MEM, LM


def compact_run(run_id, R, M, MEM, LM):
    fe = R["final_eval"]
    tj = R["timing"]
    dens = R.get("densification_events", [])
    return {
        "run_id": run_id, "arm": R["arm"], "scene": R["scene"],
        "seed": R["seed"], "iterations": R["iterations"],
        "loss_backend": R["loss_backend"]["name"],
        "loss_backend_padding": R["loss_backend"].get("padding"),
        "loss_backend_so_sha256": R["loss_backend"].get("so_sha256"),
        "binary_identity": {"so_sha256": R["binary_identity"]["so_sha256"],
                             "core_sha256": R["binary_identity"].get("core_sha256")},
        "gpu_physical": M["group_gpu_physical"],
        "launcher_wall_s": round(M["wall_launcher_s"], 2),
        "ts_start": M["ts_start"], "ts_end": M["ts_end"],
        "timing_grade": R["timing_grade"],
        "total_wall_s": round(tj["total_wall_s"], 2),
        "total_wall_min": round(tj["total_wall_s"] / 60, 2),
        "iters_per_s": round(R["iterations"] / tj["total_wall_s"], 3),
        "mean_iter_ms": round(tj["mean_iter_ms"], 3),
        "median_iter_ms": round(tj["median_iter_ms"], 3),
        "peak_vram_gb": round(MEM["peak_vram_gb"], 2),
        "initial_N": R["initial_N"], "final_N": R["final_N"],
        "psnr": fe.get("psnr"), "ssim": fe.get("ssim"),
        "lpips": fe.get("lpips"), "l1": fe.get("l1"),
        "n_eval_cameras_final": fe.get("n_eval_cameras"),
        "status": LM["status"],
        "loss_curve_checkpoints": [[r["iter"], round(r["loss"], 6), r["n_gs"],
                                    r["n_visible"], r["loss_nan"]]
                                   for r in R["per_iter"]],
        "population_trajectory": [[r["iter"], r["n_gs"], r["n_visible"]]
                                  for r in R["per_iter"]],
        "densification_events": {
            "n_events": len(dens),
            "total_clones": R["total_clones"], "total_splits": R["total_splits"],
            "total_prunes": R["total_prunes"],
            "total_opacity_resets": R["total_opacity_resets"],
            "first_event_iter": dens[0]["iter"] if dens else None,
            "last_event_iter": dens[-1]["iter"] if dens else None,
            "per_event": [[e["iter"], e["n_before"], e["n_after"], e["cloned"],
                           e["split"], e["pruned_total"]] for e in dens],
        },
        "phase_ms": tj.get("phase_ms"),
        "results_json": os.path.join(RUNS_PREFIX, run_id, "results.json"),
    }


def main(camp):
    spec = SPEC[camp]
    root = RUNS[camp]
    global RUNS_PREFIX
    RUNS_PREFIX = root

    # ---- load all runs; only CLEAN, SUCCESSFUL, FINAL runs enter stats ----
    contam = None
    cp = os.path.join(ART, "P12_CONTAMINATION.json")
    if os.path.exists(cp):
        contam = json.load(open(cp))["campaigns"].get(camp, {})
    scene_runs = {}
    missing = []
    for scene in SCENES:
        rid_num = f"{camp}_{spec['arm_num']}f_s42_{scene}"
        rid_den = f"{camp}_{spec['arm_den']}f_s42_{scene}"
        rec = {}
        for role, rid in (("num", rid_num), ("den", rid_den)):
            d = os.path.join(root, rid)
            if not (os.path.exists(os.path.join(d, "results.json"))
                    and os.path.exists(os.path.join(d, "EPIC05_SUCCESS"))):
                missing.append(rid)
                continue
            R, M, MEM, LM = load_run(rid, root)
            rec[role] = {"R": R, "M": M, "MEM": MEM, "LM": LM, "run_id": rid}
            if contam and rid in contam.get("runs", {}):
                rec[role]["clean"] = contam["runs"][rid]["clean"]
            else:
                rec[role]["clean"] = None
        scene_runs[scene] = rec

    if missing:
        print(f"MISSING/INCOMPLETE RUNS: {missing}")
        sys.exit(2)

    # ---- RUNS artifact ----
    runs_artifact = {"campaign_id": spec["prefix"],
                     "question": spec["question"],
                     "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "per_scene": {}}
    ratios = {}
    quality_deltas = []
    n_ratio = {}
    for scene in SCENES:
        rec = scene_runs[scene]
        runs_artifact["per_scene"][scene] = {
            spec["label_num"]: compact_run(rec["num"]["run_id"], rec["num"]["R"],
                                           rec["num"]["M"], rec["num"]["MEM"],
                                           rec["num"]["LM"]),
            spec["label_den"]: compact_run(rec["den"]["run_id"], rec["den"]["R"],
                                           rec["den"]["M"], rec["den"]["MEM"],
                                           rec["den"]["LM"]),
        }
        w_num = rec["num"]["R"]["timing"]["total_wall_s"]
        w_den = rec["den"]["R"]["timing"]["total_wall_s"]
        ratios[scene] = w_num / w_den
        # SPEC delta direction: p12a = num - den (GMAIN - C0);
        # p12b = den - num (A1 - A0)
        if camp == "p12a":
            dsign = 1.0
        else:
            dsign = -1.0
        quality_deltas.append({
            "scene": scene,
            "d_psnr": dsign * (rec["num"]["R"]["final_eval"]["psnr"]
                               - rec["den"]["R"]["final_eval"]["psnr"]),
            "d_ssim": dsign * (rec["num"]["R"]["final_eval"]["ssim"]
                               - rec["den"]["R"]["final_eval"]["ssim"]),
            "d_lpips": dsign * ((rec["num"]["R"]["final_eval"].get("lpips") or 0)
                                - (rec["den"]["R"]["final_eval"].get("lpips") or 0)),
            "n_ratio": rec["num"]["R"]["final_N"] / rec["den"]["R"]["final_N"],
        })
        n_ratio[scene] = rec["num"]["R"]["final_N"] / rec["den"]["R"]["final_N"]

    # ---- STATS ----
    rvals = [ratios[s] for s in SCENES]
    logr = [math.log(r) for r in rvals]
    gm = math.exp(mean(logr))
    med = median(rvals)
    sd = math.sqrt(sum((x - mean(logr)) ** 2 for x in logr) / (len(logr) - 1))
    se = sd / math.sqrt(len(logr))
    ci_lo_log = mean(logr) - T975_DF12 * se
    ci_hi_log = mean(logr) + T975_DF12 * se
    n_pos = sum(1 for r in rvals if r > 1)
    equiv_reduction = 1 - 1 / gm
    corr_pop = pearson(logr, [math.log(n_ratio[s]) for s in SCENES])

    old = spec["old_ref"]
    dlog_vs_old = math.log(gm) - math.log(old["gm"])
    rel_vs_old = gm / old["gm"] - 1

    # ---- QUALITY ----
    d_psnr = [q["d_psnr"] for q in quality_deltas]
    d_ssim = [q["d_ssim"] for q in quality_deltas]
    d_lpips = [q["d_lpips"] for q in quality_deltas]
    quality_neutral = (abs(mean(d_psnr)) <= 0.15 and abs(mean(d_ssim)) <= 0.005
                       and abs(mean(d_lpips)) <= 0.005)
    psnr_outliers = [q for q in quality_deltas if abs(q["d_psnr"]) > 0.5]

    # ---- campaign cleanliness ----
    all_clean = all(rec[role]["clean"] for rec in scene_runs.values()
                    for role in ("num", "den"))
    clean_known = all(rec[role]["clean"] is not None
                      for rec in scene_runs.values() for role in ("num", "den"))

    # ---- verdict ----
    ci_entirely_above_1 = math.exp(ci_lo_log) > 1
    majority = n_pos >= spec["positive"]["min_positive"]
    strong_pos = n_pos >= spec["strong"]["min_positive"]
    if camp == "p12a":
        if gm > 1 and strong_pos and ci_entirely_above_1 and quality_neutral and all_clean:
            verdict = "SYSTEM_REVALIDATED_STRONG"
        elif gm > 1 and majority and quality_neutral:
            verdict = "SYSTEM_REVALIDATED_POSITIVE"
        elif gm < 1 and (13 - n_pos) >= spec["strong"]["min_positive"]:
            verdict = "SYSTEM_REVERSED"
        else:
            verdict = "SYSTEM_NEUTRAL"
    else:
        if gm > 1 and strong_pos and ci_entirely_above_1 and quality_neutral and all_clean:
            verdict = "F9_REVALIDATED_STRONG"
        elif gm > 1 and majority and quality_neutral:
            verdict = "F9_REVALIDATED_POSITIVE_SMALL"
        elif gm < 1 and (13 - n_pos) >= 8:
            verdict = "F9_NEGATIVE"
        else:
            verdict = "F9_NEUTRAL"

    # ---- write artifacts ----
    P = spec["prefix"]
    with open(os.path.join(ART, f"{P}_RUNS.json"), "w") as f:
        json.dump(runs_artifact, f, indent=2)

    with open(os.path.join(ART, f"{P}_QUALITY.json"), "w") as f:
        json.dump({
            "definition": f"delta = {spec['delta_direction']} "
                          "per scene (spec direction)",
            "per_scene": quality_deltas,
            "mean_d_psnr": mean(d_psnr), "median_d_psnr": median(d_psnr),
            "mean_d_ssim": mean(d_ssim), "median_d_ssim": median(d_ssim),
            "mean_d_lpips": mean(d_lpips), "median_d_lpips": median(d_lpips),
            "psnr_outliers_gt_0p5dB": psnr_outliers,
            "quality_neutral_preregistered_band": quality_neutral,
            "band": "|mean dPSNR|<=0.15 dB AND |mean dSSIM|<=0.005 AND |mean dLPIPS|<=0.005",
        }, f, indent=2)

    with open(os.path.join(ART, f"{P}_TRAJECTORY.json"), "w") as f:
        json.dump({
            "note": "population/trajectory confound analysis; even numerically "
                    "equivalent changes may alter the optimization path via "
                    "floating-point reassociation",
            "per_scene_final_N": {
                s: {"num_N": scene_runs[s]["num"]["R"]["final_N"],
                    "den_N": scene_runs[s]["den"]["R"]["final_N"],
                    "N_ratio": round(n_ratio[s], 5)} for s in SCENES},
            "N_ratio_min": min(n_ratio.values()), "N_ratio_max": max(n_ratio.values()),
            "corr_logR_logNratio": corr_pop,
            "visible_gaussian_trajectories": {
                s: {"num": runs_artifact["per_scene"][s][spec["label_num"]]["population_trajectory"],
                    "den": runs_artifact["per_scene"][s][spec["label_den"]]["population_trajectory"]}
                for s in SCENES},
            "densification_events": {
                s: {"num": runs_artifact["per_scene"][s][spec["label_num"]]["densification_events"],
                    "den": runs_artifact["per_scene"][s][spec["label_den"]]["densification_events"]}
                for s in SCENES},
            "population_confound_assessment": {
                "corr": corr_pop,
                "interpretation": ("wall ratio must NOT be called an intrinsic "
                                   "renderer-kernel ratio if corr is strong "
                                   "(|corr|>0.5) and N-ratio spread is material"),
            },
        }, f, indent=2)

    with open(os.path.join(ART, f"{P}_STATS.json"), "w") as f:
        json.dump({
            "ratio_definition": spec["ratio_name"] +
                                f" = wall_{spec['label_num']} / wall_{spec['label_den']}",
            "wall_field": "timing.total_wall_s (pure 30K training loop; identical "
                          "field to the P8/P2 statistics)",
            "per_scene_ratios": {s: round(ratios[s], 5) for s in SCENES},
            "positive_scenes": [s for s in SCENES if ratios[s] > 1],
            "n_positive": n_pos,
            "geometric_mean": round(gm, 5),
            "median_ratio": round(med, 5),
            "ci95_log": {"lo": round(ci_lo_log, 6), "hi": round(ci_hi_log, 6),
                          "t_df12": T975_DF12,
                          "ratio_lo": round(math.exp(ci_lo_log), 5),
                          "ratio_hi": round(math.exp(ci_hi_log), 5),
                          "entirely_above_1": ci_entirely_above_1},
            "min_ratio": round(min(rvals), 5), "max_ratio": round(max(rvals), 5),
            "equivalent_wall_reduction": round(equiv_reduction, 5),
            "corr_logR_logNratio": corr_pop,
            "old_reference": old,
            "new_vs_old": {"delta_log_ratio": round(dlog_vs_old, 6),
                            "relative_change_pct": round(rel_vs_old * 100, 3),
                            "note": "descriptive comparison only; old and new "
                                    "GMs are never pooled"},
        }, f, indent=2)

    with open(os.path.join(ART, f"{P}_VERDICT.json"), "w") as f:
        json.dump({
            "campaign": camp, "verdict": verdict,
            "question": spec["question"],
            "rule_trace": {
                "gm": round(gm, 5), "gm_gt_1": gm > 1,
                "n_positive": f"{n_pos}/13",
                "strong_min_positive": spec["strong"]["min_positive"],
                "positive_min_positive": spec["positive"]["min_positive"],
                "ci95_entirely_above_1": ci_entirely_above_1,
                "quality_neutral": quality_neutral,
                "campaign_all_clean": all_clean,
                "clean_status_known": clean_known,
            },
            "stats_pointer": f"{P}_STATS.json",
            "quality_pointer": f"{P}_QUALITY.json",
            "population_pointer": f"{P}_TRAJECTORY.json",
            "ownership_note": ("system endpoint verdict, not an ownership claim; "
                               "never attribute the whole system ratio to F9/H8"
                               if camp == "p12a" else
                               "the only intended renderer difference is F9"),
        }, f, indent=2)

    # ---- report ----
    rep = []
    rep.append(f"# {P.replace('_', ' ')} Report — Strong-Baseline Revalidation"
               f" ({'current gsplat vs C0' if camp == 'p12a' else 'A0 vs A1 (F9)'})")
    rep.append("")
    rep.append(f"**Question**: {spec['question']}")
    rep.append(f"**Common loss**: P11-validated fused-SSIM (padding=\"same\"), "
               f"L = 0.8*L1 + 0.2*(1-SSIM); eval SSIM metric = SepSSIM (frozen).")
    rep.append("")
    rep.append(f"## Per-scene wall-clock ({spec['ratio_name']} = "
               f"{spec['label_num']}/{spec['label_den']})")
    rep.append("")
    rep.append(f"| scene | {spec['label_num']} wall (min) | {spec['label_den']} wall (min) | "
               f"{spec['ratio_name']} | ΔPSNR | ΔSSIM | ΔLPIPS | N-ratio |")
    rep.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SCENES:
        rn = runs_artifact["per_scene"][s][spec["label_num"]]
        rd = runs_artifact["per_scene"][s][spec["label_den"]]
        q = next(x for x in quality_deltas if x["scene"] == s)
        rep.append(f"| {s} | {rn['total_wall_min']:.1f} | {rd['total_wall_min']:.1f} "
                   f"| **{ratios[s]:.4f}** | {q['d_psnr']:+.2f} | {q['d_ssim']:+.4f} "
                   f"| {q['d_lpips']:+.4f} | {n_ratio[s]:.3f} |")
    rep.append("")
    rep.append(f"- geometric mean **{gm:.4f}**; median {med:.4f}; "
               f"min {min(rvals):.4f}; max {max(rvals):.4f}")
    rep.append(f"- {n_pos}/13 scenes with {spec['ratio_name']} > 1")
    rep.append(f"- 95% CI on GM (log-ratio t, df=12): "
               f"[{math.exp(ci_lo_log):.4f}, {math.exp(ci_hi_log):.4f}] "
               f"({'entirely > 1' if ci_entirely_above_1 else 'overlaps 1'})")
    rep.append(f"- equivalent {spec['label_den'].split('_')[0]} wall-time reduction: "
               f"{equiv_reduction*100:.2f}%")
    rep.append(f"- quality: mean ΔPSNR {mean(d_psnr):+.3f} dB (median {median(d_psnr):+.3f}), "
               f"mean ΔSSIM {mean(d_ssim):+.5f}, mean ΔLPIPS {mean(d_lpips):+.5f}; "
               f"preregistered quality-neutral band: {'PASS' if quality_neutral else 'FAIL'}")
    rep.append(f"- population: final-N ratio spans [{min(n_ratio.values()):.3f}, "
               f"{max(n_ratio.values()):.3f}]; corr(log {spec['ratio_name']}, log N-ratio) "
               f"= {corr_pop if corr_pop is None else round(corr_pop, 3)}")
    rep.append("")
    rep.append(f"## Old reference (descriptive only, never pooled)")
    rep.append(f"- {old['campaign']}: GM {old['gm']}, positive {old['positive']}")
    rep.append(f"- new fused-loss GM {gm:.4f}; Δlog-ratio {dlog_vs_old:+.4f}; "
               f"relative change {rel_vs_old*100:+.2f}%")
    rep.append("")
    rep.append(f"## Verdict")
    rep.append(f"**{verdict}**")
    rep.append("")
    with open(os.path.join(ART, f"{P}_REPORT.md"), "w") as f:
        f.write("\n".join(rep))

    print("\n".join(rep))
    print(f"\nWROTE {P}_RUNS/QUALITY/TRAJECTORY/STATS/VERDICT/REPORT artifacts")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p12a")
