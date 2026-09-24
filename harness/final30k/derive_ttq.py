#!/usr/bin/env python3
"""
derive_ttq.py — Time-to-quality derivation for the FINAL 30K benchmark.

Implements the FROZEN TTQ spec (ttq_spec.json), including the R1 censoring semantics:
  - Thresholds are derived from the FROZEN baseline reference (B1A_ACCUTILE) final-eval quality,
    NOT chosen after seeing final results.
  - time_to_PSNR: cumulative wall time at first eval checkpoint where PSNR >= threshold.
  - time_to_SSIM: cumulative wall time at first eval checkpoint where SSIM >= threshold.
  - time_to_LPIPS: cumulative wall time at first eval checkpoint where LPIPS <= threshold (lower better).
  - CENSORING (R1): for a scene that NEVER reaches the frozen threshold within 30000 steps:
        ttq = null, status = TTQ_NOT_REACHED, censor_time = total_wall_time_30000.
    censor_time is a right-censoring bound, NOT an achieved TTQ. It is NEVER treated as an
    achieved TTQ value in any comparison or geomean.
  - The primary TTQ geomean includes ONLY comparable reached/reached pairs (both baseline AND
    candidate reached the threshold) and explicitly reports reach counts.
  - TTQ_speedup = TTQ_baseline / TTQ_candidate (ratio; >1 = candidate reaches reference quality faster).

Reads training_curve.csv (step, wall_time, psnr, ssim, lpips, ...) per candidate/scene.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL30K = REPO_ROOT / "artifacts" / "final-30k"


def load_curve(root: Path, candidate: str, scene: str) -> list:
    p = root / candidate / scene / "training_curve.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "step": int(r["step"]),
                    "wall_time_s": float(r["wall_time"]),
                    "psnr": float(r["psnr"]) if r.get("psnr") not in (None, "", "null") else None,
                    "ssim": float(r["ssim"]) if r.get("ssim") not in (None, "", "null") else None,
                    "lpips": float(r["lpips"]) if r.get("lpips") not in (None, "", "null") else None,
                })
            except (KeyError, ValueError):
                continue
    return rows


def ttq_for_metric(rows: list, metric: str, threshold: float, direction: str, censor_time_s: float):
    """direction: '>=' for PSNR/SSIM, '<=' for LPIPS.

    Returns (ttq_s, reached, censor_time_s):
      reached=True  -> ttq_s = cumulative wall time at the first crossing checkpoint
      reached=False -> ttq_s = None (NOT an achieved value); censor_time_s is the
                       right-censoring bound (total wall time). NEVER treat censor_time
                       as an achieved TTQ.
    """
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        if (direction == ">=" and v >= threshold) or (direction == "<=" and v <= threshold):
            return r["wall_time_s"], True, censor_time_s
    return None, False, censor_time_s


def derive(candidate: str, baseline: str, root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenes = [s["scene_id"] for s in manifest["scenes"]]

    # Step 1: derive thresholds from the baseline reference FINAL eval (iter 30000)
    thresholds = {}
    for s in scenes:
        bcurve = load_curve(root, baseline, s)
        final_b = [r for r in bcurve if r["step"] >= 30000]
        if not final_b:
            final_b = [bcurve[-1]] if bcurve else []
        if not final_b:
            thresholds[s] = None
            continue
        fb = final_b[0]
        thresholds[s] = {"psnr": fb["psnr"], "ssim": fb["ssim"], "lpips": fb["lpips"],
                         "total_wall_s": fb["wall_time_s"]}

    out = {
        "candidate": candidate,
        "baseline": baseline,
        "censoring_semantics": "ttq=null + TTQ_NOT_REACHED + censor_time for non-reached scenes; censor_time is a right-censoring bound, NOT an achieved TTQ; geomean over comparable reached/reached pairs only",
        "thresholds": thresholds,
        "per_scene": {},
    }

    reached_pairs_psnr = []   # comparable reached/reached TTQ speedups (baseline & candidate both reached)
    reach_counts = {
        "n_scenes": len(scenes),
        "n_reached_baseline_psnr": 0,
        "n_reached_candidate_psnr": 0,
        "n_reached_pairs_psnr": 0,
        "n_not_reached_candidate_psnr": 0,
        "n_no_curve": 0,
    }

    for s in scenes:
        th = thresholds.get(s)
        ccurve = load_curve(root, candidate, s)
        bcurve = load_curve(root, baseline, s)
        if not ccurve or not bcurve or not th:
            out["per_scene"][s] = {"status": "NO_CURVE"}
            reach_counts["n_no_curve"] += 1
            continue
        total_c = ccurve[-1]["wall_time_s"]      # candidate censor time (total wall time @30000)
        total_b = bcurve[-1]["wall_time_s"]      # baseline censor time

        # Candidate TTQ (censored when not reached)
        ttq_p, r_p, cens_p = ttq_for_metric(ccurve, "psnr", th["psnr"], ">=", total_c)
        ttq_s, r_s, _ = ttq_for_metric(ccurve, "ssim", th["ssim"], ">=", total_c)
        ttq_l, r_l, _ = ttq_for_metric(ccurve, "lpips", th["lpips"], "<=", total_c)

        # Baseline TTQ to its own threshold (censored when not reached)
        ttq_p_b, r_p_b, _ = ttq_for_metric(bcurve, "psnr", th["psnr"], ">=", total_b)

        if r_p_b:
            reach_counts["n_reached_baseline_psnr"] += 1
        if r_p:
            reach_counts["n_reached_candidate_psnr"] += 1
        else:
            reach_counts["n_not_reached_candidate_psnr"] += 1

        # Comparable reached/reached pair ONLY: both baseline and candidate reached the threshold.
        # Censored scenes (either side) are EXCLUDED from the geomean — censor_time is never
        # substituted for a missing TTQ.
        if r_p and r_p_b and ttq_p and ttq_p_b:
            reached_pairs_psnr.append(ttq_p_b / ttq_p)
            reach_counts["n_reached_pairs_psnr"] += 1

        out["per_scene"][s] = {
            "thresholds": th,
            "time_to_PSNR_s": ttq_p,
            "status_PSNR": "REACHED" if r_p else "TTQ_NOT_REACHED",
            "censor_time_PSNR_s": cens_p if not r_p else None,
            "time_to_SSIM_s": ttq_s,
            "status_SSIM": "REACHED" if r_s else "TTQ_NOT_REACHED",
            "censor_time_SSIM_s": total_c if not r_s else None,
            "time_to_LPIPS_s": ttq_l,
            "status_LPIPS": "REACHED" if r_l else "TTQ_NOT_REACHED",
            "censor_time_LPIPS_s": total_c if not r_l else None,
        }

    gm = (math.exp(sum(math.log(x) for x in reached_pairs_psnr) / len(reached_pairs_psnr))
          if reached_pairs_psnr else None)
    out["ttq_speedup_to_PSNR"] = {
        "geomean": gm,
        "geomean_scope": "comparable reached/reached pairs ONLY (baseline AND candidate both reached the threshold)",
        "reach_counts": reach_counts,
        "note": ("Geomean over %d/%d scenes. Censored scenes are EXCLUDED and reported via reach_counts; "
                 "censor_time is a right-censoring bound, NOT an achieved TTQ."
                 % (reach_counts["n_reached_pairs_psnr"], reach_counts["n_scenes"])),
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="B1A_ACCUTILE")
    ap.add_argument("--root", default=str(FINAL30K))
    args = ap.parse_args()
    print(json.dumps(derive(args.candidate, args.baseline, args.root), indent=2))
