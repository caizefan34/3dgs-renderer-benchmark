#!/usr/bin/env python3
"""
aggregate_results.py — Aggregation for the FINAL 30K benchmark.

Implements the FROZEN aggregation rules (aggregation_spec.json):
  - speedup = T_baseline / T_candidate (ratio); reduction = 1 - T_candidate/T_baseline (fraction). NEVER mixed.
  - ΔPSNR/ΔSSIM/ΔLPIPS = candidate - baseline. LPIPS sign NOT flipped (lower is better).
  - A failed/OOM scene MUST NOT silently disappear: aggregates carry missing_or_failed_scenes and compute over SUCCESS only.
  - per-scene reported BEFORE any aggregate; geomean for speedup, arithmetic mean for quality deltas.

Reads per-scene timing.json + quality.json + final_status.json under
artifacts/final-30k/<candidate>/<scene>/.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL30K = REPO_ROOT / "artifacts" / "final-30k"

MANIFEST = json.loads((FINAL30K / "manifest.json").read_text(encoding="utf-8"))
SCENES = [s["scene_id"] for s in MANIFEST["scenes"]]

STATUS_VOCAB = ["SUCCESS", "OOM", "CUDA_ERROR", "NUMERIC_FAILURE", "QUALITY_FAILURE", "INTERRUPTED"]


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _amxs(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def load_scene(candidate: str, scene: str, root: Path) -> dict:
    d = root / candidate / scene
    out = {"scene": scene, "candidate": candidate, "status": "MISSING"}
    fs = d / "final_status.json"
    if fs.exists():
        st = json.loads(fs.read_text(encoding="utf-8"))
        out["status"] = st.get("status", "UNKNOWN")
    if out["status"] != "SUCCESS":
        return out
    tj = d / "timing.json"
    if tj.exists():
        t = json.loads(tj.read_text(encoding="utf-8"))
        out["T_s"] = t.get("total_wall_s")
        out["iters_per_s"] = t.get("iterations_per_s")
        out["mean_iter_ms"] = t.get("mean_iter_ms")
        out["median_iter_ms"] = t.get("median_iter_ms")
        out["forward_ms"] = t.get("forward_ms")
        out["backward_ms"] = t.get("backward_ms")
        out["nested_fb_ms"] = t.get("nested_fb_ms")
        out["optimizer_ms"] = t.get("optimizer_ms")
        out["loss_ms"] = t.get("loss_ms")
    qj = d / "quality.json"
    if qj.exists():
        q = json.loads(qj.read_text(encoding="utf-8"))
        out["psnr"] = q.get("psnr")
        out["ssim"] = q.get("ssim")
        out["lpips"] = q.get("lpips")
    mj = d / "memory.json"
    if mj.exists():
        out["peak_vram_gb"] = json.loads(mj.read_text(encoding="utf-8")).get("peak_vram_gb")
    out["final_N_GS"] = (json.loads((d / "training_results.json").read_text(encoding="utf-8")).get("final_N")
                         if (d / "training_results.json").exists() else None)
    return out


def aggregate(candidate: str, baseline: str, root: Path) -> dict:
    per_scene = {}
    for s in SCENES:
        per_scene[s] = load_scene(candidate, s, root)
    base = {s: load_scene(baseline, s, root) for s in SCENES}

    speedups, reductions = [], []
    dP, dS, dL = [], [], []
    failed = []
    for s in SCENES:
        c, b = per_scene[s], base[s]
        if c["status"] != "SUCCESS" or b["status"] != "SUCCESS":
            failed.append({"scene": s, "candidate_status": c["status"], "baseline_status": b["status"]})
            continue
        if c.get("T_s") and b.get("T_s"):
            sp = b["T_s"] / c["T_s"]
            speedups.append({"scene": s, "speedup": sp, "reduction_pct": (1 - c["T_s"] / b["T_s"]) * 100})
        if None not in (c.get("psnr"), b.get("psnr")):
            dP.append({"scene": s, "dPSNR": c["psnr"] - b["psnr"],
                       "dSSIM": (c["ssim"] - b["ssim"]) if None not in (c.get("ssim"), b.get("ssim")) else None,
                       "dLPIPS": (c["lpips"] - b["lpips"]) if None not in (c.get("lpips"), b.get("lpips")) else None})

    sp_vals = [x["speedup"] for x in speedups]
    red_vals = [x["reduction_pct"] for x in speedups]
    result = {
        "candidate": candidate,
        "baseline": baseline,
        "n_scenes": len(SCENES),
        "n_success_pairs": len(speedups),
        "missing_or_failed_scenes": failed,
        "publication_grade_for_aggregate": (len(failed) == 0),
        "speedup": {
            "geomean": _geomean(sp_vals),
            "arithmetic_mean": _amxs(sp_vals),
            "per_scene": speedups,
        },
        "time_reduction_pct": {
            "geomean": _geomean([max(r, 1e-6) for r in red_vals]) if red_vals else None,
            "arithmetic_mean": _amxs(red_vals),
            "note": "reduction = 1 - T_candidate/T_baseline (fraction, shown as %). Separate from speedup ratio.",
        },
        "quality_delta": {
            "mean_dPSNR": _amxs([x["dPSNR"] for x in dP]),
            "mean_dSSIM": _amxs([x["dSSIM"] for x in dP if x["dSSIM"] is not None]),
            "mean_dLPIPS": _amxs([x["dLPIPS"] for x in dP if x["dLPIPS"] is not None]),
            "per_scene": dP,
            "lpips_sign_note": "ΔLPIPS = candidate - baseline; NEGATIVE = improvement. Sign NOT flipped.",
        },
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="B1A_ACCUTILE")
    ap.add_argument("--root", default=str(FINAL30K))
    args = ap.parse_args()
    res = aggregate(args.candidate, args.baseline, Path(args.root))
    print(json.dumps(res, indent=2))
