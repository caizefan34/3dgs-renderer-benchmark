#!/usr/bin/env python3
"""Analyze the confirmatory_formal_30k matrix (5 methods x 11 scenes x 3 seeds).

Pairs every method against the matched control ``gsplat`` on (scene, seed),
computes paired deltas, and reports scene-block-bootstrap 95% CIs, effect
sizes, CV, and non-inferiority decisions against the pre-registered margins:

- PSNR paired-delta 95% CI lower bound >= -0.10 dB
- SSIM paired-delta 95% CI lower bound >= -0.003
- LPIPS paired-delta 95% CI upper bound <= +0.005
- end-to-end wall time >= 10% lower, speedup-ratio 95% CI lower bound > 1.0
- TTQ (time_to_quality) 95% CI upper bound <= 0 (not slower than gsplat)

Also reports peak VRAM, energy, final gaussian count, and emits the raw
per-(scene,seed) quality-vs-wall-time curves for plotting.

Usage:
    python src/scripts/analyze_confirmatory_matrix.py \
        --results-dir artifacts/training-ablation/results \
        --out artifacts/training-ablation/confirmatory-analysis.json \
        --markdown artifacts/training-ablation/confirmatory-analysis.md \
        --curves artifacts/training-ablation/confirmatory-curves.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

METRIC_SPEC = {
    "psnr_db": ("quality", "psnr_db", "higher"),
    "ssim": ("quality", "ssim", "higher"),
    "lpips": ("quality", "lpips", "lower"),
    "wall_time_seconds": ("performance", "wall_time_seconds", "lower"),
    "time_to_quality_seconds": ("performance", "time_to_quality_seconds", "lower"),
    "peak_gpu_memory_mib": ("resources", "peak_gpu_memory_mib", "lower"),
    "energy_joules": ("resources", "energy_joules", "lower"),
    "final_gaussian_count": ("resources", "final_gaussian_count", "lower"),
}

# delta-space bounds; None means no pre-registered gate for that bound
NI_MARGINS = {
    "psnr_db": {"lower": -0.10, "upper": None},
    "ssim": {"lower": -0.003, "upper": None},
    "lpips": {"lower": None, "upper": 0.005},
    "wall_time_seconds": {"lower": None, "upper": None},  # speedup-ratio gate instead
    "time_to_quality_seconds": {"lower": None, "upper": 0.0},
    "peak_gpu_memory_mib": {"lower": None, "upper": None},
    "energy_joules": {"lower": None, "upper": None},
    "final_gaussian_count": {"lower": None, "upper": None},
}

WALL_ACCEL_FRACTION = 0.10  # >=10% end-to-end wall-time reduction required
BOOTSTRAP_ITERS = 10_000
RNG_SEED = 12345


def _field(doc: dict, spec: tuple) -> float | None:
    section, key, _ = spec
    return doc.get(section, {}).get(key)


def load_results(results_dir: Path) -> dict:
    jobs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("confirmatory_formal_30k--*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("status") != "complete":
            continue
        jobs[doc["job_id"]] = doc
    return jobs


def pair_table(jobs: dict) -> dict:
    """Return {(scene, seed): {method: doc}}."""
    table: dict[tuple, dict] = defaultdict(dict)
    for doc in jobs.values():
        table[(doc["scene"], doc["seed"])][doc["method"]] = doc
    return table


def block_bootstrap_ci(deltas_by_scene: dict, rng: np.random.Generator, iters: int = BOOTSTRAP_ITERS) -> dict:
    scenes = list(deltas_by_scene)
    means = []
    for _ in range(iters):
        picks = rng.choice(scenes, size=len(scenes), replace=True)
        vals = [v for s in picks for v in deltas_by_scene[s]]
        means.append(float(np.mean(vals)))
    means.sort()
    lo = means[int(round(0.025 * (len(means) - 1)))]
    hi = means[int(round(0.975 * (len(means) - 1)))]
    return {"mean": float(np.mean(means)), "ci95_lower": lo, "ci95_upper": hi}


def analyze(jobs: dict) -> dict:
    table = pair_table(jobs)
    scene_order = sorted({k[0] for k in table})
    methods = sorted({m for row in table.values() for m in row})
    baseline = "gsplat"
    if baseline not in methods:
        raise SystemExit(f"missing baseline method {baseline!r} in results")
    methods = [m for m in methods if m != baseline]

    report = {
        "baseline": baseline,
        "methods": methods,
        "n_scenes": len(scene_order),
        "scenes": scene_order,
        "n_jobs": len(jobs),
        "paired_cells": len(table),
        "per_method": {},
        "per_scene": {},
        "curves": {},
    }

    for method in methods:
        deltas = {metric: [] for metric in METRIC_SPEC}
        deltas_by_scene = {metric: defaultdict(list) for metric in METRIC_SPEC}
        ratios = []  # baseline_wall / method_wall (speedup ratio)
        rows = []
        for scene in scene_order:
            for seed in (0, 1, 2):
                row = table.get((scene, seed))
                if not row or baseline not in row or method not in row:
                    continue
                base, cand = row[baseline], row[method]
                entry = {"scene": scene, "seed": seed}
                for metric, spec in METRIC_SPEC.items():
                    bv, cv = _field(base, spec), _field(cand, spec)
                    if bv is None or cv is None:
                        continue
                    d = cv - bv
                    deltas[metric].append(d)
                    deltas_by_scene[metric][scene].append(d)
                    entry[metric + "_baseline"] = bv
                    entry[metric + "_candidate"] = cv
                bw = _field(base, ("performance", "wall_time_seconds", "lower"))
                mw = _field(cand, ("performance", "wall_time_seconds", "lower"))
                if bw and mw:
                    ratios.append(bw / mw)
                rows.append(entry)
        report["per_scene"][method] = rows

        rng = np.random.default_rng(RNG_SEED)
        metric_report = {}
        for metric, spec in METRIC_SPEC.items():
            if not deltas[metric]:
                continue
            d = deltas[metric]
            ci = block_bootstrap_ci(deltas_by_scene[metric], rng)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            cohens = (ci["mean"] / sd) if sd > 0 else None
            cv = (sd / abs(statistics.mean(d))) if statistics.mean(d) != 0 else None
            margin = NI_MARGINS[metric]
            direction = spec[2]
            passed = None
            if margin.get("lower") is not None and direction == "higher":
                passed = ci["ci95_lower"] >= margin["lower"]
            elif margin.get("upper") is not None and direction == "lower":
                passed = ci["ci95_upper"] <= margin["upper"]
            metric_report[metric] = {
                "delta_mean": float(statistics.mean(d)),
                "delta_sd": sd,
                "delta_ci95_lower": ci["ci95_lower"],
                "delta_ci95_upper": ci["ci95_upper"],
                "cohens_dz": cohens,
                "cv": cv,
                "margin": margin,
                "ni_passed": passed,
            }
        # wall-time speedup gate
        if ratios:
            sorted_ratios = sorted(ratios)
            speedup_mean = float(np.mean(ratios))
            speedup_ci_lo = float(sorted_ratios[int(round(0.025 * (len(ratios) - 1)))])
            metric_report["wall_speedup_ratio"] = {
                "mean": speedup_mean,
                "ci95_lower": speedup_ci_lo,
                "required_min": 1.0 / (1.0 - WALL_ACCEL_FRACTION),
                "accel_passed": speedup_mean >= 1.0 / (1.0 - WALL_ACCEL_FRACTION)
                and speedup_ci_lo > 1.0,
            }
        report["per_method"][method] = metric_report

    # quality-vs-wall-time curves (baseline + all methods), per job
    for doc in sorted(jobs.values(), key=lambda d: (d["scene"], d["seed"], d["method"])):
        key = f"{doc['scene']}__s{doc['seed']}__{doc['method']}"
        report["curves"][key] = [
            {"wall_time_seconds": p.get("wall_time_seconds"), "psnr_db": p.get("psnr_db"),
             "ssim": p.get("ssim"), "lpips": p.get("lpips"), "iteration": p.get("iteration")}
            for p in doc.get("quality_curve", [])
        ]
    return report


def render_markdown(report: dict) -> str:
    lines = ["# HiGS Confirmatory Matrix Analysis (5 methods x 11 scenes x 3 seeds)",
             "", f"- baseline: `{report['baseline']}`; methods: {', '.join(report['methods'])}",
             f"- jobs analyzed: {report['n_jobs']}; paired (scene, seed) cells: {report['paired_cells']}",
             "", "## Paired deltas vs gsplat (scene-block bootstrap 95% CI)", ""]
    header = ("| method | metric | delta mean | 95% CI | cohen's dz | CV | NI margin | passed |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for method in report["methods"]:
        mr = report["per_method"][method]
        for metric in METRIC_SPEC:
            if metric not in mr:
                continue
            m = mr[metric]
            ci = f"[{m['delta_ci95_lower']:.4f}, {m['delta_ci95_upper']:.4f}]"
            margin = m["margin"]
            mstr = "-"
            if margin.get("lower") is not None:
                mstr = f">= {margin['lower']}"
            elif margin.get("upper") is not None:
                mstr = f"<= {margin['upper']}"
            dz = f"{m['cohens_dz']:.2f}" if m["cohens_dz"] is not None else "-"
            cv = f"{m['cv']:.3f}" if m["cv"] is not None else "-"
            lines.append(f"| {method} | {metric} | {m['delta_mean']:.4f} | {ci} | {dz} | {cv} | {mstr} | {m['ni_passed']} |")
        if "wall_speedup_ratio" in mr:
            w = mr["wall_speedup_ratio"]
            lines.append(
                f"| {method} | wall speedup ratio | {w['mean']:.3f} | "
                f"CI lo {w['ci95_lower']:.3f} (need > 1.0) | - | - | "
                f"mean >= {w['required_min']:.3f} | {w['accel_passed']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "artifacts" / "training-ablation" / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "training-ablation" / "confirmatory-analysis.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "artifacts" / "training-ablation" / "confirmatory-analysis.md")
    parser.add_argument("--curves", type=Path, default=ROOT / "artifacts" / "training-ablation" / "confirmatory-curves.json")
    args = parser.parse_args(argv)

    jobs = load_results(args.results_dir)
    report = analyze(jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    args.curves.write_text(json.dumps(report["curves"], indent=2), encoding="utf-8")
    print(f"analyzed {report['n_jobs']} jobs across {report['n_scenes']} scenes")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
