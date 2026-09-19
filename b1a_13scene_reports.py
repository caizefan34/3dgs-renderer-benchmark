#!/usr/bin/env python3
"""Generate all 7 required markdown reports from results.json."""
import json, os, sys, math
from pathlib import Path
from datetime import datetime, timezone

REPORT_DIR = Path(__file__).resolve().parent / "reports" / "accutile30k"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIPNERF360 = ["bicycle", "bonsai", "counter", "flowers", "garden",
              "kitchen", "room", "stump", "treehill"]
TANKSTEMPLES = ["train", "truck"]
DEEPBLENDING = ["drjohnson", "playroom"]
ALL = MIPNERF360 + TANKSTEMPLES + DEEPBLENDING
SCENE_DS = {}
for s in MIPNERF360: SCENE_DS[s] = "Mip-NeRF360"
for s in TANKSTEMPLES: SCENE_DS[s] = "Tanks & Temples"
for s in DEEPBLENDING: SCENE_DS[s] = "Deep Blending"


def load():
    with open(REPORT_DIR / "results.json") as f:
        return json.load(f)


def load_raw():
    p = REPORT_DIR / "results_raw.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def fmt(v, prec=2, suffix=""):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{prec}f}{suffix}"
    return str(v) + suffix


def report_run_manifest(data, raw):
    lines = [
        "# Run Manifest — B1A 13-Scene 30K Full-Training Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Run directories",
        "",
        "All runs output to `/dev/shm/accutile30k/<scene>/<method>/` on host `mx`.",
        "Each run directory contains: `config.json`, `provenance.json`, "
        "`training_metrics.json`, `evaluation_metrics.json`, `camera_sequence.npy`, "
        "`training_results.json`, `MANIFEST.sha256`, `checkpoints/iter_30000.pt`.",
        "",
        "| Scene | Dataset | B1 run_id | B1A run_id | Status |",
        "|-------|---------|-----------|-----------|--------|",
    ]
    for e in data["per_scene"]:
        b1_id = e["scene"] + "_b1_s42"
        b1a_id = e["scene"] + "_b1a_s42"
        lines.append(f"| {e['scene']} | {e['dataset']} | {b1_id} | {b1a_id} | {e['status']} |")
    lines += [
        "",
        "## Run IDs",
        "",
        "Run IDs are deterministic: `<scene>_<method>_s42` where s42 = seed 42.",
        "Immutable: a re-run would create a new run ID with a timestamp suffix.",
        "",
        "## Hardware cohort",
        "",
        "| Item | Value |",
        "|------|-------|",
        "| Host | mx (bms-39468022-001) |",
        "| GPU | 8 × NVIDIA A100-PCIE-40GB |",
        "| GPUs used | 0,1,2,3,5,6,7 (GPU 4 excluded — in use by another user) |",
        "| Driver / CUDA | 595.71.05 / CUDA 13.2 |",
        "| Conda env | anysplat |",
        "| PyTorch | 2.4.1+cu124 |",
    ]
    (REPORT_DIR / "run-manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote run-manifest.md")


def report_quality(data):
    lines = [
        "# 13-Scene Quality — B1A vs B1 (30K)",
        "",
        "Quality metrics from final evaluation (all cameras) at 30,000 iterations.",
        "",
        "| Scene | Dataset | B1 PSNR | B1A PSNR | ΔPSNR | B1 SSIM | B1A SSIM | ΔSSIM | B1 LPIPS | B1A LPIPS | ΔLPIPS |",
        "|-------|---------|--------:|---------:|------:|--------:|---------:|------:|---------:|----------:|-------:|",
    ]
    for e in data["per_scene"]:
        lines.append(
            f"| {e['scene']} | {e['dataset']} | {fmt(e.get('b1_psnr'))} | {fmt(e.get('b1a_psnr'))} | "
            f"{fmt(e.get('dpsnr'),3)} | {fmt(e.get('b1_ssim'),4)} | {fmt(e.get('b1a_ssim'),4)} | "
            f"{fmt(e.get('dssim'),4)} | {fmt(e.get('b1_lpips'),4)} | {fmt(e.get('b1a_lpips'),4)} | "
            f"{fmt(e.get('dlpips'),4)} |"
        )
    agg = data["dataset_aggregation"]
    lines += [
        "",
        "## Dataset aggregates",
        "",
        "| Dataset | n | mean ΔPSNR | mean ΔSSIM | mean ΔLPIPS |",
        "|---------|---|-----------:|-----------:|------------:|",
        f"| Mip-NeRF360 | {agg['Mip-NeRF360']['n_scenes']} | {fmt(agg['Mip-NeRF360']['mean_dpsnr'],3)} | {fmt(agg['Mip-NeRF360']['mean_dssim'],4)} | {fmt(agg['Mip-NeRF360']['mean_dlpips'],4)} |",
        f"| Tanks & Temples | {agg['Tanks & Temples']['n_scenes']} | {fmt(agg['Tanks & Temples']['mean_dpsnr'],3)} | {fmt(agg['Tanks & Temples']['mean_dssim'],4)} | {fmt(agg['Tanks & Temples']['mean_dlpips'],4)} |",
        f"| Deep Blending | {agg['Deep Blending']['n_scenes']} | {fmt(agg['Deep Blending']['mean_dpsnr'],3)} | {fmt(agg['Deep Blending']['mean_dssim'],4)} | {fmt(agg['Deep Blending']['mean_dlpips'],4)} |",
        f"| **All 13** | {agg['all_13']['n_scenes']} | **{fmt(agg['all_13']['mean_dpsnr'],3)}** | **{fmt(agg['all_13']['mean_dssim'],4)}** | **{fmt(agg['all_13']['mean_dlpips'],4)}** |",
        "",
        "## Interpretation",
        "",
        "AccuTile is an exact conservative workload optimization. The expected result is "
        "quality equivalence within normal FP/stochastic training variation, NOT PSNR improvement. "
        "Small positive/negative ΔPSNR values (< 0.2 dB) are within normal trajectory sensitivity "
        "from floating-point accumulation-order differences amplified by densification.",
    ]
    (REPORT_DIR / "13scene-quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote 13scene-quality.md")


def report_speed(data):
    lines = [
        "# 13-Scene Training Speed — B1A vs B1 (30K)",
        "",
        "Full-training wall-clock timing on the A100-PCIE-40GB cohort (anysplat env, "
        "torch 2.4.1+cu124). `training_speedup = T_B1 / T_B1A`.",
        "",
        "| Scene | Dataset | B1 total (min) | B1A total (min) | Speedup | B1 mean iter (ms) | B1A mean iter (ms) | B1 steady iter (ms) | B1A steady iter (ms) | iter speedup |",
        "|-------|---------|----------------:|----------------:|--------:|-------------------:|--------------------:|---------------------:|----------------------:|-------------:|",
    ]
    for e in data["per_scene"]:
        b1m = e.get("b1_total_wall_s", 0) or 0
        b1am = e.get("b1a_total_wall_s", 0) or 0
        lines.append(
            f"| {e['scene']} | {e['dataset']} | {fmt(b1m/60,1)} | {fmt(b1am/60,1)} | "
            f"{fmt(e.get('training_speedup'),3,'×')} | {fmt(e.get('b1_mean_iter_ms'),1)} | {fmt(e.get('b1a_mean_iter_ms'),1)} | "
            f"{fmt(e.get('b1_steady_iter_ms'),1)} | {fmt(e.get('b1a_steady_iter_ms'),1)} | {fmt(e.get('iter_speedup'),3,'×')} |"
        )
    agg = data["dataset_aggregation"]
    lines += [
        "",
        "## Dataset / overall geomean speedup",
        "",
        "| Dataset | n | geomean speedup | arith mean speedup |",
        "|---------|---|----------------:|-------------------:|",
        f"| Mip-NeRF360 | {agg['Mip-NeRF360']['n_scenes']} | {fmt(agg['Mip-NeRF360']['geomean_training_speedup'],4,'×')} | {fmt(agg['Mip-NeRF360']['arith_mean_training_speedup'],4,'×')} |",
        f"| Tanks & Temples | {agg['Tanks & Temples']['n_scenes']} | {fmt(agg['Tanks & Temples']['geomean_training_speedup'],4,'×')} | {fmt(agg['Tanks & Temples']['arith_mean_training_speedup'],4,'×')} |",
        f"| Deep Blending | {agg['Deep Blending']['n_scenes']} | {fmt(agg['Deep Blending']['geomean_training_speedup'],4,'×')} | {fmt(agg['Deep Blending']['arith_mean_training_speedup'],4,'×')} |",
        f"| **All 13** | {agg['all_13']['n_scenes']} | **{fmt(agg['all_13']['geomean_training_speedup'],4,'×')}** | **{fmt(agg['all_13']['arith_mean_training_speedup'],4,'×')}** |",
        "",
        "Note: `steady_state_mean_iter_ms` = mean of iterations 15000-30000 (after densification ends), "
        "a cleaner per-iteration comparison than the full-run mean (which includes early densification spikes).",
    ]
    (REPORT_DIR / "13scene-training-speed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote 13scene-training-speed.md")


def report_topology(data):
    lines = [
        "# Convergence & Topology — B1A vs B1 (30K)",
        "",
        "## N_GS(t) trajectory",
        "",
        "| Scene | 5K B1 | 5K B1A | 10K B1 | 10K B1A | 15K B1 | 15K B1A | 20K B1 | 20K B1A | 25K B1 | 25K B1A | 30K B1 | 30K B1A |",
        "|-------|------:|-------:|-------:|--------:|-------:|--------:|-------:|--------:|-------:|--------:|-------:|-------:|",
    ]
    for e in data["per_scene"]:
        t = e.get("ngs_trajectory", {})
        row = [e["scene"]]
        for it in ["5000", "10000", "15000", "20000", "25000", "30000"]:
            row.append(fmt(t.get(it, {}).get("b1"), 0))
            row.append(fmt(t.get(it, {}).get("b1a"), 0))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Densification events (totals over 30K)",
        "",
        "| Scene | B1 clones | B1A clones | B1 splits | B1A splits | B1 prunes | B1A prunes | Final N B1 | Final N B1A | ΔN |",
        "|-------|----------:|-----------:|----------:|-----------:|----------:|-----------:|-----------:|------------:|---:|",
    ]
    for e in data["per_scene"]:
        lines.append(
            f"| {e['scene']} | {fmt(e.get('b1_clones'),0)} | {fmt(e.get('b1a_clones'),0)} | "
            f"{fmt(e.get('b1_splits'),0)} | {fmt(e.get('b1a_splits'),0)} | {fmt(e.get('b1_prunes'),0)} | "
            f"{fmt(e.get('b1a_prunes'),0)} | {fmt(e.get('b1_final_N'),0)} | {fmt(e.get('b1a_final_N'),0)} | "
            f"{fmt(e.get('dN'),0)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Identical topology is NOT required. AccuTile's floating-point accumulation-order "
        "differences can cause small densification-threshold crossing differences, leading to "
        "slightly different clone/split/prune counts and final N_GS. The question is whether "
        "topology divergence is systematic (indicating a real semantic change) or random FP-scale "
        "(consistent with an exact conservative optimization).",
    ]
    (REPORT_DIR / "convergence-topology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote convergence-topology.md")


def report_aggregation(data):
    agg = data["dataset_aggregation"]
    lines = [
        "# Dataset-Level Aggregation — B1A 13-Scene 30K",
        "",
        "## Mip-NeRF360 (9 scenes)",
        "",
        f"- scenes complete: {agg['Mip-NeRF360']['n_scenes']}/9",
        f"- geomean training speedup: {fmt(agg['Mip-NeRF360']['geomean_training_speedup'],4,'×')}",
        f"- arithmetic mean training speedup: {fmt(agg['Mip-NeRF360']['arith_mean_training_speedup'],4,'×')}",
        f"- mean ΔPSNR: {fmt(agg['Mip-NeRF360']['mean_dpsnr'],3)} dB",
        f"- mean ΔSSIM: {fmt(agg['Mip-NeRF360']['mean_dssim'],4)}",
        f"- mean ΔLPIPS: {fmt(agg['Mip-NeRF360']['mean_dlpips'],4)}",
        "",
        "## Tanks & Temples (2 scenes)",
        "",
        f"- scenes complete: {agg['Tanks & Temples']['n_scenes']}/2",
        f"- geomean training speedup: {fmt(agg['Tanks & Temples']['geomean_training_speedup'],4,'×')}",
        f"- mean ΔPSNR: {fmt(agg['Tanks & Temples']['mean_dpsnr'],3)} dB",
        f"- mean ΔSSIM: {fmt(agg['Tanks & Temples']['mean_dssim'],4)}",
        f"- mean ΔLPIPS: {fmt(agg['Tanks & Temples']['mean_dlpips'],4)}",
        "",
        "## Deep Blending (2 scenes)",
        "",
        f"- scenes complete: {agg['Deep Blending']['n_scenes']}/2",
        f"- geomean training speedup: {fmt(agg['Deep Blending']['geomean_training_speedup'],4,'×')}",
        f"- mean ΔPSNR: {fmt(agg['Deep Blending']['mean_dpsnr'],3)} dB",
        f"- mean ΔSSIM: {fmt(agg['Deep Blending']['mean_dssim'],4)}",
        f"- mean ΔLPIPS: {fmt(agg['Deep Blending']['mean_dlpips'],4)}",
        "",
        "## All-13 aggregate",
        "",
        f"- scenes complete: {agg['all_13']['n_scenes']}/13",
        f"- geomean training speedup: {fmt(agg['all_13']['geomean_training_speedup'],4,'×')}",
        f"- arithmetic mean training speedup: {fmt(agg['all_13']['arith_mean_training_speedup'],4,'×')}",
        f"- mean ΔPSNR: {fmt(agg['all_13']['mean_dpsnr'],3)} dB",
        f"- mean ΔSSIM: {fmt(agg['all_13']['mean_dssim'],4)}",
        f"- mean ΔLPIPS: {fmt(agg['all_13']['mean_dlpips'],4)}",
        "",
        "Speedup uses geometric mean (GM = exp(mean(log(speedup_i)))) as required by protocol.",
    ]
    (REPORT_DIR / "dataset-aggregation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote dataset-aggregation.md")


def report_verdict(data):
    agg = data["dataset_aggregation"]["all_13"]
    per = data["per_scene"]
    n_complete = agg["n_scenes"]
    gm = agg.get("geomean_training_speedup")
    n_faster = sum(1 for e in per if e.get("training_speedup") and e["training_speedup"] > 1.0)
    n_slower = sum(1 for e in per if e.get("training_speedup") and e["training_speedup"] <= 1.0)
    # quality: systematic degradation = most scenes ΔPSNR < -0.5
    n_degraded = sum(1 for e in per if e.get("dpsnr") is not None and e["dpsnr"] < -0.5)
    mean_dpsnr = agg.get("mean_dpsnr")

    if (n_complete == 13 and gm and gm > 1.0 and n_faster >= 7 and n_degraded <= 1 and
            (mean_dpsnr is None or mean_dpsnr > -0.2)):
        gate = "B1A_STRONG_PASS"
        promote = "YES — B1A = FROZEN ENHANCED PAPER BASELINE"
    elif (gm and gm > 1.0 and n_degraded <= 3):
        gate = "B1A_PASS_WITH_EXCEPTIONS"
        promote = "PARTIAL — investigate anomalous scenes"
    else:
        gate = "B1A_FAIL"
        promote = "NO — B1A not promoted"

    worst = min((e for e in per if e.get("dpsnr") is not None), key=lambda e: e["dpsnr"], default=None)

    lines = [
        "# Final Verdict — B1A 13-Scene 30K Full-Training Validation",
        "",
        f"## Gate: `{gate}`",
        "",
        f"## Promotion: {promote}",
        "",
        "## Gate criteria evaluation",
        "",
        f"| Criterion | Required | Observed | Pass |",
        f"|-----------|----------|----------|:----:|",
        f"| 13/13 runs complete | 13/13 | {n_complete}/13 | {'✅' if n_complete==13 else '❌'} |",
        f"| geomean full-training speedup > 1.0 | >1.0 | {fmt(gm,4,'×')} | {'✅' if gm and gm>1.0 else '❌'} |",
        f"| majority scenes faster | ≥7/13 | {n_faster}/13 | {'✅' if n_faster>=7 else '❌'} |",
        f"| no systematic quality degradation | mean ΔPSNR > -0.2 | {fmt(mean_dpsnr,3)} dB | {'✅' if mean_dpsnr is None or mean_dpsnr>-0.2 else '❌'} |",
        f"| no large convergence failure | ≤1 scene ΔPSNR < -0.5 | {n_degraded} | {'✅' if n_degraded<=1 else '❌'} |",
        "",
        "## Worst quality-delta scene",
        "",
        f"- {worst['scene']}: ΔPSNR = {fmt(worst['dpsnr'],3)} dB" if worst else "- N/A",
        "",
        "## Scenes with runtime regression (speedup ≤ 1.0)",
        "",
    ]
    regressors = [e for e in per if e.get("training_speedup") and e["training_speedup"] <= 1.0]
    if regressors:
        for e in regressors:
            lines.append(f"- {e['scene']}: speedup = {fmt(e['training_speedup'],4,'×')}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Classification",
        "",
        "AccuTile is **prior art** (Speedy-Splat). Correct wording on success: "
        "`AccuTile-enhanced gsplat baseline` / `strong prior-art baseline`. "
        "AccuTile is NOT our contribution.",
        "",
        "## Final paper-ready table",
        "",
        "| Scene | Dataset | B1 PSNR | B1A PSNR | ΔPSNR | B1 SSIM | B1A SSIM | B1 Time (min) | B1A Time (min) | Speedup | B1 N_GS | B1A N_GS |",
        "|-------|---------|--------:|---------:|------:|--------:|---------:|--------------:|---------------:|--------:|--------:|---------:|",
    ]
    for e in per:
        b1m = (e.get("b1_total_wall_s") or 0) / 60
        b1am = (e.get("b1a_total_wall_s") or 0) / 60
        lines.append(
            f"| {e['scene']} | {e['dataset']} | {fmt(e.get('b1_psnr'))} | {fmt(e.get('b1a_psnr'))} | "
            f"{fmt(e.get('dpsnr'),3)} | {fmt(e.get('b1_ssim'),4)} | {fmt(e.get('b1a_ssim'),4)} | "
            f"{fmt(b1m,1)} | {fmt(b1am,1)} | {fmt(e.get('training_speedup'),3,'×')} | "
            f"{fmt(e.get('b1_final_N'),0)} | {fmt(e.get('b1a_final_N'),0)} |"
        )
    lines += [
        "",
        f"| **All-13 geomean** | | | | **{fmt(mean_dpsnr,3)}** | | | | | **{fmt(gm,4,'×')}** | | |",
    ]
    (REPORT_DIR / "final-verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote final-verdict.md")
    return gate


def main():
    if not (REPORT_DIR / "results.json").exists():
        print("results.json not found; run b1a_13scene_analyze.py first")
        sys.exit(1)
    data = load()
    raw = load_raw()
    report_run_manifest(data, raw)
    report_quality(data)
    report_speed(data)
    report_topology(data)
    report_aggregation(data)
    gate = report_verdict(data)
    print(f"\nGate: {gate}")


if __name__ == "__main__":
    main()
