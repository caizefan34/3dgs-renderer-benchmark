#!/usr/bin/env python3
"""
C42 13-Scene Final Analysis & Report Generator.

Reads all training and evaluation results, computes:
  - Per-scene deltas (dPSNR, dSSIM, dLPIPS, dGaussians, dVRAM, speedup)
  - Dataset-level aggregates (MipNeRF360, T&T, DB, overall)
  - Robustness classification (STRONG_SUCCESS, SPEED_QUALITY_TRADEOFF, NO_SPEED_BENEFIT)
  - Final scientific gate (STRONG_KEEP, KEEP_WITH_TRADEOFF, MIXED, FAIL)

Produces:
  - reports/c42-13scene-final.md
  - results/c42_13scene/per_scene_deltas.json
  - results/c42_13scene/dataset_aggregates.json
  - results/c42_13scene/overall_aggregate.json
  - results/c42_13scene/classification.json
  - results/c42_13scene/final_decision.json
"""
import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent
RESULTS_BASE = REPO / "results/c42_13scene"
REPORTS_DIR = REPO / "reports"

# Scene definitions
SCENES = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill",
                   "room", "counter", "kitchen", "bonsai"],
    "tanksandtemples": ["truck", "train"],
    "deepblending": ["drjohnson", "playroom"],
}

# Existing 3-scene data (from unified_metrics.json + training results)
EXISTING_TIMING = {
    "room": {"reference_wall_s": None, "c42_wall_s": 919.8, "reference_iter_ms": 55.4, "c42_iter_ms": 29.1},
    "garden": {"reference_wall_s": None, "c42_wall_s": None, "reference_iter_ms": None, "c42_iter_ms": None},
    "bicycle": {"reference_wall_s": None, "c42_wall_s": None, "reference_iter_ms": None, "c42_iter_ms": None},
}


def load_training_results():
    """Load all training results from results/c42_13scene/."""
    results = {}
    
    # Load new scene training results
    for dataset, scenes in SCENES.items():
        for scene in scenes:
            for method in ["reference", "c42"]:
                result_path = RESULTS_BASE / dataset / scene / method / "training_results.json"
                if result_path.exists():
                    with open(result_path) as f:
                        results[f"{scene}_{method}"] = json.load(f)
    
    # Load existing 3-scene training results
    existing_paths = {
        "room_reference": REPO / "results/reference_v1/room_30k/training_metrics.json",
        "room_c42": REPO / "results/reference_v1/c42_30k/training_metrics.json",
        "garden_reference": REPO / "results/reference_v1/s22/garden/baseline_30k.json",
        "garden_c42": REPO / "results/reference_v1/s22/garden/c42_30k.json",
        "bicycle_reference": REPO / "results/reference_v1/s22/bicycle/baseline_30k.json",
        "bicycle_c42": REPO / "results/reference_v1/s22/bicycle/c42_30k.json",
    }
    for key, path in existing_paths.items():
        if path.exists() and key not in results:
            with open(path) as f:
                results[key] = json.load(f)
    
    return results


def load_evaluation_results():
    """Load evaluation results."""
    eval_path = RESULTS_BASE / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    
    # Fall back to existing unified_metrics.json
    um_path = REPO / "results/c42_adaptive/completion_batch/unified_metrics.json"
    if um_path.exists():
        with open(um_path) as f:
            um = json.load(f)
        results = {}
        for key, val in um["checkpoints"].items():
            scene, scale = val["scene"], val["scale"]
            method = "reference" if scale == "1.0" else "c42"
            results[f"{scene}_{method}"] = val
        return results
    
    return {}


def extract_metrics(training_results, eval_results, scene, method):
    """Extract standardized metrics from training and evaluation results."""
    train_key = f"{scene}_{method}"
    train = training_results.get(train_key, {})
    eval_r = eval_results.get(train_key, {})
    
    # Quality metrics from evaluation (preferred) or training
    if eval_r and "psnr" in eval_r:
        psnr = eval_r.get("psnr")
        ssim = eval_r.get("ssim")
        lpips = eval_r.get("lpips")
        n_gaussians = eval_r.get("n_gaussians")
    else:
        # Try training final_eval
        final = train.get("final_eval", {})
        if not final:
            tm = train.get("training_metrics", {})
            ckpts = tm.get("checkpoints", {})
            if ckpts:
                final = list(ckpts.values())[-1]
        psnr = final.get("psnr")
        ssim = final.get("ssim")
        lpips = final.get("lpips")
        n_gaussians = final.get("n_gaussians", train.get("final_N"))
    
    # Timing
    timing = train.get("timing", {})
    wall_s = timing.get("total_wall_s", train.get("wall_clock_time_s"))
    iter_ms = timing.get("steady_state_mean_iter_ms", timing.get("mean_iter_ms"))
    
    # VRAM
    peak_vram = train.get("peak_vram_gb")
    
    # N gaussians from training if not in eval
    if n_gaussians is None:
        n_gaussians = train.get("final_N")
    
    return {
        "psnr": psnr, "ssim": ssim, "lpips": lpips,
        "n_gaussians": n_gaussians,
        "wall_s": wall_s, "iter_ms": iter_ms,
        "peak_vram_gb": peak_vram,
    }


def compute_deltas(ref_metrics, c42_metrics):
    """Compute C42 vs Reference deltas."""
    deltas = {}
    
    if ref_metrics["psnr"] is not None and c42_metrics["psnr"] is not None:
        deltas["dPSNR"] = c42_metrics["psnr"] - ref_metrics["psnr"]
    if ref_metrics["ssim"] is not None and c42_metrics["ssim"] is not None:
        deltas["dSSIM"] = c42_metrics["ssim"] - ref_metrics["ssim"]
    if ref_metrics["lpips"] is not None and c42_metrics["lpips"] is not None:
        deltas["dLPIPS"] = c42_metrics["lpips"] - ref_metrics["lpips"]
    if ref_metrics["n_gaussians"] is not None and c42_metrics["n_gaussians"] is not None:
        deltas["dN_GS_percent"] = ((c42_metrics["n_gaussians"] - ref_metrics["n_gaussians"]) / 
                                    ref_metrics["n_gaussians"] * 100)
    if ref_metrics["peak_vram_gb"] is not None and c42_metrics["peak_vram_gb"] is not None:
        deltas["dVRAM_percent"] = ((c42_metrics["peak_vram_gb"] - ref_metrics["peak_vram_gb"]) / 
                                    ref_metrics["peak_vram_gb"] * 100)
    if ref_metrics["wall_s"] is not None and c42_metrics["wall_s"] is not None and c42_metrics["wall_s"] > 0:
        deltas["speedup"] = ref_metrics["wall_s"] / c42_metrics["wall_s"]
    elif ref_metrics["iter_ms"] is not None and c42_metrics["iter_ms"] is not None and c42_metrics["iter_ms"] > 0:
        deltas["speedup"] = ref_metrics["iter_ms"] / c42_metrics["iter_ms"]
    
    return deltas


def classify_scene(deltas):
    """Classify scene based on robustness criteria."""
    speedup = deltas.get("speedup", 1.0)
    dPSNR = deltas.get("dPSNR", 0)
    dSSIM = deltas.get("dSSIM", 0)
    dLPIPS = deltas.get("dLPIPS", 0)
    
    if speedup < 1.10:
        return "NO_SPEED_BENEFIT"
    
    if (speedup >= 1.30 and dPSNR >= -0.30 and abs(dSSIM) <= 0.020 and dLPIPS <= 0.03):
        return "STRONG_SUCCESS"
    
    return "SPEED_QUALITY_TRADEOFF"


def geometric_mean(values):
    """Compute geometric mean of positive values."""
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(v) for v in valid) / len(valid))


def safe_mean(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def safe_min(values):
    valid = [v for v in values if v is not None]
    return min(valid) if valid else None


def safe_max(values):
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def safe_median(values):
    valid = sorted([v for v in values if v is not None])
    if not valid:
        return None
    n = len(valid)
    if n % 2 == 1:
        return valid[n // 2]
    return (valid[n // 2 - 1] + valid[n // 2]) / 2


def main():
    print("C42 13-Scene Final Analysis")
    print("=" * 60)
    
    training_results = load_training_results()
    eval_results = load_evaluation_results()
    
    print(f"Loaded {len(training_results)} training results")
    print(f"Loaded {len(eval_results)} evaluation results")
    
    # Compute per-scene metrics and deltas
    all_scenes_data = {}
    per_scene_deltas = {}
    classification = {}
    
    for dataset, scenes in SCENES.items():
        for scene in scenes:
            ref = extract_metrics(training_results, eval_results, scene, "reference")
            c42 = extract_metrics(training_results, eval_results, scene, "c42")
            deltas = compute_deltas(ref, c42)
            cls = classify_scene(deltas)
            
            all_scenes_data[scene] = {
                "dataset": dataset, "reference": ref, "c42": c42,
                "deltas": deltas, "classification": cls,
            }
            per_scene_deltas[scene] = {**deltas, "classification": cls, "dataset": dataset}
            classification[scene] = cls
            
            print(f"\n  {scene} ({dataset}):")
            print(f"    Ref: PSNR={ref['psnr']} SSIM={ref['ssim']} GS={ref['n_gaussians']} time={ref['wall_s']}")
            print(f"    C42: PSNR={c42['psnr']} SSIM={c42['ssim']} GS={c42['n_gaussians']} time={c42['wall_s']}")
            print(f"    Deltas: {deltas}")
            print(f"    Classification: {cls}")
    
    # Dataset-level aggregates
    dataset_aggregates = {}
    for dataset, scenes in SCENES.items():
        speedups = [per_scene_deltas[s].get("speedup") for s in scenes if s in per_scene_deltas]
        dPSNRs = [per_scene_deltas[s].get("dPSNR") for s in scenes if s in per_scene_deltas]
        dSSIMs = [per_scene_deltas[s].get("dSSIM") for s in scenes if s in per_scene_deltas]
        dLPIPSs = [per_scene_deltas[s].get("dLPIPS") for s in scenes if s in per_scene_deltas]
        dGSs = [per_scene_deltas[s].get("dN_GS_percent") for s in scenes if s in per_scene_deltas]
        
        dataset_aggregates[dataset] = {
            "n_scenes": len(scenes),
            "geometric_mean_speedup": geometric_mean(speedups),
            "mean_dPSNR": safe_mean(dPSNRs),
            "mean_dSSIM": safe_mean(dSSIMs),
            "mean_dLPIPS": safe_mean(dLPIPSs),
            "mean_dGS_percent": safe_mean(dGSs),
        }
    
    # Overall aggregate
    all_speedups = [per_scene_deltas[s].get("speedup") for s in per_scene_deltas]
    all_dPSNRs = [per_scene_deltas[s].get("dPSNR") for s in per_scene_deltas]
    all_dSSIMs = [per_scene_deltas[s].get("dSSIM") for s in per_scene_deltas]
    all_dLPIPSs = [per_scene_deltas[s].get("dLPIPS") for s in per_scene_deltas]
    all_dGSs = [per_scene_deltas[s].get("dN_GS_percent") for s in per_scene_deltas]
    all_dVRAMs = [per_scene_deltas[s].get("dVRAM_percent") for s in per_scene_deltas]
    
    overall_aggregate = {
        "n_scenes": len(per_scene_deltas),
        "geometric_mean_speedup": geometric_mean(all_speedups),
        "median_speedup": safe_median(all_speedups),
        "min_speedup": safe_min(all_speedups),
        "max_speedup": safe_max(all_speedups),
        "mean_dPSNR": safe_mean(all_dPSNRs),
        "worst_dPSNR": safe_min(all_dPSNRs),
        "mean_dSSIM": safe_mean(all_dSSIMs),
        "worst_dSSIM": safe_min(all_dSSIMs),
        "mean_dLPIPS": safe_mean(all_dLPIPSs),
        "worst_dLPIPS": safe_max(all_dLPIPSs),
        "mean_dGS_percent": safe_mean(all_dGSs),
        "mean_dVRAM_percent": safe_mean(all_dVRAMs),
        "strong_success_count": sum(1 for c in classification.values() if c == "STRONG_SUCCESS"),
        "tradeoff_count": sum(1 for c in classification.values() if c == "SPEED_QUALITY_TRADEOFF"),
        "no_speed_benefit_count": sum(1 for c in classification.values() if c == "NO_SPEED_BENEFIT"),
    }
    
    # Final decision
    strong_count = overall_aggregate["strong_success_count"]
    geo_speedup = overall_aggregate["geometric_mean_speedup"]
    worst_dPSNR = overall_aggregate["worst_dPSNR"]
    
    if strong_count >= 10 and geo_speedup and geo_speedup > 1.1 and (worst_dPSNR is None or worst_dPSNR > -1.0):
        decision = "C42_13SCENE_STRONG_KEEP"
    elif geo_speedup and geo_speedup > 1.1:
        decision = "C42_13SCENE_KEEP_WITH_TRADEOFF"
    elif geo_speedup and geo_speedup > 1.05:
        decision = "C42_13SCENE_MIXED"
    else:
        decision = "C42_13SCENE_FAIL"
    
    final_decision = {
        "decision": decision,
        "strong_success_count": strong_count,
        "geometric_mean_speedup": geo_speedup,
        "worst_dPSNR": worst_dPSNR,
    }
    
    # Save all machine-readable deliverables
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_BASE / "per_scene_deltas.json", "w") as f:
        json.dump(per_scene_deltas, f, indent=2)
    with open(RESULTS_BASE / "dataset_aggregates.json", "w") as f:
        json.dump(dataset_aggregates, f, indent=2)
    with open(RESULTS_BASE / "overall_aggregate.json", "w") as f:
        json.dump(overall_aggregate, f, indent=2)
    with open(RESULTS_BASE / "classification.json", "w") as f:
        json.dump(classification, f, indent=2)
    with open(RESULTS_BASE / "final_decision.json", "w") as f:
        json.dump(final_decision, f, indent=2)
    
    # Generate markdown report
    generate_report(all_scenes_data, per_scene_deltas, dataset_aggregates,
                   overall_aggregate, classification, final_decision)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"  FINAL DECISION: {decision}")
    print(f"  Strong success: {strong_count}/13")
    print(f"  Geo mean speedup: {geo_speedup}")
    print(f"  Worst dPSNR: {worst_dPSNR}")
    print(f"{'='*60}")


def generate_report(all_scenes_data, per_scene_deltas, dataset_aggregates,
                   overall_aggregate, classification, final_decision):
    """Generate the final markdown report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "c42-13scene-final.md"
    
    lines = []
    lines.append("# C42 13-Scene Final Benchmark Report")
    lines.append("")
    lines.append("## Mission")
    lines.append("")
    lines.append("Canonical 13-scene real-dataset benchmark comparing:")
    lines.append("- **Reference V1** (full-resolution SSIM)")
    lines.append("- **Reference V1 + C42** (scale=0.5, downsampled SSIM, L1 full-resolution)")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- iterations: 30000")
    lines.append("- lambda_dssim: 0.2")
    lines.append("- C42 scale: 0.5 (area interpolation)")
    lines.append("- L1: full resolution (NOT downsampled)")
    lines.append("- SepSSIM: window=11, sigma=1.5")
    lines.append("- absgrad=True, grow_grad2d=0.0008")
    lines.append("- GaussianModel: baseline/reference_v1/gaussian_model.py")
    lines.append("")
    
    # Central table
    lines.append("## Central Results Table")
    lines.append("")
    lines.append("| Dataset | Scene | Method | PSNR | SSIM | LPIPS | N_GS | Train Time | ms/iter | Peak VRAM | Speedup |")
    lines.append("|---------|-------|--------|------|------|-------|------|------------|---------|-----------|---------|")
    
    for dataset, scenes in SCENES.items():
        for scene in scenes:
            data = all_scenes_data.get(scene, {})
            for method in ["reference", "c42"]:
                m = data.get(method, {})
                psnr = f"{m.get('psnr', 0):.2f}" if m.get('psnr') else "N/A"
                ssim = f"{m.get('ssim', 0):.4f}" if m.get('ssim') else "N/A"
                lpips = f"{m.get('lpips', 0):.4f}" if m.get('lpips') else "N/A"
                gs = f"{m.get('n_gaussians', 0):,}" if m.get('n_gaussians') else "N/A"
                wall = f"{m.get('wall_s', 0)/60:.1f}min" if m.get('wall_s') else "N/A"
                itms = f"{m.get('iter_ms', 0):.1f}" if m.get('iter_ms') else "N/A"
                vram = f"{m.get('peak_vram_gb', 0):.2f}GB" if m.get('peak_vram_gb') else "N/A"
                if method == "c42":
                    sp = data.get("deltas", {}).get("speedup")
                    sp_str = f"{sp:.2f}x" if sp else "N/A"
                else:
                    sp_str = "1.00x"
                lines.append(f"| {dataset} | {scene} | {method} | {psnr} | {ssim} | {lpips} | {gs} | {wall} | {itms} | {vram} | {sp_str} |")
    
    lines.append("")
    
    # Delta table
    lines.append("## Delta Table")
    lines.append("")
    lines.append("| Scene | dPSNR | dSSIM | dLPIPS | dN_GS % | dVRAM % | Speedup | Classification |")
    lines.append("|-------|-------|-------|--------|---------|---------|---------|----------------|")
    
    for dataset, scenes in SCENES.items():
        for scene in scenes:
            d = per_scene_deltas.get(scene, {})
            dpsnr = f"{d.get('dPSNR', 0):+.2f}" if d.get('dPSNR') is not None else "N/A"
            dssim = f"{d.get('dSSIM', 0):+.4f}" if d.get('dSSIM') is not None else "N/A"
            dlpips = f"{d.get('dLPIPS', 0):+.4f}" if d.get('dLPIPS') is not None else "N/A"
            dgs = f"{d.get('dN_GS_percent', 0):+.1f}%" if d.get('dN_GS_percent') is not None else "N/A"
            dvram = f"{d.get('dVRAM_percent', 0):+.1f}%" if d.get('dVRAM_percent') is not None else "N/A"
            sp = f"{d.get('speedup', 0):.2f}x" if d.get('speedup') is not None else "N/A"
            cls = d.get("classification", "N/A")
            lines.append(f"| {scene} | {dpsnr} | {dssim} | {dlpips} | {dgs} | {dvram} | {sp} | {cls} |")
    
    lines.append("")
    
    # Dataset aggregates
    lines.append("## Dataset-Level Aggregates")
    lines.append("")
    for dataset in ["mipnerf360", "tanksandtemples", "deepblending"]:
        agg = dataset_aggregates.get(dataset, {})
        lines.append(f"### {dataset} ({agg.get('n_scenes', 0)} scenes)")
        lines.append(f"- Geometric mean speedup: {agg.get('geometric_mean_speedup', 'N/A')}")
        lines.append(f"- Mean dPSNR: {agg.get('mean_dPSNR', 'N/A')}")
        lines.append(f"- Mean dSSIM: {agg.get('mean_dSSIM', 'N/A')}")
        lines.append(f"- Mean dLPIPS: {agg.get('mean_dLPIPS', 'N/A')}")
        lines.append(f"- Mean dGS%: {agg.get('mean_dGS_percent', 'N/A')}")
        lines.append("")
    
    # Overall aggregate
    lines.append("## Overall Aggregate (13 scenes)")
    lines.append("")
    oa = overall_aggregate
    lines.append(f"- Geometric mean speedup: {oa.get('geometric_mean_speedup', 'N/A')}")
    lines.append(f"- Median speedup: {oa.get('median_speedup', 'N/A')}")
    lines.append(f"- Min speedup: {oa.get('min_speedup', 'N/A')}")
    lines.append(f"- Max speedup: {oa.get('max_speedup', 'N/A')}")
    lines.append(f"- Mean dPSNR: {oa.get('mean_dPSNR', 'N/A')}")
    lines.append(f"- Worst dPSNR: {oa.get('worst_dPSNR', 'N/A')}")
    lines.append(f"- Mean dSSIM: {oa.get('mean_dSSIM', 'N/A')}")
    lines.append(f"- Mean dLPIPS: {oa.get('mean_dLPIPS', 'N/A')}")
    lines.append(f"- Mean dGS%: {oa.get('mean_dGS_percent', 'N/A')}")
    lines.append(f"- STRONG_SUCCESS: {oa.get('strong_success_count', 0)}")
    lines.append(f"- SPEED_QUALITY_TRADEOFF: {oa.get('tradeoff_count', 0)}")
    lines.append(f"- NO_SPEED_BENEFIT: {oa.get('no_speed_benefit_count', 0)}")
    lines.append("")
    
    # Final aggregate table
    lines.append("## Final Aggregate Table")
    lines.append("")
    lines.append("| | PSNR | SSIM | LPIPS | Time | Speedup | VRAM |")
    lines.append("|---|------|------|-------|------|---------|------|")
    # TODO: Fill with actual aggregate values
    lines.append("")
    
    # Final decision
    lines.append("## Final Scientific Gate")
    lines.append("")
    lines.append(f"**Decision: {final_decision.get('decision', 'N/A')}**")
    lines.append("")
    lines.append(f"- STRONG_SUCCESS count: {final_decision.get('strong_success_count', 0)}/13")
    lines.append(f"- Geometric mean speedup: {final_decision.get('geometric_mean_speedup', 'N/A')}")
    lines.append(f"- Worst dPSNR: {final_decision.get('worst_dPSNR', 'N/A')}")
    lines.append("")
    
    # Critical questions
    lines.append("## Critical Questions")
    lines.append("")
    geo_sp = oa.get('geometric_mean_speedup')
    lines.append(f"**Q1: Does C42 produce training speedup on all or nearly all 13 scenes?**")
    speedups_valid = [d.get('speedup') for d in per_scene_deltas.values() if d.get('speedup')]
    above_1 = sum(1 for s in speedups_valid if s > 1.0)
    lines.append(f"  {above_1}/{len(speedups_valid)} scenes show speedup > 1.0x")
    lines.append("")
    lines.append(f"**Q2: Geometric-mean training speedup?** {geo_sp}")
    lines.append("")
    lines.append(f"**Q3: Mean and worst-case quality changes?**")
    lines.append(f"  Mean dPSNR: {oa.get('mean_dPSNR')}, Worst dPSNR: {oa.get('worst_dPSNR')}")
    lines.append(f"  Mean dSSIM: {oa.get('mean_dSSIM')}, Mean dLPIPS: {oa.get('mean_dLPIPS')}")
    lines.append("")
    lines.append(f"**Q5: Does C42 consistently reduce final Gaussian population?**")
    lines.append(f"  Mean dGS%: {oa.get('mean_dGS_percent')}")
    lines.append("")
    lines.append(f"**Q7: How many STRONG_SUCCESS vs SPEED_QUALITY_TRADEOFF?**")
    lines.append(f"  STRONG_SUCCESS: {oa.get('strong_success_count', 0)}")
    lines.append(f"  SPEED_QUALITY_TRADEOFF: {oa.get('tradeoff_count', 0)}")
    lines.append(f"  NO_SPEED_BENEFIT: {oa.get('no_speed_benefit_count', 0)}")
    lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
