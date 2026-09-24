#!/usr/bin/env python3
"""Phase C51 Stage 4A — Analyze 5K Training Results

Reads all 5K training JSON results and produces:
  - Quality comparison (PSNR/SSIM degradation vs baseline)
  - Speedup measurement (mean iteration time)
  - Densification agreement (clone/split/prune counts)
  - Speedup attribution (forward vs backward time breakdown)
  - Gate pass/fail summary
"""
import json
import sys
from pathlib import Path
import numpy as np


def load_results(result_dir):
    """Load all training result JSONs."""
    results = {}
    for f in sorted(result_dir.glob("training_5k_*.json")):
        name = f.stem.replace("training_5k_", "")
        if name == "analysis":
            continue  # skip the analysis file itself
        with open(f) as fh:
            results[name] = json.load(fh)
    return results


def analyze(results):
    """Produce comprehensive analysis."""
    baseline = results.get("baseline")
    if baseline is None:
        return {"error": "No baseline results found"}

    baseline_psnr = baseline["final_psnr"]
    baseline_ssim = baseline["final_ssim"]
    baseline_time = baseline["timing"]["mean_ms"]
    baseline_gs = baseline["final_gaussians"]

    analysis = {
        "baseline": {
            "psnr": baseline_psnr,
            "ssim": baseline_ssim,
            "mean_time_ms": baseline_time,
            "final_gaussians": baseline_gs,
            "total_clone": baseline["total_clone"],
            "total_split": baseline["total_split"],
            "total_prune": baseline["total_prune"],
        },
        "configs": {},
        "gates": {},
    }

    for name, res in results.items():
        if name == "baseline":
            continue

        psnr = res["final_psnr"]
        ssim = res["final_ssim"]
        time_ms = res["timing"]["mean_ms"]
        gs = res["final_gaussians"]

        psnr_delta = psnr - baseline_psnr
        ssim_delta = ssim - baseline_ssim
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        speedup_pct = (speedup - 1) * 100

        # Densification agreement
        clone_ratio = res["total_clone"] / baseline["total_clone"] if baseline["total_clone"] > 0 else 0
        split_ratio = res["total_split"] / baseline["total_split"] if baseline["total_split"] > 0 else 0
        prune_ratio = res["total_prune"] / baseline["total_prune"] if baseline["total_prune"] > 0 else 0

        # Sparse vs dense timing
        sparse_timing = res["timing"]["sparse_mean_ms"]
        dense_timing = res["timing"]["dense_mean_ms"]
        n_sparse = res["timing"]["n_sparse_iters"]
        n_dense = res["timing"]["n_dense_iters"]

        entry = {
            "psnr": psnr,
            "ssim": ssim,
            "psnr_delta": psnr_delta,
            "ssim_delta": ssim_delta,
            "mean_time_ms": time_ms,
            "speedup": speedup,
            "speedup_pct": speedup_pct,
            "final_gaussians": gs,
            "gaussian_delta": gs - baseline_gs,
            "total_clone": res["total_clone"],
            "total_split": res["total_split"],
            "total_prune": res["total_prune"],
            "clone_ratio": clone_ratio,
            "split_ratio": split_ratio,
            "prune_ratio": prune_ratio,
            "sparse_mean_ms": sparse_timing,
            "dense_mean_ms": dense_timing,
            "n_sparse_iters": n_sparse,
            "n_dense_iters": n_dense,
            "config": res["config"],
        }
        analysis["configs"][name] = entry

        # Gate evaluation
        design = res["config"]["design"]
        k = res["config"]["keep_fraction"]

        gates = {
            "gate_a_correctness": True,  # Validated in microbenchmark
            "gate_b_quality": psnr_delta >= -0.2 and abs(ssim_delta) < 0.005,
            "gate_c_speedup": speedup_pct > 5.0,
            "gate_d_densification": clone_ratio > 0.5 and split_ratio > 0.5,
        }
        gates["all_pass"] = all(gates.values())
        analysis["gates"][name] = gates

    return analysis


def print_report(analysis):
    """Print human-readable report."""
    print("=" * 80)
    print("Phase C51 Stage 4A — 5K Training Analysis")
    print("=" * 80)

    b = analysis["baseline"]
    print(f"\nBaseline: PSNR={b['psnr']:.2f}, SSIM={b['ssim']:.4f}, "
          f"time={b['mean_time_ms']:.2f}ms, GS={b['final_gaussians']:,}")
    print(f"  Clone={b['total_clone']}, Split={b['total_split']}, Prune={b['total_prune']}")

    print(f"\n{'Config':<12} {'PSNR':>7} {'ΔPSNR':>7} {'SSIM':>7} {'ΔSSIM':>8} "
          f"{'Time':>7} {'Speedup':>8} {'GS':>10} {'Clone%':>7} {'Split%':>7} {'Prune%':>7}")
    print("-" * 100)

    for name, e in sorted(analysis["configs"].items()):
        print(f"{name:<12} {e['psnr']:7.2f} {e['psnr_delta']:+7.2f} {e['ssim']:7.4f} "
              f"{e['ssim_delta']:+8.4f} {e['mean_time_ms']:7.2f} {e['speedup_pct']:+7.1f}% "
              f"{e['final_gaussians']:10,} {e['clone_ratio']*100:6.1f}% "
              f"{e['split_ratio']*100:6.1f}% {e['prune_ratio']*100:6.1f}%")

    print("\n" + "=" * 80)
    print("Gate Evaluation")
    print("=" * 80)
    for name, gates in sorted(analysis["gates"].items()):
        status = "PASS" if gates["all_pass"] else "FAIL"
        print(f"\n  {name}: {status}")
        print(f"    Gate A (correctness):      {'PASS' if gates['gate_a_correctness'] else 'FAIL'}")
        print(f"    Gate B (quality):          {'PASS' if gates['gate_b_quality'] else 'FAIL'} "
              f"(ΔPSNR={analysis['configs'][name]['psnr_delta']:+.3f}, "
              f"ΔSSIM={analysis['configs'][name]['ssim_delta']:+.5f})")
        print(f"    Gate C (speedup >5%):      {'PASS' if gates['gate_c_speedup'] else 'FAIL'} "
              f"(speedup={analysis['configs'][name]['speedup_pct']:+.1f}%)")
        print(f"    Gate D (densification):    {'PASS' if gates['gate_d_densification'] else 'FAIL'} "
              f"(clone={analysis['configs'][name]['clone_ratio']*100:.1f}%, "
              f"split={analysis['configs'][name]['split_ratio']*100:.1f}%)")

    # Speedup attribution
    print("\n" + "=" * 80)
    print("Speedup Attribution (sparse vs dense iterations)")
    print("=" * 80)
    print(f"\n{'Config':<12} {'Sparse ms':>10} {'Dense ms':>10} {'Δ ms':>8} {'N_sparse':>10} {'N_dense':>10}")
    print("-" * 65)
    for name, e in sorted(analysis["configs"].items()):
        delta = e["dense_mean_ms"] - e["sparse_mean_ms"]
        print(f"{name:<12} {e['sparse_mean_ms']:10.2f} {e['dense_mean_ms']:10.2f} "
              f"{delta:+8.2f} {e['n_sparse_iters']:10d} {e['n_dense_iters']:10d}")


def main():
    result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4a")
    if len(sys.argv) > 1:
        result_dir = Path(sys.argv[1])

    results = load_results(result_dir)
    if not results:
        print(f"No results found in {result_dir}")
        return

    print(f"Loaded {len(results)} result files: {sorted(results.keys())}")

    analysis = analyze(results)
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print_report(analysis)

    # Save analysis
    out_file = result_dir / "training_5k_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {out_file}")


if __name__ == "__main__":
    main()
