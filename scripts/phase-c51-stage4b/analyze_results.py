#!/usr/bin/env python3
"""Phase C51 Stage 4B — Analyze canonical training results."""
import json
import sys
from pathlib import Path
import numpy as np


def load_results(result_dir):
    results = {}
    for f in sorted(result_dir.glob("training_room_*.json")):
        name = f.stem.replace("training_room_", "")
        with open(f) as fh:
            results[name] = json.load(fh)
    return results


def analyze(results):
    """Produce comprehensive analysis."""
    
    # Find baseline (30K)
    baseline = results.get("baseline")
    if baseline is None:
        return {"error": "No 30K baseline results found"}

    b_psnr = baseline["final_psnr"]
    b_ssim = baseline["final_ssim"]
    b_time = baseline["timing"]["mean_ms"]
    b_gs = baseline["final_gaussians"]

    print("=" * 90)
    print("Phase C51 Stage 4B — Canonical ROOM 30K Results")
    print("=" * 90)
    print(f"\nBaseline: PSNR={b_psnr:.2f}, SSIM={b_ssim:.4f}, time={b_time:.2f}ms, GS={b_gs:,}")
    print(f"  Clone={baseline['total_clone']}, Split={baseline['total_split']}, Prune={baseline['total_prune']}")

    # 30K comparison
    print(f"\n{'Config':<20} {'PSNR':>7} {'ΔPSNR':>7} {'SSIM':>7} {'ΔSSIM':>8} "
          f"{'Time':>7} {'Speedup':>8} {'GS':>10} {'GS Ratio':>9}")
    print("-" * 100)

    for name in ["baseline", "k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        psnr = res["final_psnr"]
        ssim = res["final_ssim"]
        time_ms = res["timing"]["mean_ms"]
        gs = res["final_gaussians"]
        speedup = (b_time / time_ms - 1) * 100 if time_ms > 0 else 0
        
        if name == "baseline":
            d_psnr = 0
            d_ssim = 0
            gs_ratio = 1.0
        else:
            d_psnr = psnr - b_psnr
            d_ssim = ssim - b_ssim
            gs_ratio = gs / b_gs

        print(f"{name:<20} {psnr:7.2f} {d_psnr:+7.2f} {ssim:7.4f} {d_ssim:+8.4f} "
              f"{time_ms:7.2f} {speedup:+7.1f}% {gs:10,} {gs_ratio:8.1%}")

    # Densification comparison
    print(f"\n{'Config':<20} {'Clone':>8} {'Split':>8} {'Prune':>8} {'Clone%':>7} {'Split%':>7} {'Prune%':>7}")
    print("-" * 75)
    for name in ["baseline", "k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        clone = res["total_clone"]
        split = res["total_split"]
        prune = res["total_prune"]
        if name == "baseline":
            clone_pct = split_pct = prune_pct = 100.0
        else:
            clone_pct = clone / baseline["total_clone"] * 100
            split_pct = split / baseline["total_split"] * 100
            prune_pct = prune / baseline["total_prune"] * 100
        print(f"{name:<20} {clone:8,} {split:8,} {prune:8,} {clone_pct:6.1f}% {split_pct:6.1f}% {prune_pct:6.1f}%")

    # Timing decomposition
    print(f"\n{'Config':<20} {'Fwd+Bwd':>8} {'Dens':>7} {'Opt':>7} {'Mask':>7} {'Total':>7}")
    print("-" * 65)
    for name in ["baseline", "k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        t = res["timing"]
        print(f"{name:<20} {t['fwd_bwd_mean_ms']:8.2f} {t['dens_mean_ms']:7.2f} "
              f"{t['opt_mean_ms']:7.2f} {t['mask_mean_ms']:7.3f} {t['mean_ms']:7.2f}")

    # Sparse vs dense timing
    print(f"\n{'Config':<20} {'Sparse ms':>10} {'Dense ms':>10} {'N_sparse':>10} {'N_dense':>10}")
    print("-" * 65)
    for name in ["baseline", "k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        t = res["timing"]
        print(f"{name:<20} {t['sparse_mean_ms']:10.2f} {t['dense_mean_ms']:10.2f} "
              f"{t['n_sparse_iters']:10d} {t['n_dense_iters']:10d}")

    # 5K ablation comparison
    print("\n" + "=" * 90)
    print("5K Ablation Comparison (quality unreliable due to opacity reset at 3K)")
    print("=" * 90)
    print(f"\n{'Config':<25} {'PSNR':>7} {'SSIM':>7} {'Time':>7} {'Clone':>8} {'Split':>8} {'Prune':>8}")
    print("-" * 80)
    for name in ["baseline_5k", "k50_b1_5k", "k50_freeze_mask_5k", "k50_oracle_5k", "k50_b1_oracledens_5k"]:
        res = results.get(name)
        if res is None:
            continue
        print(f"{name:<25} {res['final_psnr']:7.2f} {res['final_ssim']:7.4f} "
              f"{res['timing']['mean_ms']:7.2f} {res['total_clone']:8,} {res['total_split']:8,} {res['total_prune']:8,}")

    # Gate evaluation for 30K
    print("\n" + "=" * 90)
    print("Gate Evaluation (30K Canonical ROOM)")
    print("=" * 90)
    for name in ["k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        psnr = res["final_psnr"]
        ssim = res["final_ssim"]
        time_ms = res["timing"]["mean_ms"]
        d_psnr = psnr - b_psnr
        d_ssim = ssim - b_ssim
        speedup_pct = (b_time / time_ms - 1) * 100
        
        gates = {
            "Gate B (quality)": d_psnr >= -0.2 and abs(d_ssim) < 0.005,
            "Gate C (speedup >5%)": speedup_pct > 5.0,
        }
        all_pass = all(gates.values())
        
        status = "PASS" if all_pass else "FAIL"
        print(f"\n  {name}: {status}")
        print(f"    Gate B: {'PASS' if gates['Gate B (quality)'] else 'FAIL'} "
              f"(ΔPSNR={d_psnr:+.3f}, ΔSSIM={d_ssim:+.5f})")
        print(f"    Gate C: {'PASS' if gates['Gate C (speedup >5%)'] else 'FAIL'} "
              f"(speedup={speedup_pct:+.1f}%)")

    # PSNR trajectory at key milestones
    print("\n" + "=" * 90)
    print("PSNR Trajectory (30K)")
    print("=" * 90)
    milestones = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000]
    
    all_traj = {}
    for name in ["baseline", "k50_b1", "k50_postdens"]:
        res = results.get(name)
        if res is None:
            continue
        traj = {t["iter"]: t for t in res.get("trajectory", [])}
        all_traj[name] = traj

    print(f"\n{'Iter':<8}", end="")
    for name in all_traj:
        print(f" {name:>14}", end="")
    print()
    print("-" * (8 + 15 * len(all_traj)))

    for m in milestones:
        print(f"{m:<8}", end="")
        for name in all_traj:
            traj = all_traj[name]
            if m in traj:
                print(f" {traj[m]['psnr']:14.2f}", end="")
            else:
                closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
                print(f" {traj.get(closest, {}).get('psnr', 0):14.2f}", end="")
        print()

    # Gaussian count trajectory
    print(f"\n{'Iter':<8}", end="")
    for name in all_traj:
        print(f" {name:>14}", end="")
    print()
    print("-" * (8 + 15 * len(all_traj)))

    for m in milestones:
        print(f"{m:<8}", end="")
        for name in all_traj:
            traj = all_traj[name]
            if m in traj:
                print(f" {traj[m]['gaussians']:14,}", end="")
            else:
                closest = min(traj.keys(), key=lambda k: abs(k - m)) if traj else 0
                print(f" {traj.get(closest, {}).get('gaussians', 0):14,}", end="")
        print()


def main():
    result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4b")
    if len(sys.argv) > 1:
        result_dir = Path(sys.argv[1])

    results = load_results(result_dir)
    if not results:
        print(f"No results found in {result_dir}")
        return

    print(f"Loaded {len(results)} result files: {sorted(results.keys())}")
    analyze(results)


if __name__ == "__main__":
    main()
