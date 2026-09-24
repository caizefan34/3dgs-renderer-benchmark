#!/usr/bin/env python3
"""Run B1 and B1A short training and compare."""
import sys, os, json, subprocess
import numpy as np

RESULTS_DIR = "/tmp/accutile_a100_results"
TRAIN_SCRIPT = "/tmp/short_train.py"
REPO_ROOT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"

def run_training(variant, gsplat_path, accutile, out_file):
    cmd = (
        f"cd /tmp && source /home/liaoyuanjun/miniforge3/etc/profile.d/conda.sh && conda activate anysplat && "
        f"CUDA_VISIBLE_DEVICES=0 python3 {TRAIN_SCRIPT} "
        f"--variant {variant} --gsplat-path {gsplat_path} "
        f"--accutile {'true' if accutile else 'false'} "
        f"--out {out_file} --scene room --num-iters 800"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
    return r

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variants = [
        ("B1", "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", False),
        ("B1A", "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", True),
    ]

    results = {}
    for name, path, accutile in variants:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        out_file = f"{RESULTS_DIR}/short_train_{name}.json"
        r = run_training(name, path, accutile, out_file)
        print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
        if r.stderr:
            # Filter out warnings
            stderr_lines = [l for l in r.stderr.split('\n') if 'Warning' not in l and 'warn' not in l.lower()]
            if stderr_lines:
                print("STDERR:", '\n'.join(stderr_lines[-10:]))
        if os.path.exists(out_file):
            with open(out_file) as f:
                results[name] = json.load(f)

    if "B1" not in results or "B1A" not in results:
        print("ERROR: Missing training results")
        return

    # Compare
    b1 = results["B1"]
    b1a = results["B1A"]

    print(f"\n{'='*70}")
    print("B1 vs B1A SHORT TRAINING COMPARISON")
    print(f"{'='*70}")

    print(f"\nInitial Gaussians: B1={b1['initial_gaussians']}, B1A={b1a['initial_gaussians']}")
    print(f"Final Gaussians: B1={b1['final_gaussians']}, B1A={b1a['final_gaussians']}")
    print(f"Final N diff: {abs(b1['final_gaussians'] - b1a['final_gaussians'])}")

    # Compare log entries
    b1_log = {e["iter"]: e for e in b1["log"]}
    b1a_log = {e["iter"]: e for e in b1a["log"]}

    common_iters = sorted(set(b1_log.keys()) & set(b1a_log.keys()))
    print(f"\n{'Iter':>6} {'B1 loss':>12} {'B1A loss':>12} {'loss diff':>12} {'B1 PSNR':>10} {'B1A PSNR':>10} {'PSNR diff':>10} {'B1 N':>8} {'B1A N':>8} {'N diff':>6}")
    max_loss_diff = 0
    max_psnr_diff = 0
    max_n_diff = 0
    for it in common_iters:
        b1e = b1_log[it]
        b1ae = b1a_log[it]
        loss_diff = abs(b1e["loss"] - b1ae["loss"])
        psnr_diff = abs(b1e["psnr"] - b1ae["psnr"])
        n_diff = abs(b1e["n_gaussians"] - b1ae["n_gaussians"])
        max_loss_diff = max(max_loss_diff, loss_diff)
        max_psnr_diff = max(max_psnr_diff, psnr_diff)
        max_n_diff = max(max_n_diff, n_diff)
        if it % 100 == 0 or it < 20:
            print(f"{it:>6} {b1e['loss']:>12.6f} {b1ae['loss']:>12.6f} {loss_diff:>12.2e} {b1e['psnr']:>10.2f} {b1ae['psnr']:>10.2f} {psnr_diff:>10.2e} {b1e['n_gaussians']:>8} {b1ae['n_gaussians']:>8} {n_diff:>6}")

    # Densification events
    print(f"\nDensification events (clone/split/prune):")
    for it in common_iters:
        b1e = b1_log[it]
        b1ae = b1a_log[it]
        if b1e["cloned"] > 0 or b1e["split"] > 0 or b1e["pruned"] > 0:
            print(f"  iter {it}: B1 clone={b1e['cloned']} split={b1e['split']} prune={b1e['pruned']} | "
                  f"B1A clone={b1ae['cloned']} split={b1ae['split']} prune={b1ae['pruned']}")

    print(f"\nMax differences: loss={max_loss_diff:.2e}, PSNR={max_psnr_diff:.2e}, N={max_n_diff}")

    # Verdict
    rel_loss_diff = max_loss_diff / max(abs(b1_log[common_iters[-1]]["loss"]), 1e-8)
    print(f"\nRelative loss diff at final: {rel_loss_diff:.2e}")

    if max_n_diff == 0 and rel_loss_diff < 1e-4:
        verdict = "PASS — B1A is semantically identical to B1 in training"
    elif max_n_diff <= 2 and rel_loss_diff < 1e-3:
        verdict = "PASS — B1A has negligible training divergence (FP-level)"
    elif max_n_diff > 10 or rel_loss_diff > 1e-2:
        verdict = "FAIL — B1A shows systematic training divergence"
    else:
        verdict = "BORDERLINE — B1A has minor training divergence, manual review needed"

    print(f"\nSEMANTIC FREEZE VERDICT: {verdict}")

    # Save comparison
    comparison = {
        "b1": b1,
        "b1a": b1a,
        "max_loss_diff": max_loss_diff,
        "max_psnr_diff": max_psnr_diff,
        "max_n_diff": max_n_diff,
        "rel_loss_diff": rel_loss_diff,
        "verdict": verdict,
    }
    comp_path = f"{RESULTS_DIR}/short_train_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved to {comp_path}")

if __name__ == "__main__":
    main()
