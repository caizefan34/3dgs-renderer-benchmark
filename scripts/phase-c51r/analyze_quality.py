#!/usr/bin/env python3
"""Analyze C51-R quality results — compare all configs against baseline."""
import json, numpy as np
from pathlib import Path

d = Path("results/a100/phase-c51r")

# Load all results
results = {}
for f in d.glob("*.json"):
    if f.stem == "final_comparison": continue
    data = json.load(open(f))
    results[f.stem] = data

if "baseline" not in results:
    print("ERROR: baseline.json not found")
    exit(1)

baseline_psnr = results["baseline"]["final_psnr"]
baseline_ssim = None  # SSIM not in eval_points, in trajectory
baseline_gs = results["baseline"]["final_gaussians"]
baseline_time = results["baseline"]["total_train_time_s"]

print("=" * 100)
print("Phase C51-R: Quality Analysis")
print("=" * 100)

# Main comparison table
print(f"\n{'Config':<20} {'PSNR':>7} {'dPSNR':>7} {'GS':>10} {'Clone':>8} {'Split':>8} {'Prune':>7} {'Time(s)':>8} {'Iter(ms)':>9}")
print("-" * 100)

config_order = ["baseline", "k90", "k80", "refresh50", "refresh100", "refresh200", "refresh500", "post_densification"]
for name in config_order:
    if name not in results: continue
    r = results[name]
    psnr = r["final_psnr"]
    dpsnr = psnr - baseline_psnr
    gs = r["final_gaussians"]
    clone = r["total_clone"]
    split = r["total_split"]
    prune = r["total_prune"]
    time_s = r["total_train_time_s"]
    iter_ms = r["iter_time_stats"]["mean"] * 1000
    print(f"{name:<20} {psnr:>7.2f} {dpsnr:>+7.2f} {gs:>10,} {clone:>8,} {split:>8,} {prune:>7,} {time_s:>8.1f} {iter_ms:>9.1f}")

# Gradient correctness table
print(f"\n{'='*100}")
print("Gradient Correctness (last measurement)")
print(f"{'='*100}")
print(f"\n{'Config':<20} {'xyz cos':>8} {'xyz L2':>8} {'scl cos':>8} {'rot cos':>8} {'op cos':>8} {'sh cos':>8}")
print("-" * 80)

for name in config_order:
    if name not in results: continue
    r = results[name]
    gm = r.get("grad_measurements", [])
    if not gm:
        print(f"{name:<20} {'N/A':>8}")
        continue
    last = gm[-1]
    c = last["correctness"]
    print(f"{name:<20} {c['xyz']['cosine']:>8.4f} {c['xyz']['rel_l2']:>8.4f} "
          f"{c['scales']['cosine']:>8.4f} {c['rotations']['cosine']:>8.4f} "
          f"{c['opacity']['cosine']:>8.4f} {c['shs']['cosine']:>8.4f}")

# Acceptance gate check
print(f"\n{'='*100}")
print("Acceptance Gate Check")
print(f"{'='*100}")
print(f"\nBaseline PSNR: {baseline_psnr:.2f}")
print(f"Gate: PSNR degradation < 0.2 dB, SSIM degradation < 0.005, gradient cosine >= 0.99\n")

for name in config_order:
    if name == "baseline" or name not in results: continue
    r = results[name]
    dpsnr = r["final_psnr"] - baseline_psnr
    gm = r.get("grad_measurements", [])
    xyz_cos = gm[-1]["correctness"]["xyz"]["cosine"] if gm else 0

    psnr_pass = abs(dpsnr) < 0.2
    cos_pass = xyz_cos >= 0.99

    # Stability check: no GS collapse or explosion
    gs_ratio = r["final_gaussians"] / baseline_gs
    stable = 0.5 < gs_ratio < 2.0

    print(f"  {name:<20}: dPSNR={dpsnr:>+6.2f} {'PASS' if psnr_pass else 'FAIL':>4}  "
          f"xyz_cos={xyz_cos:.4f} {'PASS' if cos_pass else 'FAIL':>4}  "
          f"GS_ratio={gs_ratio:.2f} {'STABLE' if stable else 'UNSTABLE':>7}")

# PSNR trajectory comparison
print(f"\n{'='*100}")
print("PSNR Trajectory Comparison")
print(f"{'='*100}")

# Get trajectory PSNR at key iters
key_iters = [0, 1000, 2000, 3000, 4000, 4999]
print(f"\n{'Iter':>6}", end="")
for name in config_order:
    if name not in results: continue
    print(f" {name:>14}", end="")
print()
print("-" * (6 + 15 * len([n for n in config_order if n in results])))

for it in key_iters:
    print(f"{it:>6}", end="")
    for name in config_order:
        if name not in results: continue
        traj = results[name].get("trajectory", [])
        psnr_at_iter = None
        for t in traj:
            if t["iter"] == it:
                psnr_at_iter = t["psnr"]
                break
        if psnr_at_iter is not None:
            print(f" {psnr_at_iter:>14.2f}", end="")
        else:
            print(f" {'---':>14}", end="")
    print()

# Error accumulation analysis (for refresh configs)
print(f"\n{'='*100}")
print("Error Accumulation: PSNR Gap vs Baseline Over Training")
print(f"{'='*100}")

baseline_traj = {t["iter"]: t["psnr"] for t in results["baseline"].get("trajectory", [])}

refresh_configs = ["refresh50", "refresh100", "refresh200", "refresh500", "k90", "k80"]
print(f"\n{'Iter':>6}", end="")
for name in refresh_configs:
    if name in results:
        print(f" {name:>14}", end="")
print(f" {'k50_v2(C51)':>14}")
print("-" * (6 + 15 * len([n for n in refresh_configs if n in results]) + 15))

# Also load C51 v2 k50 for comparison
c51_k50 = None
c51_path = Path("results/a100/phase-c51/simulation_k50_v2.json")
if c51_path.exists():
    c51_data = json.load(open(c51_path))
    c51_k50 = {ep["iter"]: ep["psnr"] for ep in c51_data["eval_points"]}

for it in key_iters:
    print(f"{it:>6}", end="")
    for name in refresh_configs:
        if name not in results: continue
        traj = results[name].get("trajectory", [])
        psnr_at = None
        for t in traj:
            if t["iter"] == it:
                psnr_at = t["psnr"]
                break
        if psnr_at is not None and it in baseline_traj:
            gap = baseline_traj[it] - psnr_at
            print(f" {gap:>+14.2f}", end="")
        else:
            print(f" {'---':>14}", end="")
    if c51_k50 and it in c51_k50 and it in baseline_traj:
        gap = baseline_traj[it] - c51_k50[it]
        print(f" {gap:>+14.2f}", end="")
    else:
        print(f" {'---':>14}", end="")
    print()

# Mechanism analysis
print(f"\n{'='*100}")
print("Mechanism Analysis")
print(f"{'='*100}")

print("\nM1: Higher K reduces quality degradation?")
k50_dpsnr = None
if c51_path.exists():
    k50_dpsnr = c51_data["eval_points"][-1]["psnr"] - baseline_psnr
k90_dpsnr = results.get("k90", {}).get("final_psnr", 0) - baseline_psnr
k80_dpsnr = results.get("k80", {}).get("final_psnr", 0) - baseline_psnr
print(f"  K50 (C51 v2): {k50_dpsnr:+.2f} dB" if k50_dpsnr is not None else "  K50: N/A")
print(f"  K80:          {k80_dpsnr:+.2f} dB")
print(f"  K90:          {k90_dpsnr:+.2f} dB")
if k50_dpsnr is not None:
    print(f"  K50→K80 improvement: {k80_dpsnr - k50_dpsnr:+.2f} dB")
    print(f"  K50→K90 improvement: {k90_dpsnr - k50_dpsnr:+.2f} dB")
    m1_pass = (k90_dpsnr > k50_dpsnr) and (abs(k90_dpsnr) < 0.2 or abs(k80_dpsnr) < 0.2)
    print(f"  M1 supported: {'YES' if m1_pass else 'NO'}")

print("\nM2: Periodic refresh reduces cumulative degradation?")
print(f"  K50 no refresh (C51 v2): {k50_dpsnr:+.2f} dB" if k50_dpsnr is not None else "  K50 no refresh: N/A")
for name in ["refresh50", "refresh100", "refresh200", "refresh500"]:
    if name in results:
        dpsnr = results[name]["final_psnr"] - baseline_psnr
        print(f"  {name}: {dpsnr:+.2f} dB")
m2_pass = any(results.get(n, {}).get("final_psnr", 0) - baseline_psnr > (k50_dpsnr or -999)
              for n in ["refresh50", "refresh100", "refresh200", "refresh500"])
print(f"  M2 supported: {'YES' if m2_pass else 'NO'}")

print("\nM3: Post-densification sparse (pending 30K experiment)...")
if "post_densification" in results:
    pd_dpsnr = results["post_densification"]["final_psnr"] - baseline_psnr
    print(f"  Post-densification (30K): {pd_dpsnr:+.2f} dB")
else:
    print("  Results not yet available (30K training in progress)")

# Save analysis
analysis = {
    "baseline_psnr": baseline_psnr,
    "baseline_gs": baseline_gs,
    "baseline_time_s": baseline_time,
    "configs": {},
    "gates": {},
    "mechanisms": {"M1": None, "M2": None, "M3": None},
}
for name in config_order:
    if name not in results: continue
    if name == "baseline": continue
    r = results[name]
    dpsnr = r["final_psnr"] - baseline_psnr
    gm = r.get("grad_measurements", [])
    xyz_cos = gm[-1]["correctness"]["xyz"]["cosine"] if gm else 0
    analysis["configs"][name] = {
        "psnr": r["final_psnr"],
        "dpsnr": dpsnr,
        "gs": r["final_gaussians"],
        "time_s": r["total_train_time_s"],
        "xyz_cos": xyz_cos,
        "clone": r["total_clone"],
        "split": r["total_split"],
        "prune": r["total_prune"],
    }
    analysis["gates"][name] = {
        "psnr_pass": abs(dpsnr) < 0.2,
        "cosine_pass": xyz_cos >= 0.99,
    }

with open(d / "quality_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to {d / 'quality_analysis.json'}")
