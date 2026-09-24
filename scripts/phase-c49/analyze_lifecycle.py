#!/usr/bin/env python3
"""Analyze C49 lifecycle profiling results."""
import json
from pathlib import Path

f = Path("results/a100/phase-c49/lifecycle_profile.json")
data = json.load(open(f))

print("=" * 80)
print("Phase C49 Experiment 1: Gaussian Lifecycle Profile — RESULTS")
print("=" * 80)

# Lifecycle summary
ls = data["lifecycle_summary"]
print(f"\n--- Lifecycle Summary ---")
print(f"  Initial Gaussians:  {ls['initial_gaussians']:,}")
print(f"  Total created:      {ls['total_created']:,}")
print(f"  Total pruned:       {ls['total_pruned']:,}")
print(f"  Final Gaussians:    {ls['final_gaussians']:,}")
print(f"  Clone count:        {ls['total_clone']:,}")
print(f"  Split count:        {ls['total_split']:,}")
print(f"  Waste ratio:        {ls['waste_ratio']:.1%}  (pruned/created)")

# Per-densification step
print(f"\n--- Per-Densification-Step Statistics ---")
print(f"{'Iter':>5} {'Pre-N':>8} {'Created':>8} {'Cloned':>7} {'Split':>7} {'Pruned':>7} {'Post-N':>8} {'Net':>7}")
print("-" * 65)
for d in data["densify_stats"]:
    print(f"{d['iter']:>5} {d['pre_N']:>8,} {d['created']:>8} {d['cloned']:>7} {d['split']:>7} {d['pruned']:>7} {d['post_N']:>8,} {d['net_change']:>+7}")

# Gradient distribution evolution
print(f"\n--- Gradient Distribution Evolution ---")
print(f"{'Iter':>5} {'N':>8} {'Med':>10} {'Mean':>10} {'Max':>10} {'Top1%':>7} {'Top10%':>7} {'Top50%':>7} {'K_90%':>7} {'G-O corr':>8} {'G-S corr':>8}")
print("-" * 100)
for gd in data["gradient_distributions"]:
    print(f"{gd['iter']:>5} {gd['n_gaussians']:>8,} {gd['grad_p50']:>10.6f} {gd['grad_mean']:>10.6f} {gd['grad_max']:>10.6f} {gd['top_1pct_fraction']*100:>6.1f}% {gd['top_10pct_fraction']*100:>6.1f}% {gd['top_50pct_fraction']*100:>6.1f}% {gd['k_for_90pct_fraction']*100:>6.1f}% {gd.get('grad_opacity_correlation',0):>8.3f} {gd.get('grad_scale_correlation',0):>8.3f}")

# Key insights
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

# Track B: Waste ratio
print(f"\nTrack B (Lazy Densification):")
print(f"  Waste ratio: {ls['waste_ratio']:.1%} — {ls['total_pruned']:,} pruned out of {ls['total_created']:,} created")
if ls['waste_ratio'] > 0.3:
    print(f"  → HIGH waste: >30% of created Gaussians are pruned")
    print(f"  → Lazy densification is PROMISING")
else:
    print(f"  → MODERATE waste: <30% pruned")

# Track C: Gradient concentration
last_gd = data["gradient_distributions"][-1] if data["gradient_distributions"] else {}
print(f"\nTrack C (Sparse Backward):")
if last_gd:
    print(f"  Top 1% of Gaussians → {last_gd['top_1pct_fraction']*100:.1f}% of total gradient")
    print(f"  Top 10% → {last_gd['top_10pct_fraction']*100:.1f}%")
    print(f"  Top 50% → {last_gd['top_50pct_fraction']*100:.1f}%")
    print(f"  90% of gradient from {last_gd['k_for_90pct_fraction']*100:.1f}% of Gaussians")
    print(f"  → If we skip bottom 68% Gaussians in backward, only 10% gradient is lost")
    print(f"  → Grad-Opacity correlation: {last_gd.get('grad_opacity_correlation',0):.3f} (weak)")
    print(f"  → Grad-Scale correlation: {last_gd.get('grad_scale_correlation',0):.3f} (near zero)")

# Track A: Criterion analysis
print(f"\nTrack A (Gradient Importance):")
if last_gd:
    print(f"  Grad-Opacity correlation: {last_gd.get('grad_opacity_correlation',0):.3f}")
    print(f"  → Weak positive correlation: high gradient does NOT strongly imply high opacity")
    print(f"  → Gradient magnitude alone misses opacity information")
    print(f"  → Adding opacity to criterion could improve densification quality")

# Training quality
print(f"\nTraining Quality:")
for ep in data["eval_points"]:
    print(f"  iter={ep['iter']:>5}  PSNR={ep['psnr']:.2f}  GS={ep['gaussians']:,}")
print(f"  Total time: {data['total_train_time_s']:.1f}s")
