#!/usr/bin/env python3
"""
Phase C52 Stage 0 — Transition Analysis
Analyze the transition matrix for densification candidates:
  previous-high → current-high
  previous-high → current-low
  previous-low  → current-high
  previous-low  → current-low

where "high" = above median EMA gradient norm.
"""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c52-stage0")

# Load baseline 5K results (candidate analysis events)
f = result_dir / "baseline_room.json"
if f.exists():
    r = json.load(open(f))
else:
    print("ERROR: baseline_room.json not found")
    exit(1)

events = r.get("candidate_analysis_events", [])
print("=" * 90)
print("Phase C52 Stage 0 — Transition Analysis (Room 5K, EMA signal)")
print("=" * 90)

print("\n### Transition Matrix (per densification event)\n")
print(f"| Iter | N Candidates | P(prev_high→curr_high) | P(prev_low→curr_high) | Pearson | Spearman |")
print(f"|------|-------------|------------------------|----------------------|---------|----------|")

for ev in events:
    n_cand = ev["n_candidates"]
    n_hh = ev["n_prev_high_curr_high"]
    n_lh = ev["n_prev_low_curr_high"]
    p_hh = n_hh / n_cand * 100 if n_cand > 0 else 0
    p_lh = n_lh / n_cand * 100 if n_cand > 0 else 0
    pearson = ev["prev_curr_pearson"]
    spearman = ev.get("prev_curr_spearman", 0.0)
    print(f"| {ev['iter']:>4} | {n_cand:>11,} | {p_hh:>22.1f}% | {p_lh:>20.1f}% | {pearson:>7.4f} | {spearman:>8.4f} |")

# Summary
pearsons = [ev["prev_curr_pearson"] for ev in events if ev["n_candidates"] > 1]
spearmans = [ev.get("prev_curr_spearman", 0.0) for ev in events if ev["n_candidates"] > 1]
p_hh_vals = [ev["n_prev_high_curr_high"] / ev["n_candidates"] * 100 for ev in events if ev["n_candidates"] > 0]
p_lh_vals = [ev["n_prev_low_curr_high"] / ev["n_candidates"] * 100 for ev in events if ev["n_candidates"] > 0]

print(f"\n### Summary Statistics\n")
print(f"| Metric | Mean | Median | Min | Max |")
print(f"|--------|------|--------|-----|-----|")
print(f"| Pearson correlation | {np.mean(pearsons):.4f} | {np.median(pearsons):.4f} | {np.min(pearsons):.4f} | {np.max(pearsons):.4f} |")
print(f"| Spearman correlation | {np.mean(spearmans):.4f} | {np.median(spearmans):.4f} | {np.min(spearmans):.4f} | {np.max(spearmans):.4f} |")
print(f"| P(prev_high→curr_high) | {np.mean(p_hh_vals):.1f}% | {np.median(p_hh_vals):.1f}% | {np.min(p_hh_vals):.1f}% | {np.max(p_hh_vals):.1f}% |")
print(f"| P(prev_low→curr_high) | {np.mean(p_lh_vals):.1f}% | {np.median(p_lh_vals):.1f}% | {np.min(p_lh_vals):.1f}% | {np.max(p_lh_vals):.1f}% |")

print(f"\n### Interpretation\n")
print(f"The EMA gradient norm (decay=0.9) has a mean Pearson correlation of {np.mean(pearsons):.3f}")
print(f"with the current densification-window average gradient among candidates.")
print(f"The median Spearman correlation is {np.median(spearmans):.3f}.")
print(f"\nP(prev_high→curr_high) = {np.mean(p_hh_vals):.1f}% means that {np.mean(p_hh_vals):.1f}% of")
print(f"densification candidates were also above-median in the EMA gradient norm.")
print(f"This indicates the EMA signal captures which Gaussians are consistently important.")
print(f"\nP(prev_low→curr_high) = {np.mean(p_lh_vals):.1f}% means that {np.mean(p_lh_vals):.1f}% of")
print(f"candidates were below-median in the EMA — these are 'surprise' candidates that")
print(f"became important despite low historical gradient.")

# Save
analysis = {
    "mean_pearson": float(np.mean(pearsons)),
    "median_pearson": float(np.median(pearsons)),
    "mean_spearman": float(np.mean(spearmans)),
    "median_spearman": float(np.median(spearmans)),
    "mean_p_prev_high_curr_high": float(np.mean(p_hh_vals)),
    "mean_p_prev_low_curr_high": float(np.mean(p_lh_vals)),
    "n_events": len(events),
    "per_event": events,
}

out_file = result_dir / "transition_analysis.json"
with open(out_file, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to {out_file}")
