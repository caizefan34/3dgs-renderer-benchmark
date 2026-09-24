#!/usr/bin/env python3
"""
Phase C52 Stage 0 — Candidate Analysis
Analyze the relationship between previous-iteration and current-iteration gradient
for densification candidates. Measures Pearson/Spearman correlation, top-K overlap.
"""
import json
from pathlib import Path
import numpy as np

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c52-stage0")

# Load all results
results = {}
for f in result_dir.glob("*_room.json"):
    name = f.stem.replace("_room", "")
    results[name] = json.load(open(f))

print("=" * 100)
print("Phase C52 Stage 0 — Candidate Analysis (Room 5K)")
print("=" * 100)

# 1. Densification event comparison
print("\n### Densification Events Summary\n")
print(f"| Config | Strategy | Budget | Total Clone | Total Split | Total Prune | Final GS |")
print(f"|--------|----------|--------|-------------|-------------|-------------|----------|")
for name in ["baseline", "uniform_75", "uniform_50", "predictive_75", "predictive_50", "oracle_75", "oracle_50"]:
    r = results.get(name)
    if r:
        cfg = r["config"]
        print(f"| {name} | {cfg['strategy']} | {cfg['budget_fraction']:.0%} | "
              f"{r['total_clone']:>10,} | {r['total_split']:>10,} | {r['total_prune']:>10,} | "
              f"{r['final_gaussians']:>8,} |")

# 2. Budget matching verification
print("\n### Budget Matching (vs Baseline)\n")
b = results.get("baseline")
if b:
    b_clone = b["total_clone"]
    b_split = b["total_split"]
    print(f"| Config | Clone Ratio | Split Ratio | Total Ops Ratio |")
    print(f"|--------|-------------|-------------|-----------------|")
    for name in ["uniform_75", "uniform_50", "predictive_75", "predictive_50", "oracle_75", "oracle_50"]:
        r = results.get(name)
        if r:
            cl_r = r["total_clone"] / b_clone * 100 if b_clone > 0 else 0
            sp_r = r["total_split"] / b_split * 100 if b_split > 0 else 0
            total_r = (r["total_clone"] + r["total_split"]) / (b_clone + b_split) * 100
            print(f"| {name} | {cl_r:>9.1f}% | {sp_r:>9.1f}% | {total_r:>13.1f}% |")

# 3. Quality comparison (pre-reset and post-reset)
print("\n### Quality Trajectory\n")
milestones = [0, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
print(f"| Iter |", end="")
for name in ["baseline", "uniform_75", "predictive_75", "oracle_75", "uniform_50", "predictive_50", "oracle_50"]:
    short = name.replace("_room", "")
    print(f" {short:>10} |", end="")
print()
print(f"|------|", end="")
for _ in range(7):
    print(f"------------|", end="")
print()

for m in milestones:
    print(f"| {m:>4} |", end="")
    for name in ["baseline", "uniform_75", "predictive_75", "oracle_75", "uniform_50", "predictive_50", "oracle_50"]:
        r = results.get(name)
        if r:
            traj = {t["iter"]: t for t in r.get("trajectory", [])}
            if m in traj:
                print(f" {traj[m]['psnr']:>10.2f} |", end="")
            else:
                print(f" {'N/A':>10} |", end="")
        else:
            print(f" {'N/A':>10} |", end="")
    print()

# 4. Matched-budget comparison (the critical test)
print("\n### Matched-Budget Comparison (Critical Test)\n")
print(f"| Budget | Uniform PSNR | Predictive PSNR | Oracle PSNR | P-U Δ | O-U Δ | P-O Gap |")
print(f"|--------|-------------|-----------------|-------------|-------|-------|---------|")

# Use pre-reset PSNR (iter 2500) since 5K final is unreliable
for budget_label, names in [("75%", ("uniform_75", "predictive_75", "oracle_75")),
                             ("50%", ("uniform_50", "predictive_50", "oracle_50"))]:
    u, p, o = names
    ru, rp, ro = results.get(u), results.get(p), results.get(o)
    if ru and rp and ro:
        # Use iter 2500 (pre-reset) for quality
        def get_psnr_at(r, it):
            traj = {t["iter"]: t for t in r.get("trajectory", [])}
            return traj.get(it, {}).get("psnr", 0)

        u_psnr = get_psnr_at(ru, 2500)
        p_psnr = get_psnr_at(rp, 2500)
        o_psnr = get_psnr_at(ro, 2500)
        pu = p_psnr - u_psnr
        ou = o_psnr - u_psnr
        po = p_psnr - o_psnr
        print(f"| {budget_label} | {u_psnr:>11.2f} | {p_psnr:>15.2f} | {o_psnr:>11.2f} | "
              f"{pu:>+5.2f} | {ou:>+5.2f} | {po:>+7.2f} |")

# Also show final (post-reset) comparison
print(f"\n| Budget | Uniform (final) | Predictive (final) | Oracle (final) | P-U Δ | O-U Δ |")
print(f"|--------|----------------|---------------------|----------------|-------|-------|")
for budget_label, names in [("75%", ("uniform_75", "predictive_75", "oracle_75")),
                             ("50%", ("uniform_50", "predictive_50", "oracle_50"))]:
    u, p, o = names
    ru, rp, ro = results.get(u), results.get(p), results.get(o)
    if ru and rp and ro:
        u_psnr = ru["final_psnr"]
        p_psnr = rp["final_psnr"]
        o_psnr = ro["final_psnr"]
        pu = p_psnr - u_psnr
        ou = o_psnr - u_psnr
        print(f"| {budget_label} | {u_psnr:>14.2f} | {p_psnr:>19.2f} | {o_psnr:>14.2f} | "
              f"{pu:>+5.2f} | {ou:>+5.2f} |")

# 5. Candidate correlation analysis
print("\n### Candidate Correlation Analysis\n")
# Use baseline's candidate analysis events (same candidates for all conditions)
b = results.get("baseline")
if b and b.get("candidate_analysis_events"):
    events = b["candidate_analysis_events"]
    print(f"Using baseline candidate analysis events ({len(events)} densification events)\n")
    print(f"| Iter | N Candidates | Prev-Curr Pearson | P(prev_high→curr_high) | P(prev_low→curr_high) |")
    print(f"|------|-------------|-------------------|------------------------|----------------------|")
    for ev in events:
        n_cand = ev["n_candidates"]
        pearson = ev["prev_curr_pearson"]
        n_hh = ev["n_prev_high_curr_high"]
        n_lh = ev["n_prev_low_curr_high"]
        p_hh = n_hh / n_cand * 100 if n_cand > 0 else 0
        p_lh = n_lh / n_cand * 100 if n_cand > 0 else 0
        print(f"| {ev['iter']:>4} | {n_cand:>11,} | {pearson:>17.4f} | {p_hh:>22.1f}% | "
              f"{p_lh:>20.1f}% |")

    # Summary statistics
    pearsons = [ev["prev_curr_pearson"] for ev in events if ev["n_candidates"] > 1]
    if pearsons:
        print(f"\nMean Pearson correlation: {np.mean(pearsons):.4f}")
        print(f"Median Pearson correlation: {np.median(pearsons):.4f}")
        print(f"Min: {np.min(pearsons):.4f}, Max: {np.max(pearsons):.4f}")
else:
    print("No candidate analysis events found in baseline.")

# 6. Per-event detailed comparison
print("\n### Per-Event Densification Details (Baseline)\n")
if b and b.get("densification_events"):
    print(f"| Iter | Candidates | Selected | Clone | Split | Prune | GS After |")
    print(f"|------|-----------|----------|-------|-------|-------|----------|")
    for ev in b["densification_events"][:15]:
        print(f"| {ev['iter']:>4} | {ev.get('n_candidates', 'N/A'):>9,} | "
              f"{ev.get('n_selected', 'N/A'):>8,} | {ev['cloned']:>5,} | {ev['split']:>5,} | "
              f"{ev['pruned']:>5,} | {ev['gaussians']:>8,} |")

# Save analysis
analysis = {
    "budget_matching": {},
    "quality_comparison": {},
    "candidate_correlation": {},
}
if b:
    for name in ["uniform_75", "uniform_50", "predictive_75", "predictive_50", "oracle_75", "oracle_50"]:
        r = results.get(name)
        if r:
            analysis["budget_matching"][name] = {
                "clone_ratio": r["total_clone"] / b["total_clone"] if b["total_clone"] > 0 else 0,
                "split_ratio": r["total_split"] / b["total_split"] if b["total_split"] > 0 else 0,
            }
    if b.get("candidate_analysis_events"):
        pearsons = [ev["prev_curr_pearson"] for ev in b["candidate_analysis_events"] if ev["n_candidates"] > 1]
        analysis["candidate_correlation"] = {
            "mean_pearson": float(np.mean(pearsons)) if pearsons else 0,
            "median_pearson": float(np.median(pearsons)) if pearsons else 0,
            "n_events": len(pearsons),
        }

out_file = result_dir / "candidate_analysis.json"
with open(out_file, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nAnalysis saved to {out_file}")
