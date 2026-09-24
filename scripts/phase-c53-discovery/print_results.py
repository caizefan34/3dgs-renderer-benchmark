#!/usr/bin/env python3
"""Print key analysis results for the report."""
import json
from pathlib import Path

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")

# Temporal horizon
with open(RESULT_DIR / "temporal_horizon.json") as f:
    t = json.load(f)

print("=== TEMPORAL HORIZON (mean Pearson across checkpoints) ===")
for scene in ["room", "garden", "bicycle"]:
    if scene in t:
        s = t[scene].get("summary", {}).get("temporal_decay", {})
        print(f"\n--- {scene} ---")
        for util in ["grad_sum", "update_xyz", "vis_count"]:
            if util in s:
                print(f"  {util}:")
                for sig in ["ema_grad_norm", "visibility_count", "opacity", "prev_grad_norm"]:
                    if sig in s[util]:
                        vals = s[util][sig]
                        d10 = vals.get("delta_10", {}).get("mean_pearson", 0)
                        d50 = vals.get("delta_50", {}).get("mean_pearson", 0)
                        d100 = vals.get("delta_100", {}).get("mean_pearson", 0)
                        print(f"    {sig}: d10={d10:.3f} d50={d50:.3f} d100={d100:.3f}")

# Densification analysis
print("\n\n=== DENSIFICATION ANALYSIS ===")
dens_file = RESULT_DIR / "densification_analysis.json"
if dens_file.exists():
    with open(dens_file) as f:
        d = json.load(f)
    for scene in ["room", "garden", "bicycle"]:
        if scene in d:
            s = d[scene].get("summary", {})
            print(f"\n--- {scene} ---")
            for outcome in ["clone_lift", "split_lift", "prune_lift"]:
                if outcome in s:
                    print(f"  {outcome}:")
                    for sig in ["ema_grad_norm", "visibility_count", "opacity", "prev_grad_norm", "scale_norm"]:
                        if sig in s[outcome]:
                            print(f"    {sig}: mean_lift={s[outcome][sig]['mean_lift']:.3f}")
        # Also show outcome counts
        if scene in d:
            for cp_key, cp_data in d[scene]["checkpoints"].items():
                oc = cp_data.get("outcome_counts", {})
                if any(oc.values()):
                    print(f"  CP {cp_key}: {oc}")
                    break  # just show first checkpoint

# Age stratification
print("\n\n=== AGE STRATIFICATION (best signal per age group) ===")
with open(RESULT_DIR / "age_stratification.json") as f:
    a = json.load(f)
for scene in ["room", "garden", "bicycle"]:
    if scene in a:
        s = a[scene].get("summary", {}).get("best_signal_by_age", {})
        print(f"\n--- {scene} ---")
        for group in ["age_lt_100", "age_100_500", "age_500_2000", "age_gt_2000"]:
            if group in s:
                print(f"  {group}:")
                for util in ["grad_sum", "update_xyz", "vis_count"]:
                    if util in s[group]:
                        bs = s[group][util]
                        print(f"    {util}: best={bs['best_signal']}, pearson={bs['best_pearson']:.3f}")
