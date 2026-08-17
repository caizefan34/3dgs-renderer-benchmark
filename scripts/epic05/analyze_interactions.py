#!/usr/bin/env python3
"""
Analyze interaction effects between optimization modules.
Computes interaction gains and classifies them as positive/neutral/negative.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_aggregated(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_interaction_effects(ablation_agg: Dict, interaction_agg: Dict) -> str:
    """
    Compute interaction gains:
    interaction_gain = combined_gain - expected_independent_gain
    where gain = baseline_mean / config_mean (speedup factor)
    """
    # Build lookup by module_id for ablation
    ablation_by_id = {}
    for r in ablation_agg["results"]:
        ablation_by_id[r["module_id"]] = r

    baseline = ablation_by_id.get("M0")
    if not baseline:
        return "Error: No baseline (M0) found in ablation results"

    b_mean = baseline["mean_ms"]

    lines = [
        "## Interaction Analysis\n",
        "| Exp | Combination | Individual Gains | Expected | Actual | Interaction | Verdict |",
        "|-----|-------------|-----------------:|--------:|------:|-----------:|---------|",
    ]

    for exp in interaction_agg["results"]:
        mod_ids = exp.get("modules", [])
        if not mod_ids or "M0" in mod_ids:
            continue

        # Get individual gains
        indiv_gains = []
        for mid in mod_ids:
            if mid in ablation_by_id:
                r = ablation_by_id[mid]
                gain = b_mean / r["mean_ms"]
                indiv_gains.append(gain)

        expected_gain = sum(indiv_gains) - (len(indiv_gains) - 1) if indiv_gains else 1.0
        actual_gain = b_mean / exp["mean_ms"]

        interaction_gain = actual_gain - expected_gain

        if interaction_gain > 0.1:
            verdict = "Positive interaction"
        elif interaction_gain < -0.1:
            verdict = "Negative interaction"
        else:
            verdict = "Neutral (additive)"

        indiv_str = " + ".join([f"{g:.3f}x" for g in indiv_gains])
        lines.append(
            f"| {exp.get('experiment_id', '?')} | {exp.get('description', '?')} | {indiv_str} | "
            f"{expected_gain:.3f}x | {actual_gain:.3f}x | {interaction_gain:+.3f}x | {verdict} |"
        )

    return "\n".join(lines)


def main():
    agg_dir = REPO_ROOT / "results" / "epic05" / "aggregated"

    for scene in ["50k", "200k", "400k"]:
        for res in ["720p", "1080p", "4k"]:
            ablation_path = agg_dir / f"ablation_{scene}_{res}.json"
            interaction_path = agg_dir / f"interaction_{scene}_{res}.json"

            ablation = load_aggregated(ablation_path)
            interaction = load_aggregated(interaction_path)

            if ablation and interaction:
                print(f"\n{'=' * 70}")
                print(f"  Interaction Analysis: {scene}, {res}")
                print(f"{'=' * 70}")
                analysis = compute_interaction_effects(ablation, interaction)
                print(analysis)

                # Save
                out_path = agg_dir / f"interaction_analysis_{scene}_{res}.md"
                with open(out_path, "w") as f:
                    f.write(analysis)
                print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
