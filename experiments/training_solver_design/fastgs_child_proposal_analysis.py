"""Describe one room child-choice pilot against both frozen proposal controls."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
new = json.loads((HERE / "fastgs_child_proposal_eval.json").read_text(encoding="utf-8"))
old27 = json.loads((HERE / "fastgs_gate_branch_64_512_eval.json").read_text(encoding="utf-8"))
old28 = json.loads((HERE / "fastgs_gate_branch_64_512_eval_20260928.json").read_text(encoding="utf-8"))
score27 = json.loads((HERE / "fastgs_event_gate_multiview_result.json").read_text(encoding="utf-8"))
score28 = json.loads((HERE / "fastgs_event_gate_multiview_result_20260928.json").read_text(encoding="utf-8"))
sources = {
    "matched_control": new["arms"],
    "prior_seed20260927": old27["arms"],
    "prior_seed20260928": old28["arms"],
}
rng = np.random.default_rng(20260927)
result = {"comparisons": {}, "cost": {}}
for step in (4164, 4612):
    chosen = new["arms"][f"choice_{step}"]["per_view"]
    for name, arms in sources.items():
        control = arms[f"control_{step}"]["per_view"]
        if [x["camera"] for x in control] != [x["camera"] for x in chosen]:
            raise AssertionError(f"unpaired cameras: {name}")
        row = {}
        for metric in ("psnr", "ssim", "lpips"):
            delta = np.array([a[metric] - b[metric] for a, b in zip(chosen, control)])
            indices = rng.integers(len(delta), size=(10000, len(delta)))
            row[metric] = {
                "mean_choice_minus_control": float(delta.mean()),
                "views_choice_better": int((delta < 0).sum() if metric == "lpips" else (delta > 0).sum()),
                "view_bootstrap_95_interval": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            }
        result["comparisons"][f"{step}_{name}"] = row

cost_keys = ("sum_base_render_wall_ms", "sum_original_loss_adjoint_wall_ms",
             "sum_union_render_wall_ms", "sum_event_gate_wall_ms")
result["cost"]["measured_two_separate_scoring_passes_ms"] = sum(
    score[k] for score in (score27, score28) for k in cost_keys)
result["cost"]["measured_components_only"] = True
result["cost"]["complete_branch_process_wall_seconds"] = {
    arm: float((HERE / f"fastgs_child_choice_{arm}.time").read_text().split()[-1])
    for arm in ("control", "choice")
}
result["cost"]["scoring_excluded_from_branch_wall"] = True
result["scope"] = ("One room split event and 512 subsequent full-resolution steps. "
                   "The 39 room views are development views; the bootstrap is descriptive "
                   "only and does not measure cross-scene uncertainty. "
                   "Scoring component timings exclude proposal construction, data load, "
                   "and scoring job startup; branch process wall is not a fair net speed test.")
(HERE / "fastgs_child_proposal_analysis.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
