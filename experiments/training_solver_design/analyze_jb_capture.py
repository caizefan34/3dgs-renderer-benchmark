"""Read the frozen SAD split capture's child-count distribution without rendering."""

import json
import sys
from collections import Counter

import torch


def main(path):
    capture = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    counts = capture["child_counts"].to(torch.int64)
    parents = capture["parents"].to(torch.int64)
    candidates = capture["structural_candidates"].to(torch.int64)
    candidate_mask = torch.isin(parents, candidates)
    result = {}
    eta = capture.get("eta")
    if eta is not None:
        eta = eta[parents].float()
        axes = torch.sqrt(torch.clamp(eta, min=1.0)).ceil().int().clamp(min=1)
        active_axes = (axes > 1).sum(dim=1)
        patterns = Counter(tuple(row.tolist()) for row in axes)
        result["axis_structure"] = {
            "active_axes_parent_hist": {str(k): int((active_axes == k).sum()) for k in range(4)},
            "active_axes_child_hist": {str(k): int(counts[active_axes == k].sum()) for k in range(4)},
            "top_12_k_patterns": [{"k": k, "parents": n} for k, n in patterns.most_common(12)],
        }
        observations = capture["model"]["tensors"]["accum_view_count"][parents].float()
        triple = active_axes == 3
        observed = observations[triple]
        born = counts[triple]
        result["three_axis_observation_count"] = {
            "meaning": "valid observation updates since last accumulator reset; not distinct camera identities",
            "parents": int(triple.sum()),
            "children": int(born.sum()),
            "parents_at_most_1": int((observed <= 1).sum()),
            "children_at_most_1": int(born[observed <= 1].sum()),
            "parents_at_most_2": int((observed <= 2).sum()),
            "children_at_most_2": int(born[observed <= 2].sum()),
            "parents_at_least_5": int((observed >= 5).sum()),
            "children_at_least_5": int(born[observed >= 5].sum()),
            "max": float(observed.max()) if observed.numel() else 0,
        }
    for label, selected in (("all", counts), ("structural_candidates", counts[candidate_mask])):
        result[label] = {
            "parents": selected.numel(),
            "children": int(selected.sum()),
            "one": int((selected == 1).sum()),
            "two": int((selected == 2).sum()),
            "three_to_eight": int(((selected >= 3) & (selected <= 8)).sum()),
            "above_eight": int((selected > 8).sum()),
            "children_from_above_eight": int(selected[selected > 8].sum()),
            "max": int(selected.max()) if selected.numel() else 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
