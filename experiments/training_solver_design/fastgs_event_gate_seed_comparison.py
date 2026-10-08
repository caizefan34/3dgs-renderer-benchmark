"""Compare frozen FastGS split-event scores across two child proposal seeds."""

import json
from pathlib import Path

import torch


HERE = Path(__file__).parent


def load(seed):
    bundle = torch.load(HERE / f"fastgs_event_gate_multiview_scores_{seed}.pt",
                        map_location="cpu", weights_only=True)
    policy = torch.load(HERE / f"fastgs_event_gate_stable_positive_mask_{seed}.pt",
                        map_location="cpu", weights_only=True)
    return bundle, policy


def main():
    a, pa = load(20260927)
    b, pb = load(20260928)
    if not torch.equal(a["family_parent_ids"], b["family_parent_ids"]):
        raise AssertionError("different parent candidates")
    if a["camera_names"] != b["camera_names"]:
        raise AssertionError("different cameras")
    ga = a["camera_family_loss_vjp"].double().mean(dim=0)
    gb = b["camera_family_loss_vjp"].double().mean(dim=0)
    sa, sb = pa["stable_positive"], pb["stable_positive"]
    significant = (ga.abs() > 1e-10) & (gb.abs() > 1e-10)
    sign_disagreement = significant & (ga * gb < 0)
    both = int((sa & sb).sum())
    either = int((sa | sb).sum())
    result = {
        "parent_count": int(ga.numel()),
        "same_parent_and_train_camera_lists": True,
        "both_mean_significant": int(significant.sum()),
        "opposite_mean_signs": int(sign_disagreement.sum()),
        "stable_positive_seed27": int(sa.sum()),
        "stable_positive_seed28": int(sb.sum()),
        "stable_positive_intersection": both,
        "stable_positive_union": either,
        "stable_positive_jaccard": both / either,
        "scope": "Two candidate child samples on the same room state/cameras. A changed score may reflect genuinely different child proposals; this comparison does not measure long-horizon utility.",
    }
    (HERE / "fastgs_event_gate_seed_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
