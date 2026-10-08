"""Freeze a two-proposal, train-view-only child-placement choice for one event."""

import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).parent
OUT = HERE / "fastgs_child_proposal_choice_20260927_20260928.pt"
SUMMARY = HERE / "fastgs_child_proposal_choice_summary.json"


def main():
    a = torch.load(HERE / "fastgs_event_gate_multiview_scores_20260927.pt",
                   map_location="cpu", weights_only=True)
    b = torch.load(HERE / "fastgs_event_gate_multiview_scores_20260928.pt",
                   map_location="cpu", weights_only=True)
    for key in ("ply_sha256", "mask_sha256", "camera_names"):
        if a[key] != b[key]:
            raise AssertionError(f"mismatched {key}")
    parents = a["family_parent_ids"].long()
    if not torch.equal(parents, b["family_parent_ids"]):
        raise AssertionError("candidate parent order differs")
    score27 = a["camera_family_loss_vjp"].double().mean(dim=0)
    score28 = b["camera_family_loss_vjp"].double().mean(dim=0)
    choose28 = score28 < score27
    chosen = torch.where(choose28, score28, score27)
    bundle = {
        "ply_sha256": a["ply_sha256"],
        "mask_sha256": a["mask_sha256"],
        "camera_names": a["camera_names"],
        "family_parent_ids": parents,
        "choose_seed20260928": choose28,
        "seed20260927": 20260927,
        "seed20260928": 20260928,
        "rule": "For each native selected split parent, take both children from the proposal seed whose mean full-resolution loss VJP over the ten fixed training cameras is smaller; ties choose 20260927.",
    }
    torch.save(bundle, OUT)
    summary = {
        "parents": int(parents.numel()),
        "choose_seed20260928": int(choose28.sum()),
        "choose_seed20260927": int((~choose28).sum()),
        "mean_vjp_seed27_sum": float(score27.sum()),
        "mean_vjp_seed28_sum": float(score28.sum()),
        "mean_vjp_chosen_sum": float(chosen.sum()),
        "scores_source": ["fastgs_event_gate_multiview_scores_20260927.pt",
                          "fastgs_event_gate_multiview_scores_20260928.pt"],
        "policy_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "scope": "One room native split event, two child proposal samples, ten fixed full-resolution training cameras. Smaller first derivative is not verified long-horizon utility; selection cost is not free.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
