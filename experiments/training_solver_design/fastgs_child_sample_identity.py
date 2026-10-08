"""Hash the author-rule full child proposals for frozen room split seeds."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scene.gaussian_model import GaussianModel  # noqa: E402


def main():
    model = GaussianModel(3, "default")
    model.load_ply(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply")
    selected = torch.load("/root/fastgs_gate_selected_4100_mask.pt", map_location="cpu", weights_only=True)["selected_splits"]
    ids = selected.nonzero().flatten().to("cuda")
    if ids.numel() != 9345:
        raise AssertionError(ids.numel())
    stds = model.get_scaling[ids].repeat(2, 1)
    result = {}
    for seed in (20260927, 20260928):
        torch.manual_seed(seed)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        result[str(seed)] = hashlib.sha256(samples.detach().cpu().numpy().tobytes()).hexdigest()
    print("FASTGS_CHILD_SAMPLE_IDENTITY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
