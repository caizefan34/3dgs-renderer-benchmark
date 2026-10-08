"""Check whether the 4100 pre-event PLY matches the saved 4099 parameters."""

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
    checkpoint = torch.load(REPO / "checkpoints/room_big_4099.pth", map_location="cpu", weights_only=False)
    params, iteration = checkpoint
    model = GaussianModel(3, "default")
    model.load_ply(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply")
    names = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
    result = {"checkpoint_iteration": iteration, "rows": int(model.get_xyz.shape[0]), "parameters": {}}
    for index, name in enumerate(names, start=1):
        original = params[index].detach().cpu()
        loaded = getattr(model, name).detach().cpu()
        if original.shape != loaded.shape:
            raise AssertionError((name, original.shape, loaded.shape))
        result["parameters"][name] = {
            "torch_equal": bool(torch.equal(original, loaded)),
            "max_abs_difference": float((original - loaded).abs().max()),
        }
    print("FASTGS_CHECKPOINT_PLY_IDENTITY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
