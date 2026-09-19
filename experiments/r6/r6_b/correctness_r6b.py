#!/usr/bin/env python3
"""Checkpoint replay gradient comparison for the R6-B rasterizer prototype."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "experiments" / "r6"))
from runtime import configure, invalidate  # noqa: E402
from r6_1_bwd_decompose import (  # noqa: E402
    ReferenceV1Config, GTDataset, SepSSIM, load_model_from_ckpt, render_with_meta,
)


def metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict:
    candidate, baseline = candidate.float(), baseline.float()
    delta = (candidate - baseline).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(delta.norm().item() / max(baseline.norm().item(), 1e-30)),
        "nan_or_inf": bool((~torch.isfinite(candidate)).any().item() or
                           (~torch.isfinite(baseline)).any().item()),
    }


def replay(mode: str, ckpt: str, config, cam, target, ssim) -> dict[str, torch.Tensor]:
    configure(mode)
    if mode != "baseline":
        invalidate()
    model, _ = load_model_from_ckpt(ckpt, config, str(REPO))
    image, _, means2d = render_with_meta(model, cam, model.active_sh_degree)
    (0.8 * F.l1_loss(image, target) + 0.2 * ssim(image, target)).backward()
    torch.cuda.synchronize()
    return {
        "mean2d_gradient": means2d.grad.detach(),
        "absgrad_densification": means2d.absgrad.detach(),
        "xyz_gradient": model._xyz.grad.detach(),
        "sh_gradient": model._shs.grad.detach(),
        "scaling_gradient": model._scaling.grad.detach(),
        "rotation_gradient": model._rotation.grad.detach(),
        "opacity_parameter_gradient": model._opacity.grad.detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-idx", type=int, default=0)
    args = parser.parse_args()
    config = ReferenceV1Config()
    config.scene, config.repo_root = args.scene, str(REPO)
    dataset = GTDataset(scene=args.scene, repo_root=str(REPO), resolution=config.resolution,
                        device="cuda", background="black")
    cam, target = dataset.get_item(args.camera_idx)
    ssim = SepSSIM(device="cuda")
    base = replay("baseline", args.ckpt, config, cam, target, ssim)
    b0 = replay("b0", args.ckpt, config, cam, target, ssim)
    b1 = replay("b1", args.ckpt, config, cam, target, ssim)
    result = {
        "scope": "checkpoint replay; rasterizer raw color/conic/opacity checks are test_r6b.py",
        "comparisons": {name: {"b0": metrics(b0[name], ref), "b1": metrics(b1[name], ref)}
                        for name, ref in base.items()},
    }
    result["passed"] = all(
        m[mode]["max_abs"] == 0.0 and not m[mode]["nan_or_inf"]
        for m in result["comparisons"].values() for mode in ("b0", "b1")
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
