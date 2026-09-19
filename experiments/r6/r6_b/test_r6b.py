#!/usr/bin/env python3
"""CUDA correctness checks for R6-B's unambiguous rasterizer buffers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from runtime import configure, invalidate


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().float(), b.detach().float()
    delta = (a - b).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(delta.norm().item() / max(b.norm().item(), 1e-30)),
        "nan_or_inf": bool((~torch.isfinite(a)).any().item() or (~torch.isfinite(b)).any().item()),
    }


def raw_raster_run(mode: str, ids: list[int], reset: bool = True) -> dict[str, torch.Tensor]:
    from gsplat.cuda._wrapper import _RasterizeToPixels

    if reset:
        configure(mode)
    if reset and mode != "baseline":
        invalidate()
    means2d = torch.tensor([[[7.5, 7.5], [11.0, 11.0]]], device="cuda", requires_grad=True)
    conics = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], device="cuda", requires_grad=True)
    colors = torch.tensor([[[0.6, 0.3, 0.2], [0.1, 0.7, 0.4]]], device="cuda", requires_grad=True)
    opacities = torch.tensor([[0.5, 0.4]], device="cuda", requires_grad=True)
    offsets = torch.zeros((1, 1, 1), device="cuda", dtype=torch.int32)
    flatten_ids = torch.tensor(ids, device="cuda", dtype=torch.int32)
    image, alpha = _RasterizeToPixels.apply(
        means2d, conics, colors, opacities, None, None, 16, 16, 16,
        offsets, flatten_ids, True,
    )
    (image.square().mean() + alpha.square().mean()).backward()
    torch.cuda.synchronize()
    return {
        "color_gradient": colors.grad,
        "opacity_gradient": opacities.grad,
        "mean2d_gradient": means2d.grad,
        "conic_gradient": conics.grad,
        "absgrad": means2d.absgrad,
    }


def main() -> None:
    torch.manual_seed(0)
    baseline = raw_raster_run("baseline", [0])
    b0 = raw_raster_run("b0", [0])
    b1 = raw_raster_run("b1", [0])
    comparisons = {name: {"b0": compare(b0[name], ref), "b1": compare(b1[name], ref)}
                   for name, ref in baseline.items()}

    # t touches Gaussian 0.  At t+1 only Gaussian 1 is in flatten_ids: row 0
    # is required to be zero even though it held an accumulated value at t.
    configure("b1")
    invalidate()
    raw_raster_run("b1", [0], reset=False)
    stale = raw_raster_run("b1", [1], reset=False)
    stale_row = stale["mean2d_gradient"][0, 0]
    stale_ok = bool(torch.count_nonzero(stale_row).item() == 0)
    result = {
        "mode": "rasterizer-only",
        "comparisons": comparisons,
        "stale_gradient": {"row": stale_row.detach().cpu().tolist(), "passed": stale_ok},
        "passed": all(
            item[variant]["max_abs"] == 0.0 and not item[variant]["nan_or_inf"]
            for item in comparisons.values() for variant in ("b0", "b1")
        ) and stale_ok,
    }
    out = Path(__file__).with_name("r6-b-local-correctness.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
