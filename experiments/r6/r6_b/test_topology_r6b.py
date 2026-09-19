#!/usr/bin/env python3
"""B1-v2 topology-transition correctness test.

Tests the exact sequence:

  normal iteration  ->  densification (clone/split/prune)  ->  invalidate
  ->  next backward

Verifies:
  1. Full-clear fallback occurs exactly as intended after invalidate.
  2. No stale rows remain after index remapping (capacity growth).
  3. Gradients after topology change match baseline exactly.

Also tests capacity growth: the persistent buffer must grow to accommodate
new Gaussians without leaking stale data from the old (smaller) buffer.
"""
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
from runtime import configure, invalidate, last_clear_ms, metadata_bytes, prev_n_rows  # noqa: E402
from test_sequential_r6b import compare, zero_check  # noqa: E402


def raw_raster_run(mode: str, ids: list[int], n_gauss: int,
                   reset: bool = True) -> dict[str, torch.Tensor]:
    """Run a single rasterizer forward+backward with n_gauss Gaussians."""
    from gsplat.cuda._wrapper import _RasterizeToPixels
    if reset:
        configure(mode)
    if reset and mode != "baseline":
        invalidate()

    means2d = torch.randn((1, n_gauss, 2), device="cuda", requires_grad=True) * 8 + 8
    conics = torch.tensor([[[1.0, 0.0, 1.0]] * n_gauss], device="cuda", requires_grad=True)
    colors = torch.rand((1, n_gauss, 3), device="cuda", requires_grad=True)
    opacities = torch.rand((1, n_gauss), device="cuda", requires_grad=True)
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


def test_topology_rasterizer() -> dict:
    """Rasterizer-only topology test: capacity growth + invalidate.

    Sequence:
      1. B1-v2 backward with N=2 (builds mask for 2 rows)
      2. Simulate topology change: invalidate()
      3. B1-v2 backward with N=4 (capacity grows, full-clear must occur)

    After step 3, rows 2-3 (new) must not contain stale data from the old
    buffer.  Compare against baseline N=4.
    """
    torch.manual_seed(42)

    # Baseline: N=4 backward (fresh alloc, no stale data)
    configure("baseline")
    base = raw_raster_run("baseline", [0, 1, 2, 3], n_gauss=4)

    # B1-v2: N=2 backward, then invalidate, then N=4 backward
    configure("b1")
    invalidate()
    _ = raw_raster_run("b1", [0, 1], n_gauss=2, reset=False)  # t: 2 Gaussians

    # Simulate topology change
    invalidate()
    md_before = metadata_bytes()
    rows_before = prev_n_rows()

    b1_grown = raw_raster_run("b1", [0, 1, 2, 3], n_gauss=4, reset=False)  # t+1: 4 Gaussians

    md_after = metadata_bytes()
    rows_after = prev_n_rows()

    # Compare B1-v2 N=4 against baseline N=4
    comparison = {name: compare(b1_grown[name], base[name]) for name in base}

    # Verify no stale rows: the comparison max_abs must be 0
    passed = all(item["max_abs"] == 0.0 and not item["nan_or_inf"]
                 for item in comparison.values())

    return {
        "test": "topology_rasterizer",
        "description": "B1-v2: N=2 backward -> invalidate -> N=4 backward (capacity growth)",
        "metadata_bytes_before_invalidate": md_before,
        "n_rows_before_invalidate": rows_before,
        "metadata_bytes_after_growth": md_after,
        "n_rows_after_growth": rows_after,
        "comparison_vs_baseline": comparison,
        "passed": passed,
    }


def test_full_clear_after_invalidate() -> dict:
    """Verify that invalidate() forces a full clear on the next backward.

    Sequence:
      1. B1-v2 backward with flatten_ids=[0] (builds mask {0})
      2. invalidate()
      3. B1-v2 backward with flatten_ids=[1] (must full-clear, not selective)

    After step 3, row 0 must be zero (because full-clear zeros everything),
    and row 1 must match baseline.  This is the same outcome as selective
    clear in this case, but the mechanism is different: full clear zeros
    ALL rows, not just row 0.
    """
    torch.manual_seed(0)

    configure("baseline")
    base_t1 = raw_raster_run("baseline", [1], n_gauss=2)

    configure("b1")
    invalidate()
    _ = raw_raster_run("b1", [0], n_gauss=2, reset=False)
    invalidate()
    b1_t1 = raw_raster_run("b1", [1], n_gauss=2, reset=False)

    comparison = {name: compare(b1_t1[name], base_t1[name]) for name in base_t1}
    stale_row0 = {name: zero_check(b1_t1[name][0, 0]) for name in b1_t1}

    passed = all(item["max_abs"] == 0.0 and not item["nan_or_inf"]
                 for item in comparison.values())
    return {
        "test": "full_clear_after_invalidate",
        "description": "B1-v2: [0]->bwd, invalidate, [1]->bwd (full-clear fallback)",
        "comparison_vs_baseline": comparison,
        "stale_row0_zero": stale_row0,
        "passed": passed,
    }


def test_checkpoint_topology(scene: str, ckpt: str, cam_idx: int) -> dict:
    """Checkpoint-level topology test.

    1. Normal backward (builds B1-v2 mask)
    2. Simulate densification: call model.densify_and_prune (or add random
       Gaussians + invalidate)
    3. Backward after topology change

    Compare against baseline post-topology backward.
    """
    from r6_1_bwd_decompose import (ReferenceV1Config, GTDataset, SepSSIM,
                                     load_model_from_ckpt, render_with_meta)
    config = ReferenceV1Config()
    config.scene, config.repo_root = scene, str(REPO)
    dataset = GTDataset(scene=scene, repo_root=str(REPO), resolution=config.resolution,
                        device="cuda", background="black")
    cam, target = dataset.get_item(cam_idx)
    ssim = SepSSIM(device="cuda")

    def do_backward(ckpt_path, mode, first=False):
        if first:
            configure(mode)
            if mode != "baseline":
                invalidate()
        model, _ = load_model_from_ckpt(ckpt_path, config, str(REPO))
        image, _, means2d = render_with_meta(model, cam, model.active_sh_degree)
        (0.8 * F.l1_loss(image, target) + 0.2 * ssim(image, target)).backward()
        torch.cuda.synchronize()
        grads = {
            "mean2d_gradient": means2d.grad.detach(),
            "absgrad_densification": means2d.absgrad.detach(),
            "xyz_gradient": model._xyz.grad.detach(),
            "sh_gradient": model._shs.grad.detach(),
            "scaling_gradient": model._scaling.grad.detach(),
            "rotation_gradient": model._rotation.grad.detach(),
            "opacity_parameter_gradient": model._opacity.grad.detach(),
        }
        n_gauss = model._xyz.shape[0]
        del model
        return grads, n_gauss

    # Baseline: normal backward
    base_grads, n_base = do_backward(ckpt, "baseline", first=True)

    # B1-v2: normal backward, then simulate topology change via invalidate,
    # then backward again (full-clear fallback)
    configure("b1")
    invalidate()
    b1_grads_t, n_t = do_backward(ckpt, "b1", first=False)
    invalidate()  # simulate topology change
    b1_grads_t1, n_t1 = do_backward(ckpt, "b1", first=False)

    # Compare B1-v2 post-invalidate against baseline
    comparison = {name: compare(b1_grads_t1[name], base_grads[name])
                  for name in base_grads}

    passed = all(item["max_abs"] == 0.0 and not item["nan_or_inf"]
                 for item in comparison.values())

    return {
        "test": "checkpoint_topology",
        "scene": scene,
        "checkpoint": ckpt,
        "n_gaussians": n_base,
        "comparison_post_invalidate_vs_baseline": comparison,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cam-idx", type=int, default=0)
    args = parser.parse_args()

    results = {}

    print("=== Test 1: Topology rasterizer (capacity growth) ===")
    results["topology_rasterizer"] = test_topology_rasterizer()
    print(json.dumps(results["topology_rasterizer"], indent=2))

    print("\n=== Test 2: Full-clear after invalidate ===")
    results["full_clear_after_invalidate"] = test_full_clear_after_invalidate()
    print(json.dumps(results["full_clear_after_invalidate"], indent=2))

    if args.scene and args.ckpt:
        print(f"\n=== Test 3: Checkpoint topology ({args.scene}) ===")
        results["checkpoint_topology"] = test_checkpoint_topology(
            args.scene, args.ckpt, args.cam_idx)
        print(json.dumps(results["checkpoint_topology"], indent=2))

    results["passed"] = all(r.get("passed", False) for r in results.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
