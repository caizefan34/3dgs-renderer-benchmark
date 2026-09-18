#!/usr/bin/env python3
"""
R4 Training Wrapper: Run existing trainer with Candidate C skip.

Wraps baseline/reference_v1/trainer.py run_training() with a monkey-patch
on _RasterizeToPixels.backward that adds skip_mask support.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python experiments/r4/r4_train_wrapper.py \
    --scene room --mode candidate_c --budget 0.05 --iterations 30000 \
    --output /mnt/storage_pool/liaoyuanjun/r4_13scene/room/candidate_c
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse, math, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "r4"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "r4" / "build"))

# Global state
_CURRENT_SKIP_MASK = None
_EXT_SKIP = None
_SKIP_MODE = "baseline"
_BUDGET = 0.05


def patched_rasterize_backward(ctx, v_render_colors, v_render_alphas):
    (means2d, conics, colors, opacities, backgrounds, masks,
     isect_offsets, flatten_ids, render_alphas, last_ids) = ctx.saved_tensors
    skip_mask = _CURRENT_SKIP_MASK
    if skip_mask is None:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
                means2d, conics, colors, opacities, backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids, render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(), ctx.absgrad)
    else:
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _EXT_SKIP.rasterize_to_pixels_bwd_with_skip(
                means2d, conics, colors, opacities, backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids, render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                skip_mask, ctx.absgrad)
    if ctx.absgrad and v_means2d_abs is not None:
        means2d.absgrad = v_means2d_abs
    v_backgrounds = None
    if ctx.needs_input_grad[4]:
        v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(-3, -2))
    return (v_means2d, v_conics, v_colors, v_opacities,
            v_backgrounds, None, None, None, None, None, None, None)


def main():
    global _SKIP_MODE, _BUDGET, _EXT_SKIP, _CURRENT_SKIP_MASK

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", choices=["baseline", "candidate_c"], required=True)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _SKIP_MODE = args.mode
    _BUDGET = args.budget

    os.makedirs(args.output, exist_ok=True)

    print(f"=== R4 Training: {args.scene} ({args.mode}) ===")
    print(f"  Iterations: {args.iterations}")
    print(f"  Budget: {args.budget}")
    print(f"  Output: {args.output}")

    # Setup skip patch
    if args.mode == "candidate_c":
        try:
            import ext_skip
            _EXT_SKIP = ext_skip
            print(f"  [R4] Loaded ext_skip: {ext_skip.__file__}")
        except ImportError as e:
            print(f"  [R4] FATAL: Cannot load ext_skip: {e}")
            sys.exit(1)

        from gsplat.cuda._wrapper import _RasterizeToPixels
        _RasterizeToPixels.backward = staticmethod(patched_rasterize_backward)
        print(f"  [R4] Patched _RasterizeToPixels.backward with skip support")

    # Create config
    from config import ReferenceV1Config
    config = ReferenceV1Config()
    config.scene = args.scene
    config.iterations = args.iterations
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed

    # If candidate_c, we need to hook into the training loop to compute
    # skip_mask before each backward. The trainer calls loss.backward()
    # internally. We'll use a pre-backward hook via torch.autograd.
    #
    # Strategy: Wrap the rasterization forward to capture metadata,
    # then use a tensor hook on the loss to set skip_mask before backward.

    if args.mode == "candidate_c":
        # Patch render_with_meta to capture metadata and set skip_mask
        import trainer as trainer_module
        original_render = trainer_module.render_with_meta

        def patched_render(model, cam, sh_degree):
            global _CURRENT_SKIP_MASK
            image, meta, means2d = original_render(model, cam, sh_degree)
            # Compute skip mask from metadata
            from compute_skip_mask import compute_skip_mask
            try:
                shs = model.get_features.detach()
                if shs.dim() == 3:
                    colors_dc = shs[:, 0, :].unsqueeze(0)
                elif shs.dim() == 4:
                    colors_dc = shs[0, :, 0, :].unsqueeze(0)
                else:
                    colors_dc = torch.ones(1, shs.shape[0], 3, device="cuda")

                _CURRENT_SKIP_MASK = compute_skip_mask(
                    meta["means2d"].detach(), meta["conics"].detach(),
                    colors_dc, meta["opacities"].detach(),
                    meta["isect_offsets"].detach(), meta["flatten_ids"].detach(),
                    16, cam.image_width, cam.image_height,
                    budget=_BUDGET,
                )
            except Exception as e:
                print(f"  [R4] skip_mask failed: {e}")
                _CURRENT_SKIP_MASK = None
            return image, meta, means2d

        trainer_module.render_with_meta = patched_render
        print(f"  [R4] Patched render_with_meta to compute skip_mask")

    # Run training
    from trainer import run_training
    run_training(config, args.output)

    # Save mode info
    info = {"scene": args.scene, "mode": args.mode, "budget": args.budget,
            "iterations": args.iterations}
    with open(os.path.join(args.output, "r4_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n=== Training Complete: {args.scene} ({args.mode}) ===")


if __name__ == "__main__":
    main()
