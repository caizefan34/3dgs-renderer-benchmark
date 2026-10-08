#!/usr/bin/env python3
"""P11 stategen: faithful DSH-A gmain-arm replay to iter 14000, saving
capture()+optimizer states at 2000/5000/10000/14000 for Gate D1.

This is STATE GENERATION for measurement only (Gate D1 needs pre-15K states):
- NOT a 30K campaign (stops at 14K);
- no algorithm change: the loop replicates publication_trainer_dsha.py
  (arm gmain) verbatim, minus LPIPS/eval (they run under no_grad and consume
  no RNG, so the trajectory is unaffected);
- intermediate states are frozen artifacts once written.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=14000)
    ap.add_argument("--save-iters", type=str, default="2000,5000,10000,14000")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    save_iters = set(int(x) for x in args.save_iters.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    ident = C.bootstrap_gmain()
    C.import_recipe()
    from config import ReferenceV1Config

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = C.REPO
    config.iterations = args.iterations

    model, dataset, scene_extent, meta = C.load_model_and_dataset(args.scene)
    n_cameras = meta["n_cameras"]
    ssim_fn = C.SepSSIM(device="cuda")

    # frozen camera sequence: identical rng/seed to the campaign trainer
    viewpoint_stack = list(range(n_cameras))
    camera_sequence = []
    rng = random.Random(42)
    for iteration in range(1, config.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        camera_sequence.append(viewpoint_stack.pop(rand_idx))
    camera_sequence = np.array(camera_sequence, dtype=np.int32)
    np.save(os.path.join(args.outdir, "camera_sequence.npy"), camera_sequence)

    print(f"[stategen {args.scene}] cameras={n_cameras} extent={scene_extent:.4f} "
          f"initial_N={meta['initial_N']} save_iters={sorted(save_iters)}", flush=True)

    t0 = time.perf_counter()
    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)

        # ---- trainer loop (verbatim order) ----
        r0, image, radii_full, means2d = C.render_with_meta(
            model, cam, model.active_sh_degree)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss.backward()

        radii = radii_full[0]
        visibility_filter = (radii > 0).any(dim=-1)

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width,
                                          height=cam.image_height)

        if (iteration > config.densify_from_iter and
                iteration < config.densify_until_iter and
                iteration % config.densification_interval == 0):
            size_threshold = (config.max_screen_size
                              if iteration > config.opacity_reset_interval else None)
            current_radii = radii.float().max(dim=-1).values
            model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii)

        if (iteration < config.densify_until_iter and
                iteration % config.opacity_reset_interval == 0):
            model.reset_opacity()

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if iteration in save_iters:
            state = model.capture()
            state["optimizer_state_dict"] = model.optimizer.state_dict()
            state["iteration"] = iteration
            state["scene"] = args.scene
            state["arm"] = "gmain-stategen"
            state["camera_index_at_next_iter"] = (
                int(camera_sequence[iteration]) if iteration < len(camera_sequence)
                else None)
            state["n_cameras"] = n_cameras
            path = os.path.join(args.outdir, f"state_{iteration}.pt")
            torch.save(state, path)
            print(f"[stategen {args.scene}] saved {path} N={state['num_points']:,}",
                  flush=True)

        if iteration % 500 == 0:
            print(f"[stategen {args.scene}] iter {iteration} "
                  f"loss={float(loss.item()):.4f} GS={model._xyz.shape[0]:,} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    summary = {
        "scene": args.scene, "iterations": args.iterations,
        "final_N": int(model._xyz.shape[0]),
        "wall_s": time.perf_counter() - t0,
        "binary_identity": ident,
        "recipe": "DSH-A gmain replay (verbatim trainer loop, no eval/lpips)",
        "save_iters": sorted(save_iters),
    }
    with open(os.path.join(args.outdir, "stategen_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stategen {args.scene}] DONE final_N={summary['final_N']:,}")


if __name__ == "__main__":
    main()
