#!/usr/bin/env python3
"""
ReproTopo0 workload capture (packet §1).

For each cell (room@10K, bicycle@10K):
  - load canonical N1 10K checkpoint (params + Adam moments + step),
  - restore the C0 model + optimizer,
  - run 500 ordinary topology-FROZEN C0 training steps (10001..10500) purely
    to grow realistic densification statistics (xyz_gradient_accum / denom),
  - at nominal 10100,10200,10300,10400,10500 capture the COMPLETE pre-event
    arrays required by the original densify/prune/reset event BEFORE mutating
    topology,
  - hash + freeze each captured event input.

The captured trace is WORKLOAD-ONLY. It is NOT canonical continuation evidence
and is never used for quality claims (packet §1, §9).
"""
import os
import sys
import json
import copy
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rt0_common as C

C0_WT = C.C0_WT
SCENE_CUDA_DIR = C.SCENE_CUDA_DIR
SRC_DIR = C.SRC_DIR
PHASE7_DIR = C.PHASE7_DIR
REF_V1_DIR = C.REF_V1_DIR


def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    from importlib import util as _ilu

    def load_ext_module(name, so_path):
        spec = _ilu.spec_from_file_location(name, so_path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    core = load_ext_module("gsplat_cuda", C.CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C.C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REF_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE  # noqa
    global _DYNAMIC_SCENE
    _DYNAMIC_SCENE = _HIGS_DYNAMIC_SCENE
    return {"c0_so_sha256": C.sha256_file(C.C0_SO),
            "core_so_sha256": C.sha256_file(C.CORE_SO)}


_DYNAMIC_SCENE = None


class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()


def build_camera_sequence(n_cameras, seed, n_iters):
    # Verbatim match to publication_trainer_p12_n1snap.py camera-sequence build:
    # generates one camera index per iteration 1..n_iters, popping from a stack
    # that refills when exhausted. Deterministic given seed.
    viewpoint_stack = list(range(n_cameras))
    rng = random.Random(seed)
    seq = []
    for _ in range(n_iters):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        idx = rng.randint(0, len(viewpoint_stack) - 1)
        seq.append(viewpoint_stack.pop(idx))
    return np.array(seq, dtype=np.int32)


def compute_scene_extent(dataset, n=50):
    centers = []
    for i in range(min(n, len(dataset))):
        centers.append(dataset.get_camera(i).camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            extent = max(extent, float(np.linalg.norm(centers[i] - centers[j])))
    return max(extent, 0.1)


def restore_model(ck_path, config, scene, device="cuda"):
    from gaussian_model import GaussianModel
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                        device=device, background="black")
    # scene_extent from live cameras (matches trainer compute_scene_extent)
    scene_extent = compute_scene_extent(dataset)
    from dataclasses import asdict
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ck, asdict(config))
    # restore optimizer moments (Adam exp_avg / exp_avg_sq per param group)
    od = ck["optimizer_state_dict"]
    model.optimizer.load_state_dict(od)
    # restore() calls training_setup() which zeros accumulators; the 10K
    # snapshot was saved POST-event so its accumulators are already ~zero, but
    # restore them explicitly from the checkpoint for exactness.
    model.xyz_gradient_accum = ck["xyz_gradient_accum"].to(device)
    model.denom = ck["denom"].to(device)
    model.max_radii2D = ck["max_radii2D"].to(device)
    model._gaussian_ids = torch.arange(model._xyz.shape[0], dtype=torch.long, device=device)
    model._next_gaussian_id = model._xyz.shape[0]
    return model, dataset, scene_extent, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--start-iter", type=int, default=10000)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--capture-iters", default="10100,10200,10300,10400,10500")
    ap.add_argument("--scenes", default="room,bicycle")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ident = bootstrap_c0()
    config = C.load_config()
    ssim_fn = SepSSIM(device="cuda")
    capture_iters = [int(x) for x in args.capture_iters.split(",") if x.strip()]
    scenes = args.scenes.split(",")

    manifest = {
        "packet": "REPROTOPO0_PACKET_20260930",
        "captured_at_utc": C.now_utc(),
        "authority": "N1 canonical trajectory + HM0 frozen C0 + DPC0 packet",
        "execution": "EPIC-05 A100 (8x A100-SXM4-80GB); local GPU NOT used",
        "binary_identity": ident,
        "config": config.to_dict(),
        "seed": args.seed,
        "start_iter": args.start_iter,
        "n_steps": args.n_steps,
        "capture_iters": capture_iters,
        "cells": {},
        "note": "WORKLOAD-ONLY trace. NOT canonical continuation evidence.",
    }

    for scene in scenes:
        cell = f"{scene}@10000"
        ck_path = C.CELLS[cell]["ckpt"]
        ck_sha = C.sha256_file(ck_path)
        assert ck_sha == C.AUTHORITATIVE["cells_4"][cell], f"ckpt sha mismatch {cell}"
        print(f"\n=== capture {cell}  ckpt_sha={ck_sha[:16]}... ===", flush=True)

        model, dataset, scene_extent, ck = restore_model(ck_path, config, scene)
        n_cameras = len(dataset)
        cam_seq = build_camera_sequence(n_cameras, args.seed, config.iterations)
        # camera_sequence is 1-indexed over iterations; align to start_iter
        # trainer built seq for iters 1..30000; we need iters start_iter+1..start_iter+n_steps
        global_seed = args.seed
        event_index = 0

        cell_dir = os.path.join(args.outdir, scene)
        os.makedirs(cell_dir, exist_ok=True)
        # freeze the restored 10K state hash
        h0, n0 = C.canonical_state_hash({
            "logical_id": model._gaussian_ids,
            "xyz": model._xyz, "shs": model._shs, "scaling": model._scaling,
            "rotation": model._rotation, "opacity": model._opacity,
        })
        manifest["cells"][cell] = {
            "scene": scene, "ckpt_path": ck_path, "ckpt_sha256": ck_sha,
            "restored_N": int(n0), "restored_state_hash": h0,
            "scene_extent": scene_extent, "n_cameras": int(n_cameras),
            "events": {},
        }

        for step in range(1, args.n_steps + 1):
            iteration = args.start_iter + step
            model.update_learning_rate(iteration)
            if iteration % config.sh_progress_interval == 0:
                model.oneupSHdegree()
            cam_idx = int(cam_seq[(iteration - 1) % len(cam_seq)])
            cam, gt = dataset.get_item(cam_idx)

            image, radii_full, means2d = render_with_meta(model, cam, model.active_sh_degree)
            L1 = F.l1_loss(image, gt)
            dssim = ssim_fn(image, gt)
            loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
            loss.backward()
            assert hasattr(means2d, "absgrad"), "C0 arm: means2d.absgrad missing"

            radii = radii_full[0]
            visibility_filter = (radii > 0).any(dim=-1)
            if iteration < config.densify_until_iter:
                model.max_radii2D[visibility_filter] = torch.max(
                    model.max_radii2D[visibility_filter],
                    radii[visibility_filter].float().max(dim=-1).values)
                model.add_densification_stats(means2d, visibility_filter,
                                              width=cam.image_width, height=cam.image_height)

            do_capture = iteration in capture_iters
            do_event = (iteration > config.densify_from_iter and
                        iteration < config.densify_until_iter and
                        iteration % config.densification_interval == 0)

            if do_capture:
                event_index += 1
                snap = capture_event_input(model, scene_extent, config, iteration,
                                           event_index, global_seed)
                snap_path = os.path.join(cell_dir, f"event_{iteration}.pt")
                torch.save(snap, snap_path)
                snap_sha = C.sha256_file(snap_path)
                manifest["cells"][cell]["events"][str(iteration)] = {
                    "event_index": event_index,
                    "iteration": iteration,
                    "snapshot_path": snap_path,
                    "snapshot_sha256": snap_sha,
                    "N_before": int(snap["N"]),
                    "n_clone_eligible": int(snap["_diag_clone_eligible"]),
                    "n_split_eligible": int(snap["_diag_split_eligible"]),
                }
                print(f"  [capture] iter={iteration} event_index={event_index} "
                      f"N={snap['N']:,} clone_el={snap['_diag_clone_eligible']} "
                      f"split_el={snap['_diag_split_eligible']} sha={snap_sha[:16]}", flush=True)

            if do_event:
                current_radii = radii.float().max(dim=-1).values
                size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
                model.densify_and_prune(
                    max_grad=config.densify_grad_threshold,
                    min_opacity=config.min_opacity,
                    extent=scene_extent,
                    max_screen_size=size_threshold,
                    radii=current_radii)
                # C0 HIGS dynamic-scene invariant: mark topology dirty before
                # next forward (matches publication_trainer_p12_n1snap.py).
                if _DYNAMIC_SCENE is not None:
                    _DYNAMIC_SCENE.mark_dirty()
            if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
                model.reset_opacity()
                if _DYNAMIC_SCENE is not None:
                    _DYNAMIC_SCENE.mark_dirty()

            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            if iteration % 50 == 0:
                print(f"  step {iteration} N={model._xyz.shape[0]:,} loss={float(loss):.4f}",
                      flush=True)

    man_path = os.path.join(args.outdir, "REPROTOPO0_CAPTURE_MANIFEST.json")
    C.emit_json(man_path, manifest)
    print(f"\nWROTE {man_path}", flush=True)


def render_with_meta(model, cam, sh_degree):
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity,
        model.get_features, sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    info = out["densification_info"]
    means2d = info["means2d"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], means2d


def capture_event_input(model, scene_extent, config, iteration, event_index, global_seed):
    """Capture the COMPLETE pre-event arrays required by the original event."""
    with torch.no_grad():
        xyz = model._xyz.detach().cpu()
        shs = model._shs.detach().cpu()
        scaling = model._scaling.detach().cpu()
        rotation = model._rotation.detach().cpu()
        opacity = model._opacity.detach().cpu()
        max_radii2D = model.max_radii2D.detach().cpu()
        xyz_gradient_accum = model.xyz_gradient_accum.detach().cpu()
        denom = model.denom.detach().cpu()
        gaussian_ids = model._gaussian_ids.detach().cpu().long()
        next_gaussian_id = int(model._next_gaussian_id)
        active_sh_degree = int(model.active_sh_degree)

        # optimizer moments (per param group, by name)
        opt = {}
        for pg in model.optimizer.param_groups:
            name = pg["name"]
            p = pg["params"][0]
            st = model.optimizer.state.get(p, None)
            if st is not None:
                opt[name] = {
                    "exp_avg": st["exp_avg"].detach().cpu().clone(),
                    "exp_avg_sq": st["exp_avg_sq"].detach().cpu().clone(),
                    "lr": float(pg["lr"]),
                }
            else:
                opt[name] = {"exp_avg": None, "exp_avg_sq": None, "lr": float(pg["lr"])}

        N = int(xyz.shape[0])
        grads = (xyz_gradient_accum / denom).squeeze(-1)
        grads = torch.nan_to_num(grads, nan=0.0)
        max_scale = torch.exp(scaling).max(dim=1).values
        clone_el = int(((grads >= config.densify_grad_threshold) &
                        (max_scale <= config.percent_dense * scene_extent)).sum())
        split_el = int(((grads >= config.densify_grad_threshold) &
                        (max_scale > config.percent_dense * scene_extent)).sum())

    snap = {
        "scene": config.scene, "iteration": iteration, "event_index": event_index,
        "global_seed": int(global_seed),
        "N": N, "scene_extent": float(scene_extent),
        "active_sh_degree": active_sh_degree, "next_gaussian_id": next_gaussian_id,
        "xyz": xyz, "shs": shs, "scaling": scaling, "rotation": rotation,
        "opacity": opacity, "max_radii2D": max_radii2D,
        "xyz_gradient_accum": xyz_gradient_accum, "denom": denom,
        "gaussian_ids": gaussian_ids,
        "optimizer": opt,
        "config": {
            "densify_grad_threshold": float(config.densify_grad_threshold),
            "min_opacity": float(config.min_opacity),
            "percent_dense": float(config.percent_dense),
            "max_screen_size": int(config.max_screen_size),
            "densification_interval": int(config.densification_interval),
            "opacity_reset_interval": int(config.opacity_reset_interval),
        },
        "_diag_clone_eligible": clone_el, "_diag_split_eligible": split_el,
    }
    return snap


if __name__ == "__main__":
    main();
