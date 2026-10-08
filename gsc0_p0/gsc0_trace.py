#!/usr/bin/env python3
"""GSC0 P0 — Gaussian State Compression trace harness (EPIC-05 only).

Semantics-preserving: imports the FROZEN reference_v1 recipe (GaussianModel,
ReferenceV1Config, GTDataset, render_with_meta, SepSSIM, compute_scene_extent)
and reproduces the canonical 30K loop VERBATIM, adding ONLY:
  - harness-side per-Gaussian lineage accumulators (visibility count, gradient
    norm sum) keyed by the frozen model._gaussian_ids (read-only bookkeeping;
    does NOT touch model state or training math)
  - read-only full-state snapshots at the 3 preregistered phases

No code modification to the frozen recipe. No training with reduced precision.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/root/3dgs-renderer-benchmark")
# ONLY gsplat tree whose rasterization() accepts the `accutile` kwarg that
# render_with_meta passes. gsplat_pkg is a private, patched copy of the
# true-accutile tree whose cuda/_backend.py builds cleanly under torch 2.9.1.
# Used with accutile=False => reference_v1 default rendering.
GSPLAT_TREE = "/root/gsc0_p0/gsplat_pkg"
REF_V1 = str(REPO_ROOT / "baseline" / "reference_v1")
PHASE7 = str(REPO_ROOT / "scripts" / "epic05" / "phase7")
SRC = str(REPO_ROOT / "src")

# --- gsplat JIT shim (torch 2.9.1 + true-accutile tree) ----------------------
# 1) Reuse the prebuilt CUDA extension dir (gsplat_cuda.so already compiled;
#    avoids a multi-minute JIT rebuild on every worker).
os.environ["TORCH_EXTENSIONS_DIR"] = "/root/gsc0_ext2"
# 2) torch 2.9.1's cpp_extension versioner crashes on the unhashable
#    extra_cuda_cflags list; wrap update_hash so unhashable values are repr'd.
import torch.utils._cpp_extension_versioner as _V  # noqa: E402

def _gsc0_hashable(v):
    try:
        hash(v)
        return True
    except TypeError:
        return False

_gsc0_orig_update_hash = _V.update_hash
_V.update_hash = lambda s, v: _gsc0_orig_update_hash(s, v) if _gsc0_hashable(v) else _gsc0_orig_update_hash(s, repr(v))

# sys.path order mirrors frozen trainer.py (reference_v1 first for GaussianModel)
sys.path.insert(0, GSPLAT_TREE)   # must precede `from gsplat import rasterization`
sys.path.insert(0, PHASE7)        # GTDataset
sys.path.insert(0, SRC)
sys.path.insert(0, REF_V1)

from gaussian_model import GaussianModel          # noqa: E402
from config import ReferenceV1Config              # noqa: E402
from colmap_reader import read_points3D_binary, sfm_to_pcd_data  # noqa: E402
from dataset import GTDataset                     # noqa: E402
import trainer as frozen_trainer                  # noqa: E402  (render_with_meta, SepSSIM, compute_scene_extent)

render_with_meta = frozen_trainer.render_with_meta
SepSSIM = frozen_trainer.SepSSIM
compute_scene_extent = frozen_trainer.compute_scene_extent

EPS = 1e-12


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ensure_cap(buf, need, device):
    if buf.numel() >= need:
        return buf
    new = torch.zeros(max(need, buf.numel() * 2), device=device, dtype=torch.float32)
    new[: buf.numel()] = buf
    return new


@torch.no_grad()
def snapshot(model, ids, vis_count, grad_sum, phase, iteration, out_dir, scene):
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    adam = {}
    for grp in model.optimizer.param_groups:
        name = grp["name"]
        p = grp["params"][0]
        st = model.optimizer.state.get(p, {})
        adam[name] = {
            "exp_avg": st.get("exp_avg", torch.zeros_like(p)).detach().cpu(),
            "exp_avg_sq": st.get("exp_avg_sq", torch.zeros_like(p)).detach().cpu(),
        }
    snap = {
        "scene": scene,
        "phase": phase,
        "iteration": iteration,
        "N": int(model._xyz.shape[0]),
        "active_sh_degree": int(model.active_sh_degree),
        "spatial_lr_scale": float(model.spatial_lr_scale),
        "ids": ids.detach().cpu(),
        "xyz": model._xyz.detach().cpu(),
        "scaling": model._scaling.detach().cpu(),
        "rotation": model._rotation.detach().cpu(),
        "opacity": model._opacity.detach().cpu(),
        "shs": model._shs.detach().cpu(),
        "grad_sum": grad_sum[ids].detach().cpu(),
        "vis_count": vis_count[ids].detach().cpu(),
        "xyz_gradient_accum": model.xyz_gradient_accum.detach().cpu(),
        "denom": model.denom.detach().cpu(),
        "max_radii2D": model.max_radii2D.detach().cpu(),
        "adam": adam,
    }
    path = os.path.join(out_dir, "traces", f"{scene}_{phase}_iter{iteration}.pt")
    torch.save(snap, path)
    rec = {
        "scene": scene, "phase": phase, "iteration": iteration,
        "N": snap["N"], "active_sh_degree": snap["active_sh_degree"],
        "file": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
    }
    print(f"[gsc0-trace] {scene} {phase} iter={iteration} N={snap['N']} "
          f"shdeg={snap['active_sh_degree']} -> {rec['file']} "
          f"({rec['bytes']/1e6:.1f} MB)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", default=30000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--accutile", default="off", choices=("off", "on"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    scene = args.scene
    accutile = (args.accutile == "on")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Preregistered phases (named). Map iteration -> phase name.
    phase_map = {1000: "early", 15000: "densify_complete", 30000: "mature"}
    phase_map = {k: v for k, v in phase_map.items() if k <= args.iterations}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = ReferenceV1Config(scene=scene, repo_root=str(REPO_ROOT),
                               iterations=args.iterations, seed=args.seed)

    print(f"[gsc0-trace] scene={scene} gpu={args.gpu} iters={args.iterations} "
          f"phases={phase_map}", flush=True)

    # === Dataset (verbatim) ===
    dataset = GTDataset(scene=scene, repo_root=str(REPO_ROOT),
                        resolution=config.resolution, device="cuda",
                        background="black")
    n_cameras = len(dataset)
    scene_extent = compute_scene_extent(dataset)
    print(f"[gsc0-trace] cameras={n_cameras} extent={scene_extent:.4f}", flush=True)

    # === Model init (verbatim) ===
    sfm_path = os.path.join(str(REPO_ROOT), "data", "datasets", "mipnerf360",
                            scene, "sparse", "0", "points3D.bin")
    points3d = read_points3D_binary(sfm_path)
    pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = model._xyz.shape[0]
    model.training_setup({
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    print(f"[gsc0-trace] initial_N={initial_N}", flush=True)

    # === Camera sequence (verbatim) ===
    viewpoint_stack = list(range(n_cameras))
    camera_sequence = []
    rng = random.Random(config.seed)
    for iteration in range(1, config.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        camera_sequence.append(viewpoint_stack.pop(rand_idx))
    camera_sequence = np.array(camera_sequence, dtype=np.int32)

    ssim_fn = SepSSIM(device="cuda")

    # === Harness-side per-id lineage accumulators (read-only bookkeeping) ===
    dev = "cuda"
    max_id = int(model._next_gaussian_id)
    vis_count = torch.zeros(max_id + 1, device=dev, dtype=torch.float32)
    grad_sum = torch.zeros(max_id + 1, device=dev, dtype=torch.float32)

    trace_records = []
    topology_events = []
    total_clones = total_splits = total_prunes = 0
    t0 = time.time()

    print(f"[gsc0-trace] starting training loop", flush=True)
    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)
        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        cam_idx = camera_sequence[iteration - 1]
        cam, gt_image = dataset.get_item(cam_idx)

        image, meta, means2d = render_with_meta(
            model, cam, model.active_sh_degree, accutile=accutile)
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss.backward()

        radii = meta["radii"][0]
        visibility_filter = (radii > 0).any(dim=-1)

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(
                means2d, visibility_filter,
                width=cam.image_width, height=cam.image_height)

        # --- harness lineage accumulators (read-only; no effect on training) ---
        ids_now = model._gaussian_ids
        need = int(ids_now.max().item()) + 1 if ids_now.numel() else 1
        if need > vis_count.numel():
            vis_count = ensure_cap(vis_count, need, dev)
            grad_sum = ensure_cap(grad_sum, need, dev)
        vis_count[ids_now[visibility_filter]] += 1
        g = getattr(means2d, "absgrad", None)
        if g is None:
            g = means2d.grad
        if g is not None:
            gg = g[0][visibility_filter, :2].clone()
            gg[:, 0] *= cam.image_width / 2.0
            gg[:, 1] *= cam.image_height / 2.0
            grad_sum[ids_now[visibility_filter]] += torch.norm(gg, dim=-1)

        # === Topology events (verbatim) ===
        if (iteration > config.densify_from_iter and
                iteration < config.densify_until_iter and
                iteration % config.densification_interval == 0):
            N_before = model._xyz.shape[0]
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            event_result = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii)
            N_after = model._xyz.shape[0]
            topology_events.append({
                "iteration": iteration, "N_before": N_before,
                "N_after": N_after, **event_result})
            total_clones += event_result["cloned"]
            total_splits += event_result["split"]
            total_prunes += event_result["pruned_total"]

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        # --- phase snapshot (read-only) ---
        if iteration in phase_map:
            rec = snapshot(model, model._gaussian_ids, vis_count, grad_sum,
                           phase_map[iteration], iteration, out_dir, scene)
            trace_records.append(rec)

        if iteration % 500 == 0:
            print(f"[gsc0-trace] {scene} iter={iteration} N={model._xyz.shape[0]} "
                  f"loss={loss.item():.5f} elapsed={time.time()-t0:.0f}s", flush=True)

    # === trace index ===
    index = {
        "scene": scene,
        "iterations": args.iterations,
        "seed": args.seed,
        "accutile": accutile,
        "initial_N": int(initial_N),
        "final_N": int(model._xyz.shape[0]),
        "total_clones": total_clones,
        "total_splits": total_splits,
        "total_prunes": total_prunes,
        "n_cameras": n_cameras,
        "scene_extent": float(scene_extent),
        "phases": phase_map,
        "trace_records": trace_records,
        "topology_event_count": len(topology_events),
        "elapsed_s": time.time() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "frozen_recipe": "baseline/reference_v1 (git abfa6c3e)",
        "gsplat_tree": GSPLAT_TREE,
    }
    with open(os.path.join(out_dir, f"trace_index_{scene}.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[gsc0-trace] DONE scene={scene} final_N={index['final_N']} "
          f"elapsed={index['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()