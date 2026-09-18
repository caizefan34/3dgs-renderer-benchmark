#!/usr/bin/env python3
"""
R4 13-Scene Training Script

Trains a single scene with either baseline (MODE0) or Candidate C (MODE2).
Designed to be launched in parallel across 8 GPUs.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python experiments/r4/r4_train_scene.py \
    --scene room \
    --data-dir data/official/mipnerf360/room \
    --images-dir data/datasets/mipnerf360/room/images \
    --sfm-ply data/official/mipnerf360/room/point_cloud.ply \
    --output /mnt/storage_pool/liaoyuanjun/r4_13scene/room/baseline \
    --mode baseline \
    --iterations 30000 \
    --budget 0.05
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add build dir for ext_skip
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
sys.path.insert(0, BUILD_DIR)

# Add experiments/r4 to path for compute_skip_mask
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_ply(path):
    """Load initial point cloud from PLY file."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    xyz = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=-1)
    return torch.from_numpy(xyz).float().cuda()


def load_cameras(path):
    with open(path) as f:
        return json.load(f)


def get_camera_batch(cameras, indices, images_dir, target_w=1920):
    """Load a batch of cameras + GT images."""
    from PIL import Image
    batch = []
    for idx in indices:
        cam = cameras[idx]
        w, h = cam["width"], cam["height"]
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = w / 2.0, h / 2.0

        position = np.array(cam["position"], dtype=np.float32)
        rot = np.array(cam["rotation"], dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = rot
        viewmat[:3, 3] = position

        # Resize
        th = (int(h * target_w / w) // 16) * 16
        tw = (target_w // 16) * 16
        sx, sy = tw / w, th / h
        fx *= sx; fy *= sy; cx *= sx; cy *= sy

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        viewmat_t = torch.from_numpy(viewmat).cuda().float()
        K_t = torch.from_numpy(K).cuda().float()

        # Load image
        img_path = os.path.join(images_dir, cam["img_name"] + ".JPG")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, cam["img_name"] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, cam["img_name"] + ".png")

        img_pil = Image.open(img_path)
        img_pil = img_pil.resize((tw, th), Image.LANCZOS)
        img = np.array(img_pil, dtype=np.float32) / 255.0
        gt = torch.from_numpy(img).cuda().float()

        batch.append((viewmat_t, K_t, tw, th, gt))
    return batch


class GaussianModel:
    """Minimal 3DGS model for training."""
    def __init__(self, xyz, device="cuda"):
        N = xyz.shape[0]
        self.xyz = xyz.clone().requires_grad_(True)
        self.rotations = torch.zeros(N, 4, device=device)
        self.rotations[:, 0] = 1.0  # identity quaternion
        self.rotations.requires_grad_(True)
        self.scales = torch.log(torch.ones(N, 3, device=device) * 0.01).requires_grad_(True)
        self.opacity = torch.zeros(N, 1, device=device).requires_grad_(True)
        # SH: DC component only (degree 0)
        self.shs = torch.zeros(N, 1, 3, device=device)
        self.shs[:, 0, :] = 0.5  # gray
        self.shs.requires_grad_(True)
        self.active_sh_degree = 0

    def params(self):
        return {
            "xyz": self.xyz,
            "rotations": self.rotations,
            "scales": self.scales,
            "opacity": self.opacity,
            "shs": self.shs,
            "active_sh_degree": self.active_sh_degree,
        }

    def optimizers(self, lr=0.0025):
        import torch.optim as optim
        params = [
            {"params": [self.xyz], "lr": lr, "name": "xyz"},
            {"params": [self.rotations], "lr": lr, "name": "rotation"},
            {"params": [self.scales], "lr": lr, "name": "scaling"},
            {"params": [self.opacity], "lr": lr, "name": "opacity"},
            {"params": [self.shs], "lr": lr, "name": "shs"},
        ]
        return optim.Adam(params, lr=lr, eps=1e-15)


# Global skip mask state
_CURRENT_SKIP_MASK = None
_EXT_SKIP = None


def patched_rasterize_backward(ctx, v_render_colors, v_render_alphas):
    """Replacement for _RasterizeToPixels.backward that uses ext_skip."""
    (
        means2d, conics, colors, opacities,
        backgrounds, masks, isect_offsets, flatten_ids,
        render_alphas, last_ids,
    ) = ctx.saved_tensors

    skip_mask = _CURRENT_SKIP_MASK

    if skip_mask is None:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
                means2d, conics, colors, opacities,
                backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids,
                render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                ctx.absgrad,
            )
    else:
        (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities) = \
            _EXT_SKIP.rasterize_to_pixels_bwd_with_skip(
                means2d, conics, colors, opacities,
                backgrounds, masks,
                ctx.width, ctx.height, ctx.tile_size,
                isect_offsets, flatten_ids,
                render_alphas, last_ids,
                v_render_colors.contiguous(), v_render_alphas.contiguous(),
                skip_mask, ctx.absgrad,
            )

    if ctx.absgrad and v_means2d_abs is not None:
        means2d.absgrad = v_means2d_abs

    v_backgrounds = None
    if ctx.needs_input_grad[4]:
        v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(-3, -2))

    return (
        v_means2d, v_conics, v_colors, v_opacities,
        v_backgrounds, None, None, None, None, None, None, None,
    )


def train_scene(args):
    """Train a single scene."""
    from gsplat import rasterization
    from gsplat.cuda._wrapper import _RasterizeToPixels
    from compute_skip_mask import compute_skip_mask

    global _CURRENT_SKIP_MASK, _EXT_SKIP

    # Load extension if Candidate C mode
    if args.mode == "candidate_c":
        try:
            import ext_skip
            _EXT_SKIP = ext_skip
            print(f"Loaded ext_skip: {ext_skip.__file__}")
        except ImportError as e:
            print(f"FATAL: Cannot load ext_skip for candidate_c mode: {e}")
            return False

    # Load data
    print(f"Loading cameras from {args.cameras_json}...")
    cameras = load_cameras(args.cameras_json)
    print(f"  {len(cameras)} cameras")

    print(f"Loading SfM point cloud from {args.sfm_ply}...")
    xyz = load_ply(args.sfm_ply)
    print(f"  {xyz.shape[0]} initial points")

    model = GaussianModel(xyz)
    optimizer = model.optimizers(lr=0.0025)

    # Training loop
    n_cameras = len(cameras)
    camera_indices = list(range(n_cameras))

    print(f"\n=== Training {args.scene} ({args.mode}) ===")
    print(f"  Iterations: {args.iterations}")
    print(f"  Budget: {args.budget}")
    print(f"  Output: {args.output}")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)

    losses = []
    skip_stats = []
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        # Random camera selection
        np.random.shuffle(camera_indices)
        cam_idx = camera_indices[0]
        batch = get_camera_batch(cameras, [cam_idx], args.images_dir)
        viewmat, K, w, h, gt = batch[0]

        # Forward
        params = model.params()
        _CURRENT_SKIP_MASK = None

        # Restore original backward
        original_backward = _RasterizeToPixels.backward
        if args.mode == "candidate_c":
            _RasterizeToPixels.backward = staticmethod(patched_rasterize_backward)

        try:
            r, a, meta = rasterization(
                means=params["xyz"], quats=params["rotations"],
                scales=params["scales"], opacities=params["opacity"],
                colors=params["shs"],
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=w, height=h, tile_size=16, packed=False,
                sh_degree=params["active_sh_degree"],
                radius_clip=0.0, eps2d=0.1,
                render_mode="RGB", absgrad=True,
            )
            image = r[0].clamp(0, 1)

            # Loss
            L1 = F.l1_loss(image, gt)
            loss = L1  # Simple L1 for now

            # Backward
            if args.mode == "candidate_c":
                # Compute skip mask from forward metadata
                means2d = meta["means2d"].detach()
                conics = meta["conics"].detach()
                opacities = meta["opacities"].detach()
                isect_offsets = meta["isect_offsets"].detach()
                flatten_ids = meta["flatten_ids"].detach()

                # DC color
                shs = params["shs"].detach()
                colors_dc = shs[:, 0, :].unsqueeze(0)

                skip_mask = compute_skip_mask(
                    means2d, conics, colors_dc, opacities,
                    isect_offsets, flatten_ids,
                    16, w, h, budget=args.budget,
                )
                _CURRENT_SKIP_MASK = skip_mask

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        finally:
            _RasterizeToPixels.backward = original_backward
            _CURRENT_SKIP_MASK = None

        losses.append(loss.item())

        if iteration % 1000 == 0:
            avg_loss = np.mean(losses[-1000:])
            elapsed = time.time() - start_time
            print(f"  iter {iteration:5d} | loss={avg_loss:.6f} | "
                  f"N={model.xyz.shape[0]} | {elapsed:.1f}s")

        # Save checkpoint
        if iteration % 5000 == 0 or iteration == args.iterations:
            ckpt_path = os.path.join(args.output, "checkpoints", f"iter_{iteration}.pt")
            torch.save({
                "xyz": model.xyz.detach().cpu(),
                "rotation": model.rotations.detach().cpu(),
                "scaling": model.scales.detach().cpu(),
                "opacity": model.opacity.detach().cpu(),
                "shs": model.shs.detach().cpu(),
                "active_sh_degree": model.active_sh_degree,
                "iteration": iteration,
                "loss": loss.item(),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final metrics
    metrics = {
        "scene": args.scene,
        "mode": args.mode,
        "iterations": args.iterations,
        "budget": args.budget,
        "final_loss": losses[-1],
        "avg_loss_last_1000": float(np.mean(losses[-1000:])),
        "n_gaussians": model.xyz.shape[0],
        "total_time_s": time.time() - start_time,
        "losses_every_100": [float(x) for x in losses[::100]],
    }
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Training complete: {args.scene} ({args.mode}) ===")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Metrics: {metrics_path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cameras-json", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--sfm-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["baseline", "candidate_c"], required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--budget", type=float, default=0.05)
    args = parser.parse_args()

    # Auto-detect paths
    if args.cameras_json is None:
        args.cameras_json = os.path.join(args.data_dir, "cameras.json")
    if args.images_dir is None:
        scene_name = args.scene
        repo_root = Path.cwd()
        args.images_dir = str(repo_root / "data" / "datasets" / "mipnerf360" / scene_name / "images")

    train_scene(args)


if __name__ == "__main__":
    main()
