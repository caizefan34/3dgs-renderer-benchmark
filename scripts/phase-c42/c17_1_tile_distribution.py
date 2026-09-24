#!/usr/bin/env python3
"""Tile intersection distribution profiler for C17-1 feasibility analysis."""
import sys, math, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import fully_fused_projection, isect_tiles
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"

def profile_scene(scene, repo_root):
    print(f"\n{'='*60}")
    print(f"Tile Intersection Distribution: {scene}")
    print(f"{'='*60}")

    dataset = GTDataset(scene=scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    n_cams = len(dataset)
    cam_indices = list(range(0, n_cams, max(1, n_cams // 6)))[:6]

    # Load model
    ckpt_path = repo_root / "results" / "a100" / "phase-c42" / "p2_checkpoints"
    ckpt_files = list(ckpt_path.glob("*iter10000*.pt")) if ckpt_path.exists() else []
    if ckpt_files and scene == "room":
        ckpt = torch.load(ckpt_files[0], map_location=DEVICE, weights_only=False)
        model = GaussianModel.from_checkpoint_state(ckpt, device=DEVICE)
        model.set_sh_degree(3)
    else:
        sfm_data = load_initial_checkpoint(scene, repo_root, device=DEVICE)
        model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
        model.init_from_sfm(xyz=sfm_data["xyz"],
            opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0],1),0.1,device=DEVICE)),
            scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
        model.set_sh_degree(3)
    print(f"  Gaussians: {model.xyz.shape[0]:,}")

    all_counts = []
    for tile_size in [16, 32]:
        print(f"\n  --- tile_size={tile_size} ---")
        scene_counts = []
        for ci in cam_indices:
            cam = dataset.get_camera(ci)
            data = model.forward()
            radii, means2d, depths, _, _ = fully_fused_projection(
                means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                radius_clip=0.0, packed=False, eps2d=0.1)
            W, H = cam.image_width, cam.image_height
            tw = (W + tile_size - 1) // tile_size
            th = (H + tile_size - 1) // tile_size
            tpg, isect_ids, flatten_ids = isect_tiles(
                means2d, radii, depths, tile_size, tw, th,
                sort=False, packed=False)
            n_isects = len(flatten_ids)
            n_tiles = tw * th

            # Extract tile_id from upper 32 bits
            # isect_ids layout: [image_id | tile_id | depth(32b)]
            # upper 32 bits = image_id | tile_id
            upper = (isect_ids >> 32).to(torch.int64)
            tile_n_bits = int(math.floor(math.log2(n_tiles))) + 1
            tile_ids = upper & ((1 << tile_n_bits) - 1)

            # Count per-tile
            counts = torch.bincount(tile_ids, minlength=n_tiles)
            nonzero = counts[counts > 0]
            scene_counts.append(nonzero.cpu().numpy())

            if ci == cam_indices[0]:
                print(f"    Camera {ci}: {W}x{H} tiles={tw}x{th}={n_tiles} n_isects={n_isects:,}")
                print(f"    Non-zero tiles: {len(nonzero)}/{n_tiles} ({100*len(nonzero)/n_tiles:.1f}%)")
                print(f"    Per-tile (non-zero): mean={nonzero.float().mean():.0f} "
                      f"p50={nonzero.float().median():.0f} "
                      f"p90={torch.quantile(nonzero.float(), 0.9):.0f} "
                      f"p95={torch.quantile(nonzero.float(), 0.95):.0f} "
                      f"p99={torch.quantile(nonzero.float(), 0.99):.0f} "
                      f"max={nonzero.max()}")

        # Aggregate across cameras
        all_c = np.concatenate(scene_counts)
        print(f"\n  Aggregate ({len(cam_indices)} cameras, tile_size={tile_size}):")
        print(f"    Total non-zero tiles: {len(all_c)}")
        print(f"    mean={np.mean(all_c):.0f}")
        print(f"    p50={np.percentile(all_c, 50):.0f}")
        print(f"    p75={np.percentile(all_c, 75):.0f}")
        print(f"    p90={np.percentile(all_c, 90):.0f}")
        print(f"    p95={np.percentile(all_c, 95):.0f}")
        print(f"    p99={np.percentile(all_c, 99):.0f}")
        print(f"    p99.9={np.percentile(all_c, 99.9):.0f}")
        print(f"    max={np.max(all_c)}")
        print(f"    entries >6144 (48KB/8B limit): {np.sum(all_c > 6144)} ({100*np.mean(all_c > 6144):.2f}%)")
        print(f"    entries >12288 (96KB/8B limit): {np.sum(all_c > 12288)} ({100*np.mean(all_c > 12288):.2f}%)")

        if tile_size == 16:
            all_counts = all_c

    return all_counts

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results = {}
    for scene in ["room", "bicycle", "garden"]:
        counts = profile_scene(scene, repo_root)
        results[scene] = {
            "mean": float(np.mean(counts)),
            "p50": float(np.percentile(counts, 50)),
            "p75": float(np.percentile(counts, 75)),
            "p90": float(np.percentile(counts, 90)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "p999": float(np.percentile(counts, 99.9)),
            "max": int(np.max(counts)),
            "pct_over_6144": float(np.mean(counts > 6144) * 100),
            "pct_over_12288": float(np.mean(counts > 12288) * 100),
        }

    import json
    save_path = repo_root / "results" / "a100" / "phase-c42" / "c17_1_tile_distribution.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()
