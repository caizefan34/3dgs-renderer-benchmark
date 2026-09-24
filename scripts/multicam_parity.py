#!/usr/bin/env python3
"""Multi-camera forward parity test: B1 vs B1A across multiple cameras for room/bicycle/garden.

Uses the true-accutile-v153 checkout for both (accutile=False for B1, accutile=True for B1A).
Renders 5 cameras per scene and compares RGB/alpha output.
"""
import sys, os, json, math
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"
sys.path.insert(0, f"{REPO_ROOT}/src")
sys.path.insert(0, REPO_ROOT)

GSPLAT_PATH = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, GSPLAT_PATH)

CKPT_BASE = f"{REPO_ROOT}/results/epic05/phase7"
CAM_BASE = f"{REPO_ROOT}/data/official/mipnerf360"

SCENES = {
    "room": {
        "ckpt": f"{CKPT_BASE}/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/room/cameras.json",
    },
    "bicycle": {
        "ckpt": f"{CKPT_BASE}/a100_30k_bicycle_t16_16/a100_30k_bicycle_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/bicycle/cameras.json",
    },
    "garden": {
        "ckpt": f"{CKPT_BASE}/a100_30k_garden_t16_16/a100_30k_garden_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/garden/cameras.json",
    },
}

NUM_CAMERAS = 5

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("model_state", ckpt)

def make_viewmat_K(cam, device="cuda"):
    rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device)
    position = torch.tensor(cam["position"], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[cam["fx"], 0., cam["width"] / 2],
                      [0., cam["fy"], cam["height"] / 2],
                      [0., 0., 1.]], dtype=torch.float32, device=device)[None]
    return viewmat[None], K, cam["width"], cam["height"]

def prep_tensors(state, device="cuda"):
    means = state["xyz"].to(device).contiguous()
    quats = state["rotations"].to(device).contiguous()
    scales = torch.exp(state["scales"].to(device)).contiguous()
    opacities = torch.sigmoid(state["opacity"].to(device).flatten()).contiguous()
    colors = state["shs"].to(device).contiguous()
    return means, quats, scales, opacities, colors, int(state.get("sh_degree", 3))

def render(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree, accutile):
    import gsplat
    from gsplat import rasterization
    kwargs = dict(sh_degree=sh_degree, absgrad=False, tile_size=16, packed=False, render_mode="RGB")
    if accutile:
        kwargs["accutile"] = True
    rgb, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        width, height, **kwargs)
    return rgb, alpha, meta

def main():
    device = "cuda"
    results = {}

    for scene, paths in SCENES.items():
        print(f"\n{'='*60}")
        print(f"Scene: {scene}")
        print(f"{'='*60}")

        state = load_checkpoint(paths["ckpt"])
        with open(paths["cams"]) as f:
            cams = json.load(f)

        means, quats, scales, opacities, colors, sh_degree = prep_tensors(state, device)
        n_gauss = means.shape[0]
        print(f"  Gaussians: {n_gauss}, SH: {sh_degree}")

        # Test multiple cameras spread across the sequence
        n_cams = len(cams)
        cam_indices = [0, n_cams // 4, n_cams // 2, 3 * n_cams // 4, n_cams - 1]
        cam_indices = cam_indices[:NUM_CAMERAS]

        scene_results = []
        for ci in cam_indices:
            cam = cams[ci]
            viewmats, Ks, width, height = make_viewmat_K(cam, device)

            # B1 (accutile=False)
            rgb_b1, alpha_b1, meta_b1 = render(
                means, quats, scales, opacities, colors,
                viewmats, Ks, width, height, sh_degree, accutile=False)

            # B1A (accutile=True)
            rgb_b1a, alpha_b1a, meta_b1a = render(
                means, quats, scales, opacities, colors,
                viewmats, Ks, width, height, sh_degree, accutile=True)

            rgb_diff = (rgb_b1 - rgb_b1a).abs()
            alpha_diff = (alpha_b1 - alpha_b1a).abs()
            isect_b1 = int(meta_b1["tiles_per_gauss"].sum())
            isect_b1a = int(meta_b1a["tiles_per_gauss"].sum())

            entry = {
                "cam_idx": ci,
                "resolution": [width, height],
                "rgb_max_abs": float(rgb_diff.max()),
                "rgb_mean_abs": float(rgb_diff.mean()),
                "alpha_max_abs": float(alpha_diff.max()),
                "alpha_mean_abs": float(alpha_diff.mean()),
                "isect_B1": isect_b1,
                "isect_B1A": isect_b1a,
                "isect_reduction": 1 - isect_b1a / isect_b1 if isect_b1 > 0 else 0,
            }
            scene_results.append(entry)
            print(f"  cam {ci} ({width}x{height}): rgb_max={entry['rgb_max_abs']:.1e} alpha_max={entry['alpha_max_abs']:.1e} "
                  f"isect B1={isect_b1:,} B1A={isect_b1a:,} ({entry['isect_reduction']:.4f})")

        results[scene] = {
            "n_gaussians": n_gauss,
            "cameras": scene_results,
            "all_rgb_bit_identical": all(r["rgb_max_abs"] == 0.0 for r in scene_results),
            "all_alpha_bit_identical": all(r["alpha_max_abs"] == 0.0 for r in scene_results),
        }

    # Summary
    print(f"\n{'='*70}")
    print("MULTI-CAMERA FORWARD PARITY SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for scene, sr in results.items():
        bit_id = sr["all_rgb_bit_identical"] and sr["all_alpha_bit_identical"]
        avg_red = np.mean([r["isect_reduction"] for r in sr["cameras"]])
        print(f"\n{scene} ({sr['n_gaussians']:,} Gaussians, {len(sr['cameras'])} cameras):")
        print(f"  RGB bit-identical: {'YES' if sr['all_rgb_bit_identical'] else 'NO'}")
        print(f"  alpha bit-identical: {'YES' if sr['all_alpha_bit_identical'] else 'NO'}")
        print(f"  avg intersection reduction: {avg_red:.4f}")
        if not bit_id:
            all_pass = False

    print(f"\nOVERALL VERDICT: {'PASS — all scenes bit-identical forward' if all_pass else 'FAIL'}")

    out_path = "/tmp/accutile_a100_results/multicam_forward_parity.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
