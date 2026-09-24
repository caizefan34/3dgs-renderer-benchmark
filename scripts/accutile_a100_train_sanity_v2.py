#!/usr/bin/env python3
"""AccuTile A100 short training semantic test (v2 - fixed viewmat shape).

Compares B1 (baseline) vs B1A (AccuTile) short training runs with:
- same seed
- same camera sequence
- same initialization
- same optimizer
- same densification

Checks: loss, Gaussian count, intersection count over time.
"""
import sys, os, json, argparse, time, math
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153")
import torch
import torch.nn.functional as F
import numpy as np
from gsplat import rasterization

CAM_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
SCENES = {"room": f"{CAM_BASE}/room/cameras.json"}


def load_cameras(scene, device="cuda"):
    path = SCENES[scene]
    with open(path) as f:
        cams = json.load(f)
    cam_list = []
    K_list = []
    for cam in cams:
        rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device)
        position = torch.tensor(cam["position"], dtype=torch.float32, device=device)
        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = -rotation @ position
        K = torch.tensor([[cam["fx"], 0., cam["width"] / 2],
                          [0., cam["fy"], cam["height"] / 2],
                          [0., 0., 1.]], dtype=torch.float32, device=device)
        cam_list.append(viewmat)
        K_list.append(K)
    return cam_list, K_list, cams[0]["width"], cams[0]["height"], len(cams)


def init_gaussians(n_init, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    means = torch.randn(n_init, 3, device=device) * 2.0
    means[:, 2] += 4.0
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_init, device=device, dtype=torch.float32)
    log_scales = torch.zeros(n_init, 3, device=device)
    raw_opacities = torch.zeros(n_init, device=device)
    colors = torch.zeros(n_init, 1, 3, device=device)  # [N, K=1, 3] for SH degree 0
    return means, quats, log_scales, raw_opacities, colors


def train_short(scene, accutile, n_iters, seed, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cam_list, K_list, width, height, n_cams = load_cameras(scene, device)
    means, quats, log_scales, raw_opacities, colors = init_gaussians(50000, device, seed)

    means = means.clone().requires_grad_(True)
    log_scales = log_scales.clone().requires_grad_(True)
    raw_opacities = raw_opacities.clone().requires_grad_(True)
    quats = quats.clone().requires_grad_(True)
    colors = colors.clone().requires_grad_(True)

    params = [
        (means, 0.00016),
        (log_scales, 0.005),
        (raw_opacities, 0.05),
        (quats, 0.001),
        (colors, 0.0025),
    ]
    optimizer = torch.optim.Adam(
        [{"params": p, "lr": lr} for p, lr in params], eps=1e-15
    )

    history = []
    cam_indices = np.random.RandomState(seed).permutation(n_cams)

    for iter_idx in range(n_iters):
        cam_idx = cam_indices[iter_idx % n_cams]
        viewmat = cam_list[cam_idx][None]  # [1, 4, 4]
        K = K_list[cam_idx][None]          # [1, 3, 3]

        optimizer.zero_grad()
        cur_scales = torch.exp(log_scales)
        cur_opacities = torch.sigmoid(raw_opacities)
        rgb, alpha, meta = rasterization(
            means, quats, cur_scales, cur_opacities, colors,
            viewmat, K, width, height, tile_size=16, packed=False,
            sh_degree=0, accutile=accutile,
        )

        loss = rgb.sum() + alpha.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        if iter_idx % 50 == 0 or iter_idx == n_iters - 1:
            entry = {
                "iter": iter_idx,
                "loss": float(loss.item()),
                "n_gaussians": int(means.shape[0]),
                "intersections": int(meta["tiles_per_gauss"].sum()),
            }
            history.append(entry)
            label = "AccuTile" if accutile else "Baseline"
            print(f"  [{label}] iter {iter_idx}: loss={entry['loss']:.4f} N={entry['n_gaussians']} isect={entry['intersections']}")

    return {
        "scene": scene, "accutile": accutile, "n_iters": n_iters, "seed": seed,
        "final_loss": history[-1]["loss"],
        "final_n_gaussians": history[-1]["n_gaussians"],
        "final_intersections": history[-1]["intersections"],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="room")
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="/tmp/accutile_a100_results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("=" * 60)
    print("Short Training Semantic Test: B1 vs B1A (AccuTile)")
    print("=" * 60)

    results = {}
    for label, accutile in [("B1", False), ("B1A", True)]:
        print(f"\n--- {label} ---")
        results[label] = train_short(args.scene, accutile, args.n_iters, args.seed)

    b1 = results["B1"]
    b1a = results["B1A"]
    rel_diff = abs(b1["final_loss"] - b1a["final_loss"]) / max(abs(b1["final_loss"]), 1e-8)
    comparison = {
        "scene": args.scene, "n_iters": args.n_iters, "seed": args.seed,
        "B1": {"final_loss": b1["final_loss"], "final_n_gaussians": b1["final_n_gaussians"]},
        "B1A": {"final_loss": b1a["final_loss"], "final_n_gaussians": b1a["final_n_gaussians"]},
        "loss_diff": abs(b1["final_loss"] - b1a["final_loss"]),
        "loss_rel_diff": rel_diff,
        "gaussian_count_diff": abs(b1["final_n_gaussians"] - b1a["final_n_gaussians"]),
        "verdict": "PASS" if rel_diff < 0.01 else "INVESTIGATE",
    }
    print(f"\n{'='*60}")
    print(json.dumps(comparison, indent=2))

    output = {"results": results, "comparison": comparison}
    with open(f"{args.out}/accutile_training_sanity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.out}/accutile_training_sanity.json")


if __name__ == "__main__":
    main()
