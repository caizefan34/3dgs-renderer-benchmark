"""Probe the frozen Room checkpoint through the actual gsplat rasterization call."""

import argparse
import json
import time
import torch
from gsplat import rasterization


def load_room():
    checkpoint = torch.load(
        "results/epic05/phase7/phase7_room_30k_16/phase7_room_30k_16_latest.pt",
        map_location="cpu", weights_only=False,
    )["model_state"]
    camera = json.load(open("data/official/mipnerf360/room/cameras.json", encoding="utf-8"))[0]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0., camera["width"] / 2],
                      [0., camera["fy"], camera["height"] / 2], [0., 0., 1.]],
                     dtype=torch.float32, device="cuda")[None]
    return checkpoint, camera, viewmat[None], K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accutile", choices=("off", "on"), required=True)
    args = parser.parse_args()
    state, camera, viewmats, Ks = load_room()
    means = state["xyz"].cuda().requires_grad_(True)
    quats = state["rotations"].cuda().requires_grad_(True)
    scales = torch.exp(state["scales"].cuda()).requires_grad_(True)
    opacities = torch.sigmoid(state["opacity"].cuda().flatten()).requires_grad_(True)
    colors = state["shs"].cuda().requires_grad_(True)
    start = time.perf_counter()
    rgb, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        camera["width"], camera["height"], tile_size=16, packed=False,
        sh_degree=int(state["sh_degree"]), absgrad=True, accutile=args.accutile == "on",
    )
    (rgb.sum() + alpha.sum()).backward()
    torch.cuda.synchronize()
    print(json.dumps({
        "accutile": args.accutile,
        "gaussian_count": int(means.shape[0]),
        "intersections": int(meta["tiles_per_gauss"].sum()),
        "rgb_sum": float(rgb.sum()),
        "alpha_sum": float(alpha.sum()),
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }))


if __name__ == "__main__":
    main()
