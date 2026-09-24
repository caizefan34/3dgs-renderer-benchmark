"""Baseline/candidate parity runner for two real cameras per scene."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("WARP_EMIT_REMOTE_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw" / "correctness"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo")
CKPTS, DATA = REPO / "results" / "epic05" / "phase7", REPO / "data" / "official" / "mipnerf360"
SCENES, CAMERAS, TILE = ("room", "bicycle", "garden"), (0, 1), 16

from gsplat.cuda._wrapper import (fully_fused_projection, isect_offset_encode, isect_tiles,
                                  rasterize_to_pixels, spherical_harmonics)  # noqa: E402


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def params(scene, cam_idx):
    state = torch.load(CKPTS / f"a100_30k_{scene}_t16_16" / f"a100_30k_{scene}_t16_16_latest.pt",
                       map_location="cpu", weights_only=False)["model_state"]
    with open(DATA / scene / "cameras.json", encoding="utf-8") as f: camera = json.load(f)[cam_idx]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation; viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0, camera["width"] / 2], [0, camera["fy"], camera["height"] / 2], [0, 0, 1]],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    return {"xyz": state["xyz"].detach().cuda().requires_grad_(), "quats": state["rotations"].detach().cuda().requires_grad_(),
            "scales": torch.exp(state["scales"].detach()).cuda().requires_grad_(),
            "opacity": state["opacity"].detach().flatten().cuda().requires_grad_(), "shs": state["shs"].detach().cuda().requires_grad_(),
            "degree": int(state["sh_degree"]), "camera": camera, "viewmat": viewmat.unsqueeze(0), "K": K}


def run(scene, cam_idx):
    p = params(scene, cam_idx); width, height = p["camera"]["width"], p["camera"]["height"]
    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    bi, ci, gi, radii, means2d, depths, conics, _ = fully_fused_projection(
        p["xyz"], None, p["quats"], p["scales"], p["viewmat"], p["K"], width, height, eps2d=.3, near_plane=.01,
        far_plane=1e10, radius_clip=0., packed=True, sparse_grad=False, calc_compensations=False,
        camera_model="pinhole", opacities=p["opacity"])
    image_ids = bi * ci
    tpg, pre_ids, pre_flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=False, segmented=False,
                                          packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gi)
    _, sorted_ids, sorted_flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=True, segmented=False,
                                              packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gi)
    offsets = isect_offset_encode(sorted_ids, 1, tw, th)
    campos = torch.inverse(p["viewmat"])[:, :3, 3]; dirs = p["xyz"][gi] - campos[bi]
    colors = torch.clamp_min(spherical_harmonics(p["degree"], dirs, p["shs"][gi], masks=(radii > 0).all(-1)) + .5, 0.)
    rgb, alpha = rasterize_to_pixels(means2d, conics, colors, p["opacity"][gi], width, height, TILE,
                                      offsets.view(1, 1, th, tw), sorted_flat, backgrounds=None, masks=None,
                                      packed=True, absgrad=False)
    (rgb.square().mean() + alpha.mean()).backward()
    torch.cuda.synchronize()
    return {"meta": {"n_visible": int(gi.numel()), "n_isects": int(tpg.sum().item()), "tile_grid": [tw, th]},
            "hashes": {"tpg": sha(tpg), "pre_ids": sha(pre_ids), "pre_flat": sha(pre_flat),
                       "sorted_ids": sha(sorted_ids), "sorted_flat": sha(sorted_flat), "offsets": sha(offsets)},
            "tensors": {"rgb": rgb.detach().cpu(), "alpha": alpha.detach().cpu(),
                        "xyz_grad": p["xyz"].grad.cpu(), "quats_grad": p["quats"].grad.cpu(),
                        "scales_grad": p["scales"].grad.cpu(), "opacity_grad": p["opacity"].grad.cpu(),
                        "shs_grad": p["shs"].grad.cpu()}}


def metrics(candidate, baseline):
    d = candidate.to(torch.float64) - baseline.to(torch.float64)
    max_abs, mean_abs = float(d.abs().max()), float(d.abs().mean())
    denom = float(baseline.to(torch.float64).abs().mean())
    return {"max_abs": max_abs, "mean_abs": mean_abs, "relative_error": mean_abs / max(denom, 1e-20)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("baseline", "candidate")); args = ap.parse_args()
    summary = {"mode": args.mode, "cases": {}}
    for scene in SCENES:
        for cam in CAMERAS:
            key = f"{scene}_cam{cam}"; result = run(scene, cam)
            artifact = OUT / f"baseline_{key}.pt"
            if args.mode == "baseline":
                torch.save(result, artifact); summary["cases"][key] = {"meta": result["meta"], "hashes": result["hashes"]}
            else:
                baseline = torch.load(artifact, weights_only=True)
                tensor_metrics = {k: metrics(v, baseline["tensors"][k]) for k, v in result["tensors"].items()}
                tensor_metrics["render_psnr"] = float("inf") if tensor_metrics["rgb"]["max_abs"] == 0 else \
                    float(-10 * math.log10(max((result["tensors"]["rgb"].float() - baseline["tensors"]["rgb"].float()).square().mean().item(), 1e-30)))
                exact = {k: result["hashes"][k] == baseline["hashes"][k] for k in result["hashes"]}
                summary["cases"][key] = {"meta": result["meta"], "exact_hash_identity": exact, "tensor_metrics": tensor_metrics}
            print(key, summary["cases"][key], flush=True)
    (OUT / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
