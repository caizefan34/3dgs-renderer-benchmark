"""A0 only: compare current AABB enumeration to upstream AccuTile/SNUGBOX.

This never alters the Reference V1 renderer.  The modern gsplat call receives
the exact projected tensors and raw opacity tensor used by Reference V1.
"""
import json
import math
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(os.environ.get("ACCUTILE_A0_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo")
PHASE7 = REPO / "results" / "epic05" / "phase7"
DATA = REPO / "data" / "official" / "mipnerf360"
TILE = 16
ALPHA = 1.0 / 255.0
EXTEND = 3.33

from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles  # noqa: E402


def quantiles(x):
    x = x.float()
    return {str(q): float(torch.quantile(x, q / 100).item()) for q in (50, 90, 95, 99)} | {"max": int(x.max().item())}


def load(scene):
    checkpoint = torch.load(PHASE7 / f"a100_30k_{scene}_t16_16" / f"a100_30k_{scene}_t16_16_latest.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    with open(DATA / scene / "cameras.json", encoding="utf-8") as f:
        camera = json.load(f)[0]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0, camera["width"] / 2], [0, camera["fy"], camera["height"] / 2], [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)
    return state, camera, viewmat.unsqueeze(0), K


def quadratic_min_on_tile(means, conics, tiles, width, height, tile_size):
    """Exact continuous minimum of positive-definite q over each image-clipped tile."""
    tx = (tiles % math.ceil(width / tile_size)).float() * tile_size
    ty = torch.div(tiles, math.ceil(width / tile_size), rounding_mode="floor").float() * tile_size
    dx0, dx1 = tx - means[:, 0], torch.minimum(tx + tile_size, torch.full_like(tx, width)) - means[:, 0]
    dy0, dy1 = ty - means[:, 1], torch.minimum(ty + tile_size, torch.full_like(ty, height)) - means[:, 1]
    a, b, c = conics.unbind(1)
    def q(dx, dy): return a * dx * dx + 2 * b * dx * dy + c * dy * dy
    candidates = [q(dx0, dy0), q(dx0, dy1), q(dx1, dy0), q(dx1, dy1)]
    for dx in (dx0, dx1):
        candidates.append(q(dx, torch.clamp(-b * dx / c, dy0, dy1)))
    for dy in (dy0, dy1):
        candidates.append(q(torch.clamp(-b * dy / a, dx0, dx1), dy))
    inside = (dx0 <= 0) & (dx1 >= 0) & (dy0 <= 0) & (dy1 >= 0)
    candidates.append(torch.where(inside, torch.zeros_like(dx0), torch.full_like(dx0, float("inf"))))
    return torch.stack(candidates).amin(0)


def pixel_support(mean, conic, opacity, tile, width, height, tile_size):
    """Match rasterization's `j+0.5`, `i+0.5`, alpha threshold test."""
    tw = math.ceil(width / tile_size)
    x0, y0 = (tile % tw) * tile_size, (tile // tw) * tile_size
    a, b, c = conic
    for y in range(y0, min(y0 + tile_size, height)):
        for x in range(x0, min(x0 + tile_size, width)):
            dx, dy = mean[0] - (x + 0.5), mean[1] - (y + 0.5)
            sigma = 0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
            alpha = min(0.999, opacity * math.exp(-sigma))
            if sigma >= 0.0 and alpha >= ALPHA:
                return True
    return False


def analyze(scene):
    state, camera, viewmat, K = load(scene)
    xyz = state["xyz"].detach().cuda()
    quats = state["rotations"].detach().cuda()
    scales = torch.exp(state["scales"].detach()).cuda()
    # Reference V1 passes this stored tensor directly to projection. Record its
    # range instead of silently applying sigmoid as a modern implementation might.
    opacities = state["opacity"].detach().flatten().cuda().float()
    width, height = camera["width"], camera["height"]
    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    with torch.no_grad():
        projection = fully_fused_projection(
            xyz, None, quats, scales, viewmat, K, width, height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=True,
            sparse_grad=False, calc_compensations=False, camera_model="pinhole", opacities=opacities)
        bi, ci, gi, _, radii, means2d, depths, conics, _ = projection
        image_ids = bi * ci
        packed_opacity = opacities[gi].contiguous()
        aabb_tpg, aabb_ids, aabb_flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=False, segmented=False, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gi)
        accu_tpg, accu_ids, accu_flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=False, segmented=False, packed=True, n_images=1, image_ids=image_ids, gaussian_ids=gi, conics=conics.float().contiguous(), opacities=packed_opacity)
        n_tiles = tw * th
        tile_bits = math.ceil(math.log2(n_tiles))
        tile_mask = (1 << tile_bits) - 1
        aabb_key = aabb_flat.to(torch.int64) * n_tiles + ((aabb_ids >> 32) & tile_mask)
        accu_key = accu_flat.to(torch.int64) * n_tiles + ((accu_ids >> 32) & tile_mask)
        aabb_key = torch.sort(aabb_key).values
        accu_key = torch.sort(accu_key).values
        positions = torch.searchsorted(aabb_key, accu_key)
        subset_failures = int(((positions >= aabb_key.numel()) | (aabb_key[positions.clamp_max(aabb_key.numel() - 1)] != accu_key)).sum().item())
        positions = torch.searchsorted(accu_key, aabb_key)
        is_retained = (positions < accu_key.numel()) & (accu_key[positions.clamp_max(accu_key.numel() - 1)] == aabb_key)
        removed = aabb_key[~is_retained]
        false_negative = 0
        pixel_false_negative = []
        min_margin = float("inf")
        batch = 1_000_000
        for start in range(0, removed.numel(), batch):
            keys = removed[start:start + batch]
            flat = torch.div(keys, n_tiles, rounding_mode="floor")
            tile = keys.remainder(n_tiles)
            qmin = quadratic_min_on_tile(means2d[flat], conics.float()[flat], tile, width, height, TILE)
            opacity = packed_opacity[flat]
            t = torch.minimum(torch.full_like(opacity, EXTEND * EXTEND), 2 * torch.log(opacity / ALPHA))
            margin = qmin - t
            false_negative += int((margin <= 1e-5).sum().item())
            min_margin = min(min_margin, float(margin.min().item()))
            candidate = margin <= 1e-5
            if candidate.any():
                candidate_flat = flat[candidate].cpu().tolist()
                candidate_tile = tile[candidate].cpu().tolist()
                candidate_mean = means2d[flat[candidate]].cpu().tolist()
                candidate_conic = conics.float()[flat[candidate]].cpu().tolist()
                candidate_opacity = packed_opacity[flat[candidate]].cpu().tolist()
                for flat_id, tile_id, mean, conic, opacity in zip(candidate_flat, candidate_tile, candidate_mean, candidate_conic, candidate_opacity):
                    if pixel_support(mean, conic, opacity, tile_id, width, height, TILE):
                        pixel_false_negative.append({"flat_id": flat_id, "tile_id": tile_id, "q_minus_t": None})
        aabb_n, accu_n = int(aabb_tpg.sum().item()), int(accu_tpg.sum().item())
        return {
            "scene": scene, "camera_index": 0, "width": width, "height": height, "tile_size": TILE,
            "n_total": int(xyz.shape[0]), "n_visible": int(gi.numel()), "n_isects_aabb": aabb_n,
            "n_isects_accutile": accu_n, "reduction": 1.0 - accu_n / aabb_n,
            "aabb_distribution": quantiles(aabb_tpg), "accutile_distribution": quantiles(accu_tpg),
            "mean_tiles_aabb": float(aabb_tpg.float().mean().item()), "mean_tiles_accutile": float(accu_tpg.float().mean().item()),
            "subset_failures": subset_failures, "removed_pairs": int(removed.numel()),
            "continuous_support_false_negative_pairs": false_negative, "minimum_removed_q_minus_t": min_margin,
            "pixel_support_false_negative_pairs": len(pixel_false_negative),
            "pixel_support_false_negative_examples": pixel_false_negative[:8],
            "opacity_stored_range": [float(opacities.min().item()), float(opacities.max().item()), float(opacities.mean().item())],
        }


def main():
    output = {"environment": {
        "reference_upstream_commit": "28e794ca44a4c25ffc39175370c5ee7b38bfcc36",
        "modern_harness_source_commit": subprocess.check_output(
            ["git", "-C", "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx", "rev-parse", "HEAD"], text=True).strip(),
        "reference_v1_commit": "02375033388d4348376b6b607ab85f551e498a77",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "protocol": "same checkpoint/camera/projected state; modern upstream AccuTile branch only",
    }, "scenes": {}}
    for scene in ("room", "bicycle", "garden"):
        result = analyze(scene)
        output["scenes"][scene] = result
        print(json.dumps(result, indent=2), flush=True)
        torch.cuda.empty_cache()
    (OUT / "a0_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
