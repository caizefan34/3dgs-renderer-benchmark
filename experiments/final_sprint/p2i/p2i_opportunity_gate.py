"""P2I P0 measurement harness; it does not alter the gsplat renderer.

The extension is an exact extraction of legacy IntersectTile.cu pass-1 math.
Its output is checked against the production isect_tiles output before timing.
All timed outer stages use CUDA events; pass-2 is the no-sort residual after
subtracting directly event-measured count and scan medians.
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(os.environ.get("P2I_REMOTE_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo")
CKPTS = REPO / "results" / "epic05" / "phase7"
DATA = REPO / "data" / "official" / "mipnerf360"
SCENES = ("room", "bicycle", "garden")
WARMUP, REPS, TILE = 20, 40, 16

sys.path.insert(0, str(REPO / "src"))
from gsplat.cuda._wrapper import (  # noqa: E402
    fully_fused_projection, isect_offset_encode, isect_tiles, rasterize_to_pixels,
    spherical_harmonics,
)


def stats(values):
    value_tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median_ms": float(value_tensor.median()),
        "mean_ms": float(value_tensor.mean()),
        "std_ms": float(value_tensor.std(unbiased=True)),
        "n": len(values),
        "samples_ms": [round(float(v), 6) for v in values],
    }


def event_time(call):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    out = call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), out


def load_scene(scene):
    path = CKPTS / f"a100_30k_{scene}_t16_16" / f"a100_30k_{scene}_t16_16_latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    opacity = state["opacity"].detach().flatten().cuda()
    with open(DATA / scene / "cameras.json", encoding="utf-8") as f:
        camera = json.load(f)[0]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0, camera["width"] / 2],
                      [0, camera["fy"], camera["height"] / 2],
                      [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)
    return {
        "xyz": state["xyz"].detach().cuda(),
        "quats": state["rotations"].detach().cuda(),
        "scales": torch.exp(state["scales"].detach()).cuda(),
        "opacity": opacity,
        "shs": state["shs"].detach().cuda(),
        "sh_degree": int(state["sh_degree"]),
        "camera": camera,
        "viewmat": viewmat.unsqueeze(0),
        "K": K,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ext = load(
        name="p2i_count_ext_v1",
        sources=[str(ROOT / "p2i_count_ext.cpp"), str(ROOT / "p2i_count_ext.cu")],
        extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=True,
        build_directory=str(ROOT / "build"),
    )
    environment = {
        "commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "tile_size": TILE, "warmup": WARMUP, "reps": REPS,
        "camera": "first real camera from each official Mip-NeRF 360 cameras.json",
        "method": "CUDA events; pass-1 is an exact extracted baseline kernel with output identity check",
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    all_results = {"environment": environment, "scenes": {}}
    for scene in SCENES:
        params = load_scene(scene)
        width, height = params["camera"]["width"], params["camera"]["height"]
        tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
        raw = {k: [] for k in ("projection", "sh", "count", "scan", "no_sort", "with_sort", "offset", "raster")}
        mismatch = None
        last = {}
        for iteration in range(WARMUP + REPS):
            projection_ms, pr = event_time(lambda: fully_fused_projection(
                params["xyz"], None, params["quats"], params["scales"], params["viewmat"], params["K"],
                width, height, eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
                packed=True, sparse_grad=False, calc_compensations=False, camera_model="pinhole",
                opacities=params["opacity"],
            ))
            bi, ci, gi, radii, means2d, depths, conics, _ = pr
            image_ids = bi * ci
            counts = torch.empty_like(depths, dtype=torch.int32)
            count_ms, _ = event_time(lambda: ext.count_tiles(means2d.contiguous(), radii.contiguous(), counts, TILE, tw, th))
            scan_ms, cumulative = event_time(lambda: torch.cumsum(counts.view(-1).to(torch.int64), 0))
            no_sort_ms, no_sort = event_time(lambda: isect_tiles(
                means2d, radii, depths, TILE, tw, th, sort=False, segmented=False, packed=True,
                n_images=1, image_ids=image_ids, gaussian_ids=gi))
            with_sort_ms, sorted_isects = event_time(lambda: isect_tiles(
                means2d, radii, depths, TILE, tw, th, sort=True, segmented=False, packed=True,
                n_images=1, image_ids=image_ids, gaussian_ids=gi))
            tpg, isect_ids, flatten_ids = sorted_isects
            offset_ms, offsets = event_time(lambda: isect_offset_encode(isect_ids, 1, tw, th))
            campos = torch.inverse(params["viewmat"])[:, :3, 3]
            dirs = params["xyz"][gi] - campos[bi]
            sh_ms, colors = event_time(lambda: torch.clamp_min(
                spherical_harmonics(params["sh_degree"], dirs, params["shs"][gi], masks=(radii > 0).all(-1)) + 0.5, 0.0))
            packed_opacity = params["opacity"][gi]
            raster_ms, _ = event_time(lambda: rasterize_to_pixels(
                means2d, conics, colors, packed_opacity, width, height, TILE,
                offsets.view(1, 1, th, tw), flatten_ids, backgrounds=None, masks=None,
                packed=True, absgrad=False))
            if iteration == WARMUP:
                diff = (counts != tpg).sum().item()
                mismatch = {"count": int(diff), "max_abs": int((counts - tpg).abs().max().item())}
            if iteration >= WARMUP:
                for key, value in (("projection", projection_ms), ("sh", sh_ms), ("count", count_ms), ("scan", scan_ms),
                                   ("no_sort", no_sort_ms), ("with_sort", with_sort_ms),
                                   ("offset", offset_ms), ("raster", raster_ms)):
                    raw[key].append(value)
            last = {"n_total_gaussians": int(params["xyz"].shape[0]), "n_visible": int(gi.numel()),
                    "n_isects": int(tpg.sum().item()), "tile_grid": [tw, th]}
        timed = {key: stats(values) for key, values in raw.items()}
        # The production wrapper exposes only count+scan+emit, so emit is the
        # residual after independently event-measured count and scan.
        emit_median = timed["no_sort"]["median_ms"] - timed["count"]["median_ms"] - timed["scan"]["median_ms"]
        sort_median = timed["with_sort"]["median_ms"] - timed["no_sort"]["median_ms"]
        forward_median = (timed["projection"]["median_ms"] + timed["sh"]["median_ms"] + timed["with_sort"]["median_ms"] +
                          timed["offset"]["median_ms"] + timed["raster"]["median_ms"])
        all_results["scenes"][scene] = {
            "meta": last, "tile_count_identity": mismatch, "timings": timed,
            "derived": {"emit_median_ms": emit_median, "sort_median_ms": sort_median,
                        "forward_median_ms": forward_median,
                        "count_pass_share": timed["count"]["median_ms"] / forward_median,
                        "predicted_forward_gain_upper_bound": timed["count"]["median_ms"] / forward_median,
                        "derivation": "emit=no_sort-count-scan; sort=with_sort-no_sort; no component overlaps"},
        }
        print(scene, json.dumps(all_results["scenes"][scene], indent=2), flush=True)
        torch.cuda.empty_cache()
    with (OUT / "p2i_opportunity_gate.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with (OUT / "p2i_opportunity_gate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["scene", "count_ms", "forward_ms", "count_share", "n_visible", "n_isects", "mismatches"])
        for scene, result in all_results["scenes"].items():
            writer.writerow([scene, result["timings"]["count"]["median_ms"], result["derived"]["forward_median_ms"],
                             result["derived"]["count_pass_share"], result["meta"]["n_visible"], result["meta"]["n_isects"],
                             result["tile_count_identity"]["count"]])


if __name__ == "__main__":
    main()
