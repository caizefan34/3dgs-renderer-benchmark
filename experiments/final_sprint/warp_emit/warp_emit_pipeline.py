"""Full candidate timing with an isolated gsplat source tree.

The package selected by PYTHONPATH has only the WARP_ALL second-pass edit.
All other stages and Python calls match P2I's frozen component protocol.
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

ROOT = Path(os.environ.get("WARP_EMIT_REMOTE_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo")
CKPTS, DATA = REPO / "results" / "epic05" / "phase7", REPO / "data" / "official" / "mipnerf360"
SCENES, WARMUP, REPS, TILE = ("room", "bicycle", "garden"), 20, 40, 16

from gsplat.cuda._wrapper import (fully_fused_projection, isect_offset_encode, isect_tiles,
                                  rasterize_to_pixels, spherical_harmonics)  # noqa: E402


def stats(values):
    v = torch.tensor(values, dtype=torch.float64)
    return {"median_ms": float(v.median()), "mean_ms": float(v.mean()), "std_ms": float(v.std(unbiased=True)),
            "n": len(values), "samples_ms": [round(float(x), 6) for x in values]}


def event_time(call):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); out = call(); end.record(); end.synchronize()
    return start.elapsed_time(end), out


def load_scene(scene):
    state = torch.load(CKPTS / f"a100_30k_{scene}_t16_16" / f"a100_30k_{scene}_t16_16_latest.pt",
                       map_location="cpu", weights_only=False)["model_state"]
    with open(DATA / scene / "cameras.json", encoding="utf-8") as f: camera = json.load(f)[0]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation; viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0, camera["width"] / 2], [0, camera["fy"], camera["height"] / 2], [0, 0, 1]],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    return {"xyz": state["xyz"].detach().cuda(), "quats": state["rotations"].detach().cuda(),
            "scales": torch.exp(state["scales"].detach()).cuda(), "opacity": state["opacity"].detach().flatten().cuda(),
            "shs": state["shs"].detach().cuda(), "sh_degree": int(state["sh_degree"]),
            "camera": camera, "viewmat": viewmat.unsqueeze(0), "K": K}


def main():
    env = {"commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "tile_size": TILE, "warmup": WARMUP, "reps": REPS,
           "variant": "isolated gsplat WARP_ALL; 128-thread CTA / 4 warps", "camera": "first real official camera"}
    result = {"environment": env, "scenes": {}}
    for scene in SCENES:
        p = load_scene(scene); width, height = p["camera"]["width"], p["camera"]["height"]
        tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
        raw = {k: [] for k in ("projection", "sh", "no_sort", "with_sort", "offset", "raster")}
        last = {}
        for it in range(WARMUP + REPS):
            projection_ms, pr = event_time(lambda: fully_fused_projection(
                p["xyz"], None, p["quats"], p["scales"], p["viewmat"], p["K"], width, height, eps2d=.3,
                near_plane=.01, far_plane=1e10, radius_clip=0., packed=True, sparse_grad=False,
                calc_compensations=False, camera_model="pinhole", opacities=p["opacity"]))
            bi, ci, gi, radii, means2d, depths, conics, _ = pr; image_ids = bi * ci
            no_sort_ms, _ = event_time(lambda: isect_tiles(means2d, radii, depths, TILE, tw, th, sort=False,
                                                            segmented=False, packed=True, n_images=1,
                                                            image_ids=image_ids, gaussian_ids=gi))
            with_sort_ms, sorted_isects = event_time(lambda: isect_tiles(means2d, radii, depths, TILE, tw, th, sort=True,
                                                                           segmented=False, packed=True, n_images=1,
                                                                           image_ids=image_ids, gaussian_ids=gi))
            tpg, isect_ids, flatten_ids = sorted_isects
            offset_ms, offsets = event_time(lambda: isect_offset_encode(isect_ids, 1, tw, th))
            campos = torch.inverse(p["viewmat"])[:, :3, 3]; dirs = p["xyz"][gi] - campos[bi]
            sh_ms, colors = event_time(lambda: torch.clamp_min(
                spherical_harmonics(p["sh_degree"], dirs, p["shs"][gi], masks=(radii > 0).all(-1)) + .5, 0.))
            raster_ms, _ = event_time(lambda: rasterize_to_pixels(
                means2d, conics, colors, p["opacity"][gi], width, height, TILE, offsets.view(1, 1, th, tw),
                flatten_ids, backgrounds=None, masks=None, packed=True, absgrad=False))
            if it >= WARMUP:
                for k, x in (("projection", projection_ms), ("sh", sh_ms), ("no_sort", no_sort_ms),
                             ("with_sort", with_sort_ms), ("offset", offset_ms), ("raster", raster_ms)): raw[k].append(x)
            last = {"n_visible": int(gi.numel()), "n_isects": int(tpg.sum().item()), "tile_grid": [tw, th]}
        timing = {k: stats(v) for k, v in raw.items()}
        forward = sum(timing[k]["median_ms"] for k in ("projection", "sh", "with_sort", "offset", "raster"))
        result["scenes"][scene] = {"meta": last, "timings": timing,
            "derived": {"sort_median_ms": timing["with_sort"]["median_ms"] - timing["no_sort"]["median_ms"],
                        "forward_median_ms": forward}}
        print(scene, json.dumps(result["scenes"][scene], indent=2), flush=True)
    (OUT / "w1_full_pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "w1_full_pipeline.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["scene", "visible", "isects", "projection", "no_sort", "sort", "offset", "raster", "forward"])
        for s, r in result["scenes"].items():
            t, d = r["timings"], r["derived"]; w.writerow([s, r["meta"]["n_visible"], r["meta"]["n_isects"],
                t["projection"]["median_ms"], t["no_sort"]["median_ms"], d["sort_median_ms"], t["offset"]["median_ms"],
                t["raster"]["median_ms"], d["forward_median_ms"]])


if __name__ == "__main__": main()
