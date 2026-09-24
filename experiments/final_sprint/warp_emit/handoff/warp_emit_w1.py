"""W1 isolated emit experiment. Does not modify the renderer.

It feeds exact production projection/count/scan state to an extracted serial
baseline and to WARP_ALL. The candidate retains baseline logical output index k
and is checked bit-for-bit before timing.
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
OUT = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo")
CKPTS = REPO / "results" / "epic05" / "phase7"
DATA = REPO / "data" / "official" / "mipnerf360"
SCENES = ("room", "bicycle", "garden")
WARMUP, REPS, TILE = 20, 40, 16
BUCKETS = ((1, 4), (5, 16), (17, 32), (33, 64), (65, 128), (129, 256),
           (257, 512), (513, 1024), (1025, None))

sys.path.insert(0, str(REPO / "src"))
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles  # noqa: E402


def stats(values):
    v = torch.tensor(values, dtype=torch.float64)
    return {"median_ms": float(v.median()), "mean_ms": float(v.mean()),
            "std_ms": float(v.std(unbiased=True)), "n": len(values),
            "samples_ms": [round(float(x), 6) for x in values]}


def event_time(call):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); out = call(); end.record(); end.synchronize()
    return start.elapsed_time(end), out


def load_scene(scene):
    path = CKPTS / f"a100_30k_{scene}_t16_16" / f"a100_30k_{scene}_t16_16_latest.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
    with open(DATA / scene / "cameras.json", encoding="utf-8") as f:
        camera = json.load(f)[0]
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation; viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[camera["fx"], 0, camera["width"] / 2],
                      [0, camera["fy"], camera["height"] / 2], [0, 0, 1]],
                     dtype=torch.float32, device="cuda").unsqueeze(0)
    return {"xyz": state["xyz"].detach().cuda(), "quats": state["rotations"].detach().cuda(),
            "scales": torch.exp(state["scales"].detach()).cuda(),
            "opacity": state["opacity"].detach().flatten().cuda(), "camera": camera,
            "viewmat": viewmat.unsqueeze(0), "K": K}


def bucket_metrics(counts):
    cpu = counts.cpu()
    total_gauss, total_work = int(cpu.numel()), int(cpu.sum())
    rows = []
    for lo, hi in BUCKETS:
        mask = cpu >= lo if hi is None else (cpu >= lo) & (cpu <= hi)
        n, work = int(mask.sum()), int(cpu[mask].sum())
        label = f">{lo - 1}" if hi is None else f"{lo}–{hi}"
        rows.append({"bucket": label, "gaussians": n, "intersections": work,
                     "gaussian_fraction": n / total_gauss if total_gauss else 0.0,
                     "work_fraction": work / total_work if total_work else 0.0})
    positive = cpu[cpu > 0].to(torch.float64)
    return {"distribution": {"mean": float(positive.mean()), "p50": float(torch.quantile(positive, .50)),
                             "p90": float(torch.quantile(positive, .90)), "p95": float(torch.quantile(positive, .95)),
                             "p99": float(torch.quantile(positive, .99)), "max": int(positive.max())},
            "buckets": rows}


def main():
    ext = load(name="warp_emit_ext_w1", sources=[str(ROOT / "warp_emit_ext.cpp"), str(ROOT / "warp_emit_ext.cu")],
               extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=True, build_directory=str(ROOT / "build"))
    env = {"commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "tile_size": TILE, "warmup": WARMUP, "reps": REPS,
           "camera": "first real official Mip-NeRF 360 camera", "variant": "WARP_ALL"}
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    result = {"environment": env, "scenes": {}}
    for scene in SCENES:
        p = load_scene(scene); width, height = p["camera"]["width"], p["camera"]["height"]
        tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
        # Project and produce production baseline once. Projection is frozen;
        # its outputs drive both emit implementations for all warmups/reps.
        _, projection = event_time(lambda: fully_fused_projection(
            p["xyz"], None, p["quats"], p["scales"], p["viewmat"], p["K"], width, height,
            eps2d=.3, near_plane=.01, far_plane=1e10, radius_clip=0., packed=True,
            sparse_grad=False, calc_compensations=False, camera_model="pinhole", opacities=p["opacity"]))
        bi, ci, gi, radii, means2d, depths, _, _ = projection
        image_ids = (bi * ci).contiguous()
        tpg, prod_ids, prod_flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=False,
                                                segmented=False, packed=True, n_images=1,
                                                image_ids=image_ids, gaussian_ids=gi)
        cumulative = torch.cumsum(tpg.to(torch.int64), 0); n_isects = int(cumulative[-1].item()) if len(cumulative) else 0
        def alloc():
            return (torch.empty(n_isects, dtype=torch.int64, device="cuda"),
                    torch.empty(n_isects, dtype=torch.int32, device="cuda"))
        serial_ids, serial_flat = alloc()
        ext.serial_emit(means2d.contiguous(), radii.contiguous(), depths.contiguous(), image_ids,
                        cumulative, TILE, tw, th, serial_ids, serial_flat)
        identity = {"production_vs_serial_isect_ids": int((prod_ids != serial_ids).sum().item()),
                    "production_vs_serial_flatten_ids": int((prod_flat != serial_flat).sum().item())}
        timings = {"serial": [], "warp_128": [], "warp_256": []}
        warp_identity = {}
        for it in range(WARMUP + REPS):
            sid, sflat = alloc()
            ms, _ = event_time(lambda: ext.serial_emit(means2d.contiguous(), radii.contiguous(), depths.contiguous(), image_ids,
                                                        cumulative, TILE, tw, th, sid, sflat))
            if it >= WARMUP: timings["serial"].append(ms)
            for tpb in (128, 256):
                wid, wflat = alloc()
                ms, _ = event_time(lambda t=tpb: ext.warp_emit(means2d.contiguous(), radii.contiguous(), depths.contiguous(), image_ids,
                                                                cumulative, TILE, tw, th, t, wid, wflat))
                if it == WARMUP:
                    warp_identity[str(tpb)] = {"isect_ids_mismatch": int((prod_ids != wid).sum().item()),
                                                "flatten_ids_mismatch": int((prod_flat != wflat).sum().item())}
                if it >= WARMUP: timings[f"warp_{tpb}"].append(ms)
        timing_stats = {k: stats(v) for k, v in timings.items()}
        best = min(("warp_128", "warp_256"), key=lambda k: timing_stats[k]["median_ms"])
        result["scenes"][scene] = {"meta": {"n_visible": int(gi.numel()), "n_isects": n_isects,
                                             "tile_grid": [tw, th], "intersections_per_visible": n_isects / int(gi.numel())},
                                   "identity": {**identity, "warp_vs_production": warp_identity},
                                   "workload": bucket_metrics(tpg), "timings": timing_stats,
                                   "best_warp": best,
                                   "derived": {"best_emit_speedup": timing_stats["serial"]["median_ms"] / timing_stats[best]["median_ms"]}}
        print(scene, json.dumps(result["scenes"][scene], indent=2), flush=True)
        torch.cuda.empty_cache()
    (OUT / "w1_isolated_emit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "w1_isolated_emit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["scene", "visible", "isects", "isects_per_visible", "serial_ms", "warp128_ms", "warp256_ms", "best", "best_speedup", "id_mismatches", "flat_mismatches"])
        for scene, r in result["scenes"].items():
            i = r["identity"]; w.writerow([scene, r["meta"]["n_visible"], r["meta"]["n_isects"], r["meta"]["intersections_per_visible"],
                r["timings"]["serial"]["median_ms"], r["timings"]["warp_128"]["median_ms"], r["timings"]["warp_256"]["median_ms"], r["best_warp"], r["derived"]["best_emit_speedup"],
                i["warp_vs_production"]["128"]["isect_ids_mismatch"], i["warp_vs_production"]["128"]["flatten_ids_mismatch"]])


if __name__ == "__main__":
    main()
