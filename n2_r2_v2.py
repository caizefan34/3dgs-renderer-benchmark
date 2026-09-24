#!/usr/bin/env python3
"""N2-R2: A100 raster-backward + E2E timing benchmark (low-level API).

Isolates rasterize_to_pixels_bwd kernel timing by using the low-level API:
  1. Projection (no_grad) -> means2d, conics, depths, radii
  2. isect_tiles (no_grad) -> isect_offsets, flatten_ids
  3. rasterize_to_pixels fwd (with grad on detached inputs)
  4. Time rasterize_to_pixels bwd ONLY (no projection_bwd, no SH_bwd)

Also measures E2E (full forward + backward via high-level API) for context.

Protocol:
- warmup >= 50 iterations
- timed >= 300 iterations
- >= 5 independent timing repetitions
- CUDA Events with explicit synchronization
- Interleaved baseline/CPCB order
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import subprocess
import numpy as np

TIMING_RUNNER = r'''
import sys, os, json, math, torch
import numpy as np

def main():
    gsplat_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    output_path = sys.argv[3]
    config_json = json.loads(sys.argv[4])

    sys.path.insert(0, gsplat_path)
    for mod in list(sys.modules.keys()):
        if "gsplat" in mod:
            del sys.modules[mod]
    import gsplat
    from gsplat import rasterization
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode,
        rasterize_to_pixels, spherical_harmonics
    )

    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    N = ckpt["num_points"]
    print(f"  N={N}", flush=True)

    means3d = ckpt["xyz"].to(device).float()
    quats = ckpt["rotation"].to(device).float()
    scales = ckpt["scaling"].to(device).float()
    opacities_raw = ckpt["opacity"].to(device).float()
    shs = ckpt["shs"].to(device).float()
    sh_degree = ckpt.get("active_sh_degree", 3)
    opacities = torch.sigmoid(opacities_raw)

    repo_root = os.path.expanduser("~/3dgs-renderer-benchmark")
    camera_path = os.path.join(repo_root, "data", "official", "mipnerf360", "room", "cameras.json")
    with open(camera_path) as f:
        cameras_data = json.load(f)

    if isinstance(cameras_data, dict):
        cam_list = cameras_data.get("cameras", list(cameras_data.values()))
    elif isinstance(cameras_data, list):
        cam_list = cameras_data

    cam_entry = cam_list[0]
    orig_W = int(cam_entry.get("width", 1920))
    orig_H = int(cam_entry.get("height", 1080))
    fx = float(cam_entry.get("fx", 1000))
    fy = float(cam_entry.get("fy", 1000))
    W, H = 960, 540
    fx = fx * W / orig_W
    fy = fy * H / orig_H
    rot = np.asarray(cam_entry["rotation"], dtype=np.float32)
    pos = np.asarray(cam_entry["position"], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos
    viewmat = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32).unsqueeze(0).to(device)
    K = torch.tensor([[fx, 0, W/2.0], [0, fy, H/2.0], [0, 0, 1]], dtype=torch.float32).unsqueeze(0).to(device)

    print(f"  Camera 0: {W}x{H}", flush=True)

    warmup = config_json.get("warmup", 50)
    timed = config_json.get("timed", 300)
    n_reps = config_json.get("n_reps", 5)

    tile_size = 16
    tile_width = math.ceil(W / float(tile_size))
    tile_height = math.ceil(H / float(tile_size))

    # ---- Phase 1: Precompute projection + intersection (no grad) ----
    with torch.no_grad():
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means3d, None, quats, scales, viewmat, K, W, H,
            eps2d=0.1, packed=False, calc_compensations=False,
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height, packed=False,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
        isect_offsets = isect_offsets.reshape(1, tile_height, tile_width)

        # SH evaluation (no grad) to get colors for rasterize_to_pixels
        campos = torch.inverse(viewmat)[..., :3, 3]
        dirs = means3d.unsqueeze(0) - campos.unsqueeze(1)  # [1, N, 3]
        masks = (radii > 0).all(dim=-1)  # [1, N]
        colors = spherical_harmonics(sh_degree, dirs, shs.unsqueeze(0), masks=masks)  # [1, N, 3]
        opacities_2d = opacities.unsqueeze(0)  # [1, N]

    print(f"  Projection + SH done. n_isects={isect_ids.shape[0]}", flush=True)

    # Random v_render for backward
    torch.manual_seed(123)
    v_rc = torch.randn(1, H, W, 3, device=device) * 0.1
    v_ra = torch.randn(1, H, W, 1, device=device) * 0.01

    # ---- Phase 2: Isolated rasterize_to_pixels_bwd timing ----
    # Create leaf tensors ONCE to avoid memory leak from repeated clone()
    m2d = means2d.detach().clone().requires_grad_(True)
    con = conics.detach().clone().requires_grad_(True)
    col = colors.detach().clone().requires_grad_(True)
    opa = opacities_2d.detach().clone().requires_grad_(True)

    all_reps = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}", flush=True)

        # Warmup
        for _ in range(warmup):
            m2d.grad = None; con.grad = None; col.grad = None; opa.grad = None
            # means2d.absgrad is set by the backward kernel
            rc, ra = rasterize_to_pixels(
                m2d, con, col, opa, W, H, tile_size,
                isect_offsets, flatten_ids, packed=False, absgrad=True,
            )
            torch.autograd.backward((rc, ra), (v_rc, v_ra))
            torch.cuda.synchronize()

        # Timed
        bwd_times = []
        for _ in range(timed):
            m2d.grad = None; con.grad = None; col.grad = None; opa.grad = None
            rc, ra = rasterize_to_pixels(
                m2d, con, col, opa, W, H, tile_size,
                isect_offsets, flatten_ids, packed=False, absgrad=True,
            )

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.autograd.backward((rc, ra), (v_rc, v_ra))
            e.record()
            torch.cuda.synchronize()
            bwd_times.append(s.elapsed_time(e))

        bwd_times = np.array(bwd_times)
        rep_r = {
            "rep": rep,
            "raster_bwd_mean_ms": float(bwd_times.mean()),
            "raster_bwd_median_ms": float(np.median(bwd_times)),
            "raster_bwd_std_ms": float(bwd_times.std()),
            "raster_bwd_p5_ms": float(np.percentile(bwd_times, 5)),
            "raster_bwd_p95_ms": float(np.percentile(bwd_times, 95)),
            "n_warmup": warmup,
            "n_timed": timed,
        }
        all_reps.append(rep_r)
        print(f"    raster_bwd={rep_r['raster_bwd_mean_ms']:.3f}ms", flush=True)
        torch.cuda.empty_cache()

    # Free precomputed tensors to avoid OOM in E2E section
    del m2d, con, col, opa, means2d, conics, depths, radii, colors, opacities_2d
    del isect_ids, flatten_ids, isect_offsets, tiles_per_gauss
    del compensations, dirs, masks, campos
    torch.cuda.empty_cache()

    # ---- Phase 3: E2E timing (full forward + backward via high-level API) ----
    e2e_reps = []
    means3d.requires_grad_(True)
    quats.requires_grad_(True)
    scales.requires_grad_(True)
    opacities_raw.requires_grad_(True)
    shs.requires_grad_(True)

    for rep in range(n_reps):
        # Warmup
        for _ in range(warmup):
            means3d.grad = None; quats.grad = None; scales.grad = None
            opacities_raw.grad = None; shs.grad = None
            rc, ra, meta = rasterization(
                means=means3d, quats=quats, scales=scales, opacities=torch.sigmoid(opacities_raw),
                colors=shs.unsqueeze(0), viewmats=viewmat, Ks=K,
                width=W, height=H, tile_size=16, packed=False,
                sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
                render_mode="RGB", absgrad=True,
            )
            torch.autograd.backward((rc, ra), (v_rc, v_ra))
            torch.cuda.synchronize()

        e2e_times = []
        bwd_times_e2e = []
        for _ in range(timed):
            means3d.grad = None; quats.grad = None; scales.grad = None
            opacities_raw.grad = None; shs.grad = None
            s_e = torch.cuda.Event(enable_timing=True)
            e_e = torch.cuda.Event(enable_timing=True)
            s_b = torch.cuda.Event(enable_timing=True)
            e_b = torch.cuda.Event(enable_timing=True)

            s_e.record()
            rc, ra, meta = rasterization(
                means=means3d, quats=quats, scales=scales, opacities=torch.sigmoid(opacities_raw),
                colors=shs.unsqueeze(0), viewmats=viewmat, Ks=K,
                width=W, height=H, tile_size=16, packed=False,
                sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
                render_mode="RGB", absgrad=True,
            )
            e_e.record()
            torch.cuda.synchronize()
            fwd_ms = s_e.elapsed_time(e_e)

            s_b.record()
            torch.autograd.backward((rc, ra), (v_rc, v_ra))
            e_b.record()
            torch.cuda.synchronize()
            bwd_ms = s_b.elapsed_time(e_b)

            e2e_times.append(fwd_ms + bwd_ms)
            bwd_times_e2e.append(bwd_ms)

        e2e_times = np.array(e2e_times)
        bwd_times_e2e = np.array(bwd_times_e2e)
        e2e_r = {
            "rep": rep,
            "e2e_mean_ms": float(e2e_times.mean()),
            "e2e_median_ms": float(np.median(e2e_times)),
            "e2e_std_ms": float(e2e_times.std()),
            "e2e_p5_ms": float(np.percentile(e2e_times, 5)),
            "e2e_p95_ms": float(np.percentile(e2e_times, 95)),
            "total_bwd_mean_ms": float(bwd_times_e2e.mean()),
            "fwd_mean_ms": float((e2e_times - bwd_times_e2e).mean()),
        }
        e2e_reps.append(e2e_r)
        print(f"    e2e={e2e_r['e2e_mean_ms']:.3f}ms total_bwd={e2e_r['total_bwd_mean_ms']:.3f}ms", flush=True)
        torch.cuda.empty_cache()

    raster_bwd_means = [r["raster_bwd_mean_ms"] for r in all_reps]
    e2e_means = [r["e2e_mean_ms"] for r in e2e_reps]
    total_bwd_means = [r["total_bwd_mean_ms"] for r in e2e_reps]
    fwd_means = [r["fwd_mean_ms"] for r in e2e_reps]

    final = {
        "checkpoint": ckpt_path,
        "N": N,
        "resolution": f"{W}x{H}",
        "n_cameras": 1,
        "warmup": warmup,
        "timed": timed,
        "n_reps": n_reps,
        "raster_bwd_reps": all_reps,
        "e2e_reps": e2e_reps,
        "raster_bwd_mean_ms": float(np.mean(raster_bwd_means)),
        "raster_bwd_median_ms": float(np.median(raster_bwd_means)),
        "raster_bwd_std_ms": float(np.std(raster_bwd_means)),
        "raster_bwd_p5_ms": float(np.percentile(raster_bwd_means, 5)),
        "raster_bwd_p95_ms": float(np.percentile(raster_bwd_means, 95)),
        "total_bwd_mean_ms": float(np.mean(total_bwd_means)),
        "fwd_mean_ms": float(np.mean(fwd_means)),
        "e2e_mean_ms": float(np.mean(e2e_means)),
        "e2e_median_ms": float(np.median(e2e_means)),
        "e2e_std_ms": float(np.std(e2e_means)),
    }
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  Saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
'''


def run_timing(gsplat_path, ckpt_path, output_path, config):
    runner_path = output_path.replace(".json", "_runner.py")
    with open(runner_path, "w") as f:
        f.write(TIMING_RUNNER)
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, runner_path, gsplat_path, ckpt_path,
         output_path, json.dumps(config)],
        capture_output=True, text=True, env=env, timeout=1800
    )
    print(f"  stdout: {result.stdout[-500:]}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[-1000:]}")
        return None
    with open(output_path) as f:
        return json.load(f)


def main():
    work = "/tmp/gsplat_n2_cpcb"
    baseline_path = os.path.join(work, "gsplat_baseline")
    cpcb_path = os.path.join(work, "gsplat_cpcb")
    results_dir = os.path.expanduser("~/3dgs-renderer-benchmark/results/n2_cpcb")
    timing_dir = os.path.join(results_dir, "timing_output_v2")
    os.makedirs(timing_dir, exist_ok=True)

    checkpoints = {
        "5K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_5000.pt"),
        "15K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_15000.pt"),
        "30K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
    }

    config = {"warmup": 50, "timed": 300, "n_reps": 3}

    all_timing = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"R2 TIMING (v2): Room {ckpt_name}")
        print(f"{'='*60}")

        for rep in range(config["n_reps"]):
            if rep % 2 == 0:
                order = [("baseline", baseline_path), ("cpcb", cpcb_path)]
            else:
                order = [("cpcb", cpcb_path), ("baseline", baseline_path)]

            for label, gp in order:
                print(f"\n  {ckpt_name} rep {rep+1}/{config['n_reps']} - {label}")
                out_file = os.path.join(timing_dir, f"{ckpt_name}_{label}_rep{rep}.json")
                if os.path.exists(out_file):
                    print(f"    exists, skip")
                    continue
                r = run_timing(gp, ckpt_path, out_file, config)
                if r:
                    print(f"    raster_bwd={r['raster_bwd_mean_ms']:.3f}ms e2e={r['e2e_mean_ms']:.3f}ms")

        for label in ["baseline", "cpcb"]:
            rep_files = sorted([f for f in os.listdir(timing_dir)
                                if f.startswith(f"{ckpt_name}_{label}_rep") and f.endswith(".json")])
            if not rep_files:
                continue
            rbwd_means, e2e_means, tbwd_means, fwd_means, all_reps_data = [], [], [], [], []
            for rf in rep_files:
                with open(os.path.join(timing_dir, rf)) as f:
                    d = json.load(f)
                rbwd_means.append(d["raster_bwd_mean_ms"])
                e2e_means.append(d["e2e_mean_ms"])
                tbwd_means.append(d["total_bwd_mean_ms"])
                fwd_means.append(d["fwd_mean_ms"])
                all_reps_data.append(d)

            key = f"room_{ckpt_name}_{label}"
            all_timing[key] = {
                "raster_bwd_mean_ms": float(np.mean(rbwd_means)),
                "raster_bwd_median_ms": float(np.median(rbwd_means)),
                "raster_bwd_std_ms": float(np.std(rbwd_means)),
                "raster_bwd_p5_ms": float(np.percentile(rbwd_means, 5)),
                "raster_bwd_p95_ms": float(np.percentile(rbwd_means, 95)),
                "total_bwd_mean_ms": float(np.mean(tbwd_means)),
                "fwd_mean_ms": float(np.mean(fwd_means)),
                "e2e_mean_ms": float(np.mean(e2e_means)),
                "e2e_median_ms": float(np.median(e2e_means)),
                "e2e_std_ms": float(np.std(e2e_means)),
                "n_reps": len(rep_files),
                "reps": all_reps_data,
            }
            print(f"  {key}: raster_bwd={all_timing[key]['raster_bwd_mean_ms']:.3f}ms e2e={all_timing[key]['e2e_mean_ms']:.3f}ms")

    # Compute comparison
    summary = {}
    for ckpt_name in ["5K", "15K", "30K"]:
        bk = f"room_{ckpt_name}_baseline"
        ck = f"room_{ckpt_name}_cpcb"
        if bk in all_timing and ck in all_timing:
            brb = all_timing[bk]["raster_bwd_mean_ms"]
            crb = all_timing[ck]["raster_bwd_mean_ms"]
            btwd = all_timing[bk]["total_bwd_mean_ms"]
            ctwd = all_timing[ck]["total_bwd_mean_ms"]
            be = all_timing[bk]["e2e_mean_ms"]
            ce = all_timing[ck]["e2e_mean_ms"]

            raster_bwd_reduction = 1 - crb / brb if brb > 0 else 0
            raster_bwd_speedup = brb / crb if crb > 0 else 0

            comp = {
                "baseline_raster_bwd_ms": brb,
                "cpcb_raster_bwd_ms": crb,
                "raster_bwd_reduction_pct": raster_bwd_reduction * 100,
                "raster_bwd_speedup": raster_bwd_speedup,
                "baseline_total_bwd_ms": btwd,
                "cpcb_total_bwd_ms": ctwd,
                "total_bwd_reduction_pct": (1 - ctwd / btwd) * 100 if btwd > 0 else 0,
                "baseline_e2e_ms": be,
                "cpcb_e2e_ms": ce,
                "e2e_reduction_pct": (1 - ce / be) * 100 if be > 0 else 0,
            }
            all_timing[f"room_{ckpt_name}_comparison"] = comp
            summary[ckpt_name] = comp

            # Decision gate
            if raster_bwd_reduction * 100 < 2:
                gate = "DROP"
            elif raster_bwd_reduction * 100 < 5:
                gate = "KEEP_SYSTEMS"
            elif raster_bwd_reduction * 100 < 10:
                gate = "STRONG_KEEP"
            else:
                gate = "PROMOTE"
            comp["gate"] = gate
            print(f"\n  {ckpt_name} GATE: {gate} (raster_bwd reduction={raster_bwd_reduction*100:.2f}%)")

    for ckpt_name in ["5K", "15K", "30K"]:
        out_file = os.path.join(results_dir, f"room_{ckpt_name.lower()}_timing.json")
        d = {}
        for label in ["baseline", "cpcb"]:
            k = f"room_{ckpt_name}_{label}"
            if k in all_timing:
                d[label] = all_timing[k]
        ck = f"room_{ckpt_name}_comparison"
        if ck in all_timing:
            d["comparison"] = all_timing[ck]
        with open(out_file, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Saved {out_file}")

    with open(os.path.join(results_dir, "e2e_timing.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved e2e_timing.json")


if __name__ == "__main__":
    main()
