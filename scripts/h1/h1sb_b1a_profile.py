#!/usr/bin/env python3
"""H1-SB: B1A (clean gsplat + True AccuTile) profiling.

Uses the true-accutile tree at /mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153.
Runs rasterization() with accutile=True, packed=False (AccuTile requires packed=False).
Same checkpoints, cameras, resolution, warmup/measure as H1.
"""
import os, sys, json, csv, math, gc, argparse
import torch
import numpy as np

# Use the true-accutile tree
ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, ACCUTILE_TREE)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from plyfile import PlyData
import gsplat
print("gsplat file:", gsplat.__file__)
print("gsplat version:", gsplat.__version__)

from gsplat.rendering import rasterization

SH_DEGREE = 3

SCENES = {
    "train": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json",
        "native_w": 1959, "native_h": 1090,
    },
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "native_w": 3114, "native_h": 2075,
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "native_w": 4946, "native_h": 3286,
    },
}

def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    def arr(name): return torch.tensor(v[name], dtype=torch.float32, device=device)
    means = torch.stack([arr("x"), arr("y"), arr("z")], dim=-1)
    sh0 = torch.stack([arr("f_dc_0"), arr("f_dc_1"), arr("f_dc_2")], dim=-1).unsqueeze(1)
    opacities = torch.sigmoid(arr("opacity"))
    scales = torch.stack([arr("scale_0"), arr("scale_1"), arr("scale_2")], dim=-1)
    quats = torch.stack([arr("rot_0"), arr("rot_1"), arr("rot_2"), arr("rot_3")], dim=-1)
    K_SH = 16
    f_rest = []
    for i in range(1, K_SH):
        f_rest.append(torch.stack([arr("f_rest_%d" % (3*(i-1)+j)) for j in range(3)], dim=-1))
    f_rest = torch.stack(f_rest, dim=1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = sh0.squeeze(1)
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh

def load_cameras(cams_path, width, height, device):
    cams = json.load(open(cams_path))
    for cam in cams:
        scale = width / float(cam["width"])
        cam["fx_scaled"] = float(cam["fx"]) * scale
        cam["fy_scaled"] = float(cam["fy"]) * scale
    return cams

def make_viewmat_K(cam, width, height, device):
    R = np.asarray(cam["rotation"], dtype=np.float64)
    p = np.asarray(cam["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
    scale = width / float(cam["width"])
    K = np.array([[float(cam["fx"])*scale, 0, (width-1)/2],
                  [0, float(cam["fy"])*scale, (height-1)/2], [0,0,1]], dtype=np.float64)
    vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return vm, K

def time_repeated(fn, warmup, measure, label=""):
    times = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(measure):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_measure": measure,
    }

def profile_b1a(means, quats, scales, opacities, sh, vm, K, width, height, device,
                warmup, measure, cam_id):
    """Profile B1A (accutile=True) for one camera."""
    # Forward only
    def fwd_fn():
        with torch.no_grad():
            out = rasterization(
                means=means.unsqueeze(0), quats=quats.unsqueeze(0),
                scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
                colors=sh.unsqueeze(0), viewmats=vm, Ks=K, width=width, height=height,
                sh_degree=SH_DEGREE, packed=False, radius_clip=0.0,
                accutile=True,
            )
        return out[0], out[1]

    # Get render for correctness
    render, alpha = fwd_fn()
    render_save = render.detach().clone()

    # Time forward
    fwd_timing = time_repeated(fwd_fn, warmup, measure, "B1A_fwd")

    # Forward + backward
    target = render_save.detach()

    def fb_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        out = rasterization(
            means=m.unsqueeze(0), quats=q.unsqueeze(0), scales=s.unsqueeze(0),
            opacities=o.unsqueeze(0), colors=c.unsqueeze(0),
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=SH_DEGREE, packed=False, radius_clip=0.0,
            accutile=True,
        )
        loss = (out[0] - target).pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()

    fb_timing = time_repeated(fb_fn, warmup, measure, "B1A_fb")
    bwd_ms = fb_timing["median_ms"] - fwd_timing["median_ms"]

    # Workload metrics
    with torch.no_grad():
        out = rasterization(
            means=means.unsqueeze(0), quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
            colors=sh.unsqueeze(0), viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=SH_DEGREE, packed=False, radius_clip=0.0,
            accutile=True,
        )
    info = out[2] if len(out) > 2 else {}
    radii = info.get("radii", None)
    n_visible = int(((radii > 0).any(dim=-1)).sum().item()) if radii is not None else -1
    tiles_per_gauss = info.get("tiles_per_gauss", None)
    n_isects = int(tiles_per_gauss.sum().item()) if tiles_per_gauss is not None else -1
    n_tiles = math.ceil(width / 16) * math.ceil(height / 16)

    return {
        "camera_id": cam_id,
        "width": width, "height": height,
        "B1A_forward_ms": fwd_timing["median_ms"],
        "B1A_forward_timing": fwd_timing,
        "B1A_fwd_bwd_ms": fb_timing["median_ms"],
        "B1A_fwd_bwd_timing": fb_timing,
        "B1A_backward_ms": bwd_ms,
        "B1A_render": render_save.reshape(-1)[:10].tolist(),  # first 10 pixels for correctness
        "workload": {
            "N_total": int(means.shape[0]),
            "N_visible": n_visible,
            "N_isects": n_isects,
            "total_tiles": n_tiles,
            "mean_tiles_per_gauss": float(tiles_per_gauss.float().mean().item()) if tiles_per_gauss is not None else -1,
            "accutile": True,
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="train,room,bicycle")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--max-long-side", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda:0"
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = args.scenes.split(",")
    all_results = {}

    for scene_name in scenes:
        cfg = SCENES[scene_name]
        nw, nh = cfg["native_w"], cfg["native_h"]
        if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
            scale = args.max_long_side / max(nw, nh)
            width = max(1, int(round(nw * scale)))
            height = max(1, int(round(nh * scale)))
        else:
            width, height = nw, nh
        print("\n[scene] %s %dx%d (native %dx%d)" % (scene_name, width, height, nw, nh))

        means, quats, scales, opacities, sh = load_ply(cfg["ply"], device)
        cams = load_cameras(cfg["cams"], width, height, device)
        n_cams = len(cams)
        cam_indices = [0, n_cams // 2, n_cams - 1]
        print("[cams] selected: %s" % cam_indices)

        scene_results = {}
        for ci in cam_indices:
            cam = cams[ci]
            vm, K = make_viewmat_K(cam, width, height, device)
            print("[profile] %s cam %d..." % (scene_name, ci))
            r = profile_b1a(means, quats, scales, opacities, sh, vm, K,
                           width, height, device, args.warmup, args.measure, cam.get("id", str(ci)))
            r["scene"] = scene_name
            r["camera_idx"] = ci
            scene_results[ci] = r
            fname = "%s_cam%d.json" % (scene_name, ci)
            with open(os.path.join(args.out_dir, fname), "w") as f:
                json.dump(r, f, indent=2, default=str)
            print("  B1A_fwd=%.3fms B1A_fb=%.3fms B1A_bwd=%.3fms N_isects=%d" % (
                r["B1A_forward_ms"], r["B1A_fwd_bwd_ms"], r["B1A_backward_ms"],
                r["workload"]["N_isects"]))
            gc.collect(); torch.cuda.empty_cache()

        all_results[scene_name] = scene_results

    # Save combined
    with open(os.path.join(args.out_dir, "all_b1a_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n[DONE] All B1A results saved to %s" % args.out_dir)

if __name__ == "__main__":
    main()