#!/usr/bin/env python3
"""EPIC-05 P7 fixed-state replay: renderer intrinsic execution timing.

Per (scene, arm) invocation (separate process per arm -- pybind isolation):
  arms: b1a (AccuTile tree + b1a ext), a0 (c0 ext + HIGS_DISABLE_F9=1),
        c0  (c0 ext, frozen substrate env)
  scenes: room / bicycle / garden
  state: frozen {scene}_30000.ply (identical Gaussian state for all arms),
         cam0, 2048-max-side, eps2d=0.3 for ALL arms (matched substrate)
  measures (CUDA events, 20 warmup, 100 samples x 5 reps, median of rep
  medians):
    forward_only / direct_backward / forward_plus_backward
  NO optimizer, NO densification, NO parameter update.

Output: p7_fixed_state/{scene}/p7_replay_{scene}_{arm}.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["b1a", "a0", "c0"], required=True)
ap.add_argument("--scene", required=True, choices=["room", "bicycle", "garden"])
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--eps2d", type=float, default=0.3)
ap.add_argument("--state", choices=["ply", "ckpt"], default="ply",
                help="ply: frozen MX reference 30K state (P5/P6 protocol); "
                     "ckpt: the matched b1ae03_s42 30K final state from THIS "
                     "campaign (training-final population; identical file for "
                     "all arms)")
ap.add_argument("--outdir", default="/root/3dgs-publication-epic05/p7_fixed_state")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

REPO = "/root/3dgs-renderer-benchmark"
FROZEN = "/root/epic05_frozen"
CACHE = "/root/epic05_build_cache"
C0_SO = f"{CACHE}/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = f"{CACHE}/core/gsplat_cuda/gsplat_cuda.so"
SCENE_DIR = f"{CACHE}/scene/gsplat_scene_cuda"
B1A_SO = f"{CACHE}/b1a_ext/gsplat_cuda_final30k/gsplat_cuda_final30k.so"
ACCUTILE_TREE = f"{FROZEN}/gsplat-true-accutile-v153"
C0_WT = f"{FROZEN}/higs_c0_final30k_worktree"

# frozen substrate env (identical to the C0 30K runs)
os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
os.environ["HIGS_BWD_ABSGRAD"] = "1"
os.environ.pop("HIGS_BWD_H8_MR", None)
os.environ.pop("HIGS_DISABLE_F9", None)
if args.arm == "a0":
    os.environ["HIGS_DISABLE_F9"] = "1"

import torch  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ext_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def main():
    so_sha = sha256_file(B1A_SO if args.arm == "b1a" else C0_SO)
    if args.arm == "b1a":
        mod = load_ext_module("gsplat_cuda_final30k", B1A_SO)
        sys.modules["gsplat.csrc"] = mod
        sys.path.insert(0, ACCUTILE_TREE)
        sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
        sys.path.insert(0, f"{REPO}/src")
        sys.path.insert(0, f"{REPO}/baseline/reference_v1")
        import gsplat
        import inspect
        sig = inspect.signature(gsplat.rasterization)
        assert "accutile" in sig.parameters, "accutile param missing -- wrong tree"
        from gsplat import rasterization

        def fwd():
            r, _, meta = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=sh,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=3, radius_clip=0.0,
                eps2d=args.eps2d, render_mode="RGB", absgrad=True, accutile=True)
            return r[0]
    else:
        sys.path.insert(0, C0_WT)
        sys.path.insert(0, SCENE_DIR)
        core = load_ext_module("gsplat_cuda", CORE_SO)
        sys.modules["gsplat.csrc"] = core
        exp = load_ext_module(
            "experimental_gaussian_render_inference_scene_cuda", C0_SO)
        sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
        sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
        sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
        sys.path.insert(0, f"{REPO}/src")
        sys.path.insert(0, f"{REPO}/baseline/reference_v1")
        from gsplat.experimental import rasterize_gaussian_higs_dynamic

        def fwd():
            out = rasterize_gaussian_higs_dynamic(
                means, quats, scales, opacities, sh,
                sh_degree=3,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB", eps2d=args.eps2d)
            return out["frame"].reshape(cam.image_height, cam.image_width, 3)

    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

    dev = torch.cuda.get_device_properties(0)
    print(f"P7 replay scene={args.scene} arm={args.arm} gpu={dev.name} state={args.state}")

    if args.state == "ply":
        state_path = f"{FROZEN}/p5_reference_plys/{args.scene}_30000.ply"
        ply = load_ply(state_path, device="cuda")
        state_sha = sha256_file(state_path)
    else:
        state_path = (f"/root/3dgs-publication-epic05/full30k/"
                      f"b1ae03_s42_{args.scene}/ckpt_30000.pt")
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        state_sha = sha256_file(state_path)
        assert ck["arm"] == "b1a" and int(ck["iteration"]) == 30000
        ply = {
            "xyz": ck["xyz"].cuda(),
            "rotations": ck["rotation"].cuda(),
            # ckpt stores raw log/logit params, same convention as load_ply
            "scales": ck["scaling"].cuda(),
            "opacity": ck["opacity"].cuda(),
            "shs": ck["shs"].cuda(),
        }

    cams = load_cameras_from_json(
        f"{REPO}/data/official/mipnerf360/{args.scene}/cameras.json", device="cpu")
    c0 = cams[0]
    w, h = c0.image_width, c0.image_height
    s = 2048.0 / max(w, h)
    tw, th = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    cams = resize_cameras(cams, tw, th)
    cam = cams[0]
    for attr in ("viewmatrix", "K"):
        setattr(cam, attr, getattr(cam, attr).to("cuda"))

    means = ply["xyz"].clone().requires_grad_(True)
    quats = torch.nn.functional.normalize(
        ply["rotations"].clone(), dim=-1).requires_grad_(True)
    scales = torch.exp(ply["scales"].clone()).requires_grad_(True)
    opacities = torch.sigmoid(ply["opacity"].clone()).requires_grad_(True)
    sh = ply["shs"].clone().contiguous().requires_grad_(True)
    n = means.shape[0]
    leaves = [means, quats, scales, opacities, sh]

    g = torch.Generator(device="cuda").manual_seed(4200)
    V = torch.randn((th, tw, 3), generator=g, device="cuda")

    def zero():
        for t in leaves:
            t.grad = None

    # ---- reference pass: loss + render checksum (informational) ----
    zero()
    frame = fwd()
    loss = (frame * V).sum()
    loss.backward()
    render_sum = float(frame.detach().sum().item())
    loss_v = float(loss.item())
    del frame, loss
    zero()

    # ---- warmup ----
    for _ in range(args.warmup):
        zero()
        f = fwd()
        (f * V).sum().backward()
    torch.cuda.synchronize()

    # ---- timing: 3 modes x reps x samples ----
    modes = ("forward_only", "direct_backward", "forward_plus_backward")
    all_reps = {m: [] for m in modes}
    for _ in range(args.reps):
        for mode in modes:
            times = []
            for _s in range(args.samples):
                zero()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                if mode == "direct_backward":
                    f = fwd()
                    l = (f * V).sum()
                    torch.cuda.synchronize()
                    e0.record()
                    l.backward()
                else:
                    torch.cuda.synchronize()
                    e0.record()
                    f = fwd()
                    if mode == "forward_plus_backward":
                        (f * V).sum().backward()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
            all_reps[mode].append(statistics.median(times))

    out = {
        "experiment": "EPIC05_P7_FIXED_STATE_REPLAY",
        "scene": args.scene,
        "arm": args.arm,
        "eps2d": args.eps2d,
        "n_gaussians": int(n),
        "resolution": [tw, th],
        "gpu_name": dev.name,
        "so_sha256": so_sha,
        "env": {k: os.environ.get(k) for k in
                ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_ABSGRAD",
                 "HIGS_BWD_H8_MR", "HIGS_DISABLE_F9")},
        "loss_reference": loss_v,
        "render_sum_reference": render_sum,
        "protocol": {"warmup": args.warmup, "samples": args.samples,
                     "reps": args.reps, "cotangent_seed": 4200,
                     "state_tier": args.state,
                     "state": state_path,
                     "state_sha256": state_sha,
                     "camera": "cam0, 2048-max-side resize (P5 protocol)"},
        "timing_ms": {
            m: {
                "median": statistics.median(all_reps[m]),
                "mean": statistics.fmean(all_reps[m]),
                "min": min(all_reps[m]),
                "rep_medians_ms": all_reps[m],
            } for m in modes
        },
    }
    os.makedirs(f"{args.outdir}/{args.scene}", exist_ok=True)
    path = f"{args.outdir}/{args.scene}/p7_replay_{args.scene}_{args.arm}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {path}")
    for m in modes:
        print(f"  {m:22s} {out['timing_ms'][m]['median']:.3f} ms")


if __name__ == "__main__":
    main()
