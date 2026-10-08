#!/usr/bin/env python3
"""EPIC-05 P8 fixed-state replay: per (scene, arm) dump / timing phases on
the P7 30K b1a frozen states. Decomposes the end-to-end P8A ratios into
intrinsic per-iteration renderer speed at IDENTICAL state (Q5 support).

Arms:
  b1a  : frozen publication B1A (AccuTile), eps2d=0.3
  a0   : frozen A0 (c0 ext, HIGS_DISABLE_F9=1, substrate env)
  c0   : frozen C0 system (c0 ext, H8_MR=1, F9 on -- the publication system)
  gmain: gsplat main @ 512d366b, standard rasterization() path, eps2d=0.3
Scenes: room / bicycle / garden.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["b1a", "a0", "c0", "gmain"], required=True)
ap.add_argument("--scene", required=True, choices=["room", "bicycle", "garden"])
ap.add_argument("--phase", choices=["dump", "timing", "loop"], required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--samples", type=int, default=100)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--eps2d", type=float, default=0.3)
ap.add_argument("--outroot", default="/root/3dgs-publication-epic05/p8_replay")
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
GMAIN_TREE = "/root/epic05_p8/gsplat-main-512d366b"
GMAIN_SO = "/root/epic05_p8/build/torchext/gsplat_cuda/gsplat_cuda.so"

# base substrate env; per-arm overrides BELOW follow the trainer exactly
os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
os.environ["HIGS_BWD_ABSGRAD"] = "1"
os.environ["HIGS_BWD_H8_MR"] = "1"
os.environ.pop("HIGS_DISABLE_F9", None)
if args.arm == "a0":
    os.environ["HIGS_DISABLE_F9"] = "1"
    os.environ.pop("HIGS_BWD_H8_MR", None)

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


def setup_arm():
    """Returns (fwd, so_sha, extra_meta). fwd() -> (frame, alpha, means2d, meta)."""
    if args.arm == "b1a":
        so_sha = sha256_file(B1A_SO)
        mod = load_ext_module("gsplat_cuda_final30k", B1A_SO)
        sys.modules["gsplat.csrc"] = mod
        sys.path.insert(0, ACCUTILE_TREE)
        sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
        sys.path.insert(0, f"{REPO}/src")
        sys.path.insert(0, f"{REPO}/baseline/reference_v1")
        import gsplat
        import inspect
        assert "accutile" in inspect.signature(gsplat.rasterization).parameters
        from gsplat import rasterization

        def fwd():
            r, alpha, meta = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=sh,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=3, radius_clip=0.0,
                eps2d=args.eps2d, render_mode="RGB", absgrad=True,
                accutile=True)
            return r[0], alpha[0], meta["means2d"], meta
    elif args.arm == "gmain":
        so_sha = sha256_file(GMAIN_SO)
        mod = load_ext_module("gsplat_cuda", GMAIN_SO)
        sys.modules["gsplat_cuda"] = mod
        sys.modules["gsplat.csrc"] = mod
        sys.path.insert(0, GMAIN_TREE)
        sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
        sys.path.insert(0, f"{REPO}/src")
        sys.path.insert(0, f"{REPO}/baseline/reference_v1")
        import gsplat
        import inspect
        sig = inspect.signature(gsplat.rasterization)
        assert "accutile" not in sig.parameters
        assert "absgrad" in sig.parameters
        assert sig.parameters["rasterize_mode"].default == "classic"
        from gsplat import rasterization

        def fwd():
            r, alpha, meta = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=sh,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=3, radius_clip=0.0,
                eps2d=args.eps2d, render_mode="RGB", absgrad=True)
            return r[0], alpha[0], meta["means2d"], meta
    else:  # a0 / c0 (same binary, env-switched)
        so_sha = sha256_file(C0_SO)
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
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            alpha = out.get("alpha")
            if alpha is not None:
                alpha = alpha.reshape(cam.image_height, cam.image_width, 1)
                if alpha.dim() == 3:
                    alpha = alpha.unsqueeze(-1)
            info = out["densification_info"]
            return frame, alpha, info["means2d"], out
    return fwd, so_sha, {}


def main():
    global means, quats, scales, opacities, sh, cam
    for p in (f"{REPO}/scripts/epic05/phase7", f"{REPO}/src",
              f"{REPO}/baseline/reference_v1"):
        if p not in sys.path:
            sys.path.insert(0, p)
    fwd, so_sha, _ = setup_arm()
    from benchmark_framework import load_cameras_from_json, resize_cameras

    dev = torch.cuda.get_device_properties(0)
    state_path = (f"/root/3dgs-publication-epic05/full30k/"
                  f"b1ae03_s42_{args.scene}/ckpt_30000.pt")
    state_sha = sha256_file(state_path)
    cam_json = f"{REPO}/data/official/mipnerf360/{args.scene}/cameras.json"
    cam_json_sha = sha256_file(cam_json)

    ck = torch.load(state_path, map_location="cpu", weights_only=False)
    assert ck["arm"] == "b1a" and int(ck["iteration"]) == 30000
    means = ck["xyz"].cuda().requires_grad_(True)
    quats = torch.nn.functional.normalize(
        ck["rotation"].cuda(), dim=-1).requires_grad_(True)
    scales = torch.exp(ck["scaling"].cuda()).requires_grad_(True)
    opacities = torch.sigmoid(ck["opacity"].cuda()).requires_grad_(True)
    sh = ck["shs"].cuda().contiguous().requires_grad_(True)
    n = means.shape[0]
    leaves = [means, quats, scales, opacities, sh]

    cams = load_cameras_from_json(cam_json, device="cpu")
    c0 = cams[0]
    w, h = c0.image_width, c0.image_height
    s = 2048.0 / max(w, h)
    tw, th = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    cams = resize_cameras(cams, tw, th)
    cam = cams[0]
    for attr in ("viewmatrix", "K"):
        setattr(cam, attr, getattr(cam, attr).to("cuda"))
    cam_hash = hashlib.sha256(
        cam.viewmatrix.detach().cpu().numpy().tobytes()
        + cam.K.detach().cpu().numpy().tobytes()
        + f"{tw}x{th}".encode()).hexdigest()

    g = torch.Generator(device="cuda").manual_seed(4200)
    V = torch.randn((th, tw, 3), generator=g, device="cuda")

    def zero():
        for t in leaves:
            t.grad = None

    base = {
        "experiment": "EPIC05_P8_REPLAY",
        "arm": args.arm,
        "scene": args.scene,
        "phase": args.phase,
        "eps2d": args.eps2d,
        "n_gaussians": int(n),
        "resolution": [tw, th],
        "gpu_name": dev.name,
        "so_sha256": so_sha,
        "state_path": state_path,
        "state_sha256": state_sha,
        "camera_json_sha256": cam_json_sha,
        "camera_hash": cam_hash,
        "env": {k: os.environ.get(k) for k in
                ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_ABSGRAD",
                 "HIGS_BWD_H8_MR", "HIGS_DISABLE_F9")},
    }
    outdir = f"{args.outroot}/{args.scene}"
    os.makedirs(outdir, exist_ok=True)

    if args.phase == "dump":
        zero()
        frame, alpha, means2d, meta = fwd()
        means2d.retain_grad()
        loss = (frame * V).sum()
        loss.backward()
        grads = {
            "means": means.grad.detach().clone(),
            "quats": quats.grad.detach().clone(),
            "scales": scales.grad.detach().clone(),
            "opacities": opacities.grad.detach().clone(),
            "sh": sh.grad.detach().clone(),
        }
        m2d_grad = getattr(means2d, "grad", None)
        if m2d_grad is not None:
            grads["means2d"] = m2d_grad.detach().clone()
        ab = getattr(means2d, "absgrad", None)
        if ab is not None:
            grads["absgrad"] = ab.detach().clone()
        finite_ok = all(bool(torch.isfinite(v).all()) for v in grads.values())
        finite_ok = finite_ok and bool(torch.isfinite(frame).all())
        meta_out = dict(base)
        meta_out.update({
            "loss": float(loss.item()),
            "render_sum": float(frame.sum().item()),
            "render_mean": float(frame.mean().item()),
            "render_finite": bool(torch.isfinite(frame).all()),
            "alpha_sum": (float(alpha.sum().item())
                          if alpha is not None else None),
            "grads_finite": finite_ok,
            "grad_keys": sorted(grads.keys()),
        })
        torch.save({"grads": grads,
                    "render": frame.detach().clone(),
                    "alpha": (alpha.detach().clone()
                              if alpha is not None else None),
                    "V": V.clone()},
                   f"{outdir}/dump_{args.scene}_{args.arm}.pt")
        with open(f"{outdir}/dumpmeta_{args.scene}_{args.arm}.json", "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"WROTE {outdir}/dump_{args.scene}_{args.arm}.pt")
        print(f"loss={meta_out['loss']:.4f} "
              f"render_sum={meta_out['render_sum']:.2f} "
              f"grads_finite={finite_ok}")
        return

    def one_iter(mode):
        if mode == "fwd":
            fwd()
        else:
            f, a, m2d, meta = fwd()
            l = (f * V).sum()
            l.backward()

    if args.phase == "loop":
        for _ in range(args.warmup):
            zero()
            one_iter("fb")
        torch.cuda.synchronize()
        import time
        for _ in range(10):
            zero()
            one_iter("fb")
        torch.cuda.synchronize()
        print("LOOP_DONE")
        return

    # ---------------- phase: timing ----------------
    for _ in range(args.warmup):
        zero()
        one_iter("fb")
    torch.cuda.synchronize()
    modes = ("fwd", "bwd", "fb")
    stats = {}
    raw = {}
    for mode in modes:
        reps = []
        for _ in range(args.reps):
            times = []
            for _s in range(args.samples):
                zero()
                if mode == "bwd":
                    f, a, m2d, meta = fwd()
                    l = (f * V).sum()
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    l.backward()
                    e1.record()
                    torch.cuda.synchronize()
                else:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    e0.record()
                    if mode == "fwd":
                        fwd()
                    else:
                        f, a, m2d, meta = fwd()
                        l = (f * V).sum()
                        l.backward()
                    e1.record()
                    torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
            q = sorted(times)
            reps.append({
                "median": statistics.median(times),
                "mean": statistics.fmean(times),
                "p25": q[len(q) // 4],
                "p75": q[(3 * len(q)) // 4],
                "min": q[0],
                "max": q[-1],
            })
        meds = [r["median"] for r in reps]
        stats[mode] = {
            "median_of_rep_medians": statistics.median(meds),
            "mean_of_rep_medians": statistics.fmean(meds),
            "rep_stats": reps,
        }
        raw[mode] = meds
    out = dict(base)
    out["timing_ms"] = stats
    out["raw_rep_medians"] = raw
    out["protocol"] = {"warmup": args.warmup, "samples": args.samples,
                       "reps": args.reps, "cotangent_seed": 4200}
    with open(f"{outdir}/timing_{args.scene}_{args.arm}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {outdir}/timing_{args.scene}_{args.arm}.json")
    for m in modes:
        print(f"  {m:4s} {stats[m]['median_of_rep_medians']:.3f} ms")


if __name__ == "__main__":
    main()
