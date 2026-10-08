#!/usr/bin/env python3
"""P12 support lane: per-arm runtime feature-switch probe (evidence only).

Runs in an isolated subprocess per arm (each arm needs its own binding of the
frozen extension module into sys.modules, so arms cannot share a process).

The probe runs on REAL frozen state: the room scene, the frozen SfM init plus
the P8 30K checkpoint (when present), and the first camera of the frozen
training sequence -- the same tensors the trainer hands to the renderer.

For GMAIN_FUSED (arm=gmain):
  * bootstrap asserts (no accutile kwarg, absgrad present, rasterize_mode
    default == classic) -- proves the pristine gsplat main tree is bound
  * one gsplat classic rasterization() at sh_degree=3 proving the classic path
    works and exposes NO HiGS/F9 concept in its metadata keys
  * backward grad finiteness / L2 norms through the classic autograd path

For the HiGS arms (c0 / a0 / a1):
  * F9 gate matrix: forward at sh_degree 0..3, recording
    out["metadata"]["f9_enabled"] -- F9's frozen contract requires
    sh_degree==3, so a 200-iteration preflight runs at sh_degree 0 where F9 is
    armed but cannot engage
  * main forward+backward at sh_degree=3 (contract degree) with the exact
    trainer call arguments; frame stats and per-parameter grad norms are
    recorded so A0 and A1 can be compared on identical inputs
  * adversarial env-consumption test (AFTER the records above): setting
    HIGS_BWD_H8_MR=1 without HIGS_BWD_SCALAR_ADJOINT must raise the frozen
    TORCH_CHECK -- proves the C++ kernel reads the switch env per launch
  * binary string census of the loaded .so (unique HIGS_* strings)

Writes one JSON per arm. No timing is recorded (this is not a benchmark).
No renderer/loss configuration is modified: the probe consumes the frozen
binaries and the frozen env switches exactly as the P12 arms do.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SWITCH_VARS = ("HIGS_DISABLE_F9", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR",
               "HIGS_BWD_ABSGRAD", "HIGS_PX_RUNTIME", "GSPLAT_H8_MR")
FULL30K = "/root/3dgs-publication-epic05/full30k"
TRAIN_CFG = {
    "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
    "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
    "rotation_lr": 0.001, "percent_dense": 0.01,
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def env_snapshot():
    return {k: os.environ.get(k) for k in SWITCH_VARS}


def so_string_census(path):
    """Unique HIGS_*/GSPLAT_H8_MR strings in a .so (binary patch evidence)."""
    try:
        out = subprocess.run(["strings", "-a", path], capture_output=True,
                             text=True, timeout=300).stdout
    except Exception as e:
        return {"error": str(e)}
    higs = sorted(set(re.findall(r"HIGS_[A-Z0-9_]+", out)))
    h8 = sorted(set(re.findall(r"GSPLAT_H8_MR", out)))
    f9 = sorted(set(re.findall(r"higs_gatherless_projected_producer", out)))
    return {"n_HIGS_strings": len(higs), "HIGS_strings": higs,
            "GSPLAT_H8_MR": h8, "higs_gatherless_projected_producer": f9}


def gpu_identity():
    props = None
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        props = {"name": p.name, "total_memory_gb": round(p.total_memory / 2**30, 2),
                 "uuid": str(getattr(p, "uuid", None)),
                 "multi_processor_count": p.multi_processor_count}
    except Exception as e:
        props = {"error": str(e)}
    smi = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=uuid,name,driver_version,clocks.sm,"
         "temperature.gpu,power.draw", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return {"torch_device_properties": props, "nvidia_smi_csv": smi,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def compute_scene_extent(dataset, n_samples=50):
    import numpy as np
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        centers.append(dataset.get_camera(i).camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            extent = max(extent, float(np.linalg.norm(centers[i] - centers[j])))
    return max(extent, 0.1)


def load_real_state(T, scene, ckpt_path, cam_pos=0):
    """Frozen SfM init + frozen checkpoint + frozen camera (trainer semantics)."""
    import numpy as np
    import torch
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset

    repo = T.REPO
    dataset = GTDataset(scene=scene, repo_root=repo, resolution="1080p",
                        device="cuda", background="black")
    extent = compute_scene_extent(dataset)
    config = ReferenceV1Config()
    sfm_path = os.path.join(repo, "data", "datasets", "mipnerf360", scene,
                            "sparse", "0", "points3D.bin")
    points3d = read_points3D_binary(sfm_path)
    pcd = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd, spatial_lr_scale=extent)
    model.training_setup(TRAIN_CFG)
    meta = {"scene": scene, "scene_extent": float(extent),
            "initial_N": int(model._xyz.shape[0]), "n_cameras": len(dataset),
            "state_source": None, "state_sha256": None, "state_iteration": None}
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.restore(state, TRAIN_CFG)
        meta["state_source"] = ckpt_path
        meta["state_sha256"] = sha256_file(ckpt_path)
        meta["state_iteration"] = state.get("iteration")
        meta["state_N"] = int(model._xyz.shape[0])
    seq_path = os.path.join(FULL30K, f"gmain03_s42_{scene}",
                            "camera_sequence.npy")
    if os.path.exists(seq_path):
        seq = np.load(seq_path)
        cam_idx = int(seq[cam_pos])
        meta["camera_sequence_sha256"] = sha256_file(seq_path)
    else:
        cam_idx = cam_pos
    cam, gt = dataset.get_item(cam_idx)
    meta["camera_index"] = cam_idx
    meta["image_wh"] = [int(cam.image_width), int(cam.image_height)]
    meta["active_sh_degree"] = int(model.active_sh_degree)
    return model, cam, gt, meta


def grad_report(torch, grads, names=("means", "quats", "scales", "opacities",
                                     "colors")):
    return {
        "grad_finite": {n: (g is not None and bool(torch.isfinite(g).all()))
                        for n, g in zip(names, grads)},
        "grad_l2_norm": {n: (float(torch.linalg.norm(g)) if g is not None else None)
                         for n, g in zip(names, grads)},
    }


def probe_gmain(T, scene, ckpt, out):
    import torch
    import inspect
    ident = T.bootstrap_gmain()
    import gsplat
    sig = inspect.signature(gsplat.rasterization)
    model, cam, gt, mmeta = load_real_state(T, scene, ckpt)
    rec = {
        "arm": "gmain",
        "bootstrap_ident": ident,
        "gsplat_file": gsplat.__file__,
        "rasterization_params": list(sig.parameters),
        "has_accutile_kwarg": "accutile" in sig.parameters,
        "rasterize_mode_default": sig.parameters["rasterize_mode"].default,
        "binary_census_gsplat_so": so_string_census(T.GMAIN_SO),
        "so_sha256": sha256_file(T.GMAIN_SO),
        "state": mmeta,
    }
    means, quats, scales, opacities, colors = (
        model.get_xyz, model.get_rotation, model.get_scaling,
        model.get_opacity, model.get_features)
    r, alpha, meta = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.3,
        render_mode="RGB", absgrad=True)
    fwd = r[0]
    meta_keys = sorted(meta.keys()) if isinstance(meta, dict) else None
    with torch.no_grad():
        fwd_stats = {"shape": list(fwd.shape),
                     "finite": bool(torch.isfinite(fwd).all()),
                     "mean": float(fwd.mean()), "min": float(fwd.min()),
                     "max": float(fwd.max()),
                     "n_radii_positive": int((meta["radii"] > 0).any(dim=-1).sum())}
    rec["forward"] = {**fwd_stats, "metadata_keys": meta_keys,
                      "metadata_has_f9": bool(meta_keys and any("f9" in k for k in meta_keys))}
    loss = (fwd.clamp(0, 1) * gt).mean()
    grads = torch.autograd.grad(loss, [means, quats, scales, opacities, colors],
                                allow_unused=True)
    rec["backward"] = {"loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
                       **grad_report(torch, grads)}
    del model
    torch.cuda.empty_cache()
    out.update(rec)


def probe_higs(T, arm, scene, ckpt, out):
    import torch
    if arm == "c0":
        ident = T.bootstrap_c0()
    else:
        ident = T.bootstrap_a(arm)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    model, cam, gt, mmeta = load_real_state(T, scene, ckpt)
    rec = {
        "arm": arm,
        "bootstrap_ident": ident,
        "env_after_bootstrap": env_snapshot(),
        "state": mmeta,
        "binary_census_ext_so": so_string_census(T.C0_SO),
        "binary_census_core_so": so_string_census(T.CORE_SO),
        "so_sha256_ext": sha256_file(T.C0_SO),
        "so_sha256_core": sha256_file(T.CORE_SO),
    }
    means, quats, scales, opacities, colors = (
        model.get_xyz, model.get_rotation, model.get_scaling,
        model.get_opacity, model.get_features)
    viewmats = cam.viewmatrix.unsqueeze(0).unsqueeze(0)
    Ks = cam.K.unsqueeze(0).unsqueeze(0)
    wh = (cam.image_width, cam.image_height)

    # ---- F9 gate matrix: forward at sh_degree 0..3 (no backward) ----
    gate_matrix = {}
    with torch.no_grad():
        for d in (0, 1, 2, 3):
            od = rasterize_gaussian_higs_dynamic(
                means, quats, scales, opacities, colors,
                sh_degree=d, viewmats=viewmats, Ks=Ks, width=wh[0], height=wh[1],
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB", eps2d=0.3)
            mdd = od["metadata"]
            gate_matrix[f"sh_degree_{d}"] = {
                "f9_enabled": bool(mdd.get("f9_enabled", False)),
                "f9_key_present": "f9_enabled" in mdd,
                "backward_backend": mdd.get("backward_backend"),
                "n_visible": int(mdd.get("n_visible", -1)),
            }
            del od
    rec["f9_gate_matrix"] = gate_matrix

    # ---- main forward+backward at the F9 contract degree ----
    out_d = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opacities, colors,
        sh_degree=3, viewmats=viewmats, Ks=Ks, width=wh[0], height=wh[1],
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    md = out_d["metadata"]
    frame = out_d["frame"].reshape(wh[1], wh[0], 3)
    with torch.no_grad():
        frame_stats = {"frame_finite": bool(torch.isfinite(frame).all()),
                       "frame_mean": float(frame.mean()),
                       "frame_abs_mean": float(frame.abs().mean()),
                       "frame_shape": list(frame.shape)}
    rec["forward"] = {
        "f9_enabled": bool(md.get("f9_enabled", False)),
        "f9_key_present": "f9_enabled" in md,
        "backward_backend": md.get("backward_backend"),
        "native_available": md.get("native_available"),
        "n_visible": int(md.get("n_visible", -1)),
        "packed_dtype": str(md.get("packed_dtype")),
        "sh_compression_mode": str(md.get("sh_compression_mode")),
        "culling_ratio": float(md.get("culling_ratio", -1.0)),
        **frame_stats,
        "metadata_keys": sorted(md.keys()),
        "metadata_has_scalar_adjoint_key": any("scalar" in k for k in md.keys()),
        "metadata_has_h8_key": any("h8" in k.lower() for k in md.keys()),
    }
    loss = (frame.clamp(0, 1) * gt).mean()
    grads = torch.autograd.grad(loss, [means, quats, scales, opacities, colors],
                                allow_unused=True)
    rec["backward"] = {"loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
                       **grad_report(torch, grads)}
    del out_d, frame

    # ---- adversarial env-consumption test (after the records above) ----
    # synthetic mini-scene: the check targets the C++ TORCH_CHECK, not the data
    n = 256
    sg = torch.Generator(device="cpu").manual_seed(0)
    s_means = (torch.rand(n, 3, generator=sg) - 0.5).to("cuda")
    s_quats = torch.randn(n, 4, generator=sg).to("cuda")
    s_quats = s_quats / s_quats.norm(dim=-1, keepdim=True)
    s_scales = torch.full((n, 3), -2.0).exp().to("cuda")
    s_opac = torch.full((n,), 0.6).to("cuda")
    s_col = torch.rand(n, 16, 3, generator=sg).to("cuda")
    s_means.requires_grad_(True)
    saved = {k: os.environ.get(k) for k in ("HIGS_BWD_H8_MR",
                                            "HIGS_BWD_SCALAR_ADJOINT")}
    os.environ.pop("HIGS_BWD_SCALAR_ADJOINT", None)
    os.environ["HIGS_BWD_H8_MR"] = "1"
    h8_err = None
    try:
        od2 = rasterize_gaussian_higs_dynamic(
            s_means, s_quats, s_scales, s_opac, s_col,
            sh_degree=3, viewmats=viewmats, Ks=Ks, width=wh[0], height=wh[1],
            backward_mode="higs_native", enable_culling=False,
            camera_model="pinhole", render_mode="RGB", eps2d=0.3)
        od2["frame"].reshape(wh[1], wh[0], 3).sum().backward()
        h8_err = "NO_ERROR"
    except Exception as e:
        h8_err = f"{type(e).__name__}: {e}"
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    rec["env_consumption_test"] = {
        "scenario": "HIGS_BWD_H8_MR=1 with HIGS_BWD_SCALAR_ADJOINT unset",
        "expected": "RuntimeError containing 'requires HIGS_BWD_SCALAR_ADJOINT'",
        "observed": h8_err[:400] if h8_err else None,
        "torch_check_fired": bool(h8_err and "requires HIGS_BWD_SCALAR_ADJOINT"
                                  in h8_err),
        "env_restored": env_snapshot(),
    }
    del model
    torch.cuda.empty_cache()
    out.update(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gmain", "c0", "a0", "a1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--ckpt", default=None,
                    help="frozen checkpoint (default: full30k gmain03_s42_<scene> "
                         "ckpt_30000.pt)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    record = {
        "phase": "P12 support lane - runtime switch probe (evidence only, no timing)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "process": {"pid": os.getpid(), "cwd": os.getcwd()},
        "env_before_bootstrap": env_snapshot(),
        "gpu": None,
        "result": None,
        "error": None,
    }
    sys.path.insert(0, "/root/epic05_p12_support")
    import publication_trainer_p12 as T

    ckpt = args.ckpt or os.path.join(
        FULL30K, f"gmain03_s42_{args.scene}", "ckpt_30000.pt")
    try:
        record["gpu"] = gpu_identity()
        res = {}
        if args.arm == "gmain":
            probe_gmain(T, args.scene, ckpt, res)
        else:
            probe_higs(T, args.arm, args.scene, ckpt, res)
        record["result"] = res
    except Exception as e:
        import traceback
        record["error"] = {"type": type(e).__name__, "message": str(e),
                           "traceback": traceback.format_exc()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"PROBE_WROTE {args.out} arm={args.arm} "
          f"error={bool(record['error'])}", flush=True)
    if record["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
