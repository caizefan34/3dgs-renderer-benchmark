#!/usr/bin/env python3
"""A-arm functional gate for the P2 cumulative ablation (publication phase).

Runs one render+backward per ablation configuration in SEPARATE subprocesses
(env switches are process-global), on ONE GPU (auto-picked least-loaded; this
is a FUNCTIONAL gate, not timing), then compares:

  G1  forward frame BIT-IDENTICAL across a0/a1/a2/c0 (F9 is an exact
      reassociation; SCALAR_ADJOINT/H8-MR are backward-only -> all four arms
      must produce the same forward frame bitwise)
  G2  backward gradients relatively close across arms (exact algebraic
      reassociations -> tiny FP-level rel-err, threshold 1e-3)
  G3  means2d.absgrad present in ALL arms (HIGS_BWD_ABSGRAD=1)
  G4  PX launch-shape invariance: c0 (PX=2) vs c0px0 (PX=0) gradients equal

Usage (parent):  python a_arm_gate.py --repo <repo_root> [--gpu N]
Worker mode:    python a_arm_gate.py --worker <arm> --out <pt> [--gpu N]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ARMS = ["a0", "a1", "a2", "c0", "c0px0"]


def pick_gpu():
    import subprocess as sp
    r = sp.run(["nvidia-smi", "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits"], capture_output=True, text=True)
    best, best_mb = None, 1 << 30
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        mb = int(parts[1])
        if mb < best_mb:
            best, best_mb = int(parts[0]), mb
    return best, best_mb


def worker(arm, out_path, repo):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, repo)
    import publication_trainer as PT

    if arm in ("a0", "a1", "a2"):
        ident = PT.bootstrap_a(arm)
    elif arm == "c0":
        ident = PT.bootstrap_a("a2")
        os.environ["HIGS_BWD_H8_MR"] = "1"
    elif arm == "c0px0":
        ident = PT.bootstrap_a("a2")
        os.environ["HIGS_BWD_H8_MR"] = "1"
        os.environ["HIGS_PX_RUNTIME"] = "0"
    else:
        raise SystemExit(f"bad arm {arm}")

    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gaussian_model import GaussianModel
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset

    dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p",
                        device="cuda", background="black")
    centers = [dataset.get_camera(i).camera_center.cpu().numpy() for i in range(20)]
    extent = max(np.linalg.norm(centers[i] - centers[j])
                 for i in range(20) for j in range(i + 1, 20))
    extent = max(float(extent), 0.1)

    sfm_path = os.path.join(repo, "data", "datasets", "mipnerf360", "room",
                            "sparse", "0", "points3D.bin")
    pcd_data = sfm_to_pcd_data(read_points3D_binary(sfm_path), sh_degree=3)
    model = GaussianModel(max_sh_degree=3)
    model.create_from_pcd(pcd_data, spatial_lr_scale=extent)
    model.training_setup({
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
        "rotation_lr": 0.001, "percent_dense": 0.01,
    })

    cam, gt = dataset.get_item(0)
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling,
        model.get_opacity, model.get_features,
        sh_degree=model.active_sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    info = out["densification_info"]
    means2d = info["means2d"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)

    loss = F.l1_loss(frame.clamp(0, 1), gt)
    loss.backward()

    gxyz = model._xyz.grad.detach().float()
    gop = model._opacity.grad.detach().float()
    gsc = model._scaling.grad.detach().float()
    has_abs = hasattr(means2d, "absgrad") and means2d.absgrad is not None
    abs_norm = float(means2d.absgrad.norm()) if has_abs else None

    torch.save({
        "arm": arm, "ident_env": ident["env"],
        "frame": frame.detach().cpu(),
        "loss": float(loss.item()),
        "gxyz": gxyz.cpu(), "gop": gop.cpu(), "gsc": gsc.cpu(),
        "absgrad_present": has_abs, "absgrad_norm": abs_norm,
        "n_gaussians": int(model._xyz.shape[0]),
    }, out_path)
    print(f"[{arm}] N={model._xyz.shape[0]} loss={float(loss.item()):.6f} "
          f"absgrad={has_abs} |gxyz|={float(gxyz.norm()):.4f}")


def relerr(a, b):
    denom = max(float(b.norm()), 1e-12)
    return float((a - b).norm()) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/liaoyuanjun/3dgs-renderer-benchmark")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--worker")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.out, args.repo)
        return

    gpu, used_mb = pick_gpu() if args.gpu is None else (args.gpu, -1)
    print(f"functional gate on GPU {gpu} (least loaded, {used_mb} MiB used)")
    tmp = "/mnt/storage_pool/liaoyuanjun/pubphase/gate"
    os.makedirs(tmp, exist_ok=True)

    for arm in ARMS:
        out_pt = os.path.join(tmp, f"{arm}.pt")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--worker", arm, "--out", out_pt,
                            "--repo", args.repo],
                           env=env, capture_output=True, text=True, timeout=900)
        print(r.stdout[-600:])
        if r.returncode != 0:
            print(f"WORKER {arm} FAILED:\n{r.stderr[-1200:]}")
            sys.exit(1)

    import torch
    data = {a: torch.load(os.path.join(tmp, f"{a}.pt")) for a in ARMS}

    checks = {}
    base = data["a0"]
    # G1: forward frame bit-identical across all arms
    for a in ARMS:
        checks[f"G1_frame_bitwise_a0_vs_{a}"] = bool(torch.equal(base["frame"], data[a]["frame"]))
    # G2: gradient closeness (exact reassociations -> tiny rel err)
    for a in ARMS[1:]:
        checks[f"G2_gxyz_relerr_a0_vs_{a}"] = relerr(base["gxyz"], data[a]["gxyz"])
    checks["G2_gxyz_relerr_a2_vs_c0"] = relerr(data["a2"]["gxyz"], data["c0"]["gxyz"])
    # G3: absgrad present everywhere
    for a in ARMS:
        checks[f"G3_absgrad_present_{a}"] = bool(data[a]["absgrad_present"])
    # G4: PX invariance
    checks["G4_gxyz_relerr_px2_vs_px0"] = relerr(data["c0"]["gxyz"], data["c0px0"]["gxyz"])
    checks["G4_frame_bitwise_px2_vs_px0"] = bool(torch.equal(data["c0"]["frame"], data["c0px0"]["frame"]))

    g1_ok = all(checks[f"G1_frame_bitwise_a0_vs_{a}"] for a in ARMS)
    g2_ok = all(checks[k] < 1e-3 for k in checks if k.startswith("G2_"))
    g3_ok = all(checks[f"G3_absgrad_present_{a}"] for a in ARMS)
    g4_ok = checks["G4_gxyz_relerr_px2_vs_px0"] < 1e-3 and checks["G4_frame_bitwise_px2_vs_px0"]
    verdict = "A_ARM_FUNCTIONAL_GATE_PASS" if (g1_ok and g2_ok and g3_ok and g4_ok) else "A_ARM_FUNCTIONAL_GATE_FAIL"

    result = {
        "verdict": verdict,
        "gpu": gpu, "gpu_used_mb_at_pick": used_mb,
        "timing_grade": "FUNCTIONAL_ONLY (shared/loaded GPU; no timing claims)",
        "n_gaussians": {a: data[a]["n_gaussians"] for a in ARMS},
        "loss": {a: data[a]["loss"] for a in ARMS},
        "env": {a: data[a]["ident_env"] for a in ARMS},
        "checks": checks,
        "criteria": {
            "G1": "forward frame bit-identical across a0/a1/a2/c0 (F9 exact; SA/H8 backward-only)",
            "G2": "gxyz rel-err < 1e-3 across arms (exact algebraic reassociation)",
            "G3": "means2d.absgrad present in all arms",
            "G4": "PX=2 vs PX=0 gradients and frame identical (launch shape, not algorithm)",
        },
    }
    out_json = os.path.join(tmp, "a_arm_functional_gate.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2)[:2400])
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()
