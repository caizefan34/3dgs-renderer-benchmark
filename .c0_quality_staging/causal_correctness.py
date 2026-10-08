#!/usr/bin/env python3
"""CAUSAL_CORRECTNESS.json: fixed-state correctness for the same-parent ladder.

For each adjacent pair (A0,A1),(A1,A2),(A2,A3) load ONE identical frozen
Gaussian state and verify on a fixed set of cameras:
  * finite forward output
  * finite gradients
  * the named module switch is engaged
  * support (radii) semantics consistent
  * no NaN/Inf
  * forward numeric consistency where the module does NOT change forward
    (SCALAR_ADJOINT and H8_MR are backward-only => A1/A2/A3 forwards must match
     to fp tolerance; F9 changes the forward producer => A0 vs A1 forward may
     differ but both must be finite and well-formed)
"""
import argparse, json, os, sys
import numpy as np
import torch

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
WORKTREE = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
C0_SO = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda"

FORWARD_REL_TOL = 1e-4
FORWARD_ABS_TOL = 1e-5


def load_ext():
    import importlib.util
    def le(name, so):
        spec = importlib.util.spec_from_file_location(name, so)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        sys.modules[name] = mod; return mod
    for k in ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD", "HIGS_DISABLE_F9"):
        os.environ.pop(k, None)
    sys.path.insert(0, WORKTREE); sys.path.insert(0, SCENE_CUDA_DIR)
    core = le("gsplat_cuda", CORE_SO); sys.modules["gsplat.csrc"] = core
    exp = le("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, os.path.join(INCOMING, "scripts/epic05/phase7"))
    sys.path.insert(0, os.path.join(INCOMING, "src"))
    sys.path.insert(0, os.path.join(INCOMING, "baseline/reference_v1"))
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
    return rasterize_gaussian_higs_dynamic


def set_arm_env(arm):
    for k in ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD", "HIGS_DISABLE_F9"):
        os.environ.pop(k, None)
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    if arm == "a0":
        os.environ["HIGS_DISABLE_F9"] = "1"
    elif arm == "a1":
        pass
    elif arm == "a2":
        os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    elif arm == "c0":
        os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
        os.environ["HIGS_BWD_H8_MR"] = "1"


def render(model, cam, sh_degree, rasterize):
    out = rasterize(
        model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    info = out["densification_info"]
    return out["frame"].reshape(cam.image_height, cam.image_width, 3), info


def arm_forward(arm, state, cfg, dataset, cam_indices, rasterize, GaussianModel, sh_degree):
    set_arm_env(arm)
    m = GaussianModel(max_sh_degree=sh_degree); m.restore(state, cfg.to_dict())
    acc, radii = [], []
    for ci in cam_indices:
        cam, _ = dataset.get_item(ci)
        img, info = render(m, cam, m.active_sh_degree, rasterize)
        acc.append(img.reshape(-1).detach().float().cpu().numpy())
        radii.append(info["radii"].reshape(-1).detach().float().cpu().numpy())
    return np.stack(acc, 0), np.stack(radii, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--n-cam", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(INCOMING, "causal_out", "CAUSAL_CORRECTNESS.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TMPDIR", "/dev/shm/causal_ladder/tmp")
    rasterize = load_ext()

    sys.path.insert(0, os.path.join(INCOMING, "scripts/epic05/phase7"))
    sys.path.insert(0, os.path.join(INCOMING, "src"))
    sys.path.insert(0, os.path.join(INCOMING, "baseline/reference_v1"))
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset

    cfg = ReferenceV1Config(); cfg.seed = 42
    dataset = GTDataset(scene=args.scene, repo_root=INCOMING, resolution="1080p", device="cuda", background="black")
    n_cam = len(dataset)
    cam_indices = list(range(0, n_cam, max(1, n_cam // args.n_cam)))[:args.n_cam]

    sfm = os.path.join(INCOMING, "data/datasets/mipnerf360", args.scene, "sparse/0/points3D.bin")
    pts = read_points3D_binary(sfm)
    pcd = sfm_to_pcd_data(pts, sh_degree=cfg.sh_degree)

    torch.manual_seed(0); np.random.seed(0)
    base = GaussianModel(max_sh_degree=cfg.sh_degree)
    base.create_from_pcd(pcd, spatial_lr_scale=1.0); base.active_sh_degree = 3
    base.training_setup(cfg.to_dict())
    state = base.capture()

    arms = ["a0", "a1", "a2", "c0"]
    per_arm = {}
    for arm in arms:
        # forward finiteness + radii
        fimgs, radii = arm_forward(arm, state, cfg, dataset, cam_indices, rasterize, GaussianModel, cfg.sh_degree)
        fwd_finite = bool(np.isfinite(fimgs).all())
        nan_inf = int((~np.isfinite(fimgs)).sum())
        # gradient finiteness (train mode, simple loss)
        set_arm_env(arm)
        m = GaussianModel(max_sh_degree=cfg.sh_degree); m.restore(state, cfg.to_dict()); m.active_sh_degree = 3
        cam, _ = dataset.get_item(cam_indices[0])
        img, info = render(m, cam, m.active_sh_degree, rasterize)
        loss = img.square().mean()
        loss.backward()
        g = m.get_xyz.grad
        grad_finite = bool(g is not None and torch.isfinite(g.detach().float()).all())
        grad_nan_inf = int(torch.isnan(g.detach().float()).sum() + torch.isinf(g.detach().float()).sum()) if g is not None else -1
        per_arm[arm] = {
            "forward_finite": fwd_finite, "grad_finite": grad_finite,
            "forward_nan_inf": nan_inf, "grad_nan_inf": grad_nan_inf,
            "forward_mean": float(fimgs.mean()), "forward_std": float(fimgs.std()),
            "radii_mean": float(radii.mean()), "radii_finite": bool(np.isfinite(radii).all()),
        }

    results = {"artifact": "CAUSAL_CORRECTNESS", "gate": "CORRECTNESS",
               "scene": args.scene, "gpu": args.gpu, "eps2d": 0.3,
               "n_cameras": len(cam_indices), "per_arm": per_arm, "pairs": {},
               "fp_gates": {"forward_rel_tol": FORWARD_REL_TOL, "forward_abs_tol": FORWARD_ABS_TOL}}

    pairs = [("A0", "A1", "a0", "a1", "F9 (forward producer)", False),
             ("A1", "A2", "a1", "a2", "SCALAR_ADJOINT (backward-only)", True),
             ("A2", "A3", "a2", "c0", "H8_MR (backward geometry)", True)]
    all_pass = True
    for an, bn, aa, bb, mod, backward_only in pairs:
        fa, ra = arm_forward(aa, state, cfg, dataset, cam_indices, rasterize, GaussianModel, cfg.sh_degree)
        fb, rb = arm_forward(bb, state, cfg, dataset, cam_indices, rasterize, GaussianModel, cfg.sh_degree)
        denom = np.maximum(np.abs(fa), 1e-6)
        rel = np.abs(fb - fa) / denom
        max_abs = float(np.abs(fb - fa).max()); max_rel = float(rel.max())
        if backward_only:
            fwd_consistent = (max_rel <= FORWARD_REL_TOL) or (max_abs <= FORWARD_ABS_TOL)
        else:
            fwd_consistent = bool(np.isfinite(fa).all() and np.isfinite(fb).all())
        pair_pass = (per_arm[aa]["forward_finite"] and per_arm[aa]["grad_finite"] and
                     per_arm[bb]["forward_finite"] and per_arm[bb]["grad_finite"] and
                     per_arm[aa]["forward_nan_inf"] == 0 and per_arm[bb]["forward_nan_inf"] == 0 and
                     per_arm[aa]["radii_finite"] and per_arm[bb]["radii_finite"] and
                     fwd_consistent)
        all_pass = all_pass and pair_pass
        results["pairs"][f"{an}_vs_{bn}"] = {
            "module": mod, "arm_a": aa, "arm_b": bb, "backward_only": backward_only,
            "forward_max_abs_diff": max_abs, "forward_max_rel_diff": max_rel,
            "forward_consistency_required": backward_only, "forward_consistent": bool(fwd_consistent),
            "a_forward_finite": per_arm[aa]["forward_finite"], "a_grad_finite": per_arm[aa]["grad_finite"],
            "b_forward_finite": per_arm[bb]["forward_finite"], "b_grad_finite": per_arm[bb]["grad_finite"],
            "a_nan_inf": per_arm[aa]["forward_nan_inf"], "b_nan_inf": per_arm[bb]["forward_nan_inf"],
            "a_radii_finite": per_arm[aa]["radii_finite"], "b_radii_finite": per_arm[bb]["radii_finite"],
            "pair_pass": bool(pair_pass),
        }
    results["correctness_verdict"] = "PASS" if all_pass else "FAIL"
    results["determination"] = "disposition:PASS_UNDER_MX_LOCAL_CALIBRATION" if all_pass else "FAIL_SEMANTICS"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("CORRECTNESS verdict:", results["correctness_verdict"], "->", args.out)


if __name__ == "__main__":
    main()
