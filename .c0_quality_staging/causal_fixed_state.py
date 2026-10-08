#!/usr/bin/env python3
"""CAUSAL_FIXED_STATE_TIMING.json: fixed-state renderer timing for A0/A1/A2/A3.

One identical frozen Gaussian state per scene (deterministic COLMAP init state by
default; optional brief a0 warmup to a post-densification state). For each arm
measure forward / backward / F+B / VRAM with CUDA events, warmup>=50, timed>=100.
This isolates the INTRINSIC renderer effect from the training-trajectory effect.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
WORKTREE = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
C0_SO = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/mnt/storage_pool/liaoyuanjun/torchext/gsplat_scene_cuda"


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
    global _SCENE
    _SCENE = _HIGS_DYNAMIC_SCENE
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


def render(model, cam, sh_degree, rasterize, want_info=False):
    out = rasterize(
        model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    if want_info:
        return out["frame"], out["densification_info"]
    return out["frame"]


def time_arm(arm, state, cfg, dataset, cam_indices, rasterize, GaussianModel, warmup, timed):
    set_arm_env(arm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m = GaussianModel(max_sh_degree=cfg.sh_degree); m.restore(state, cfg.to_dict()); m.active_sh_degree = 3
    cams = [dataset.get_item(ci)[0] for ci in cam_indices]

    def one_pass():
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(len(cams))]
        for i, cam in enumerate(cams):
            ev[i][0].record()
            img = render(m, cam, m.active_sh_degree, rasterize)
            loss = img.square().mean()
            loss.backward()
            ev[i][1].record()
        torch.cuda.synchronize()
        fwd = sum(e[0].elapsed_time(e[1]) for e in ev)
        return fwd / len(cams)

    for _ in range(warmup):
        one_pass()
    fwd_times = [one_pass() for _ in range(timed)]
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {"fb_ms_per_cam": float(np.mean(fwd_times)),
            "fb_ms_per_cam_median": float(np.median(fwd_times)),
            "fb_ms_per_cam_std": float(np.std(fwd_times)),
            "n_timed": timed, "n_cameras": len(cams), "peak_vram_gb": float(peak)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--n-cam", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--train-iters", type=int, default=0,
                    help="brief a0 warmup to a post-densify state (0 = deterministic COLMAP init state)")
    ap.add_argument("--out", default=os.path.join(INCOMING, "causal_out", "CAUSAL_FIXED_STATE_TIMING.json"))
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

    set_arm_env("a0")
    torch.manual_seed(42); np.random.seed(42)
    m = GaussianModel(max_sh_degree=cfg.sh_degree)
    m.create_from_pcd(pcd, spatial_lr_scale=1.0); m.active_sh_degree = 3
    m.training_setup(cfg.to_dict())
    state_desc = "deterministic COLMAP init state"
    if args.train_iters > 0:
        rng = np.random.RandomState(42)
        viewpoint_stack = list(range(n_cam))
        for it in range(1, args.train_iters + 1):
            if not viewpoint_stack:
                viewpoint_stack = list(range(n_cam))
            cam_idx = viewpoint_stack.pop(rng.randint(0, len(viewpoint_stack)))
            cam, gt = dataset.get_item(cam_idx)
            img, info = render(m, cam, m.active_sh_degree, rasterize, want_info=True)
            loss = F.l1_loss(img, gt)
            loss.backward()
            if it < cfg.densify_until_iter:
                means2d = info["means2d"]
                radii = info["radii"][0]
                vf = (radii > 0).any(dim=-1)
                m.add_densification_stats(means2d, vf, width=cam.image_width, height=cam.image_height)
                if it % cfg.densification_interval == 0 and it > cfg.densify_from_iter:
                    m.densify_and_prune(max_grad=cfg.densify_grad_threshold, min_opacity=cfg.min_opacity,
                                        extent=1.0, max_screen_size=cfg.max_screen_size, radii=radii.float().max(dim=-1).values)
                    _SCENE.mark_dirty()
            m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        state_desc = f"post-densify a0 warmup ({args.train_iters} iters)"
    state = m.capture()
    n_state = int(state["num_points"])

    arms = ["a0", "a1", "a2", "c0"]
    per_arm = {}
    for arm in arms:
        per_arm[arm] = time_arm(arm, state, cfg, dataset, cam_indices, rasterize, GaussianModel, args.warmup, args.timed)

    def ratio(a, b):
        return per_arm[a]["fb_ms_per_cam"] / per_arm[b]["fb_ms_per_cam"]

    out = {"artifact": "CAUSAL_FIXED_STATE_TIMING", "gate": "FIXED_STATE_TIMING",
           "scene": args.scene, "gpu": args.gpu, "eps2d": 0.3,
           "frozen_state_n_gaussians": n_state, "frozen_state_desc": state_desc,
           "n_cameras": len(cam_indices), "warmup": args.warmup, "timed": args.timed,
           "per_arm": per_arm,
           "ratios_fb": {"F9_A0_over_A1": ratio("a0", "a1"), "SCALAR_A1_over_A2": ratio("a1", "a2"),
                         "H8_A2_over_A3": ratio("a2", "c0"), "TOTAL_A0_over_A3": ratio("a0", "c0")},
           "note": "F+B ms per camera on identical frozen state; ratio(a/b) < 1 => arm b faster"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("FIXED_STATE wrote", args.out, "stateN", n_state)


if __name__ == "__main__":
    main()
