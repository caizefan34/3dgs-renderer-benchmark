#!/usr/bin/env python
"""DPC0 shared harness: DPC backward bootstrap + deterministic camera splits.

DPC isolation: authoritative C0 forward/op/renderer stay loaded; ONLY
higs_rasterize_backward is swapped to the DPC build (same name/signature,
race-free primitive-centric impl). This preserves the exact frozen forward
and isolates the raster backward under test.
"""
import sys, os, hashlib, importlib.util, datetime
sys.path.insert(0, "/root/epic05_n1")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/src")
import numpy as np
import torch
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")
C0_SO = C.C0_SO
PARAM_NAMES = C.PARAM_NAMES
GRAD_ORDER = ("_xyz", "_shs", "_scaling", "_rotation", "_opacity")
CELLS = ("room@10000", "room@15000", "bicycle@10000", "bicycle@15000")


def cell_scene_iter(cell):
    scene, it = cell.split("@")
    return scene, int(it)


def load_dpc():
    spec = importlib.util.spec_from_file_location(
        "dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["dpc0_gaussian_render_inference_scene_cuda"] = mod
    return mod


def bootstrap(arm="dpc"):
    """arm='dpc' -> C0 forward + DPC backward; arm='c0' -> pure C0."""
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C.C0_WT)
    sys.path.insert(0, C.SCENE_CUDA_DIR)
    core = C.load_ext_module("gsplat_cuda", C.CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = C.load_ext_module(
        "experimental_gaussian_render_inference_scene_cuda", C.C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, C.PHASE7_DIR)
    sys.path.insert(0, C.SRC_DIR)
    sys.path.insert(0, C.REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: F401
    from gsplat.experimental.render.functional.gaussian_inference import (  # noqa: F401
        _HIGS_DYNAMIC_SCENE)
    if arm == "dpc":
        dpc = load_dpc()
        exp.higs_rasterize_backward = dpc.higs_rasterize_backward
    return exp


def setup_state(scene, it):
    ckpt = C.load_checkpoint(scene, it)
    model = C.build_model(ckpt)
    C.scene_reset()
    ds = C.load_scene_dataset(scene)
    return model, ds


def render_loss(model, ds, cam_idx):
    cam, gt = ds.get_item(cam_idx)
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    fused = C.make_fused_loss()
    L = fused(img, gt)
    params = [getattr(model, n) for n in PARAM_NAMES]
    return L, params, img, gt, cam


def grads_cpu(model, params):
    out = []
    for p in params:
        g = p.grad
        out.append(None if g is None else g.detach().to("cpu", torch.float32).numpy())
    return out


def hash_arrays(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(b"NULL" if a is None else np.ascontiguousarray(a, dtype="<f4").tobytes())
    return h.hexdigest()


# ---- deterministic camera splits (mipnerf360 convention: every 8th = holdout)
def split_indices(n):
    holdout = [i for i in range(n) if i % 8 == 0]
    train = [i for i in range(n) if i % 8 != 0]
    return train, holdout


def midpoint_stratified(idxs, k):
    """Pick k indices by dividing the sorted list into k equal strata and
    taking each stratum's midpoint. Deterministic."""
    if len(idxs) <= k:
        return list(idxs)
    out = []
    for s in range(k):
        lo = s * len(idxs) // k
        hi = (s + 1) * len(idxs) // k
        mid = (lo + hi) // 2
        out.append(idxs[mid])
    return out


def peer_train(train_list, src_pos, k=3):
    """k deterministic train cameras nearest to src_pos in the train list."""
    order = sorted(range(len(train_list)), key=lambda p: (abs(p - src_pos), p))
    return [train_list[p] for p in order[:k]]


def cell_cameras(scene):
    """Return dict with source(8), peer(3 per source), holdout(8) cams."""
    ds = C.load_scene_dataset(scene)
    n = len(ds)
    train, holdout = split_indices(n)
    srcs = midpoint_stratified(train, 8)
    holds = midpoint_stratified(holdout, 8)
    peers = {}
    for s in srcs:
        pos = train.index(s)
        peers[s] = peer_train(train, pos, 3)
    return {"n": n, "train": train, "holdout": holdout,
            "sources": srcs, "holdout8": holds, "peers": peers}
