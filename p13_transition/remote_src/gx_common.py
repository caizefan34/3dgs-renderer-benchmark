"""Shared helpers for the GRC3D M0/M1 diagnostic on EPIC-05.

Bootstraps the frozen GMAIN strong baseline (gsplat main @512d366b + the
P11-validated fused-SSIM training loss) by reusing the P12 trainer module's
own bootstrap_gmain(), so the renderer identity cannot drift from the
publication baseline. Read-only with respect to the campaign repo.
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = "/root/3dgs-renderer-benchmark"
P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
GRC3D_ROOT = "/root/epic05_grc3d"
P11_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load_p12_trainer():
    spec = importlib.util.spec_from_file_location("p12trainer", P12_TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p12trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def boot_gmain(gpu):
    """Bind CUDA_VISIBLE_DEVICES, run the P12 gmain bootstrap, verify identity."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t = _load_p12_trainer()
    ident = t.bootstrap_gmain()
    import gsplat
    assert os.path.realpath(gsplat.__file__).startswith(t.GMAIN_TREE), \
        f"gsplat resolved outside the pinned tree: {gsplat.__file__}"
    return t, ident


def check_fused_lock():
    import fused_ssim
    import fused_ssim_cuda
    so_sha = sha256_file(fused_ssim_cuda.__file__)
    if so_sha != P11_FUSED_SO_SHA256:
        raise RuntimeError(f"fused_ssim_cuda.so sha256 {so_sha} != P11 lock")
    return {"module_file": fused_ssim.__file__,
            "so_path": fused_ssim_cuda.__file__, "so_sha256": so_sha}


def fused_loss(pred_hwc, gt_hwc):
    """Exact P12 trainer training-loss semantics (BCHW contiguous 0.8 L1 + 0.2
    (1-fused SSIM))."""
    import fused_ssim
    b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    g = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    l1 = torch.nn.functional.l1_loss(b, g)
    ssim = fused_ssim.fused_ssim(b, g, padding="same", train=True)
    return 0.8 * l1 + 0.2 * (1.0 - ssim)


def add_recipe_paths():
    """Same shared-recipe path entries the P12 trainer bootstrap installs."""
    for p in (os.path.join(REPO, "scripts", "epic05", "phase7"),
              os.path.join(REPO, "src"),
              os.path.join(REPO, "baseline", "reference_v1")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_dataset(scene):
    add_recipe_paths()
    from dataset import GTDataset
    return GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                     device="cuda", background="black")


def lr_config():
    from config import ReferenceV1Config
    c = ReferenceV1Config()
    return {"position_lr_init": c.position_lr_init,
            "position_lr_final": c.position_lr_final,
            "position_lr_delay_mult": c.position_lr_delay_mult,
            "position_lr_max_steps": c.position_lr_max_steps,
            "feature_lr": c.feature_lr, "opacity_lr": c.opacity_lr,
            "scaling_lr": c.scaling_lr, "rotation_lr": c.rotation_lr,
            "percent_dense": c.percent_dense}


def restore_model(ckpt_path):
    from gaussian_model import GaussianModel
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(ck, lr_config())
    return model, ck


def bundle_params_from_model(model):
    """Adapted export mapping (see m0_audit mapping_audit): the reference_v1
    GaussianModel stores raw optimization variables as _xyz/_rotation/_scaling/
    _opacity and a single _shs [N, 16, 3] tensor (NOT graphdeco's
    _features_dc/_features_rest)."""
    with torch.no_grad():
        shs = model._shs.detach()
        return {"means": model._xyz.detach().clone(),
                "quats": model._rotation.detach().clone(),
                "log_scales": model._scaling.detach().clone(),
                "opacity_logits": model._opacity.detach().reshape(-1).clone(),
                "sh0": shs[:, :1].clone(), "shN": shs[:, 1:].clone()}


def camera_dict(cam, device="cuda"):
    return {"viewmat": cam.viewmatrix.detach().to(device).float(),
            "K": cam.K.detach().to(device).float(),
            "height": int(cam.image_height), "width": int(cam.image_width)}


def camera_identity(cam, tw, th):
    m = (cam.viewmatrix.detach().cpu().numpy().tobytes()
         + cam.K.detach().cpu().numpy().tobytes() + f"{tw}x{th}".encode())
    return hashlib.sha256(m).hexdigest()


def gpu_inventory():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.strip().splitlines()]


def run_identity(t, ident, gpu):
    import platform
    return {"host": platform.node(), "gpu_index_requested": gpu,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_before": gpu_inventory(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gsplat_file": __import__("gsplat").__file__,
            "gsplat_so_sha256": ident["so_sha256"], "gmain_tree": t.GMAIN_TREE,
            "gmain_pin": ident["base_commit"]}


def add_grc3d_path():
    if GRC3D_ROOT not in sys.path:
        sys.path.insert(0, GRC3D_ROOT)
    return GRC3D_ROOT


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        import json
        json.dump(obj, f, indent=2, allow_nan=False)
    return path
