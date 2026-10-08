#!/usr/bin/env python3
"""P18-P0 Phase 1 — isolated QSCAN reference validation (protocol section 8).

Compares DIRECT vs QSCAN4/8/16 reconstruction of G(x)=exp(q(x)) on:
  1) synthetic conics (sweeps of positions/sizes/tilts/opacities)
  2) REAL room / bicycle 16K conics (forward-captured by the frozen C0 raster)
Records: max_abs_G_error, max_rel_G_error (G>1e-8), mean_abs_G_error,
alpha branch agreement, NaN/Inf. Both forward exp2 and backward exp domains.
NO speed claim in this phase.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(os.environ.get("P18_OUT", "/root/epic05_p18"))
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
AT = 1.0 / 255.0
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")
FROZEN = {
    "core_so": "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
    "c0_ext_so": "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
}
CKPT = {"room": "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt",
        "bicycle": "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt"}


def jwrite(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, allow_nan=False)
    print(f"[p18] wrote {p}", flush=True)


def log(m):
    print(f"[p18 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ext_module(name, so_path):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    for p in (C0_WT, SCENE_CUDA_DIR, PHASE7, SRC, REF_V1):
        if p not in sys.path:
            sys.path.insert(0, p)
    core = load_ext_module("gsplat_cuda", FROZEN["core_so"])
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", FROZEN["c0_ext_so"])
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp


def build_ext(src):
    from torch.utils.cpp_extension import load
    return load(name="p18_qscan_ref", sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-use_fast_math", "-gencode=arch=compute_80,code=sm_80"])


def metrics(d, q):
    d = d.reshape(-1); q = q.reshape(-1)
    diff = (q - d).abs()
    rel_mask = d > 1e-8
    max_rel = float((diff[rel_mask] / d[rel_mask]).max()) if rel_mask.any() else None
    bd = (d < AT).reshape(-1); bq = (q < AT).reshape(-1)
    return {"max_abs_G_error": float(diff.max()), "max_rel_G_error": max_rel,
            "mean_abs_G_error": float(diff.mean()), "n_samples": int(d.numel()),
            "alpha_branch_mismatch": int((bd != bq).sum()),
            "nan_inf_count": int((~torch.isfinite(q)).sum()) + int((~torch.isfinite(d)).sum()),
            "branch_agreement": 1.0 - int((bd != bq).sum()) / max(1, int(d.numel()))}


def run_one(ext, a, b, c, variant, exp2):
    od, oq, om, on = ext.qscan_ref(a, b, c, variant, exp2, 1)
    return metrics(od, oq)


def abc_from_conic(conic, means2d):
    c00, c11, c22 = conic[:, 0], conic[:, 1], conic[:, 2]
    mx, my = means2d[:, 0], means2d[:, 1]
    A = 0.5 * c00
    B = c11 + c00 * mx
    C = 0.5 * c22 * my * my + c11 * mx * my + 0.5 * c00 * (mx * mx + my * my) - c00 * mx
    return A, B, C


def main():
    src = os.environ.get("P18_CU", str(OUT / "p18_qscan_ref.cu"))
    ext = build_ext(src)
    log("extension built")

    # ---------------- synthetic conics ----------------
    torch.manual_seed(42)
    rows = []
    for mx in (-4.0, -1.0, 0.0, 1.0, 4.0):
        for my in (-3.0, 0.0, 3.0):
            for s in (0.3, 1.0, 3.0, 8.0):
                for opac in (0.1, 0.5, 0.9, 0.99):
                    for tilt in (0.0, 0.3, 0.7):
                        c00 = 1.0 / (s * s); c22 = c00; c11 = tilt * c00
                        A = 0.5 * c00
                        B = c11 + c00 * mx
                        C = 0.5 * c22 * my * my + c11 * mx * my + 0.5 * c00 * (mx * mx + my * my) - c00 * mx
                        rows.append([A, B, C])
    a_s = torch.tensor([r[0] for r in rows], device=DEV)
    b_s = torch.tensor([r[1] for r in rows], device=DEV)
    c_s = torch.tensor([r[2] for r in rows], device=DEV)
    log(f"synthetic rows: {a_s.numel()}")
    synth = {}
    for vname, v in (("QSCAN4", 0), ("QSCAN8", 1), ("QSCAN16", 2)):
        var = torch.full((a_s.numel(),), v, device=DEV, dtype=torch.int32)
        synth[vname] = {"exp2": run_one(ext, a_s, b_s, c_s, var, True),
                        "exp":  run_one(ext, a_s, b_s, c_s, var, False)}

    # ---------------- real conics ----------------
    real = {}
    try:
        bootstrap_c0()
        from dataset import GTDataset
        from gaussian_model import GaussianModel
        from config import ReferenceV1Config
        from gsplat.experimental.render.functional import gaussian_inference as gi
        camset = json.load(open(OUT / "P18_CAMERA_SET.json"))
        for scene in ("room", "bicycle"):
            cfg = ReferenceV1Config(); cfg.scene = scene
            ds = GTDataset(scene=scene, repo_root=REPO, resolution="1080p", device=DEV, background="black")
            model = GaussianModel(max_sh_degree=cfg.sh_degree)
            ckpt = torch.load(CKPT[scene], map_location="cpu", weights_only=False)
            model.restore(ckpt, {
                "position_lr_init": cfg.position_lr_init, "position_lr_final": cfg.position_lr_final,
                "position_lr_delay_mult": cfg.position_lr_delay_mult, "position_lr_max_steps": cfg.position_lr_max_steps,
                "feature_lr": cfg.feature_lr, "opacity_lr": cfg.opacity_lr,
                "scaling_lr": cfg.scaling_lr, "rotation_lr": cfg.rotation_lr,
                "percent_dense": cfg.percent_dense})
            model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            N = int(model._xyz.shape[0])
            gi._HIGS_DYNAMIC_SCENE.mark_dirty()
            ci = camset["scenes"][scene]["probe_ids"][0]
            cam, gt = ds.get_item(ci)
            W, H = int(cam.image_width), int(cam.image_height)
            K = torch.as_tensor(cam.K.detach().cpu().numpy(), device=DEV, dtype=torch.float32)
            from gsplat.experimental.render.kernels import _backend as _ib
            backend = _ib._C
            m = model
            vm = cam.viewmatrix.unsqueeze(0).contiguous()
            Kc = K.unsqueeze(0).contiguous()
            with torch.no_grad():
                cam_pos = backend.higs_camera_positions_from_viewmats(vm)
                (radii, means2d, depths, conics, op_bc, colors_eval) = backend.higs_gatherless_projected_producer(
                    torch.arange(N, device=DEV, dtype=torch.int64),
                    m._xyz.detach().contiguous(),
                    F.normalize(m._rotation.detach(), dim=-1).contiguous(),
                    torch.exp(m._scaling.detach()).contiguous(),
                    torch.sigmoid(m._opacity.detach()).squeeze(-1).contiguous(),
                    m.get_features.detach().contiguous(),
                    vm, Kc, cam_pos, W, H, 0.3, 0.01, 1e10, 0.0)
            radii = radii.reshape(N, 2); conics = conics.reshape(N, 3).float(); means2d = means2d.reshape(N, 2).float()
            log(f"{scene}: N={N} conics captured")
            valid = (radii[:, 0] > 0) & (radii[:, 1] > 0) & torch.isfinite(conics).all(-1)
            idx = torch.nonzero(valid, as_tuple=False).flatten()
            if idx.numel() > 2048:
                idx = idx[:: idx.numel() // 2048][:2048]
            A, B, C = abc_from_conic(conics[idx].contiguous().to(DEV), means2d[idx].contiguous().to(DEV))
            real[scene] = {"n_visible": int(valid.sum()), "n_sampled": int(idx.numel()), "cam": ci, "per_variant": {}}
            for vname, v in (("QSCAN4", 0), ("QSCAN8", 1), ("QSCAN16", 2)):
                var = torch.full((A.numel(),), v, device=DEV, dtype=torch.int32)
                real[scene]["per_variant"][vname] = {"exp2": run_one(ext, A, B, C, var, True),
                                                     "exp":  run_one(ext, A, B, C, var, False)}
            del model, ds
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        real["error"] = str(e); real["tb"] = traceback.format_exc()
        log(f"real conics FAILED: {e}")

    out = {
        "phase": "P18-P0", "artifact": "P18_REFERENCE_VALIDATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "isolated DIRECT vs QSCAN validity only; NO speed claim (protocol section 8)",
        "synthetic_conics": {"n_rows": int(a_s.numel()), "variants": synth},
        "real_conics": real,
        "thresholds": {"alpha_threshold": AT, "gamma": 2e-6, "rel_g_floor": 1e-8},
    }
    jwrite(str(OUT / "P18_REFERENCE_VALIDATION.json"), out)
    for v, dom in (("QSCAN4", "exp2"), ("QSCAN16", "exp")):
        s = synth[v][dom]
        log(f"synthetic {v}/{dom}: max_abs={s['max_abs_G_error']:.3e} max_rel={s['max_rel_G_error']} branch_mm={s['alpha_branch_mismatch']}")
    log("reference validation done")


if __name__ == "__main__":
    main()
