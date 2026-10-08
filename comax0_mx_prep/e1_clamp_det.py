#!/usr/bin/env python3
"""E1_NUMERICS_SHARED_GPU_AMENDMENT: focused determinism gate for the CLAMP
production renderer (comax_render_eta_ckpt_clamp: power-clamp>=0 + TF32-off) on the
exact views that were nondeterministic-NaN under the non-clamp oracle
(COMAX0_E1_SHARED_DET2: globals 96,240,304 = NaN; 0,40 = clean).

Render-only (no training, no_grad). NUMERICS ONLY; NO timing/performance claim.
For each view renders twice under the shared-GPU setup and requires:
  - NaN mask identical run-to-run (nan_mask_match), AND
  - max_abs_diff over non-NaN pixels <= frozen tol (e2_rgb_abs_max).
Strict bitwise hash_match is also reported. Gate per view = nan_mask_match AND
(max_abs_diff_excl_nan <= tol). Global gate = all views pass.

Guard: require_guarded_process() BEFORE torch/CUDA (proof supplied by the
run_e1_shared_ckpt.py launcher). Co-resident user train/*.py NOT touched.
"""
import os
import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mx_guard import require_guarded_process

CFG, PROOF = require_guarded_process()  # BEFORE torch/CUDA

os.environ["COMAX0_POWER_CLAMP"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# --- gsplat prebuilt-extension shim (load BEFORE any gsplat-triggering import) ---
import importlib.util as _ilu
import glob as _glob


def _install_gsplat_shim():
    cands = [os.path.expanduser("~/.cache/torch_extensions/py311_cu118/gsplat_cuda/gsplat_cuda.so")]
    cands += sorted(_glob.glob(os.path.expanduser("~/.cache/torch_extensions/*/gsplat_cuda/gsplat_cuda.so")), reverse=True)
    so = next((p for p in cands if Path(p).is_file()), None)
    if so is None:
        return "FAIL:no_prebuilt_so"
    try:
        spec = _ilu.spec_from_file_location("gsplat_cuda", so)
        m = _ilu.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.modules["gsplat.csrc"] = m
        from gsplat.cuda import _backend as _bb
        if getattr(_bb, "_C", None) is None:
            _bb._C = m
        if not hasattr(m, "fully_fused_projection") and hasattr(m, "projection_ewa_3dgs_fused_fwd"):
            m.fully_fused_projection = m.projection_ewa_3dgs_fused_fwd
        return "OK:" + so
    except Exception as _e:  # noqa
        return "FAIL:" + repr(_e)


GSPLAT_SHIM = _install_gsplat_shim()
if GSPLAT_SHIM.startswith("FAIL"):
    print(json.dumps({"stage": "e1_setup", "status": "BLOCKED_GSPLAT_SHIM", "gsplat_shim": GSPLAT_SHIM}), flush=True)
    raise SystemExit(3)

ROOT = Path(CFG["work_root"]) / "package" / "COMAX0_MX_DSH"
PREREG = json.loads((ROOT / "config" / "prereg.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "comax_gs"))
import common
from common import eval_sh
from comax_eta_ckpt_clamp import comax_render_eta_ckpt as _comax_ckpt_clamp


def render_comax_clamp(model, cam, groups, eta=1.0, background=None, tile=16):
    if background is None:
        background = torch.zeros(3, device=model.get_xyz.device)
    cam_center = cam.viewmatrix.inverse()[:3, 3]
    dirs = F.normalize(model.get_xyz - cam_center, dim=-1)
    colors = (eval_sh(model.get_sh_degree, dirs, model.get_features) + 0.5).clamp(0, 1).contiguous()
    from gsplat.rendering import fully_fused_projection
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=model.get_xyz, covars=None, quats=model.get_rotation, scales=model.get_scaling,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=0.0, packed=False, opacities=model.get_opacity)
    chunk_tiles = int(os.environ.get("COMAX0_CKPT_CHUNK_TILES", "8"))
    out = _comax_ckpt_clamp(means2d[0].contiguous(), conics[0].contiguous(), depths[0].contiguous(),
                            radii[0].contiguous(), model.get_opacity.contiguous(), colors,
                            groups.contiguous(), cam.image_width, cam.image_height, eta=eta,
                            tile=tile, backgrounds=background, chunk_tiles=chunk_tiles)
    return out["rgb"]


common.render_comax = render_comax_clamp


def main():
    out_dir = Path(os.environ["COMAX0_OUTPUT_DIR"])
    dev = "cuda"
    scene, seed = "room", 17
    tol = float(PREREG["e2_rgb_abs_max"])
    views_env = os.environ.get("COMAX0_E1_DET_VIEWS", "0,40,96,240,304")
    test_idx = [int(x) for x in views_env.replace(" ", "").split(",") if x != ""]

    ck = common.load_checkpoint(CFG["checkpoint"], device=dev)
    groups = common.load_groups(scene, seed, device=dev)
    sys.path.insert(0, common.REPO)
    from scripts.epic05.phase7.dataset import GTDataset
    ds = GTDataset(scene=scene, repo_root=common.REPO, resolution="1080p", device=dev, background="black")
    bg = torch.zeros(3, device=dev)

    torch.cuda.reset_peak_memory_stats(torch.device(dev))

    def render_view(i):
        m = common.params_from_ckpt(ck, device=dev)
        for _p in (m._xyz, m._shs, m._opacity, m._scaling, m._rotation):
            _p.requires_grad_(False)
        cam, _gt = ds.get_item(i)
        with torch.no_grad():
            pred = render_comax_clamp(m, cam, groups, eta=1.0, background=bg).clamp(0, 1)
        del m
        torch.cuda.empty_cache()
        return pred.detach()

    rows = []
    all_pass = True
    for i in test_idx:
        a = render_view(i)
        b = render_view(i)
        fa = a.float(); fb = b.float()
        na = torch.isnan(fa); nb = torch.isnan(fb)
        nan_mask_match = bool((na == nb).all().item())
        nan_count_a = int(na.sum().item()); nan_count_b = int(nb.sum().item())
        finite_mask = ~(na | nb)
        if bool(finite_mask.any().item()):
            d_excl = float((fa[finite_mask] - fb[finite_mask]).abs().max().item())
        else:
            d_excl = 0.0
        ha = hashlib.sha256(fa.cpu().numpy().tobytes()).hexdigest()
        hb = hashlib.sha256(fb.cpu().numpy().tobytes()).hexdigest()
        hash_match = ha == hb
        finite_a = bool(torch.isfinite(fa).all().item())
        finite_b = bool(torch.isfinite(fb).all().item())
        view_pass = bool(nan_mask_match and (d_excl <= tol))
        all_pass = all_pass and view_pass
        rows.append({
            "global_view": int(i),
            "hash_match": hash_match,
            "nan_mask_match": nan_mask_match,
            "nan_count_run_a": nan_count_a,
            "nan_count_run_b": nan_count_b,
            "finite_run_a": finite_a,
            "finite_run_b": finite_b,
            "max_abs_diff_excl_nan": d_excl,
            "tolerance_rgb_abs_max": tol,
            "view_pass": view_pass,
        })

    peak_alloc = int(torch.cuda.max_memory_allocated(torch.device(dev)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(dev)))
    gate = {
        "pass": bool(all_pass),
        "renderer": "comax_render_eta_ckpt_clamp (power-clamp>=0, TF32-off)",
        "mode": "render_only_twice_nograd",
        "tolerance_rgb_abs_max": tol,
        "views": rows,
        "peak_e1_alloc_mib": round(peak_alloc / 1048576, 1),
        "peak_e1_reserved_mib": round(peak_reserved / 1048576, 1),
    }
    (out_dir / "DETERMINISM.json").write_text(json.dumps(gate, indent=2, allow_nan=False))
    with (out_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"stage": "e1_clamp_determinism", "pass": bool(all_pass),
                      "renderer": "clamp", "n_views": len(rows),
                      "views_pass": sum(1 for r in rows if r["view_pass"]),
                      "peak_reserved_mib": gate["peak_e1_reserved_mib"]}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
