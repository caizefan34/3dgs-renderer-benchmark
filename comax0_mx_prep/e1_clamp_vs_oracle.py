#!/usr/bin/env python3
"""E1_NUMERICS_SHARED_GPU_AMENDMENT: does the CLAMP production renderer match the
trusted NON-CLAMP oracle (comax_render_eta_ckpt) on the oracle's finite pixels?

Motivation: DET2 proved the non-clamp oracle is NaN-nondeterministic on ~3/5 views
(inf from non-PSD conics -> float reassociation -> NaN mask varies). The clamp
renderer (power>=0, TF32-off) eliminated the NaN and is bitwise-deterministic
(e1_clamp_det2: 5/5 views, nan_count=0). This test answers the follow-up: is the
clamp a FAITHFUL production renderer (matches the oracle where the oracle is finite,
<= frozen e2_rgb_abs_max) or does it DIVERGE?

Render-only (no_grad). NUMERICS ONLY; NO timing/performance claim. For each view
renders once with the oracle and once with the clamp; reports oracle/clamp NaN
counts, max_abs_diff on co-finite pixels, and the count of pixels the clamp
"fixed" (oracle NaN, clamp finite).

Guard: require_guarded_process() BEFORE torch/CUDA (proof from launcher).
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mx_guard import require_guarded_process

CFG, PROOF = require_guarded_process()  # BEFORE torch/CUDA

os.environ["COMAX0_POWER_CLAMP"] = "1"

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
from comax_eta_ckpt import comax_render_eta_ckpt as _oracle_scan
from comax_eta_ckpt_clamp import comax_render_eta_ckpt as _clamp_scan


def _project(model, cam, groups):
    cam_center = cam.viewmatrix.inverse()[:3, 3]
    dirs = F.normalize(model.get_xyz - cam_center, dim=-1)
    colors = (eval_sh(model.get_sh_degree, dirs, model.get_features) + 0.5).clamp(0, 1).contiguous()
    from gsplat.rendering import fully_fused_projection
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=model.get_xyz, covars=None, quats=model.get_rotation, scales=model.get_scaling,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=0.0, packed=False, opacities=model.get_opacity)
    return means2d, conics, depths, radii, colors


def render_with(scan_fn, model, cam, groups, bg, chunk_tiles=8):
    means2d, conics, depths, radii, colors = _project(model, cam, groups)
    out = scan_fn(means2d[0].contiguous(), conics[0].contiguous(), depths[0].contiguous(),
                  radii[0].contiguous(), model.get_opacity.contiguous(), colors,
                  groups.contiguous(), cam.image_width, cam.image_height, eta=1.0,
                  tile=16, backgrounds=bg, chunk_tiles=chunk_tiles)
    return out["rgb"]


def main():
    out_dir = Path(os.environ["COMAX0_OUTPUT_DIR"])
    tol = float(PREREG["e2_rgb_abs_max"])
    views_env = os.environ.get("COMAX0_CVO_VIEWS", "0,240")
    test_idx = [int(x) for x in views_env.replace(" ", "").split(",") if x != ""]
    chunk_tiles = int(os.environ.get("COMAX0_CKPT_CHUNK_TILES", "8"))

    ck = common.load_checkpoint(CFG["checkpoint"], device="cuda")
    groups = common.load_groups("room", 17, device="cuda")
    sys.path.insert(0, common.REPO)
    from scripts.epic05.phase7.dataset import GTDataset
    ds = GTDataset(scene="room", repo_root=common.REPO, resolution="1080p", device="cuda", background="black")
    bg = torch.zeros(3, device="cuda")

    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    rows = []
    for i in test_idx:
        m = common.params_from_ckpt(ck, device="cuda")
        for _p in (m._xyz, m._shs, m._opacity, m._scaling, m._rotation):
            _p.requires_grad_(False)
        cam, _gt = ds.get_item(i)
        with torch.no_grad():
            ro = render_with(_oracle_scan, m, cam, groups, bg, chunk_tiles).clamp(0, 1).float()
            rc = render_with(_clamp_scan, m, cam, groups, bg, chunk_tiles).clamp(0, 1).float()
        no = torch.isnan(ro); nc = torch.isnan(rc)
        cof = ~(no | nc)
        d_cof = float((ro[cof] - rc[cof]).abs().max().item()) if bool(cof.any().item()) else 0.0
        fixed = int((no & ~nc).sum().item())
        rows.append({
            "global_view": int(i),
            "oracle_nan_count": int(no.sum().item()),
            "clamp_nan_count": int(nc.sum().item()),
            "cofinite_pixels": int(cof.sum().item()),
            "max_abs_diff_cofinite": d_cof,
            "pixels_clamp_fixed": fixed,
            "tolerance_rgb_abs_max": tol,
            "clamp_matches_oracle_cofinite": bool(d_cof <= tol),
        })
        del m, ro, rc
        torch.cuda.empty_cache()

    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device("cuda")))
    gate = {
        "pass": bool(all(r["clamp_matches_oracle_cofinite"] for r in rows)),
        "comparison": "clamp(comax_render_eta_ckpt_clamp) vs oracle(comax_render_eta_ckpt)",
        "mode": "render_once_each_nograd",
        "tolerance_rgb_abs_max": tol,
        "chunk_tiles": chunk_tiles,
        "views": rows,
        "peak_e1_reserved_mib": round(peak_reserved / 1048576, 1),
    }
    (out_dir / "CLAMP_VS_ORACLE.json").write_text(json.dumps(gate, indent=2, allow_nan=False))
    print(json.dumps({"stage": "e1_clamp_vs_oracle", "pass": gate["pass"],
                      "views": [(r["global_view"], r["oracle_nan_count"], r["clamp_nan_count"],
                                 r["max_abs_diff_cofinite"], r["pixels_clamp_fixed"]) for r in rows],
                      "peak_reserved_mib": gate["peak_e1_reserved_mib"]}), flush=True)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
