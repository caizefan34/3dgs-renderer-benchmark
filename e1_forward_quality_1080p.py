#!/usr/bin/env python
"""E1_NUMERICS_SHARED_GPU_AMENDMENT: FORWARD-QUALITY-vs-GT numerics at the
preregistered primary resolution (1920x1080). NUMERICS ONLY; NO timing claim.

Fills the 1080p forward-quality gap. H0 measured quality-vs-GT only at 720p
(frozen kNN-64 grouping, PSNR 23.826 / SSIM 0.7893 vs GT). CLAMP_VS_ORACLE proved
the clamp is a faithful NaN-free production renderer but only vs the oracle, not vs
GT. This renders a deterministic subset of the FROZEN test split with (a) the
strongest qualified baseline (gsplat) and (b) the faithful NaN-free CLAMP CoMax
production renderer (eta=1, NO training), and reports PSNR/SSIM/LPIPS vs GT and
delta = CoMax - baseline against the preregistered quality gate.

This is a FORWARD-QUALITY measurement on the unrefined iter_15000 checkpoint, NOT
the full paired-refinement E1 pilot (1000 paired training steps), which is
budget-infeasible with the reference renderer. It does NOT change the master verdict.

Guard: require_guarded_process() BEFORE torch/CUDA (proof written by run_on_mx.py).
Render-only (no_grad). Provenance flags set; performance_evidence_valid=false.
"""
import os
import sys
import json
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> mx_guard
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
    print(json.dumps({"stage": "e1_fwd_quality_setup", "status": "BLOCKED_GSPLAT_SHIM", "gsplat_shim": GSPLAT_SHIM}), flush=True)
    raise SystemExit(3)

ROOT = Path(CFG["work_root"]) / "package" / "COMAX0_MX_DSH"
PREREG = json.loads((ROOT / "config" / "prereg.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "comax_gs"))
import common
from common import eval_sh
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


def render_comax_clamp(model, cam, groups, bg, chunk_tiles=8):
    means2d, conics, depths, radii, colors = _project(model, cam, groups)
    out = _clamp_scan(means2d[0].contiguous(), conics[0].contiguous(), depths[0].contiguous(),
                      radii[0].contiguous(), model.get_opacity.contiguous(), colors,
                      groups.contiguous(), cam.image_width, cam.image_height, eta=1.0,
                      tile=16, backgrounds=bg, chunk_tiles=chunk_tiles)
    return out["rgb"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        if not math.isfinite(o):
            return None
    return o


def main():
    out_dir = Path(os.environ["COMAX0_OUTPUT_DIR"])
    stride = int(os.environ.get("COMAX0_FQ_STRIDE", "7"))
    max_views = int(os.environ.get("COMAX0_FQ_MAX_VIEWS", "8"))
    chunk_tiles = int(os.environ.get("COMAX0_CKPT_CHUNK_TILES", "8"))

    ck = common.load_checkpoint(CFG["checkpoint"], device="cuda")
    groups = common.load_groups("room", 17, device="cuda")
    sys.path.insert(0, common.REPO)
    from scripts.epic05.phase7.dataset import GTDataset
    ds = GTDataset(scene="room", repo_root=common.REPO, resolution="1080p", device="cuda", background="black")
    n = len(ds)
    train_idx, test_idx = common.frozen_split(n)
    subset = test_idx[::stride][:max_views]  # deterministic, defined BEFORE measuring

    met = common.Metrics(device="cuda")
    bg = torch.zeros(3, device="cuda")

    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    rows = []
    for i in subset:
        m = common.params_from_ckpt(ck, device="cuda")
        for _p in (m._xyz, m._shs, m._opacity, m._scaling, m._rotation):
            _p.requires_grad_(False)
        cam, gt = ds.get_item(i)
        with torch.no_grad():
            pred_base = common.render_baseline(m, cam, background=bg)
            pred_comax = render_comax_clamp(m, cam, groups, bg, chunk_tiles=chunk_tiles)
        nan_base = int(torch.isnan(pred_base).sum().item())
        nan_comax = int(torch.isnan(pred_comax).sum().item())
        row = {
            "view_global": int(i),
            "baseline_psnr": float(met.psnr(pred_base, gt)),
            "baseline_ssim": float(met.ssim(pred_base, gt)),
            "baseline_lpips": float(met.lpips(pred_base, gt)),
            "comax_psnr": float(met.psnr(pred_comax, gt)),
            "comax_ssim": float(met.ssim(pred_comax, gt)),
            "comax_lpips": float(met.lpips(pred_comax, gt)),
            "baseline_nan_frac": nan_base / float(pred_base.numel()),
            "comax_nan_frac": nan_comax / float(pred_comax.numel()),
        }
        row["delta_psnr_db"] = row["comax_psnr"] - row["baseline_psnr"]
        row["delta_ssim"] = row["comax_ssim"] - row["baseline_ssim"]
        row["delta_lpips"] = row["comax_lpips"] - row["baseline_lpips"]
        rows.append(row)
        del m, pred_base, pred_comax
        torch.cuda.empty_cache()

    def agg(k):
        return float(np.mean([r[k] for r in rows]))

    d_psnr = agg("delta_psnr_db")
    d_ssim = agg("delta_ssim")
    d_lpips = agg("delta_lpips")
    complete = all(math.isfinite(x) for x in (d_psnr, d_ssim, d_lpips))
    if complete:
        quality_forward_pass = bool(
            d_psnr >= PREREG["quality_psnr_delta_min_db"] and
            d_ssim >= PREREG["quality_ssim_delta_min"] and
            d_lpips <= PREREG["quality_lpips_delta_max"])
    else:
        quality_forward_pass = None

    summary = {
        "schema": "comax0.e1_forward_quality_1080p.v1",
        "stage": "e1",
        "scene": "room",
        "seed": 17,
        "resolution": PREREG["primary_resolution"],
        "measured_on_mx": True,
        "hostname": PROOF["hostname"],
        "gpu_uuid": PROOF["gpu_uuid"],
        "renderer_baseline": "gsplat fully_fused_projection + alpha-composite (strongest qualified baseline)",
        "renderer_comax": "comax_eta_ckpt_clamp (power-clamp>=0, TF32-off), eta=1, chunk_tiles=" + str(chunk_tiles),
        "training": "NONE (forward-only, unrefined iter_15000 checkpoint)",
        "note": "FORWARD-QUALITY numerics at 1080p, NOT the full paired-refinement E1 pilot. Fills the 720p->1080p quality gap (H0 measured 720p only). Does NOT change the master verdict.",
        "subset_rule": "frozen test split, stride=" + str(stride) + ", max=" + str(max_views) + " views, defined BEFORE measuring",
        "test_split_size": int(len(test_idx)),
        "views": rows,
        "baseline_mean": {"psnr": agg("baseline_psnr"), "ssim": agg("baseline_ssim"), "lpips": agg("baseline_lpips")},
        "comax_mean": {"psnr": agg("comax_psnr"), "ssim": agg("comax_ssim"), "lpips": agg("comax_lpips")},
        "delta_mean": {"psnr_db": d_psnr, "ssim": d_ssim, "lpips": d_lpips},
        "metrics_complete": bool(complete),
        "thresholds": {"psnr_min": PREREG["quality_psnr_delta_min_db"],
                        "ssim_min": PREREG["quality_ssim_delta_min"],
                        "lpips_max": PREREG["quality_lpips_delta_max"]},
        "quality_forward_pass": quality_forward_pass,
        "peak_gpu_mem_mib": float(torch.cuda.max_memory_allocated(torch.device("cuda"))) / (1024.0 * 1024.0),
        "shared_gpu_numerics_only": True,
        "performance_evidence_valid": False,
        "timing_claim": False,
        "amendment": "E1_NUMERICS_SHARED_GPU_AMENDMENT",
    }
    (out_dir / "E1_FORWARD_QUALITY_1080P.json").write_text(json.dumps(_clean(summary), indent=2, allow_nan=False))
    print(json.dumps(_clean(summary), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
