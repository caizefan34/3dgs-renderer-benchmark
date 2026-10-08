#!/usr/bin/env python
"""probe_cuda_room - ROOM-SCALE production CUDA correctness (numerics only, NO timing).

The E2-prep anchor (COMAX0_E2_CORRECTNESS_PREP) validated the fused CUDA dense + envelope
kernels on an 8-gaussian / 1-tile SYNTHETIC fixture. That does not prove room-scale
correctness. This probe drives the REAL room checkpoint (scene=room, seed=17) through the
preregistered _fixture at larger scale and compares the production CUDA path against the
trusted tiled oracle comax_render_eta (which probe_real validates == comax_scan == native).

Comparisons (all <= e2_rgb_abs_max = 2e-5):
  cuda_dense    vs comax_render_eta   (production dense kernel vs trusted tiled oracle)
  cuda_envelope vs cuda_dense         (envelope fast path must be bit/near-exact vs dense)
  cuda_envelope vs comax_render_eta

Correctness-only. NO wall-clock, throughput, bandwidth, or E2 timing claim.
performance_evidence_valid=false, timing_run=false. Does NOT advance the master verdict.
"""
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "comax_gs"))
PROD = Path(os.environ.get("COMAX0_PROD_ROOT", "/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD"))
for _p in (str(PROD / "src"), str(PROD / "logs" / "torch_ext" / "comax_prod")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["COMAX0_PROD_ROOT"] = str(PROD)

import torch
import torch.nn.functional as F
from mx_guard import require_guarded_process
CFG, PROOF = require_guarded_process()  # guard BEFORE torch/CUDA heavy work

import common
from common import eval_sh
import run_e2
from comax_eta import comax_render_eta
from gsplat.rendering import isect_tiles, isect_offset_encode
import comax_prod_ext as cpe
import comax_envelope_ext as cee

DEV = "cuda"
F64 = torch.float64
PREREG_ABS = 2e-5
TILE = 16


def _abs_pct(vals):
    a = torch.as_tensor(vals, dtype=torch.float64).flatten().abs()
    if a.numel() == 0:
        return {"max": 0.0, "p50": 0.0, "p99": 0.0}
    return {"max": float(a.max()), "p50": float(torch.quantile(a, 0.50)),
            "p99": float(torch.quantile(a, 0.99))}


def _tiling(means2d, radii, depths, tw, th):
    m3 = means2d.unsqueeze(0) if means2d.dim() == 2 else means2d
    r3 = radii.unsqueeze(0) if radii.dim() == 2 else radii
    d3 = depths.unsqueeze(0) if depths.dim() == 1 else depths
    _tpg, _isect_ids, flatten_ids = isect_tiles(
        means2d=m3, radii=r3.clamp(min=0), depths=d3, tile_size=TILE,
        tile_width=tw, tile_height=th, sort=True, packed=False, n_images=1)
    offs_enc = isect_offset_encode(_isect_ids, 1, tw, th).reshape(-1).to(torch.long)
    n_isects = int(flatten_ids.numel())
    tile_offsets = torch.cat([offs_enc, torch.tensor([n_isects], device=means2d.device, dtype=torch.long)])
    # kernel reads tile_offsets.data_ptr<int>() => MUST be int32 (synthetic harness used int32).
    return flatten_ids.to(torch.int32), tile_offsets.to(torch.int32)


def _render_case(n_gauss, patch):
    sub, cam, groups, patch = run_e2._fixture("room", 17, n_gauss, patch, DEV)
    rgb_fast, proj = run_e2._fast_render(sub, cam, groups, patch, DEV)  # trusted tiled oracle
    m2, d, con, rad, model, colors = proj
    means2d = m2[0].contiguous()
    conics = con[0].contiguous()
    depths = d[0].contiguous()
    radii = rad[0].contiguous()
    opacity = model.get_opacity.contiguous()
    cols = colors.contiguous()
    grp = groups.contiguous().to(torch.int32)
    W = H = patch
    tw = (W + TILE - 1) // TILE
    th = (H + TILE - 1) // TILE
    bg = torch.zeros(3, device=DEV)
    tile_ids, tile_offsets = _tiling(means2d, radii, depths, tw, th)
    n_isect = int(tile_ids.numel())
    # production dense CUDA
    d_rgb, d_a, d_ctr = cpe.comax_render_dense(means2d, conics, opacity, cols, grp,
                                               tile_ids, tile_offsets, W, H, tw, th, background=bg)
    torch.cuda.synchronize()  # surface any dense-kernel illegal access HERE, before envelope
    # production envelope CUDA
    e_rgb, e_a, e_ctr = None, None, None
    env_err = None
    if not os.environ.get("PROBE_SKIP_ENVELOPE"):
        try:
            bundle = cee.comax_build_envelope(means2d, conics, opacity, cols, depths, grp,
                                              tile_ids, tile_offsets, tw * th, tw, th, W, H, background=bg)
            if os.environ.get("PROBE_DEBUG"):
                print("DBG n_tiles", bundle["n_tiles"], "TG", bundle["TG"], flush=True)
                print("DBG tile_group_count", bundle["tile_group_count"].tolist(), flush=True)
                print("DBG tile_groups", bundle["tile_groups"].tolist(), flush=True)
                print("DBG tile_offsets", tile_offsets.tolist(), flush=True)
                print("DBG n_isect", int(tile_ids.numel()), "groups_max", int(grp.max()), "G_distinct", int(grp.unique().numel()), flush=True)
            torch.cuda.synchronize()  # surface build_envelope host-prep errors
            e_rgb, e_a, e_ctr = cee.comax_render_envelope(bundle)
            torch.cuda.synchronize()  # surface envelope-kernel illegal access HERE
        except Exception as exc:  # noqa: BLE001
            e_rgb = None
            env_err = repr(exc)
    fast = rgb_fast.to(F64).reshape(-1, 3)
    d_f = d_rgb.to(F64).reshape(-1, 3)
    e_f = e_rgb.to(F64).reshape(-1, 3) if e_rgb is not None else None
    res = {"n_gauss": n_gauss, "patch": patch, "tw": tw, "th": th, "n_tiles": tw * th,
           "n_isect": n_isect, "groups_G": int(grp.unique().numel()) if grp.numel() else 0}
    res["oracle_nan"] = int(rgb_fast.isnan().sum())
    res["dense_nan"] = int(d_rgb.isnan().sum())
    res["envelope_nan"] = int(e_rgb.isnan().sum()) if e_rgb is not None else None
    # dense vs oracle on co-finite pixels (oracle may NaN on non-PSD conics at high overdraw)
    m = torch.isfinite(fast) & torch.isfinite(d_f)
    n_cmp = int(m.sum())
    res["dense_vs_oracle"] = _abs_pct((d_f - fast)[m].cpu()) if n_cmp else {"max": 0.0, "p50": 0.0, "p99": 0.0}
    res["dense_vs_oracle"]["pass"] = res["dense_vs_oracle"]["max"] <= PREREG_ABS
    res["dense_vs_oracle"]["n_compared"] = n_cmp
    res["dense_counters"] = d_ctr.tolist()
    if e_f is not None:
        # PRIMARY gate: envelope vs dense (both CUDA, dense is correct + NaN-free)
        res["envelope_vs_dense"] = _abs_pct((e_f - d_f).cpu())
        res["envelope_vs_dense"]["pass"] = res["envelope_vs_dense"]["max"] <= PREREG_ABS
        me = torch.isfinite(fast) & torch.isfinite(e_f)
        res["envelope_vs_oracle"] = _abs_pct((e_f - fast)[me].cpu()) if int(me.sum()) else {"max": 0.0, "p50": 0.0, "p99": 0.0}
        res["envelope_vs_oracle"]["pass"] = res["envelope_vs_oracle"]["max"] <= PREREG_ABS
        res["envelope_counters"] = e_ctr.tolist() if e_ctr is not None else None
    else:
        res["envelope_error"] = env_err
    res["case_pass"] = bool(res["dense_vs_oracle"]["pass"] and
                            (e_f is None or res.get("envelope_vs_dense", {}).get("pass", False)))
    return res


def main():
    import datetime
    cases = []
    _spec = os.environ.get("PROBE_CASES", "32,32;512,32;1024,64")
    _grid = [tuple(int(x) for x in c.split(",")) for c in _spec.split(";") if c.strip()]
    for (ng, pt) in _grid:
        try:
            cases.append(_render_case(ng, pt))
        except Exception as exc:  # noqa: BLE001
            cases.append({"n_gauss": ng, "patch": pt, "error": repr(exc), "traceback": traceback.format_exc()})
    all_pass = all(c.get("case_pass", False) for c in cases) and all("error" not in c for c in cases)
    payload = {
        "schema": "comax0.cuda_room_correctness.v1",
        "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": os.environ.get("COMAX0_RUN_ID"),
        "host": CFG.get("expected_hostname"),
        "gpu_uuid": PROOF.get("gpu_uuid"),
        "runner_pid": PROOF.get("runner_pid"),
        "scene": "room", "seed": 17, "tile": TILE,
        "tolerance_rgb_abs_max": PREREG_ABS,
        "oracle": "comax_render_eta (trusted tiled; probe_real-validated == comax_scan == native)",
        "performance_evidence_valid": False,
        "timing_run": False,
        "shared_gpu_numerics_only": False,
        "cases": cases,
        "all_pass": all_pass,
        "note": ("Room-scale production CUDA correctness vs trusted tiled oracle. "
                 "Numerics only; NO timing/throughput/bandwidth claim; does NOT advance verdict."),
    }
    out = os.environ.get("COMAX0_OUTPUT_DIR", ".")
    with open(os.path.join(out, "COMAX_CUDA_ROOM_CORRECTNESS.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
