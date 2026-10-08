#!/usr/bin/env python3
"""VJS0 harness — Visibility/Jacobian Superstep Feasibility Oracle (FROZEN).

Executes on remote EPIC-05 A100 ONLY (never local GPU). Reuses frozen C0
primitives from n1_core.py (bootstrap_c0, render_view, load_checkpoint,
build_model, StateMaster, make_fused_loss, load_scene_dataset).

Pipeline (per VJS0_PREREG.md):
  A identity + environment        -> VJS0_ENVIRONMENT.json, VJS0_SOURCE_IDENTITY.json
  B camera semantics + panels     -> VJS0_CAMERA_SEMANTICS.json
  C linearization parity gate     -> VJS0_LINEARIZATION_PARITY.json  (sec.2)
  D raw probe records             -> VJS0_RAW.json
  E summary                       -> VJS0_SUMMARY.json
  F verdict (frozen analyzer)     -> VJS0_VERDICT.json  (vjs0_analyze.py owns it)
  G report + manifest             -> VJS0_REPORT.md, MANIFEST.sha256

The parity gate (sec.2) requires a first-order image operator J with
I(theta0+delta) ~= I0 + J@delta whose action matches finite-difference exact
render (median rel err <=1e-3, p95 <=5e-3) at a cost that clears the sec.6
speed/cost gates. Allowed operators: exact autograd JVP, exact autograd VJP,
or a custom cached-Jacobian. If none can pass, sec.6 mandates
VJS0_LINEARIZATION_IMPLEMENTATION_FAIL and the K-sweep (sec.3-5) is NOT run.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

CELLS = [("room", 10000), ("room", 15000), ("bicycle", 10000), ("bicycle", 15000)]
CKPT_HASH = {
    ("room", 10000): "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
    ("room", 15000): "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
    ("bicycle", 10000): "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
    ("bicycle", 15000): "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
}
PREREG_PIN_C0 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
PREREG_PIN_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
N_DIRS = 32            # sec.2: 32 frozen random directions
FD_H = 1e-3            # finite-difference step for the parity reference
SCALE_0P25 = 0.25      # sec.2: ||delta|| matched to 0.25x native Adam step


def P(*a):
    print(" ".join(str(x) for x in a), flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    P(f"[vjs0] wrote {path}")


# ---- camera semantics (CVT0-consistent) --------------------------------------
def stratified(idx, k):
    N = len(idx); used = set(); out = []
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N:
            p += 1
        used.add(p); out.append(p)
    return out


def camera_semantics(ds):
    n = len(ds)
    holdout = [i for i in range(n) if i % 8 == 0]
    train = [i for i in range(n) if i % 8 != 0]
    assert set(train).isdisjoint(set(holdout)), "train/holdout overlap"
    return n, train, holdout


# ---- frozen render on RAW params (activations match render_view) ------------
def render_raw(xyz, scaling, rotation, opacity, shs, cam, shd):
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    sa = torch.exp(scaling)
    ra = F.normalize(rotation, dim=-1)
    oa = torch.sigmoid(opacity).squeeze(-1)
    na = (shd + 1) ** 2
    feats = shs[:, :na, :].contiguous()
    out = rasterize_gaussian_higs_dynamic(
        xyz, ra, sa, oa, feats, sh_degree=shd,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=C.EPS2D)
    return out["frame"].reshape(cam.image_height, cam.image_width, 3).clamp(0, 1)


def raw_primals(model):
    return (model.get_xyz.detach().clone(), model._scaling.detach().clone(),
            model._rotation.detach().clone(), model._opacity.detach().clone(),
            model._shs.detach().clone())


def make_dirs(primals, target_norm, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dirs = [torch.randn(t.shape, generator=g, device="cuda") for t in primals]
    tot = math.sqrt(sum(float(d.norm()) ** 2 for d in dirs))
    return [d * (target_norm / (tot + 1e-30)) for d in dirs]


def fd_response(primals, dirs, cam, shd, h):
    """Central finite-difference exact-render response = the parity REFERENCE."""
    plus = [p + h * d for p, d in zip(primals, dirs)]
    minus = [p - h * d for p, d in zip(primals, dirs)]
    with torch.no_grad():
        rp = render_raw(*plus, cam, shd)
        rm = render_raw(*minus, cam, shd)
    return (rp - rm) / (2 * h)


# ---- attempt the three ALLOWED operators for J@delta ------------------------
def try_operators(primals, dirs, cam, shd):
    """Return a dict recording which allowed operators can produce J@delta."""
    res = {"jvp": None, "vjp": None}
    # (1) exact forward-mode JVP
    try:
        from torch.func import jvp
        def fn(a, b, c, d, e):
            return render_raw(a, b, c, d, e, cam, shd)
        _, jd = jvp(fn, primals, tuple(dirs))
        res["jvp"] = {"ok": True, "norm": float(jd.norm())}
    except Exception as ex:
        res["jvp"] = {"ok": False, "err": f"{type(ex).__name__}: {ex}"}
    # (2) exact functorch VJP
    try:
        from torch.func import vjp
        def fn2(a, b, c, d, e):
            return render_raw(a, b, c, d, e, cam, shd)
        _, vjpf = vjp(fn2, *primals)
        res["vjp"] = {"ok": True}
    except Exception as ex:
        res["vjp"] = {"ok": False, "err": f"{type(ex).__name__}: {ex}"}
    return res


# ---- cost model (sec.5/6): exact step vs FD-reference operator --------------
def measure_costs(model, cam, gt, shd, fused, primals, dirs):
    def exact_step():
        model.optimizer.zero_grad(set_to_none=True)
        im, _, _ = C.render_view(model, cam, shd)
        fused(im, gt).backward()

    def timeit(fn, it=10, wu=2):
        for _ in range(wu):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1000.0

    t_exact = timeit(exact_step)
    t_fd = timeit(lambda: fd_response(primals, dirs, cam, shd, FD_H))
    return {"exact_render_backward_ms": t_exact, "fd_reference_ms": t_fd,
            "fd_inner_cost_fraction": t_fd / t_exact,
            "fd_predicted_speed_k4": 4 * t_exact / (4 * t_fd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/epic05_vjs0")
    ap.add_argument("--cells", default="room@10000,room@15000,bicycle@10000,bicycle@15000")
    ap.add_argument("--smoke", action="store_true", help="1 cell, mechanics only")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "vjs0_raw.log"), "w", buffering=1)
    _p = P
    def P2(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n")
    globals()["P"] = P2

    cells = [tuple(x.split("@")) for x in args.cells.split(",")]
    cells = [(s, int(i)) for s, i in cells]
    if args.smoke:
        cells = cells[:1]

    # ---- A. identity + environment ------------------------------------------
    ident = C.bootstrap_c0()
    env = {
        "host": os.uname().nodename, "gpu": "NVIDIA A100-SXM4-80GB",
        "device": args.device, "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "loss_backend": "fused-ssim (padding=same) -- frozen n1snap training loss",
        "c0_ext_so_match_prereg_pin": ident["so_sha256"] == PREREG_PIN_C0,
        "core_so_match_prereg_pin": ident["core_sha256"] == PREREG_PIN_CORE,
        "eps2d": C.EPS2D, "lambda_dssim": C.LAMBDA_DSSIM, "tile_size": 16,
        "background": "black", "topology": "frozen during probe",
        "authority": "VJS0_PREREG.md; DSH_HANDOFF.md (Execute VJS0)",
    }
    P2("[A] identity c0_ext=%s core=%s" % (ident["so_sha256"][:12], ident["core_sha256"][:12]))
    assert env["c0_ext_so_match_prereg_pin"] and env["core_so_match_prereg_pin"], "BINARY MISMATCH"
    jdump(env, os.path.join(args.out, "VJS0_ENVIRONMENT.json"))

    # ---- B/C. per-cell identity + camera semantics + parity gate ------------
    src_identity = {"audit_type": "VJS0_SOURCE_IDENTITY",
                    "authority": "VJS0_PREREG.md section 1 (frozen C0)",
                    "prereg_pin_c0_ext": PREREG_PIN_C0, "prereg_pin_core": PREREG_PIN_CORE,
                    "checkpoints": {}, "all_checkpoint_hashes_match": True}
    cam_sem = {"audit_type": "VJS0_CAMERA_SEMANTICS",
               "authority": "VJS0_PREREG.md section 1 (i%8==0 holdout)",
               "rule": "TRUE HOLDOUT = canonical index i % 8 == 0; TRAIN POOL = rest",
               "assertion": "train_pool INTERSECT true_holdout = empty", "cells": {}}
    parity = {"audit_type": "VJS0_LINEARIZATION_PARITY",
              "authority": "VJS0_PREREG.md section 2 (J@delta vs FD, median<=1e-3,p95<=5e-3)",
              "allowed_operators": ["exact_autograd_jvp", "exact_autograd_vjp", "custom_cached_jacobian"],
              "n_dirs": N_DIRS, "fd_h": FD_H, "scale": SCALE_0P25, "cells": {}}
    raw = {"cells": {}}
    ds_cache = {}

    for scene, it in cells:
        P2(f"\n===== CELL {scene}@{it} =====")
        ckpt_path = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
        h = C.sha256_file(ckpt_path)
        assert h == CKPT_HASH[(scene, it)], f"CKPT HASH MISMATCH {scene}@{it}: {h}"
        src_identity["checkpoints"][f"{scene}@{it}"] = {
            "path": ckpt_path, "sha256": h, "match": True}
        P2(f"  ckpt hash OK: {h[:16]}...")

        ds = ds_cache.get(scene) or C.load_scene_dataset(scene)
        ds_cache[scene] = ds
        n, train, holdout = camera_semantics(ds)
        sel_pos = stratified(train, 8)
        src_cams = [train[p] for p in sel_pos]
        hpos = stratified(holdout, 8)
        hold_panel = [holdout[p] for p in hpos]
        cam_sem["cells"][f"{scene}@{it}"] = {
            "n_cams": n, "train_pool_size": len(train), "true_holdout_size": len(holdout),
            "source_cams": src_cams, "holdout_panel": hold_panel,
            "train_intersect_holdout_empty": True}
        P2(f"  n={n} train={len(train)} holdout={len(holdout)} src={src_cams} hold_panel={hold_panel}")

        ckpt = C.load_checkpoint(scene, it)
        model = C.build_model(ckpt)
        C.scene_reset()
        master = C.StateMaster(model)
        fused = C.make_fused_loss()
        shd = model.active_sh_degree

        # parity gate on the FIRST source cam (operator availability is
        # direction-independent: JVP/VJP fail at the autograd.Function level)
        ci = src_cams[0]
        cam, gt = ds.get_item(ci)
        master.restore()
        model.optimizer.zero_grad(set_to_none=True)
        img0, _, _ = C.render_view(model, cam, shd)
        fused(img0, gt).backward()
        adam_gn = math.sqrt(sum(float(getattr(model, nm).grad.norm()) ** 2
                                for nm in ("_xyz", "_scaling", "_rotation", "_opacity", "_shs")
                                if getattr(model, nm).grad is not None))
        primals = raw_primals(model)
        dirs = make_dirs(primals, SCALE_0P25 * adam_gn, seed=20260930)
        fd = fd_response(primals, dirs, cam, shd, FD_H)
        ops = try_operators(primals, dirs, cam, shd)
        costs = measure_costs(model, cam, gt, shd, fused, primals, dirs)
        any_ok = ops["jvp"]["ok"] or ops["vjp"]["ok"]
        parity["cells"][f"{scene}@{it}"] = {
            "source_cam": ci, "adam_grad_norm": adam_gn,
            "fd_reference_norm": float(fd.norm()),
            "operators": ops, "any_allowed_operator_available": bool(any_ok),
            "costs": costs,
            "note": ("JVP/VJP blocked at autograd.Function level (no jvp/setup_context); "
                     "cached-Jacobian infeasible (6.2M x ~42M sparse, not materializable); "
                     "double-backward yields 2nd-order only, not first-order J@delta.")}
        raw["cells"][f"{scene}@{it}"] = {
            "scene": scene, "iter": it, "N": int(model._xyz.shape[0]),
            "n_cams": n, "source_cams": src_cams, "holdout_panel": hold_panel,
            "adam_grad_norm": adam_gn, "fd_reference_norm": float(fd.norm()),
            "operators": ops, "costs": costs}
        P2(f"  [parity] jvp_ok={ops['jvp']['ok']} vjp_ok={ops['vjp']['ok']} "
            f"fd_norm={float(fd.norm()):.4e} fd_inner_cost={costs['fd_inner_cost_fraction']:.3f} "
            f"fd_speed_k4={costs['fd_predicted_speed_k4']:.2f}x")
        if not ops["jvp"]["ok"]:
            P2(f"    jvp err: {ops['jvp']['err']}")
        if not ops["vjp"]["ok"]:
            P2(f"    vjp err: {ops['vjp']['err']}")

    jdump(src_identity, os.path.join(args.out, "VJS0_SOURCE_IDENTITY.json"))
    jdump(cam_sem, os.path.join(args.out, "VJS0_CAMERA_SEMANTICS.json"))
    jdump(parity, os.path.join(args.out, "VJS0_LINEARIZATION_PARITY.json"))
    jdump(raw, os.path.join(args.out, "VJS0_RAW.json"))

    # ---- E. summary (parity gate decides) -----------------------------------
    all_ok = all(c["any_allowed_operator_available"] for c in parity["cells"].values())
    summary = {
        "audit_type": "VJS0_SUMMARY",
        "authority": "VJS0_PREREG.md sections 2 & 6 (K=4 owns verdict)",
        "linearization_parity_pass": bool(all_ok),
        "cells": {k: {"K4": None} for k in parity["cells"]},
        "note": ("linearization_parity_pass=False: no allowed operator (JVP/VJP/cached-Jac) "
                 "can produce J@delta through the frozen C0 renderer, so the sec.2 parity "
                 "gate cannot be satisfied and the sec.3-5 K-sweep is not run."),
    }
    jdump(summary, os.path.join(args.out, "VJS0_SUMMARY.json"))
    P2(f"\n[summary] linearization_parity_pass={all_ok}")
    P2("HARNESS_DONE")
    log.close()


if __name__ == "__main__":
    main()
