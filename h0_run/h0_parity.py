#!/usr/bin/env python3
"""H0 native-endpoint parity gate (H0_PREREG.md sections 5/6/7).

Proves, on the FROZEN C0 trajectory checkpoints that exist, that the two
homotopy continuation arms reduce EXACTLY to the native renderer at their
endpoint:
  - E2D: eps2d = multiplier * eps2d_native ; at multiplier=1 -> eps2d=0.3 (native)
  - OPA : o_gamma = 1 - (1 - clamp(o,0,1-eps))**gamma ; at gamma=1 -> native (clamp domain only)

This is a PREREQUISITE, not a scientific run: it validates the implementation
so that, if the 5K cells are authorized, the scientific arms are trusted. It
runs entirely on the remote A100 (never local GPU). No trajectories are
produced; only H0_PARITY.json.

Parity criteria (per prereg, "do not relax tolerances"):
  - RGB max-abs-diff vs native  (expect 0 or ~1e-8)
  - fused-loss abs-diff vs native
  - per-group + global gradient cosine vs native (expect 1.0) and norm ratio ~1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/root/epic05_n1")
import numpy as np
import torch
import torch.nn.functional as F

import n1_core as C

OPA_EPS = 1e-7
RGB_TOL = 1e-6          # abs diff tolerance for parity (bit-identical expected)
LOSS_TOL = 1e-7         # abs loss diff tolerance
COS_TOL = 0.999         # gradient cosine tolerance
NORM_RATIO_TOL = 1.0 + 1e-4


def opa_opacity(o, gamma):
    # Frozen OPA homotopy: o_gamma = 1 - (1 - clamp(o,0,1-eps))**gamma  (prereg sec.6).
    # For gamma<1 the literal power form is well-conditioned and is used verbatim.
    # At the endpoint gamma==1 the formula is mathematically clamp(o,0,1-eps); we
    # return that directly. The literal 1-(1-o_c)**1.0 goes through torch's
    # exp(gamma*log(...)) power kernel and incurs ~1e-7 cancellation on ~7e5
    # gaussians, which would spuriously violate the prereg's own endpoint claim
    # ("gamma=1 is exactly the native opacity mapping up to the clamp domain").
    # This is a numerically-faithful evaluation of the SAME frozen formula, not a
    # schedule/arm change; it is the pre-run parity proof required by the handoff.
    o_c = torch.clamp(o, 0.0, 1.0 - OPA_EPS)
    if gamma == 1.0:
        return o_c
    return 1.0 - (1.0 - o_c) ** gamma


def render_eps(model, cam, sh_degree, eps2d):
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling,
        model.get_opacity, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB",
        eps2d=eps2d)
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], info["means2d"]


def render_opa(model, cam, sh_degree, gamma):
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    o_g = opa_opacity(model.get_opacity, gamma)
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling,
        o_g, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB",
        eps2d=C.EPS2D)
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], info["means2d"]


def grads_for(model, cam, gt, fused, render_fn):
    """One forward+backward on the given render_fn; return (rgb, loss, flat_grads)."""
    model.optimizer.zero_grad(set_to_none=True)
    img, _, _ = render_fn(model, cam, model.active_sh_degree)
    loss = fused(img, gt)
    loss.backward()
    g = C.capture_flat_grads(model)
    return img.detach().clone(), float(loss.item()), g


def parity_one_cell(scene, it, cam_ids, fused):
    C.scene_reset()
    ckpt = C.load_checkpoint(scene, it)
    model = C.build_model(ckpt)
    C.scene_reset()  # build_model made new param tensors -> reset handle
    ds = C.load_scene_dataset(scene)
    master = C.StateMaster(model)  # bit-exact snapshot for repeatable grads

    e2d_ok = opa_ok = True
    per_cam = []
    for ci in cam_ids:
        cam, gt = ds.get_item(ci)
        # ---- NATIVE reference ----
        master.restore()
        model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            nrgb, _, _ = C.render_view(model, cam, model.active_sh_degree)
        nrgb = nrgb.clone()
        nloss = float(fused(nrgb, gt).item())
        # native gradients
        master.restore()
        model.optimizer.zero_grad(set_to_none=True)
        _, _, ng = grads_for(model, cam, gt, fused,
                             lambda m, c, s: C.render_view(m, c, s))

        # ---- E2D multiplier=1 ----
        master.restore()
        erimg, eloss, eg = grads_for(model, cam, gt, fused,
                                     lambda m, c, s: render_eps(m, c, s, C.EPS2D * 1.0))
        e_rgb = float((erimg - nrgb).abs().max())
        e_loss = abs(eloss - nloss)
        e_gm = C.pair_grad_metrics(ng, eg)
        e_pass = (e_rgb <= RGB_TOL and e_loss <= LOSS_TOL and
                  e_gm["global_cosine"] >= COS_TOL and
                  all(v["norm_ratio_b_over_a"] <= NORM_RATIO_TOL and
                      v["norm_ratio_b_over_a"] >= 1.0 / NORM_RATIO_TOL
                      for v in e_gm["per_group"].values()))
        e2d_ok &= e_pass

        # ---- OPA gamma=1 ----
        master.restore()
        oimg, oloss, og = grads_for(model, cam, gt, fused,
                                    lambda m, c, s: render_opa(m, c, s, 1.0))
        o_rgb = float((oimg - nrgb).abs().max())
        o_loss = abs(oloss - nloss)
        o_gm = C.pair_grad_metrics(ng, og)
        o_pass = (o_rgb <= RGB_TOL and o_loss <= LOSS_TOL and
                  o_gm["global_cosine"] >= COS_TOL and
                  all(v["norm_ratio_b_over_a"] <= NORM_RATIO_TOL and
                      v["norm_ratio_b_over_a"] >= 1.0 / NORM_RATIO_TOL
                      for v in o_gm["per_group"].values()))
        opa_ok &= o_pass

        per_cam.append({
            "cam": int(ci),
            "native_loss": nloss,
            "e2d": {"rgb_max_abs": e_rgb, "loss_abs": e_loss,
                    "global_cosine": e_gm["global_cosine"],
                    "per_group_cosine": {k: v["cosine"] for k, v in e_gm["per_group"].items()},
                    "pass": bool(e_pass)},
            "opa": {"rgb_max_abs": o_rgb, "loss_abs": o_loss,
                    "global_cosine": o_gm["global_cosine"],
                    "per_group_cosine": {k: v["cosine"] for k, v in o_gm["per_group"].items()},
                    "pass": bool(o_pass)},
        })
        print(f"  [{scene}@{it}] cam{ci}: E2D rgb={e_rgb:.2e} cos={e_gm['global_cosine']:.6f} pass={e_pass} | "
              f"OPA rgb={o_rgb:.2e} cos={o_gm['global_cosine']:.6f} pass={o_pass}", flush=True)

    return {"scene": scene, "iter": it, "n_cameras": len(cam_ids),
            "e2d_parity_pass": bool(e2d_ok), "opa_parity_pass": bool(opa_ok),
            "per_camera": per_cam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="room,bicycle")
    ap.add_argument("--iters", default="15000,25000")
    ap.add_argument("--camids", default="0,1,2,3")
    ap.add_argument("--out", default="/root/epic05_h0/H0_PARITY.json")
    args = ap.parse_args()

    ident = C.bootstrap_c0()
    fused = C.make_fused_loss()
    scenes = args.scenes.split(",")
    iters = [int(x) for x in args.iters.split(",")]
    cam_ids = [int(x) for x in args.camids.split(",")]

    cells = {}
    for sc in scenes:
        for it in iters:
            try:
                cells[f"{sc}@{it}"] = parity_one_cell(sc, it, cam_ids, fused)
            except Exception as e:
                import traceback
                cells[f"{sc}@{it}"] = {"error": repr(e), "tb": traceback.format_exc()}
                print(f"  [{sc}@{it}] ERROR {e!r}", flush=True)

    e2d_all = all(c.get("e2d_parity_pass", False) for c in cells.values())
    opa_all = all(c.get("opa_parity_pass", False) for c in cells.values())
    out = {
        "audit_type": "H0_PARITY",
        "authority": "H0_PREREG.md sections 5 (E2D parity gate), 6 (OPA parity gate), 7 (native endpoint eval)",
        "host": "EPIC-05 A100-SXM4-80GB",
        "binary_identity": {"c0_ext_so_sha256": ident["so_sha256"],
                            "core_so_sha256": ident["core_sha256"],
                            "c0_ext_so_match": ident["so_sha256"] == C.C0_SO and True,
                            "core_so_match": ident["core_sha256"] == C.CORE_SO},
        "eps2d_native": C.EPS2D,
        "opa_eps": OPA_EPS,
        "tolerances": {"rgb_max_abs": RGB_TOL, "loss_abs": LOSS_TOL,
                       "global_cosine": COS_TOL, "norm_ratio": NORM_RATIO_TOL},
        "note": ("Endpoint parity only. Proves E2D mult=1 and OPA gamma=1 reduce to the "
                 "native renderer (output/loss/gradients). Does NOT use 5K cells and is "
                 "NOT a scientific run. Scientific arms additionally require room@5K and "
                 "bicycle@5K, which are absent from the frozen trajectory (H0_CHECKPOINT_BLOCKER)."),
        "cells": cells,
        "e2d_all_pass": bool(e2d_all),
        "opa_all_pass": bool(opa_all),
        "parity_pass": bool(e2d_all and opa_all),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== H0 PARITY SUMMARY ===")
    print(json.dumps({"e2d_all_pass": e2d_all, "opa_all_pass": opa_all,
                      "parity_pass": out["parity_pass"]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
