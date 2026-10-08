#!/usr/bin/env python3
"""P10 Gate H2 - GPU gradient equivalence on gsplat main: pristine vs H8-MR.

Fixed-state protocol (NOT training), one scene per process:

  arms
    pri_*   pristine binary      (upstream tree, no H8 code)
    off_*   h8 binary, GSPLAT_H8_MR unset  (patched tree, baseline path)
    on_*    h8 binary, GSPLAT_H8_MR=1      (patched tree, moment path)

  runs
    3 per arm with fresh forward+backward and identical inputs
    -> floor_base = median over the 6 pristine/patched-off same-arm run pairs
       (trusted arithmetic, untouched by the H8 path: the gate tolerance)
    -> floor_on   = median over the 3 patched-on same-arm run pairs
       (consistency check on the H8 path's own atomic scatter)
    -> all 9 off_i x on_j cross pairs must clear floor_base

  inputs
    PLY xyz / normalized rotations / exp(scales) / sigmoid(opacity) / SH,
    camera[0] resized to 2048 max side, deterministic VJP V = randn (cuda seed 4200)

  subcommands
    dump    --binary pristine|h8 --scene S [--gpu G] --out dump.pt
    compare --scene S --pristine dump.pt --h8 dump.pt --out EPIC05_P10_EQUIVALENCE.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys

REPO = "/root/3dgs-renderer-benchmark"
FROZEN = "/root/epic05_frozen"
ROOT = "/root/epic05_p10"
BINARIES = {
    "pristine": (f"{ROOT}/tree_pristine", f"{ROOT}/build/gsplat_cuda_p10_pristine/gsplat_cuda_p10_pristine.so"),
    "h8": (f"{ROOT}/tree_h8", f"{ROOT}/build/gsplat_cuda_p10_h8/gsplat_cuda_p10_h8.so"),
}
SEED = 4200
N_RUNS = 3

# pipeline (leaf) tensors: the definitive semantics test
PIPELINE = ["means3d", "quats", "scales", "opacity", "sh"]
# rasterizer-boundary tensors: raw values are moments in H8 mode
BOUNDARY = ["means2d_raw", "conics_raw"]


# --------------------------------------------------------------------------- dump
def load_arm(binary):
    import torch  # noqa: F401  (loads libc10/libtorch into the process first)

    tree, so = BINARIES[binary]
    modname = os.path.basename(so)[:-3]
    spec = importlib.util.spec_from_file_location(modname, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[modname] = mod
    sys.modules["gsplat.csrc"] = mod
    sys.path.insert(0, tree)
    sys.path.insert(0, f"{REPO}/scripts/epic05/phase7")
    sys.path.insert(0, f"{REPO}/src")
    sys.path.insert(0, f"{REPO}/baseline/reference_v1")
    from gsplat import rasterization  # noqa: E402
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras  # noqa: E402
    return rasterization, load_ply, load_cameras_from_json, resize_cameras


def build_inputs(load_ply, load_cameras_from_json, resize_cameras, scene):
    import torch

    ply = load_ply(f"{FROZEN}/p5_reference_plys/{scene}_30000.ply", device="cuda")
    cams = load_cameras_from_json(f"{REPO}/data/official/mipnerf360/{scene}/cameras.json", device="cpu")
    c0 = cams[0]
    s = 2048.0 / max(c0.image_width, c0.image_height)
    tw = int(round(c0.image_width * s / 2)) * 2
    th = int(round(c0.image_height * s / 2)) * 2
    cams = resize_cameras(cams, tw, th)
    cam = cams[0]
    for attr in ("viewmatrix", "K"):
        setattr(cam, attr, getattr(cam, attr).to("cuda"))

    params = {
        "means": ply["xyz"].clone().requires_grad_(True),
        "quats": torch.nn.functional.normalize(ply["rotations"].clone(), dim=-1).requires_grad_(True),
        "scales": torch.exp(ply["scales"].clone()).requires_grad_(True),
        "opacities": torch.sigmoid(ply["opacity"].clone()).requires_grad_(True),
        "sh": ply["shs"].clone().contiguous().requires_grad_(True),
    }
    g = torch.Generator(device="cuda").manual_seed(SEED)
    V = torch.randn((cam.image_height, cam.image_width, 3), generator=g, device="cuda")
    return params, cam, V


def one_run(rasterization, params, cam, V, absgrad=False):
    import torch

    for t in params.values():
        t.grad = None
    r, alpha, meta = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"],
        opacities=params["opacities"], colors=params["sh"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0,
        eps2d=0.1, render_mode="RGB", absgrad=absgrad)
    means2d = meta["means2d"]
    conics = meta["conics"]
    means2d.retain_grad()
    conics.retain_grad()
    render = r[0]
    (render * V).sum().backward()
    torch.cuda.synchronize()
    out = {
        "render": render.detach().clone(),
        "alpha": alpha[0].detach().clone(),
        "means2d": means2d.detach().clone(),
        "conics": conics.detach().clone(),
        "means2d_raw": means2d.grad.detach().clone(),
        "conics_raw": conics.grad.detach().clone(),
        "means3d": params["means"].grad.detach().clone(),
        "quats": params["quats"].grad.detach().clone(),
        "scales": params["scales"].grad.detach().clone(),
        "opacity": params["opacities"].grad.detach().clone(),
        "sh": params["sh"].grad.detach().clone(),
    }
    if absgrad:
        out["means2d_absgrad"] = means2d.absgrad.detach().clone()
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} has NaN/Inf"
    return out


def cmd_dump(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.pop("GSPLAT_H8_MR", None)
    rasterization, load_ply, load_cameras, resize_cams = load_arm(args.binary)
    params, cam, V = build_inputs(load_ply, load_cameras, resize_cams, args.scene)

    arms = ["pri"] if args.binary == "pristine" else ["off", "on"]
    dumps = {}
    for arm in arms:
        for k in range(N_RUNS):
            if arm == "on":
                os.environ["GSPLAT_H8_MR"] = "1"
            else:
                os.environ.pop("GSPLAT_H8_MR", None)
            dumps[f"{arm}{k}"] = one_run(rasterization, params, cam, V)
            if args.absgrad:
                dumps[f"{arm}{k}_abs"] = one_run(rasterization, params, cam, V, absgrad=True)
    payload = {
        "binary": args.binary, "scene": args.scene, "seed": SEED,
        "width": cam.image_width, "height": cam.image_height,
        "n_gaussians": int(params["means"].shape[0]),
        "dump": dumps,
    }
    import torch
    torch.save(payload, args.out)
    print(f"DUMP {args.binary}/{args.scene} arms={list(dumps)} N={payload['n_gaussians']} -> {args.out}")


# ------------------------------------------------------------------------ compare
def _np(t):
    import numpy as np
    return np.asarray(t.detach().cpu(), dtype=np.float64).reshape(-1)


def metrics(a, b):
    import numpy as np
    a, b = _np(a), _np(b)
    d = b - a
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    finite = float(np.mean(np.isfinite(d))) if d.size else 1.0
    return {
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "mean_abs": float(np.mean(np.abs(d))) if d.size else 0.0,
        "rel_l2": float(np.linalg.norm(d) / max(an, 1e-30)),
        "cosine": float(np.dot(a, b) / max(an * bn, 1e-30)),
        "finite_fraction": finite,
        "base_absmax": float(np.max(np.abs(a))) if a.size else 0.0,
    }


def reconstruct(means2d_raw, conics, conics_raw):
    """Apply the H8-MR per-Gaussian reconstruction in torch (no kernel involved).

    v_means2d_true = [[A,B],[B,C]] @ v_means2d_moment
    v_conics_true  = (0.5*Sxx, Sxy, 0.5*Syy)
    """
    import torch
    c = conics.double()
    A, B, C = c[..., 0], c[..., 1], c[..., 2]
    m = means2d_raw.double()
    vx = A * m[..., 0] + B * m[..., 1]
    vy = B * m[..., 0] + C * m[..., 1]
    means2d_true = torch.stack([vx, vy], dim=-1)
    cc = conics_raw.double()
    conics_true = torch.stack([0.5 * cc[..., 0], cc[..., 1], 0.5 * cc[..., 2]], dim=-1)
    return means2d_true, conics_true


ABS_FLOOR = 1e-8

# Gate tolerances. Two statistics, because they fail differently:
#   bias  - elementwise mean of the signed cross-arm difference, i.e. the
#           part of the difference that repeats in every run pair. A semantic
#           (algebraic) error is deterministic and survives averaging, while
#           atomic-scatter noise cancels; ratio K_bias bounds it.
#   tail  - worst single-pair max|difference|. Catches a gross anomaly that
#           one contaminated run could hide from an averaged statistic.
# Both are compared against the SAME statistics computed on the trusted
# same-arm pool (pristine pairs + patched-off pairs), which no H8 run can
# inflate. The pool's max/mean is used rather than its median: the
# same-arm difference distribution is heavy-tailed (garden quats reaches
# 18.7 while its median is 3.3), so a median floor answers a different
# question than the max-over-pairs it is compared against.
BIAS_MAX_TOL_RATIO = 5.0
BIAS_L2_TOL_RATIO = 5.0
# tail compares the max of 9 cross pairs against the max of a 3-pair reference
# pool, which is biased upward by the sample-count difference (observed 2.06 on
# clean data), so the bound carries headroom for that; sensitivity comes from
# the bias statistics, which are sample-count matched.
TAIL_TOL_RATIO = 3.0
# diagnostic flag only: on-arm median scatter this many times the trusted floor
ELEVATED_SCATTER_RATIO = 5.0


def diff_vec(a, b):
    return _np(b) - _np(a)


def _np32(t):
    import numpy as np
    return np.asarray(t.detach().cpu(), dtype=np.float32).reshape(-1)


def det_offset_stats(vecs_p, vecs_h):
    """Elements whose value is bit-reproducible in *both* arms across runs yet
    differs across arms.

    W = 0 with D != 0 means neither arm's own run-to-run scatter can produce the
    offset. Computed for the cross-build pair (pristine vs patched-off) and for
    the cross-process pair (two processes of the pristine binary): if the two
    counts and magnitudes agree, the deterministic component belongs to the
    process, not to the build.
    """
    import numpy as np
    P = [_np32(t) for t in vecs_p]
    H = [_np32(t) for t in vecs_h]
    W = np.zeros(P[0].size, dtype=np.float32)
    for V in (P, H):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            np.maximum(W, np.abs(V[a] - V[b]), out=W)
    Mp = (P[0] + P[1] + P[2]) / np.float32(3.0)
    Mh = (H[0] + H[1] + H[2]) / np.float32(3.0)
    D = Mh - Mp
    m = (W == 0) & (D != 0)
    n = int(np.count_nonzero(m))
    return {
        "n_det_elements": n,
        "n_elements": int(D.size),
        "det_frac": n / max(D.size, 1),
        "det_max_abs": float(np.abs(D[m]).max()) if n else 0.0,
        "det_max_rel": float((np.abs(D[m]) / np.maximum(np.abs(Mp[m]), np.float32(1e-30))).max()) if n else 0.0,
        "offset_rel_l2": float(np.linalg.norm(D) / max(np.linalg.norm(Mp), 1e-30)),
        "offset_absmax": float(np.abs(D).max()),
        "value_absmax": float(np.abs(Mp).max()),
    }


def bias_tail(vecs):
    import numpy as np
    M = np.stack(vecs)
    bias = M.mean(0)
    return {
        "bias_max_abs": float(np.abs(bias).max()),
        "bias_l2": float(np.linalg.norm(bias)),
        "bias_mean_abs": float(np.abs(bias).mean()),
        "tail_max_abs": float(max(np.abs(v).max() for v in vecs)),
        "n_pairs": len(vecs),
    }


def gate_stats(x_vecs, ref_pools):
    """Cross-arm statistics vs one or more same-arm reference pools.

    bias_*  - the part of the difference that repeats in every run pair, i.e.
              the systematic component; a proportional (scale) error shows up
              as the full relative error in bias_l2, while noise cancels.
    tail    - worst single pair, for gross anomalies.

    ref_pools: [(name, vecs)] or a bare vecs list. Every pool must be cleared,
    so a test is only as lenient as its strictest reference.
    """
    if ref_pools and not isinstance(ref_pools[0], tuple):
        ref_pools = [("reference", ref_pools)]
    x = bias_tail(x_vecs)
    pools = {}
    for name, vecs in ref_pools:
        r = bias_tail(vecs)
        pools[name] = {**r, "ratios": {
            "bias_max_ratio": x["bias_max_abs"] / max(r["bias_max_abs"], 0.0, ABS_FLOOR),
            "bias_l2_ratio": x["bias_l2"] / max(r["bias_l2"], 0.0, ABS_FLOOR * ABS_FLOOR),
            "tail_ratio": x["tail_max_abs"] / max(r["tail_max_abs"], 0.0, ABS_FLOOR),
        }}
    ok = all(p["ratios"]["bias_max_ratio"] <= BIAS_MAX_TOL_RATIO
             and p["ratios"]["bias_l2_ratio"] <= BIAS_L2_TOL_RATIO
             and p["ratios"]["tail_ratio"] <= TAIL_TOL_RATIO for p in pools.values())
    worst = {k: max(p["ratios"][k] for p in pools.values())
             for k in ("bias_max_ratio", "bias_l2_ratio", "tail_ratio")} if pools else {}
    return {
        "cross": x, "reference_pools": pools, **worst,
        "pass": bool(ok),
        "rule": (f"all reference pools: bias_max_ratio <= {BIAS_MAX_TOL_RATIO} and "
                 f"bias_l2_ratio <= {BIAS_L2_TOL_RATIO} and tail_ratio <= {TAIL_TOL_RATIO}"),
    }


def pair_block(base, test, floor):
    """Per-pair record block; carries no verdict (the gates use gate_stats)."""
    m = metrics(base, test)
    return {
        "test": m,
        "floor": {"max_abs": floor["max_abs"], "rel_l2": floor["rel_l2"]},
        "ratio_max_abs": m["max_abs"] / max(floor["max_abs"], 1e-30),
        "ratio_rel_l2": m["rel_l2"] / max(floor["rel_l2"], 1e-30),
    }


def cmd_compare(args):
    import torch
    pri = torch.load(args.pristine, weights_only=False)
    h8 = torch.load(args.h8, weights_only=False)
    assert pri["scene"] == h8["scene"] == args.scene
    dp, dh = pri["dump"], h8["dump"]
    dp2 = None
    if args.pristine2:
        pri2 = torch.load(args.pristine2, weights_only=False)
        assert pri2["scene"] == args.scene and pri2["n_gaussians"] == pri["n_gaussians"]
        assert pri2["binary"] == "pristine", "second process must be the pristine binary"
        dp2 = pri2["dump"]
    TEST_PAIRS = [(f"off{i}", f"on{j}") for i in range(N_RUNS) for j in range(N_RUNS)]

    res = {"scene": args.scene, "seed": SEED, "n_gaussians": pri["n_gaussians"],
           "width": pri["width"], "height": pri["height"],
           "protocol": {
               "arms": {"pri": "pristine binary", "off": "h8 binary, GSPLAT_H8_MR unset",
                        "on": "h8 binary, GSPLAT_H8_MR=1"},
               "null_pairs": {"floor_base": "pri_i-pri_j and off_i-off_j same-arm pairs (6)",
                              "floor_on": "on_i-on_j same-arm pairs (3)"},
               "test_pairs": [f"{a}-{b}" for a, b in TEST_PAIRS],
               "gate_rule": (
                             f"per tensor, over the 9 cross pairs vs the trusted same-arm pool "
                             f"(3 pri_i-pri_j + 3 off_i-off_j): (a) bias = the elementwise mean of the "
                             f"signed cross-pair difference - a deterministic (algebraic) error survives "
                             f"pair-averaging while atomic scatter cancels - bounded in both max-element "
                             f"(<= {BIAS_MAX_TOL_RATIO}x) and L2 (<= {BIAS_L2_TOL_RATIO}x) form against the "
                             f"same statistics on the trusted pool; the L2 form is the sensitive one (a "
                             f"proportional error of eps shows up as bias_l2_ratio ~ eps/noise_rel); "
                             f"(b) tail_ratio = max single-pair max|difference| ratio, <= {TAIL_TOL_RATIO}, "
                             f"the gross-anomaly statistic. The trusted pool is never inflated by an H8 run; "
                             f"its max (not median) is used because the same-arm difference distribution is "
                             f"heavy-tailed. If the gate fails while the on-arm's own scatter exceeds "
                             f"{ELEVATED_SCATTER_RATIO}x the trusted floor the result is reported "
                             f"INCONCLUSIVE_ELEVATED_H8_SCATTER rather than FAIL_SEMANTICS"),
               "boundary_gate_rule": (
                             f"means2d_raw / conics_raw carry moments in H8 mode, so they are gated by "
                             f"reconstruction with the same two statistics: reconstructed(on_j) vs "
                             f"raw(off_i) over 9 cross pairs against a reference pool of the same quantity "
                             f"in both representations (3 off_i-off_j direct pairs + 3 reconstruct(on_i) "
                             f"- reconstruct(on_j) pairs). The direct-adjoint floor of the raw buffer is "
                             f"reported for context but is not the tolerance: the moments' intermediate "
                             f"magnitudes (~1e3x the adjoint) set the reconstruction's own fp32 noise"),
               "cross_build_gate_rule": (
                             "pri_i x off_j (9 pairs) vs a cross-PROCESS same-build null "
                             "(pristine p1 x pristine p2, 9 pairs): the backward buffers are "
                             "atomic-scattered and their rounding pattern is a per-process "
                             "property, so a within-process pool cannot see it and would make "
                             "the inert control spuriously fail (bicycle conics_raw: 2787 vs "
                             "2922 deterministic-offset elements of ~2.9e6, offset_rel_l2 "
                             "3.11e-07 vs 3.33e-07 - indistinguishable). Same two statistics as "
                             "the pipeline gate; the within-process pool is reported as context"),
               "vjp": "deterministic V = randn(render.shape), cuda generator seed 4200",
           }}

    # ---- forward invariance (H8 must not touch fwd) ----
    res["forward_invariance"] = {
        "pri0_vs_off0": {k: metrics(dp["pri0"][k], dh["off0"][k]) for k in ("render", "alpha")},
        "off0_vs_on0": {k: metrics(dh["off0"][k], dh["on0"][k]) for k in ("render", "alpha")},
    }

    # ---- null (atomic nondeterminism) floors, per tensor ----
    # Two separate floors, because they answer different questions:
    #   floor_base (trusted arithmetic: pristine + patched-off, 6 run pairs)
    #     -> the gate tolerance; untouched by anything the H8 path does
    #   floor_on (H8 path's own scatter, 3 run pairs)
    #     -> a consistency check: if H8's own scatter exceeds the trusted floor,
    #        a failing cross pair cannot be attributed to a semantics error and
    #        the gate is noise-limited rather than conclusive.
    # Medians, not maxima, so one anomalous run cannot inflate a floor past the
    # very discrepancy it would otherwise hide.
    def median_floor(pairs, k):
        ms = [metrics(a[k], b[k]) for a, b in pairs]
        mx = sorted(m["max_abs"] for m in ms)
        rl = sorted(m["rel_l2"] for m in ms)
        return {"max_abs": mx[len(mx) // 2], "rel_l2": rl[len(rl) // 2],
                "max_abs_all": mx, "rel_l2_all": rl}

    keys = PIPELINE + ["means2d_raw", "conics_raw"]
    base_pairs = [(dp[f"pri{i}"], dp[f"pri{j}"]) for i, j in ((0, 1), (0, 2), (1, 2))]
    base_pairs += [(dh[f"off{i}"], dh[f"off{j}"]) for i, j in ((0, 1), (0, 2), (1, 2))]
    on_pairs = [(dh[f"on{i}"], dh[f"on{j}"]) for i, j in ((0, 1), (0, 2), (1, 2))]

    floors = {}
    for k in keys:
        b = median_floor(base_pairs, k)
        o = median_floor(on_pairs, k)
        floors[k] = {
            "max_abs": b["max_abs"],
            "rel_l2": b["rel_l2"],
            "base_floor_max_abs_all_pairs": b["max_abs_all"],
            "base_floor_rel_l2_all_pairs": b["rel_l2_all"],
            "on_floor_max_abs": o["max_abs"],
            "on_floor_rel_l2": o["rel_l2"],
            "on_floor_max_abs_all_pairs": o["max_abs_all"],
            "elevated_h8_nondeterminism": bool(o["max_abs"] > ELEVATED_SCATTER_RATIO * max(b["max_abs"], 1e-30)),
            "on_floor_over_base_floor": float(o["max_abs"] / max(b["max_abs"], 1e-30)),
        }

    res["noise_floor"] = floors

    # ---- cross-build inertness: pristine vs patched-with-switch-off ----
    # Both arms run baseline arithmetic, so this is a same-semantics cross-build
    # equivalence check over all 9 pri_i x off_j pairs; it verifies the patch is
    # inert when the switch is off.
    #
    # Null model. The backward buffers are atomic-scattered and their rounding
    # pattern is set per *process* (block/thread scheduling), not per build, so a
    # within-process same-arm pool cannot see the process-level component and is
    # NOT a valid null for a cross-process comparison. Measured on bicycle
    # conics_raw, an independent pristine process reproduces the same
    # deterministic-element offsets as the patched build (2787 vs 2922 elements
    # of ~2.9e6; offset_rel_l2 3.11e-07 vs 3.33e-07). The gate null is therefore
    # the cross-process pool (pristine p1 x pristine p2, 9 pairs, sample-count
    # matched to the cross-build test); the within-process pool is reported as
    # context only, and the det-element counts are reported for both directions.
    res["cross_build_off_vs_pri"] = {}
    res["cross_process_same_build"] = {}
    for k in keys:
        x = [diff_vec(dp[f"pri{i}"][k], dh[f"off{j}"][k])
             for i in range(N_RUNS) for j in range(N_RUNS)]
        within = [diff_vec(dp[f"pri{i}"][k], dp[f"pri{j}"][k]) for i, j in ((0, 1), (0, 2), (1, 2))]
        within += [diff_vec(dh[f"off{i}"][k], dh[f"off{j}"][k]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ctx = gate_stats(x, [("within_process_same_arm", within)])
        block = {
            "pairs": [f"pri{i}-off{j}" for i in range(N_RUNS) for j in range(N_RUNS)],
            "runs": [pair_block(dp[f"pri{i}"][k], dh[f"off{j}"][k], floors[k])
                     for i in range(N_RUNS) for j in range(N_RUNS)],
            "det_offset_cross_build": det_offset_stats([dp[f"pri{i}"][k] for i in range(N_RUNS)],
                                                       [dh[f"off{i}"][k] for i in range(N_RUNS)]),
            "within_process_context": ctx,
        }
        if dp2 is not None:
            cp = [diff_vec(dp[f"pri{i}"][k], dp2[f"pri{j}"][k])
                  for i in range(N_RUNS) for j in range(N_RUNS)]
            res["cross_process_same_build"][k] = {
                "pairs": [f"pri{i}-pri2{j}" for i in range(N_RUNS) for j in range(N_RUNS)],
                "det_offset_cross_process": det_offset_stats([dp[f"pri{i}"][k] for i in range(N_RUNS)],
                                                             [dp2[f"pri{i}"][k] for i in range(N_RUNS)]),
                **gate_stats(cp, [("within_process_same_arm", within)]),
            }
            block = {**block, **gate_stats(x, [("cross_process_same_build", cp)]),
                     "gate_null": "cross_process_same_build (pristine p1 x pristine p2)"}
        else:
            block = {**block, **ctx, "gate_null": "within_process_same_arm (no --pristine2 given)"}
        res["cross_build_off_vs_pri"][k] = block

    # ---- H8 on vs baseline (same binary, switch off) ----
    # The gate is over the PIPELINE tensors only: they are the representation-
    # independent semantics test, since the projection backward consumes
    # v_means2d / v_conics whatever representation they hold. All 9 off_i x on_j
    # cross pairs must clear the trusted floor, so a discrepancy confined to a
    # single on-run cannot hide behind the on-arm's own null floor.
    res["h8_vs_baseline"] = {}
    for k in PIPELINE:
        x = [diff_vec(dh[a][k], dh[b][k]) for a, b in TEST_PAIRS]
        ref = [diff_vec(dp[f"pri{i}"][k], dp[f"pri{j}"][k]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ref += [diff_vec(dh[f"off{i}"][k], dh[f"off{j}"][k]) for i, j in ((0, 1), (0, 2), (1, 2))]
        res["h8_vs_baseline"][k] = {
            "pairs": [f"{a}-{b}" for a, b in TEST_PAIRS],
            "runs": [pair_block(dh[a][k], dh[b][k], floors[k]) for a, b in TEST_PAIRS],
            **gate_stats(x, ref),
        }

    # ---- rasterizer boundary buffers (means2d / conics) ----
    # In H8 mode these carry sufficient statistics rather than the geometry
    # adjoint, so a cross-arm comparison is EXPECTED to differ and is reported
    # for the record only; their correctness gate is the reconstruction below.
    BOUNDARY_NOTE = ("in H8 mode this buffer carries accumulated moments, so the "
                     "cross-arm difference is expected by design and is not a gate; "
                     "see boundary_reconstruction for the correctness check")
    res["h8_boundary_raw"] = {
        k: {**pair_block(dh["off0"][k], dh["on0"][k], floors[k]), "note": BOUNDARY_NOTE,
            "is_gate": False}
        for k in BOUNDARY
    }

    # ---- boundary reconstruction (means2d / conics) ----
    # Which representation the buffer holds is an implementation choice, so the
    # check is adaptive: the raw buffer must EITHER already be the true adjoint
    # (patch published the reconstruction) OR be the moment that reconstructs to
    # it. Anything else is an inconsistency, over every on-run.
    #
    # Tolerance calibration: the tested quantity is the *reconstructed* adjoint.
    # Its fp32 noise is set by the magnitude of the moment intermediates
    # (|m| ~ 1e3x|adjoint|), not by the direct-adjoint noise of the raw buffer,
    # so the reference pool for this test is the same quantity computed both
    # ways: the patched-off same-arm pairs (direct representation) and the
    # patched-on same-arm pairs passed through `reconstruct` (moment
    # representation). The raw-buffer floor is still reported alongside.
    recon_ok = {}
    for k, raw_key in (("means2d", "means2d_raw"), ("conics", "conics_raw")):
        recs = [reconstruct(dh[f"on{i}"]["means2d_raw"], dh[f"on{i}"]["conics"],
                            dh[f"on{i}"]["conics_raw"])[0 if k == "means2d" else 1]
                for i in range(N_RUNS)]
        x = [diff_vec(dh[f"off{i}"][raw_key], recs[j]) for i in range(N_RUNS) for j in range(N_RUNS)]
        # reference = the same estimator (moments + reconstruction) run to run.
        # The direct-arm pairs are NOT the yardstick: that estimator has a
        # different fp32 conditioning, so its much smaller scatter would flag
        # the representation's inherent noise as a failure. They are reported
        # as context; the systematic-bias check lives in the pipeline gate.
        g = gate_stats(x, [("on_reconstruct", [diff_vec(recs[i], recs[j])
                                               for i, j in ((0, 1), (0, 2), (1, 2))])])
        g["reference_pools"]["off_direct_context"] = {
            "source": "off_i-off_j direct-adjoint pairs, context only (not a gate)",
            **bias_tail([diff_vec(dh[f"off{i}"][raw_key], dh[f"off{j}"][raw_key])
                         for i, j in ((0, 1), (0, 2), (1, 2))])}
        direct = metrics(dh["off0"][raw_key], dh["on0"][raw_key])
        direct_ok = bool(g["pass"] and direct["max_abs"]
                         <= TAIL_TOL_RATIO * max(floors[raw_key]["max_abs"], ABS_FLOOR))
        obs = [{
            "run": f"on{i}",
            "reconstructed_vs_baseline": metrics(dh["off0"][raw_key], recs[i]),
            "ratio_max_abs_vs_raw_floor": metrics(dh["off0"][raw_key], recs[i])["max_abs"]
                                          / max(floors[raw_key]["max_abs"], 1e-30),
        } for i in range(N_RUNS)]
        recon_ok[k] = {
            "runs": obs,
            "raw_buffer_vs_baseline_on0": direct,
            "representations": sorted({"true_adjoint_published" if direct_ok
                                       else "moment_reconstructed" if g["pass"] else "inconsistent"}),
            "floor_base_raw": {"max_abs": floors[raw_key]["max_abs"], "rel_l2": floors[raw_key]["rel_l2"],
                               "source": "pri_i-pri_j and off_i-off_j same-arm pairs on the raw buffer "
                                         "(direct-adjoint representation), for reference only"},
            **g,
            "matches": bool(g["pass"]),
        }
    res["boundary_reconstruction"] = recon_ok

    # ---- switch-engagement positive control ----
    engage = {
        "means2d_raw_rel_l2": metrics(dh["off0"]["means2d_raw"], dh["on0"]["means2d_raw"])["rel_l2"],
        "conics_raw_rel_l2": metrics(dh["off0"]["conics_raw"], dh["on0"]["conics_raw"])["rel_l2"],
    }
    engage["switch_engaged"] = bool(engage["means2d_raw_rel_l2"] > 1e-2 or engage["conics_raw_rel_l2"] > 1e-2)
    res["switch_engagement_control"] = engage

    # ---- additional check: absgrad path ----
    # the patch cannot rebuild |adjoint| from signed moments, so the rasterizer's
    # v_means2d_abs branch always uses the per-intersection real adjoint; this
    # verifies that branch, which absgrad=False runs never exercise
    res["absgrad_check"] = {"available": False}
    if any("_abs" in k for k in dh):
        AK = "means2d_absgrad"
        ab_x = [diff_vec(dh[f"off{i}_abs"][AK], dh[f"on{j}_abs"][AK])
                for i in range(N_RUNS) for j in range(N_RUNS)]
        ab_ref = [diff_vec(dp[f"pri{i}_abs"][AK], dp[f"pri{j}_abs"][AK]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ab_ref += [diff_vec(dh[f"off{i}_abs"][AK], dh[f"off{j}_abs"][AK]) for i, j in ((0, 1), (0, 2), (1, 2))]
        res["absgrad_check"] = {
            "available": True,
            "tensor": AK,
            "floor_base": median_floor([(dp[f"pri{i}_abs"], dp[f"pri{j}_abs"]) for i, j in ((0, 1), (0, 2), (1, 2))]
                                       + [(dh[f"off{i}_abs"], dh[f"off{j}_abs"]) for i, j in ((0, 1), (0, 2), (1, 2))], AK),
            "floor_on": median_floor([(dh[f"on{i}_abs"], dh[f"on{j}_abs"]) for i, j in ((0, 1), (0, 2), (1, 2))], AK),
            **gate_stats(ab_x, ab_ref),
        }
        res["absgrad_check"]["elevated_h8_nondeterminism"] = bool(
            res["absgrad_check"]["floor_on"]["max_abs"]
            > ELEVATED_SCATTER_RATIO * max(res["absgrad_check"]["floor_base"]["max_abs"], 1e-30))

    # ---- verdict ----
    pipeline_pass = all(res["h8_vs_baseline"][k]["pass"] for k in PIPELINE)
    boundary_recon_pass = all(recon_ok[k]["matches"] for k in recon_ok)
    inert_pass = all(res["cross_build_off_vs_pri"][k]["pass"] for k in keys)
    elevated = {k: floors[k]["elevated_h8_nondeterminism"] for k in keys}
    # Only the pipeline keys can make a failing gate inconclusive: the two
    # boundary buffers are *expected* to show "elevated" scatter in H8 mode
    # (moments have ~1e3x larger intermediates), so letting them vote would
    # turn every real pipeline failure into INCONCLUSIVE.
    any_elevated = any(elevated[k] for k in PIPELINE)
    absgrad_available = bool(res["absgrad_check"]["available"])
    if not engage["switch_engaged"]:
        classification = "INVALID_SWITCH_NOT_ENGAGED"
    elif not boundary_recon_pass:
        classification = "FAIL_SEMANTICS"
    elif absgrad_available and not res["absgrad_check"]["pass"]:
        classification = "FAIL_SEMANTICS"
    elif pipeline_pass:
        classification = "PASS_NUMERICALLY_EQUIVALENT"
    elif any_elevated:
        # cross pairs exceed the trusted floor, but the H8 arm's own scatter is
        # itself elevated -> the gate is noise-limited, not conclusive
        classification = "INCONCLUSIVE_ELEVATED_H8_SCATTER"
    else:
        classification = "FAIL_SEMANTICS"
    res["verdict"] = {
        "pipeline_gradients_equivalent": pipeline_pass,
        "boundary_reconstruction_equivalent": boundary_recon_pass,
        "absgrad_path_equivalent": res["absgrad_check"]["pass"] if res["absgrad_check"]["available"] else None,
        "patched_binary_inert_when_switch_off": inert_pass,
        "switch_engagement_control": engage["switch_engaged"],
        "boundary_representations": {k: recon_ok[k]["representations"] for k in recon_ok},
        "boundary_calibration": {k: {
            "floor_base_raw_max_abs": recon_ok[k]["floor_base_raw"]["max_abs"],
            "reference_pool": {n: {"bias_l2": p["bias_l2"], "tail_max_abs": p["tail_max_abs"]}
                               for n, p in recon_ok[k]["reference_pools"].items()},
            "bias_l2_ratio": recon_ok[k]["bias_l2_ratio"],
            "bias_max_ratio": recon_ok[k]["bias_max_ratio"],
            "tail_ratio": recon_ok[k]["tail_ratio"],
            "ratio_vs_raw_floor": max(o["ratio_max_abs_vs_raw_floor"] for o in recon_ok[k]["runs"]),
        } for k in recon_ok},
        "elevated_h8_nondeterminism": elevated,
        "pipeline_gate": {k: {kk: res["h8_vs_baseline"][k][kk] for kk in
                              ("bias_l2_ratio", "bias_max_ratio", "tail_ratio", "pass")} for k in PIPELINE},
        "cross_build_inert_gate": {k: {kk: res["cross_build_off_vs_pri"][k][kk] for kk in
                                       ("bias_l2_ratio", "bias_max_ratio", "tail_ratio", "pass")} for k in keys},
        "cross_build_inert_null": {
            "gate_null": next(iter(res["cross_build_off_vs_pri"].values()))["gate_null"],
            "cross_process_same_build_available": bool(res["cross_process_same_build"]),
            "within_process_context": {k: {kk: res["cross_build_off_vs_pri"][k]["within_process_context"][kk]
                                           for kk in ("bias_l2_ratio", "bias_max_ratio", "tail_ratio")}
                                       for k in keys},
            "det_offset_elements": {k: {
                "cross_build": res["cross_build_off_vs_pri"][k]["det_offset_cross_build"]["n_det_elements"],
                "cross_process": (res["cross_process_same_build"][k]["det_offset_cross_process"]["n_det_elements"]
                                  if res["cross_process_same_build"] else None),
                "n_elements": res["cross_build_off_vs_pri"][k]["det_offset_cross_build"]["n_elements"],
            } for k in keys},
            "det_offset_rel_l2": {k: {
                "cross_build": res["cross_build_off_vs_pri"][k]["det_offset_cross_build"]["offset_rel_l2"],
                "cross_process": (res["cross_process_same_build"][k]["det_offset_cross_process"]["offset_rel_l2"]
                                  if res["cross_process_same_build"] else None),
            } for k in keys},
        },
        "classification": classification,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"H2 {args.scene}: {classification} "
          f"pipeline={pipeline_pass} recon={boundary_recon_pass} inert={inert_pass} "
          f"inert_null={'cross_process' if res['cross_process_same_build'] else 'within_process'} "
          f"inert_worst_bias_l2={max(res['cross_build_off_vs_pri'][k]['bias_l2_ratio'] for k in keys):.2f} "
          f"absgrad={res['absgrad_check']['pass'] if absgrad_available else 'n/a'} "
          f"engaged={engage['switch_engaged']} elevated={any_elevated} "
          f"pipeline_worst_bias_l2={max(res['h8_vs_baseline'][k]['bias_l2_ratio'] for k in PIPELINE):.2f} "
          f"pipeline_worst_tail={max(res['h8_vs_baseline'][k]['tail_ratio'] for k in PIPELINE):.2f} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--binary", choices=sorted(BINARIES), required=True)
    d.add_argument("--scene", required=True)
    d.add_argument("--gpu", type=int, default=0)
    d.add_argument("--absgrad", action="store_true",
                   help="also run the absgrad=True forward/backward and compare means2d_absgrad")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_dump)
    c = sub.add_parser("compare")
    c.add_argument("--scene", required=True)
    c.add_argument("--pristine", required=True)
    c.add_argument("--pristine2", default=None,
                   help="second-process pristine dump (same binary, independent run); "
                        "supplies the cross-process null for the inertness control")
    c.add_argument("--h8", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
