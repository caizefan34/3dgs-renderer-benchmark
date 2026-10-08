#!/usr/bin/env python3
"""BWD-ARITH1 correctness + timing harness (MX A100).
Monkey-patches _C_arith0 (R1 build) into production diff_rasterize.
R0: env BWD1_CACHE_ENABLE unset -> production recompute.
R1: env BWD1_CACHE_ENABLE=1 -> exact FP32 forward Gaussian cache reuse.
"""
import sys, os, json, argparse, math
import numpy as np
import torch

import bwd1_common as BC
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import mx_probe as MP
import mx_perf as PERF


def set_mode(mode):
    BC.set_mode(mode)


def img_diff(a, b):
    d = (a.detach().float() - b.detach().float())
    mse = float((d ** 2).mean())
    psnr = round(10 * math.log10(1.0 / mse), 3) if mse > 0 else 99.0
    return {"bit_identical": bool(torch.equal(a, b)), "max_abs": float(d.abs().max()),
            "mean_abs": float(d.abs().mean()), "mse": mse, "psnr_db": psnr}


def grad_report(g0, g1):
    out = {}
    for k in ["means", "scales", "rotations", "opacities", "sh0", "shr"]:
        a = g0[k].detach().float(); b = g1[k].detach().float()
        d = (a - b).reshape(-1)
        na = float(torch.linalg.norm(a)); nb = float(torch.linalg.norm(b))
        relL2 = float(torch.linalg.norm(d) / (na + 1e-30))
        cos = float(torch.dot(a.reshape(-1), b.reshape(-1)) / (na * nb + 1e-30))
        mx = float(d.abs().max()) if d.numel() else 0.0
        mean = float(d.abs().mean()) if d.numel() else 0.0
        nfinite = int((~torch.isfinite(b)).sum())
        bitwise = float((a.view(torch.int32) == b.view(torch.int32)).float().mean()) if a.numel() else 1.0
        out[k] = {"relL2": relL2, "cos": cos, "max_abs": mx, "mean_abs": mean,
                  "nonfinite": nfinite, "bitwise_eq_rate": bitwise,
                  "gnorm_R0": na, "gnorm_R1": nb}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--modes", default="r0,r1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--correctness_only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    # Monkey-patch _C_arith0 (R1 build) into production diff_rasterize
    C = BC.patch_C()
    trainer, dataset, train_ds, sampler, ad, g = MP.build(a.config, a.checkpoint)
    views = list(train_ds); view = views[a.view]
    N = g.means.shape[0]

    res = {"schema": "BWD_ARITH1_CORRECTNESS_TIMING", "scene": a.scene, "view": a.view,
           "n_gaussians": int(N), "warmup": a.warmup, "iters": a.iters}

    # ---- Forward determinism floor: R0 vs R0 ----
    set_mode("r0"); imgA = PERF.h_fwd(g, view).detach().clone()
    imgB = PERF.h_fwd(g, view).detach().clone()
    res["forward_determinism_R0vR0"] = img_diff(imgA, imgB)
    # ---- Forward R0 vs R1 ----
    set_mode("r0"); img0 = PERF.h_fwd(g, view).detach().clone()
    set_mode("r1"); img1 = PERF.h_fwd(g, view).detach().clone()
    set_mode("r0")
    res["forward_correctness_R0vR1"] = img_diff(img0, img1)
    fd = res["forward_determinism_R0vR0"]; fc = res["forward_correctness_R0vR1"]
    res["forward_gate"] = {
        "R0vR0_max_abs": fd["max_abs"], "R0vR1_max_abs": fc["max_abs"],
        "R0vR0_psnr": fd["psnr_db"], "R0vR1_psnr": fc["psnr_db"],
        "pass": bool(fc["max_abs"] <= fd["max_abs"] * 1.5 + 1e-9 and fc["psnr_db"] >= 60.0),
    }
    print("DBG_FWD_DETERM R0vR0 bit=%s psnr=%.3f max_abs=%.3e" % (fd["bit_identical"], fd["psnr_db"], fd["max_abs"]), flush=True)
    print("DBG_FWD_R0vR1 bit=%s psnr=%.3f max_abs=%.3e pass=%s" % (fc["bit_identical"], fc["psnr_db"], fc["max_abs"], res["forward_gate"]["pass"]), flush=True)

    # ---- Gradient correctness (R0 vs R1), with R0-vs-R0 floor ----
    set_mode("r0")
    img0, gg0 = PERF.h_fwd_bwd(g, view); img0.sum().backward()
    img0b, gg0b = PERF.h_fwd_bwd(g, view); img0b.sum().backward()
    set_mode("r1")
    img1, gg1 = PERF.h_fwd_bwd(g, view); img1.sum().backward()
    set_mode("r0")
    res["gradient_determinism_R0vR0"] = grad_report(gg0, gg0b)
    res["gradient_correctness_R0vR1"] = grad_report(gg0, gg1)
    gd = res["gradient_determinism_R0vR0"]; gc = res["gradient_correctness_R0vR1"]
    worst_relL2_floor = max(v["relL2"] for v in gd.values())
    worst_relL2 = max(v["relL2"] for v in gc.values())
    worst_cos_floor = min(v["cos"] for v in gd.values())
    worst_cos = min(v["cos"] for v in gc.values())
    any_nonfinite = any(v["nonfinite"] > 0 for v in gc.values())
    fam_gate = {}
    for k in ["means", "scales", "rotations", "opacities", "sh0", "shr"]:
        f = gd[k]["relL2"]; r = gc[k]["relL2"]
        tol = max(3.0 * f, 1e-5)
        fam_gate[k] = {"floor_relL2": f, "R0vR1_relL2": r, "ratio": r / max(f, 1e-12),
                       "bitwise_R1": gc[k]["bitwise_eq_rate"],
                       "pass": bool(r <= tol and gc[k]["nonfinite"] == 0)}
    directly_affected_bitwise = bool(gc["means"]["bitwise_eq_rate"] >= 0.99999 and gc["shr"]["bitwise_eq_rate"] >= 0.99999)
    all_within_floor = all(v["pass"] for v in fam_gate.values())
    res["gradient_gate"] = {
        "worst_relL2_floor_R0vR0": worst_relL2_floor, "worst_relL2_R0vR1": worst_relL2,
        "worst_cos_floor_R0vR0": worst_cos_floor, "worst_cos_R0vR1": worst_cos,
        "any_nonfinite": any_nonfinite,
        "per_family": fam_gate,
        "directly_affected_bitwise_identical": directly_affected_bitwise,
        "all_within_nondeterminism_floor": all_within_floor,
        "pass": bool(all_within_floor and directly_affected_bitwise and not any_nonfinite),
        "note": "Production forward is run-to-run nondeterministic (CUB radix sort/atomics); R0vR0 floor ~1e-3 relL2. Gate is floor-relative: R1 must not exceed nondeterminism tolerance. Directly-affected families (means,shr) bitwise identical proves exact cache reuse.",
    }
    print("DBG_GRAD_FLOOR worst_relL2=%.3e worst_cos=%.8f" % (worst_relL2_floor, worst_cos_floor), flush=True)
    print("DBG_GRAD_R0vR1 worst_relL2=%.3e worst_cos=%.8f nonfinite=%s pass=%s"
          % (worst_relL2, worst_cos, any_nonfinite, res["gradient_gate"]["pass"]), flush=True)

    if a.correctness_only:
        json.dump(res, open(a.out, "w"), indent=2)
        print("CORR_END", flush=True)
        return

    # ---- Timing (R0 vs R1) ----
    modes = a.modes.split(",")
    timing = {}
    for mode in modes:
        set_mode(mode)
        fwd_med, fwd_min, fwd_max, fwd_vram = PERF.bench_fwd(lambda: PERF.h_fwd(g, view), a.warmup, a.iters)
        torch.cuda.empty_cache()
        bwd_med, bwd_min, bwd_max, bwd_vram = PERF.bench_bwd(lambda: PERF.h_fwd_bwd(g, view), a.warmup, a.iters)
        torch.cuda.empty_cache()
        timing[mode] = {
            "fwd_ms_median": fwd_med, "fwd_ms_min": fwd_min, "fwd_ms_max": fwd_max,
            "bwd_ms_median": bwd_med, "bwd_ms_min": bwd_min, "bwd_ms_max": bwd_max,
            "fwd_vram_gb": round(fwd_vram / 1e9, 4), "bwd_vram_gb": round(bwd_vram / 1e9, 4),
        }
        print("DBG_TIMING %s fwd_med=%.4f bwd_med=%.4f bwd_min=%.4f fwdVRAM=%.3fGB bwdVRAM=%.3fGB"
              % (mode, fwd_med, bwd_med, bwd_min, fwd_vram / 1e9, bwd_vram / 1e9), flush=True)
    set_mode("r0")
    res["timing"] = timing
    if "r0" in timing and "r1" in timing:
        t0, t1 = timing["r0"], timing["r1"]
        d_write = t1["fwd_ms_median"] - t0["fwd_ms_median"]
        d_bwd = t0["bwd_ms_median"] - t1["bwd_ms_median"]
        d_net = d_bwd - d_write
        res["deltas"] = {
            "delta_T_write_ms": d_write,
            "delta_T_bwd_ms": d_bwd,
            "delta_T_net_ms": d_net,
            "S_backward": round(t0["bwd_ms_median"] / t1["bwd_ms_median"], 4) if t1["bwd_ms_median"] else None,
            "S_renderer": round((t0["fwd_ms_median"] + t0["bwd_ms_median"]) /
                                (t1["fwd_ms_median"] + t1["bwd_ms_median"]), 4)
            if (t1["fwd_ms_median"] + t1["bwd_ms_median"]) else None,
        }
        print("DBG_DELTA " + json.dumps(res["deltas"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("TIMING_END", flush=True)


if __name__ == "__main__":
    main()
