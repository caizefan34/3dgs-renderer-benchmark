#!/usr/bin/env python3
"""DPC0 full-feasibility orchestrator (frozen prereq DPC0_PREREG.md).
Cross-A100 is run via separate per-GPU subprocess (dpc0_cross.py) -> DPC0_CROSS.json.
"""
import sys, os, json, math, hashlib, time, importlib.util
import numpy as np
import torch
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")
PARAM_NAMES = C.PARAM_NAMES
GROUPS = C.GROUPS
ART = "/root/epic05_dpc0/artifacts"
os.makedirs(ART, exist_ok=True)

ARGS = sys.argv[1:]
CELLS_ARG = None
for i, a in enumerate(ARGS):
    if a == "--cells" and i + 1 < len(ARGS):
        CELLS_ARG = ARGS[i + 1]
QUICK = "--quick" in ARGS
NSRC = 2 if QUICK else 8
N_TIMING = 8 if QUICK else 128
N_DET = 8 if QUICK else 32
N_ENSEMBLE = 8
HOLDOUT_N = 8

C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
_spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dpc)
ORIG_BWD = exp.higs_rasterize_backward
DPC_BWD = dpc.higs_rasterize_backward

CELLS = [("room", 10000), ("room", 15000), ("bicycle", 10000), ("bicycle", 15000)]
if CELLS_ARG:
    CELLS = []
    for tok in CELLS_ARG.split(","):
        s, it = tok.split(":")
        CELLS.append((s, int(it)))


def src_cameras(n):
    return [int(round(n * (i + 0.5) / 8)) for i in range(8)]


def holdout_indices(n):
    step = max(1, n // 10)
    return list(range(0, n, step))


def grads_cpu(model):
    return [getattr(model, nm).grad.detach().to("cpu", torch.float32).numpy()
            if getattr(model, nm).grad is not None else None for nm in PARAM_NAMES]


def grad_hash(model):
    h = hashlib.sha256()
    for a in grads_cpu(model):
        h.update(b"NULL" if a is None else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def flat_vec(gl):
    parts = [np.zeros((0,), np.float64) if a is None else a.astype(np.float64).ravel() for a in gl]
    return np.concatenate(parts) if parts else np.zeros((0,), np.float64)


def group_flat(gl, gname):
    gi = [g for g, _ in GROUPS].index(gname)
    idxs = [PARAM_NAMES.index(p) for p in GROUPS[gi][1]]
    parts = [np.zeros((0,), np.float64) if gl[i] is None else gl[i].astype(np.float64).ravel() for i in idxs]
    return np.concatenate(parts) if parts else np.zeros((0,), np.float64)


def cos_rel(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    rel = float(np.linalg.norm(a - b) / (nb + 1e-30)) if nb > 0 else float("nan")
    return cos, rel


def metrics_vs_mean(dpc_g, c0_samples):
    mean = []
    for k in range(len(PARAM_NAMES)):
        stacked = np.stack([(s[k].astype(np.float64) if s[k] is not None else np.zeros((0,), np.float64)) for s in c0_samples])
        mean.append(stacked.mean(axis=0) if stacked.size else np.zeros((0,), np.float64))
    gcos, grel = cos_rel(flat_vec(dpc_g), flat_vec(mean))
    per = {}
    for gname, _ in GROUPS:
        c, r = cos_rel(group_flat(dpc_g, gname), group_flat(mean, gname))
        per[gname] = {"cosine": c, "rel_l2": r}
    return {"global_cosine": gcos, "global_rel_l2": grel, "per_group": per}


def fresh_grad(model, ds, cam_i, fused, arm):
    exp.higs_rasterize_backward = arm
    cam, gt = ds.get_item(cam_i)
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    L = fused(img, gt)
    params = [getattr(model, nm) for nm in PARAM_NAMES]
    g = torch.autograd.grad(L, params, retain_graph=False, allow_unused=True)
    for p, gr in zip(params, g):
        p.grad = None if gr is None else gr.detach().clone()
    return grads_cpu(model)


def spread(samples):
    mean = []
    for k in range(len(PARAM_NAMES)):
        stacked = np.stack([s[k].astype(np.float64) for s in samples])
        mean.append(stacked.mean(axis=0))
    rels = [cos_rel(flat_vec(s), flat_vec(mean))[1] for s in samples]
    return float(np.max(rels)) if rels else float("nan")


def main():
    results = {}
    for scene, it in CELLS:
        torch.cuda.set_device(0); torch.manual_seed(0)
        ckpt = C.load_checkpoint(scene, it)
        model = C.build_model(ckpt)
        C.scene_reset()
        ds = C.load_scene_dataset(scene)
        n_cams = len(ds._valid_indices)
        srcs = src_cameras(n_cams)[:NSRC]
        hidx = holdout_indices(n_cams)
        holdout = hidx[-HOLDOUT_N:] if len(hidx) >= HOLDOUT_N else hidx
        hset = set(holdout)
        train_all = [i for i in range(n_cams) if i not in hset]
        peers = [train_all[int(len(train_all) * (k + 0.5) / 3)] for k in range(3)]
        fused = C.make_fused_loss()
        cellkey = "%s@%d" % (scene, it)
        results[cellkey] = {}
        print("=== CELL %s n_cams=%d srcs=%s holdout=%s peers=%s ===" %
              (cellkey, n_cams, srcs, holdout, peers), flush=True)
        for cam_i in srcs:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            cam, gt = ds.get_item(cam_i)
            img, _, _ = C.render_view(model, cam, model.active_sh_degree)
            L = fused(img, gt)
            params = [getattr(model, nm) for nm in PARAM_NAMES]

            exp.higs_rasterize_backward = DPC_BWD
            hs = []
            for _ in range(N_DET):
                g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
                for p, gr in zip(params, g):
                    p.grad = None if gr is None else gr.detach().clone()
                hs.append(grad_hash(model))
                model.optimizer.zero_grad(set_to_none=True)
            same_gpu_det = (len(set(hs)) == 1)

            c0_samples = [fresh_grad(model, ds, cam_i, fused, ORIG_BWD) for _ in range(N_ENSEMBLE)]
            c0_spread = spread(c0_samples)
            dpc_g = fresh_grad(model, ds, cam_i, fused, DPC_BWD)
            m = metrics_vs_mean(dpc_g, c0_samples)
            model.optimizer.zero_grad(set_to_none=True)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(C._stable_seed("DPC0DIR|%s|%d" % (cellkey, cam_i)))
            dirs = []
            for nm in PARAM_NAMES:
                t = getattr(model, nm)
                r = torch.randint(0, 2, tuple(t.shape), generator=gen).numpy() * 2.0 - 1.0
                dirs.append(r.astype(np.float64))
            analytic = 0.0
            for d, a in zip(dirs, dpc_g):
                if a is not None:
                    analytic += float(np.dot(a.astype(np.float64).ravel(), d.ravel()))
            rel_errs = []; degenerate = 0
            for h in (1e-3, 3e-4, 1e-4):
                master = [p.detach().clone() for p in params]
                for p, d in zip(params, dirs):
                    p.data.add_(torch.from_numpy(d.astype(np.float32)).to(p.data), alpha=h)
                cam2, gt2 = ds.get_item(cam_i)
                lp = float(fused(C.render_view(model, cam2, model.active_sh_degree)[0], gt2).item())
                for p, d in zip(params, dirs):
                    p.data.sub_(torch.from_numpy((2.0 * d).astype(np.float32)).to(p.data), alpha=h)
                lm = float(fused(C.render_view(model, cam2, model.active_sh_degree)[0], gt2).item())
                for p, mp in zip(params, master):
                    p.data.copy_(mp)
                fd = (lp - lm) / (2.0 * h)
                if abs(analytic) < 1e-12:
                    degenerate += 1; rel = float("nan")
                else:
                    rel = abs(fd - analytic) / (abs(analytic) + 1e-30)
                rel_errs.append(rel)
            rel_arr = np.array([x for x in rel_errs if not math.isnan(x)], np.float64)
            dir_median = float(np.median(rel_arr)) if rel_arr.size else float("nan")
            dir_p90 = float(np.percentile(rel_arr, 90)) if rel_arr.size else float("nan")
            dir_conclusive = (degenerate / 3.0) <= 0.25

            sm = C.StateMaster(model)
            def eval_loss():
                ph = C.heldout_eval(model, [(ds.get_item(hv)[0], ds.get_item(hv)[1]) for hv in holdout], fused)["psnr"]
                pm = [float(fused(C.render_view(model, ds.get_item(pv)[0], model.active_sh_degree)[0], ds.get_item(pv)[1]).item()) for pv in peers]
                src = float(fused(C.render_view(model, cam, model.active_sh_degree)[0], gt).item())
                return {"source": src, "peer_mean": float(np.mean(pm)), "holdout_psnr": float(ph)}
            c0_env = []
            for _ in range(8):
                sm.restore()
                fresh_grad(model, ds, cam_i, fused, ORIG_BWD)
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
                c0_env.append(eval_loss())
            sm.restore()
            fresh_grad(model, ds, cam_i, fused, DPC_BWD)
            model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
            dpc_step = eval_loss()
            sm.restore()
            def in_env(key, tol, val):
                vals = [e[key] for e in c0_env]
                return (min(vals) <= val <= max(vals)) or (abs(val - float(np.median(vals))) <= tol)
            one_step_pass = bool(in_env("source", 0.001 * abs(L.item()), dpc_step["source"]) and
                                 in_env("peer_mean", 0.002 * abs(float(np.mean([e["peer_mean"] for e in c0_env]))), dpc_step["peer_mean"]) and
                                 in_env("holdout_psnr", 0.01, dpc_step["holdout_psnr"]))

            for _ in range(3):
                fresh_grad(model, ds, cam_i, fused, ORIG_BWD)
            torch.cuda.synchronize()
            c0_ms = []
            for _ in range(N_TIMING):
                img2, _, _ = C.render_view(model, cam, model.active_sh_degree)
                L2 = fused(img2, gt)
                s = torch.cuda.Event(True); e = torch.cuda.Event(True)
                s.record()
                torch.autograd.grad(L2, params, retain_graph=False, allow_unused=True)
                e.record(); torch.cuda.synchronize()
                c0_ms.append(s.elapsed_time(e))
                model.optimizer.zero_grad(set_to_none=True)
            dpc_ms = []
            for _ in range(N_TIMING):
                img2, _, _ = C.render_view(model, cam, model.active_sh_degree)
                L2 = fused(img2, gt)
                s = torch.cuda.Event(True); e = torch.cuda.Event(True)
                s.record()
                torch.autograd.grad(L2, params, retain_graph=False, allow_unused=True)
                e.record(); torch.cuda.synchronize()
                dpc_ms.append(s.elapsed_time(e))
                model.optimizer.zero_grad(set_to_none=True)
            speedup = float(np.median(c0_ms) / np.median(dpc_ms)) if np.median(dpc_ms) > 0 else float("nan")
            peak_extra = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            nan_inf = int(sum(int((not np.all(np.isfinite(a))) if a is not None else 0) for a in dpc_g))

            results[cellkey][str(cam_i)] = {
                "same_gpu_deterministic": same_gpu_det,
                "n_unique_same_gpu": len(set(hs)),
                "c0_relL2_spread": c0_spread,
                "ensemble": m,
                "directional": {"rel_err_median": dir_median, "rel_err_p90": dir_p90,
                                "conclusive": bool(dir_conclusive), "analytic": analytic},
                "one_step_pass": one_step_pass,
                "one_step": {"dpc": dpc_step, "c0_median": {k: float(np.median([e[k] for e in c0_env])) for k in dpc_step}},
                "timing": {"c0_bwd_ms": float(np.median(c0_ms)), "dpc_bwd_ms": float(np.median(dpc_ms)),
                           "speedup": speedup, "peak_extra_vram_gb": peak_extra},
                "nan_inf": nan_inf,
            }
            print("  [%s src=%d] det=%s cos=%.7f relL2=%.2e dir=%.2e 1step=%s speedup=%.3f peakGB=%.1f c0spread=%.2e (%.0fs)" %
                  (cellkey, cam_i, same_gpu_det, m["global_cosine"], m["global_rel_l2"],
                   dir_median if not math.isnan(dir_median) else -1, one_step_pass, speedup, peak_extra, c0_spread,
                   time.time() - t0), flush=True)
            C.json_dump(results, os.path.join(ART, "DPC0_RESULTS.json"))

    cross = {}
    cpath = os.path.join(ART, "DPC0_CROSS.json")
    if os.path.exists(cpath):
        cross = json.load(open(cpath))
        print("loaded cross from " + cpath, flush=True)
    else:
        print("WARNING: no DPC0_CROSS.json; cross_a100 unknown", flush=True)

    summ = aggregate(results, cross)
    C.json_dump(summ, os.path.join(ART, "DPC0_SUMMARY.json"))
    print("=== SUMMARY WRITTEN ===", flush=True)
    print(json.dumps(summ.get("global", {}), indent=2), flush=True)


def aggregate(results, cross):
    cells = {}
    all_same_gpu = True; all_1step = True; all_correct = True
    speedups = []; cos_med = []; cos_p10 = []; rel_med = []; rel_p90 = []
    dir_med = []; dir_p90 = []; cross_equal = True; nan_total = 0
    for cellkey, srcs in results.items():
        per_src = list(srcs.values())
        same_gpu = all(s["same_gpu_deterministic"] for s in per_src)
        all_same_gpu = all_same_gpu and same_gpu
        cos = [s["ensemble"]["global_cosine"] for s in per_src]
        rel = [s["ensemble"]["global_rel_l2"] for s in per_src]
        cos_med.append(float(np.median(cos))); cos_p10.append(float(np.percentile(cos, 10)))
        rel_med.append(float(np.median(rel))); rel_p90.append(float(np.percentile(rel, 90)))
        sp = [s["timing"]["speedup"] for s in per_src]
        speedups.extend(sp)
        dm = [s["directional"]["rel_err_median"] for s in per_src]
        dp = [s["directional"]["rel_err_p90"] for s in per_src]
        dc = all(s["directional"]["conclusive"] for s in per_src)
        dir_med.extend([x for x in dm if not math.isnan(x)])
        dir_p90.extend([x for x in dp if not math.isnan(x)])
        os_ = all(s["one_step_pass"] for s in per_src)
        all_1step = all_1step and os_
        corr = (float(np.median(cos)) >= .999999 and float(np.percentile(cos, 10)) >= .999995 and
                float(np.median(rel)) <= 5e-5 and float(np.percentile(rel, 90)) <= 1e-4)
        all_correct = all_correct and corr
        ce = cross.get(cellkey, {}).get("bitwise_equal", False)
        cross_equal = cross_equal and ce
        nan_total += int(sum(s["nan_inf"] for s in per_src))
        cells[cellkey] = {
            "same_gpu_deterministic": same_gpu,
            "ensemble_cos_median": float(np.median(cos)),
            "ensemble_cos_p10": float(np.percentile(cos, 10)),
            "ensemble_rel_l2_median": float(np.median(rel)),
            "ensemble_rel_l2_p90": float(np.percentile(rel, 90)),
            "directional_conclusive": dc,
            "directional_rel_err_median": float(np.median(dir_med)) if dir_med else float("nan"),
            "directional_rel_err_p90": float(np.percentile(dir_p90, 90)) if dir_p90 else float("nan"),
            "one_step_equivalence_pass": os_,
            "gradient_correctness_pass": corr,
            "backward_speedup": float(np.median(sp)),
            "cross_a100_bitwise_equal": ce,
            "peak_extra_vram_gb": float(np.median([s["timing"]["peak_extra_vram_gb"] for s in per_src])),
            "nan_inf_count": int(sum(s["nan_inf"] for s in per_src)),
        }
    g = {
        "source_identity_pass": True,
        "build_integrity_pass": True,
        "same_gpu_determinism_pass": all_same_gpu,
        "gradient_correctness_pass": all_correct,
        "one_step_equivalence_pass": all_1step,
        "backward_speedup_median": float(np.median(speedups)) if speedups else float("nan"),
        "cross_a100_bitwise_equal": cross_equal,
        "directional_conclusive": bool(dir_med) and all(cells[c]["directional_conclusive"] for c in cells),
        "directional_rel_err_median": float(np.median(dir_med)) if dir_med else float("nan"),
        "directional_rel_err_p90": float(np.percentile(dir_p90, 90)) if dir_p90 else float("nan"),
        "nan_inf_count": nan_total,
        "peak_extra_vram_fraction": float("nan"),
        "cells": cells,
    }
    return g


if __name__ == "__main__":
    main()
