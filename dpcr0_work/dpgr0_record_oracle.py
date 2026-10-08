#!/usr/bin/env python3
"""DPGR0 Phase 2 record oracle: R = sum_i ceil(m_i/32) / sum_i m_i per source.
Uses the DPC0 timed .so partial-key multiplicities (identical producer math to
the verified DPC0 .so; getters only). NO DPGR0 kernel is run.
Gate (PREREG S5): R<=0.25 every cell median, <=0.35 every individual source.
"""
import sys, os, json, math, hashlib, datetime
sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np
import torch
import n1_core as C
import dpc0_common as U

CHUNK = 32  # FROZEN chunk size (PREREG S3)
gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.cuda.set_device(gpu); torch.manual_seed(0)

TIMED_SO = ("/root/.cache/torch_extensions/py310_cu128/"
            "dpc0timing_gaussian_render_inference_scene_cuda/"
            "dpc0timing_gaussian_render_inference_scene_cuda.so")

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

import importlib.util
spec = importlib.util.spec_from_file_location("dpc0timing_gaussian_render_inference_scene_cuda", TIMED_SO)
timed = importlib.util.module_from_spec(spec); spec.loader.exec_module(timed)
exp = U.bootstrap("c0")
exp.higs_rasterize_backward = timed.higs_rasterize_backward  # timed DPC backward

out = {"gate": "DPGR0_RECORD_ORACLE", "chunk_size": CHUNK, "timestamp":
       datetime.datetime.utcnow().isoformat() + "Z", "gpu": gpu,
       "timed_so_sha256": sha_file(TIMED_SO), "cells": {}}

all_src_ok = True
all_med_ok = True
for cell in U.CELLS:
    scene, it = U.cell_scene_iter(cell)
    model, ds = U.setup_state(scene, it)
    n_cams = len(ds._valid_indices)
    sources = [int(round(n_cams * (i + 0.5) / 8)) for i in range(8)]  # DPC0 authoritative
    L, params, img, gt, cam = U.render_loss(model, ds, sources[0])
    torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    model.optimizer.zero_grad(set_to_none=True)
    per_src = {}
    for ci in sources:
        L, params, img, gt, cam = U.render_loss(model, ds, ci)
        g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        for p, gr in zip(params, g):
            p.grad = None if gr is None else gr.detach().clone()
        model.optimizer.zero_grad(set_to_none=True)
        pk = timed.dpc0_get_partial_key().detach().to("cpu", torch.int32).numpy()
        N = int(timed.dpc0_get_N())
        gauss = pk.astype(np.int64) % N
        counts = np.bincount(gauss, minlength=N)
        nz = counts[counts > 0]
        m = nz.astype(np.int64)
        sum_m = int(m.sum())
        sum_ceil = int(np.ceil(m.astype(np.float64) / CHUNK).sum())
        R = sum_ceil / sum_m if sum_m > 0 else float("nan")
        per_src[str(ci)] = {
            "cam": int(ci), "n_intersections": sum_m,
            "n_gaussians_with_partials": int(m.size),
            "sum_ceil_m_over_32": sum_ceil, "R_records": R,
            "max_m": int(m.max()), "mean_m": float(m.mean()),
            "median_m": float(np.median(m)), "p90_m": float(np.percentile(m, 90)),
        }
        print("ORACLE %s src=%d R=%.5f sum_m=%d sum_ceil=%d maxm=%d meanm=%.3f p90=%.0f" %
              (cell, ci, R, sum_m, sum_ceil, m.max(), m.mean(), np.percentile(m, 90)), flush=True)
    Rs = [v["R_records"] for v in per_src.values()]
    med = float(np.median(Rs))
    cell_ok_med = med <= 0.25
    cell_ok_src = all(r <= 0.35 for r in Rs)
    all_med_ok = all_med_ok and cell_ok_med
    all_src_ok = all_src_ok and cell_ok_src
    out["cells"][cell] = {"per_source": per_src, "R_median": med,
                          "R_max": float(np.max(Rs)),
                          "gate_cell_median_le_0.25": bool(cell_ok_med),
                          "gate_all_source_le_0.35": bool(cell_ok_src)}
    print("CELL %s R_median=%.5f R_max=%.5f med_ok=%s src_ok=%s" %
          (cell, med, float(np.max(Rs)), cell_ok_med, cell_ok_src), flush=True)

out["record_headroom_pass"] = bool(all_med_ok and all_src_ok)
out["gate_median_le_0.25"] = bool(all_med_ok)
out["gate_all_source_le_0.35"] = bool(all_src_ok)
os.makedirs("/root/epic05_dpgr0/artifacts", exist_ok=True)
with open("/root/epic05_dpgr0/artifacts/DPGR0_RECORD_ORACLE.json", "w") as f:
    json.dump(out, f, indent=2)
print("RECORD_HEADROOM_PASS=%s" % out["record_headroom_pass"], flush=True)
print("ORACLE_DONE", flush=True)