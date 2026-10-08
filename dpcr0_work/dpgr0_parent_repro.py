#!/usr/bin/env python3
"""DPGR0 Phase 1: reproduce ONE DPC0 source/cell full-backward median within +/-15%.
Uses the VERIFIED DPC0 .so (monkey-patched as the backward), times the DPC
backward (render once, 32 timed backward reps). Authoritative value from
DPC0_TIMING.json cells[cell][src]["dpc_bwd_ms"].
"""
import sys, os, json, importlib.util, datetime
sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CELL = "room@10000"; SRC = "19"
torch.cuda.set_device(gpu); torch.manual_seed(0)
exp = U.bootstrap("c0")
dpc = U.load_dpc()
exp.higs_rasterize_backward = dpc.higs_rasterize_backward

TIMING = json.load(open("/root/epic05_dpc0/artifacts/DPC0_TIMING.json"))["cells"]
auth = float(TIMING[CELL][SRC]["dpc_bwd_ms"])

scene, it = U.cell_scene_iter(CELL)
model, ds = U.setup_state(scene, it)
n_cams = len(ds._valid_indices)
sources = [int(round(n_cams * (i + 0.5) / 8)) for i in range(8)]
cam_idx = int(SRC)
L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
# warmup
for _ in range(9):
    torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    model.optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
ts = []
for _ in range(32):
    g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, g):
        p.grad = None if gr is None else gr.detach().clone()
    torch.cuda.synchronize()
    model.optimizer.zero_grad(set_to_none=True)
# timed with events
for _ in range(3):
    torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    model.optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
fe0 = torch.cuda.Event(enable_timing=True); fe1 = torch.cuda.Event(enable_timing=True)
fulls = []
for _ in range(32):
    fe0.record()
    torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    fe1.record(); torch.cuda.synchronize()
    fulls.append(fe0.elapsed_time(fe1))
    model.optimizer.zero_grad(set_to_none=True)
dpc_total = float(np.median(fulls))
# render residual (non-stage1) approximated: DPC total - stage1 not available here;
# Phase 1 gate compares DPC total backward median to authoritative dpc_bwd_ms.
rel = abs(dpc_total - auth) / auth
out = {"gate": "DPGR0_PARENT_REPRO", "cell": CELL, "src": int(SRC), "gpu": gpu,
       "authoritative_dpc_bwd_ms": auth, "reproduced_dpc_bwd_ms": dpc_total,
       "rel_err": rel, "pass_within_15pct": bool(rel <= 0.15),
       "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
os.makedirs("/root/epic05_dpgr0/artifacts", exist_ok=True)
with open("/root/epic05_dpgr0/artifacts/DPGR0_PARENT_REPRO.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))