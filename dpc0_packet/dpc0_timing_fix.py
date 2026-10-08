#!/usr/bin/env python3
"""DPC0 corrected timing. Matches orchestrator source selection exactly.
Usage: dpc0_timing_fix.py <gpu> <cell> [n_rep]
Times C0 raster backward vs DPC total backward per source (orchestrator sources),
128 reps CUDA-event, peak VRAM, speedup, peak_extra_vram_fraction.
Writes runs/dpc0_timing_fix/{cell}.json (per-source rows keyed by cam_idx).
"""
import sys, os, json, datetime
sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np
import torch
import n1_core as C
import dpc0_common as U

gpu = int(sys.argv[1]); cell = sys.argv[2]
nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 128
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell)
U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
n_cams = len(ds._valid_indices)
# orchestrator source selection: [round(n*(i+0.5)/8) for i in range(8)]
sources = [int(round(n_cams * (i + 0.5) / 8)) for i in range(8)]
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
C0_BWD = exp.higs_rasterize_backward
DPC_BWD = U.load_dpc().higs_rasterize_backward

start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)


def time_arm(arm, cam_idx, reps):
    exp.higs_rasterize_backward = DPC_BWD if arm == "dpc" else C0_BWD
    L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    for _ in range(8):
        torch.autograd.grad(L, params, retain_graph=True)
        model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        start_ev.record()
        torch.autograd.grad(L, params, retain_graph=True)
        end_ev.record()
        torch.cuda.synchronize()
        ts.append(start_ev.elapsed_time(end_ev))
        model.optimizer.zero_grad(set_to_none=True)
    ts.sort()
    return ts[len(ts)//2], ts[int(0.9*(len(ts)-1))], ts[0]


def peak_gb():
    return torch.cuda.max_memory_allocated() / (1024**3)


rows = []
for si, cam_idx in enumerate(sources):
    torch.cuda.reset_peak_memory_stats()
    c0_med, c0_p90, c0_min = time_arm("c0", cam_idx, nrep)
    vram_c0 = peak_gb()
    torch.cuda.reset_peak_memory_stats()
    dpc_med, dpc_p90, dpc_min = time_arm("dpc", cam_idx, nrep)
    vram_dpc = peak_gb()
    speedup = c0_med / dpc_med if dpc_med > 0 else float("nan")
    extra_frac = max(0.0, vram_dpc - vram_c0) / max(vram_c0, 1e-6)
    row = {"cell": cell, "gpu": gpu, "source_rank": si, "cam_idx": int(cam_idx),
           "c0_bwd_ms_median": c0_med, "dpc_total_bwd_ms_median": dpc_med,
           "c0_bwd_ms_p90": c0_p90, "dpc_total_bwd_ms_p90": dpc_p90,
           "c0_peak_vram_gb": vram_c0, "dpc_peak_vram_gb": vram_dpc,
           "peak_extra_vram_gb": max(0.0, vram_dpc - vram_c0),
           "peak_extra_vram_fraction": extra_frac,
           "speedup": speedup}
    rows.append(row)
    print("TFIX " + json.dumps(row), flush=True)


def agg(key, fn):
    v = [r[key] for r in rows if r[key] == r[key]]
    return fn(v) if v else None
out = {"cell": cell, "gpu": gpu, "n_rep": nrep, "sources": sources,
       "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
       "c0_bwd_ms_median": agg("c0_bwd_ms_median", lambda v: float(np.median(v))),
       "dpc_total_bwd_ms_median": agg("dpc_total_bwd_ms_median", lambda v: float(np.median(v))),
       "backward_speedup": agg("speedup", lambda v: float(np.median(v))),
       "backward_speedup_min": agg("speedup", min),
       "peak_extra_vram_fraction": agg("peak_extra_vram_fraction", max),
       "rows": rows}
os.makedirs("/root/epic05_dpc0/runs/dpc0_timing_fix", exist_ok=True)
with open("/root/epic05_dpc0/runs/dpc0_timing_fix/%s.json" % cell, "w") as f:
    json.dump(out, f, indent=2)
print("TFIX_CELL_DONE " + json.dumps({k:v for k,v in out.items() if k!="rows"}), flush=True)
