#!/usr/bin/env python
"""DPGR0 full per-cell phase: det32 + D1(vs DPC0) + D2(vs C0-ensemble) + D3(one-step) + timing(128).
Usage: dpgr0_phase.py <gpu> <cell> [n_rep_timing] [n_rep_det]
Writes runs/dpgr0_phase/<cell>.json
"""
import sys, os, json, datetime, math, importlib.util
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U

gpu = int(sys.argv[1]); cell = sys.argv[2]
NREP_T = int(sys.argv[3]) if len(sys.argv) > 3 else 128
NREP_D = int(sys.argv[4]) if len(sys.argv) > 4 else 32
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell)
U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
cc = U.cell_cameras(scene)
sources = cc["sources"]; peers_map = cc["peers"]; holdout8 = cc["holdout8"]
PN = U.PARAM_NAMES
PT = {n: getattr(model, n) for n in PN}
LR = {"_xyz": 0.00016 * model.spatial_lr_scale, "_shs": 0.0025,
      "_scaling": 0.005, "_rotation": 0.001, "_opacity": 0.025}
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
C0_BWD = exp.higs_rasterize_backward
DPC_BWD = U.load_dpc().higs_rasterize_backward
DPGR0_SO = ("/root/.cache/torch_extensions/py310_cu128/"
            "dpgr0_gaussian_render_inference_scene_cuda/dpgr0_gaussian_render_inference_scene_cuda.so")
spec = importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", DPGR0_SO)
DPGR0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
DPGR0_BWD = DPGR0.higs_rasterize_backward_dpgr0

def set_arm(arm):
    exp.higs_rasterize_backward = {"c0": C0_BWD, "dpc": DPC_BWD, "dpgr0": DPGR0_BWD}[arm]

def raw_flat(params):
    out = []
    for p in params:
        g = p.grad
        out.append(np.zeros(p.numel(), np.float64) if g is None
                   else g.detach().to("cpu", torch.float32).numpy().astype(np.float64).ravel())
    return np.concatenate(out)

def optunit_flat(params):
    out = []
    for n, p in zip(PN, params):
        g = p.grad
        v = (np.zeros(p.numel(), np.float64) if g is None
             else g.detach().to("cpu", torch.float32).numpy().astype(np.float64).ravel())
        out.append(v * LR[n])
    return np.concatenate(out)

def cos_relL2(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return cos, float(np.linalg.norm(a - b) / max(nb, 1e-300))

def render_bwd(bwd, cam_idx, retain):
    set_arm(bwd)
    L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    grads = torch.autograd.grad(L, params, retain_graph=retain, allow_unused=True)
    for p, gr in zip(params, grads):
        p.grad = None if gr is None else gr.detach().clone()
    return L, params, gt, cam

def render_img(cam_idx):
    img, _, _ = C.render_view(model, ds.get_item(cam_idx)[0], model.active_sh_degree)
    return img

fused = C.make_fused_loss()
def fused_loss_val(img, gt):
    return float(fused(img, gt).item())
def psnr_val(img, gt):
    b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    mse = float(((b - g) ** 2).mean().item())
    return -10 * math.log10(max(mse, 1e-12))

sevs = torch.cuda.Event(enable_timing=True); seve = torch.cuda.Event(enable_timing)

rows = []
for si, cam_idx in enumerate(sources):
    cam_idx = int(cam_idx)
    # ---- det32 (same GPU, same input via retain_graph) ----
    L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    model.optimizer.zero_grad(set_to_none=True)
    hashes = []
    for r in range(NREP_D):
        grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        for p, gr in zip(params, grads):
            p.grad = None if gr is None else gr.detach().clone()
        arrs = U.grads_cpu(model, params)
        hashes.append(U.hash_arrays(arrs))
    uniq = sorted(set(hashes))
    det_ok = (len(uniq) == 1)

    # ---- D1: DPGR0 vs DPC0 raw grads ----
    set_arm("dpc"); Ld, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    gd = torch.autograd.grad(Ld, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, gd): p.grad = None if gr is None else gr.detach().clone()
    ref = raw_flat(params)
    set_arm("dpgr0"); Lp, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    gp = torch.autograd.grad(Lp, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, gp): p.grad = None if gr is None else gr.detach().clone()
    cand = raw_flat(params)
    d1_cos, d1_rl2 = cos_relL2(ref, cand)
    d1_nan = int(np.count_nonzero(~np.isfinite(cand)))
    model.optimizer.zero_grad(set_to_none=True)

    # ---- D2: DPGR0 optimizer-unit grads vs 8-run C0 ensemble mean ----
    sm = C.StateMaster(model)
    c0s = []
    for k in range(8):
        sm.restore(); render_bwd("c0", cam_idx, False); c0s.append(optunit_flat(params))
        model.optimizer.zero_grad(set_to_none=True)
    sm.restore(); render_bwd("dpgr0", cam_idx, False); dpg_opt = optunit_flat(params); dpg_raw = raw_flat(params)
    model.optimizer.zero_grad(set_to_none=True)
    c0_mean = np.mean(np.stack(c0s), axis=0)
    d2_cos, d2_rl2 = cos_relL2(dpg_opt, c0_mean)

    # ---- D3: one-step native Adam equivalence ----
    eval_cams = [cam_idx] + list(peers_map[cam_idx]) + list(holdout8)
    gts = {ci: ds.get_item(ci)[1] for ci in eval_cams}
    L0 = fused_loss_val(render_img(cam_idx), gt)
    def eval_loss():
        ph = C.heldout_eval(model, [(ds.get_item(hv)[0], ds.get_item(hv)[1]) for hv in holdout8], fused)["psnr"]
        pm = [fused_loss_val(render_img(pv), gts[pv]) for pv in eval_cams[1:4]]
        src = fused_loss_val(render_img(cam_idx), gt)
        return {"source": src, "peer_mean": float(np.mean(pm)), "holdout_psnr": float(ph)}
    def run_update(arm):
        sm.restore(); render_bwd(arm, cam_idx, False)
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        return eval_loss()
    c0_env = [run_update("c0") for _ in range(8)]
    dpg_step = run_update("dpgr0")
    sm.restore()
    def in_env(key, tol, val):
        vals = [e[key] for e in c0_env]
        return (min(vals) <= val <= max(vals)) or (abs(val - float(np.median(vals))) <= tol)
    one_step_pass = bool(in_env("source", 0.001 * abs(L0), dpg_step["source"]) and
                         in_env("peer_mean", 0.002 * abs(float(np.mean([e["peer_mean"] for e in c0_env]))), dpg_step["peer_mean"]) and
                         in_env("holdout_psnr", 0.01, dpg_step["holdout_psnr"]))

    # ---- timing (fresh render each rep; time backward only) ----
    def time_bwd(bwd, reps):
        set_arm(bwd)
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            img2, _, _ = C.render_view(model, ds.get_item(cam_idx)[0], model.active_sh_degree)
            L2 = fused(img2, gts.get(cam_idx, gt))
            sevs.record()
            torch.autograd.grad(L2, [PT[n] for n in PN], retain_graph=False, allow_unused=True)
            seve.record(); torch.cuda.synchronize()
            ts.append(sevs.elapsed_time(seve))
            model.optimizer.zero_grad(set_to_none=True)
        return float(np.median(ts)), float(np.percentile(ts, 90)), torch.cuda.max_memory_allocated() / (1024**2)
    c0_med, c0_p90, c0_vram = time_bwd("c0", NREP_T)
    dpc_med, dpc_p90, dpc_vram = time_bwd("dpc", NREP_T)
    dpg_med, dpg_p90, dpg_vram = time_bwd("dpgr0", NREP_T)
    # capture DPGR0 stage times + stats on a fresh call
    render_bwd("dpgr0", cam_idx, False)
    st = DPGR0.dpgr0_get_stage_times(); stats = DPGR0.dpgr0_get_stats()
    model.optimizer.zero_grad(set_to_none=True)
    n_isects, n_seg, n_chunked, total_chunks, pstride, N, records, record_bytes = stats
    record_ratio = records / max(n_isects, 1)
    speedup_vs_c0 = c0_med / dpg_med if dpg_med > 0 else float("nan")
    speedup_vs_dpc0 = dpc_med / dpg_med if dpg_med > 0 else float("nan")
    extra_vram_frac = max(0.0, dpg_vram - c0_vram) / max(c0_vram, 1e-6)

    rows.append({
        "cell": cell, "gpu": gpu, "source_rank": si, "cam_idx": cam_idx,
        "same_gpu_deterministic": det_ok, "n_unique_same_gpu": len(uniq),
        "dpc_cos": d1_cos, "dpc_rel_l2": d1_rl2, "dpc_nan_inf": d1_nan,
        "c0_cos": d2_cos, "c0_rel_l2": d2_rl2,
        "one_step_pass": one_step_pass,
        "one_step": {"dpgr0": dpg_step, "c0_median": {k: float(np.median([e[k] for e in c0_env])) for k in dpg_step}},
        "c0_bwd_ms_median": c0_med, "dpc_bwd_ms_median": dpc_med, "dpgr0_bwd_ms_median": dpg_med,
        "dpgr0_bwd_ms_p90": dpg_p90,
        "speedup_vs_c0": speedup_vs_c0, "speedup_vs_dpc0": speedup_vs_dpc0,
        "c0_peak_vram_mib": c0_vram, "dpgr0_peak_vram_mib": dpg_vram,
        "extra_vram_fraction": extra_vram_frac,
        "record_ratio": record_ratio, "records": int(records), "n_isects": int(n_isects),
        "n_seg": int(n_seg), "n_chunked": int(n_chunked), "total_chunks": int(total_chunks),
        "stage_times_range_gather_reduce_ms": st,
        "nan_inf_count": d1_nan,
    })
    print("PHASE " + json.dumps({k: rows[-1][k] for k in
        ("cell", "source_rank", "cam_idx", "same_gpu_deterministic", "dpc_cos", "dpc_rel_l2",
         "c0_cos", "c0_rel_l2", "one_step_pass", "speedup_vs_c0", "speedup_vs_dpc0",
         "record_ratio", "dpgr0_bwd_ms_median", "extra_vram_fraction", "nan_inf_count")}), flush=True)

def med(v): return float(np.median(v))
def p10(v): return float(np.percentile(v, 10))
def p90(v): return float(np.percentile(v, 90))
out = {
    "artifact": "DPGR0_PHASE", "cell": cell, "gpu": gpu,
    "n_rep_timing": NREP_T, "n_rep_det": NREP_D,
    "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "same_gpu_determinism_pass": all(r["same_gpu_deterministic"] for r in rows),
    "dpc_cos_median": med([r["dpc_cos"] for r in rows]), "dpc_cos_p10": p10([r["dpc_cos"] for r in rows]),
    "dpc_rel_l2_median": med([r["dpc_rel_l2"] for r in rows]), "dpc_rel_l2_p90": p90([r["dpc_rel_l2"] for r in rows]),
    "c0_cos_median": med([r["c0_cos"] for r in rows]), "c0_cos_p10": p10([r["c0_cos"] for r in rows]),
    "c0_rel_l2_median": med([r["c0_rel_l2"] for r in rows]), "c0_rel_l2_p90": p90([r["c0_rel_l2"] for r in rows]),
    "one_step_equivalence_pass": all(r["one_step_pass"] for r in rows),
    "speedup_vs_c0_median": med([r["speedup_vs_c0"] for r in rows]),
    "speedup_vs_dpc0_median": med([r["speedup_vs_dpc0"] for r in rows]),
    "record_ratio_median": med([r["record_ratio"] for r in rows]),
    "extra_vram_fraction": max(r["extra_vram_fraction"] for r in rows),
    "nan_inf_count": int(sum(r["nan_inf_count"] for r in rows)),
    "stage_times_median_ms": {
        "range_build": med([r["stage_times_range_gather_reduce_ms"][0] for r in rows]),
        "gather": med([r["stage_times_range_gather_reduce_ms"][1] for r in rows]),
        "reduce": med([r["stage_times_range_gather_reduce_ms"][2] for r in rows])},
    "rows": rows,
}
os.makedirs("/root/epic05_dpgr0/runs/dpgr0_phase", exist_ok=True)
with open("/root/epic05_dpgr0/runs/dpgr0_phase/%s.json" % cell, "w") as f:
    json.dump(out, f, indent=2)
print("PHASE_CELL_DONE " + json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)

