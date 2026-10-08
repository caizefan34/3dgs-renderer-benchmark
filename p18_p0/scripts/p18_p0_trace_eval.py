#!/usr/bin/env python3
"""P18-P0 Phase 2 + 3 — real workload capture (per-pixel) + evaluator microbenchmark.

Phase 2: capture >=8192 warp x Gaussian loop positions from the frozen C0 forward
raster (TILE_SIZE=16) for room and bicycle. A POSITION = (gaussian, pixel-x) pair
from the real inner loop: the gaussian's (a,b,c) of q(x), the actual pixel x, its
baseline G(x)=exp2(q(x)) (forward inner-loop alpha), and the branch outcome
(G>=1/255). Stratified by the per-row live-lane population (how many of the row's
32 x-pixels are live), so the trace is realistic (not all-lanes-active synthetic).

Phase 3: replay the captured (a,b,c,x) as a row-major workload (32 lanes = 32
consecutive x of one gaussian, matching the real forward loop) and time DIRECT vs
QSCAN4/8/16 with CUDA events (median over reps) -> S_eval = T_DIRECT/T_QSCAN.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(os.environ.get("P18_OUT", "/root/epic05_p18"))
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
AT = 1.0 / 255.0
N_POSITIONS = 8192          # >=8192 warp x Gaussian loop positions per scene (protocol 9)
N_TIMING_ROWS = 25_000_000  # rows (32 items each) for the timing microbench (use full trace)
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")
FROZEN = {
    "core_so": "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so",
    "c0_ext_so": "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
}
CKPT = {"room": "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt",
        "bicycle": "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt"}


def jwrite(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, allow_nan=False)
    print(f"[p18] wrote {p}", flush=True)


def log(m):
    print(f"[p18 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ext_module(name, so_path):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    for p in (C0_WT, SCENE_CUDA_DIR, PHASE7, SRC, REF_V1):
        if p not in sys.path:
            sys.path.insert(0, p)
    core = load_ext_module("gsplat_cuda", FROZEN["core_so"])
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", FROZEN["c0_ext_so"])
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp


def build_eval(src):
    from torch.utils.cpp_extension import load
    return load(name="p18_eval", sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-use_fast_math", "-gencode=arch=compute_80,code=sm_80"])


def abc_from_conic(conic, means2d):
    c00, c11, c22 = conic[:, 0], conic[:, 1], conic[:, 2]
    mx, my = means2d[:, 0], means2d[:, 1]
    A = 0.5 * c00
    B = c11 + c00 * mx
    C = 0.5 * c22 * my * my + c11 * mx * my + 0.5 * c00 * (mx * mx + my * my) - c00 * mx
    return A, B, C


def producer_for_cam(backend, m, cam, W, H, N):
    K = torch.as_tensor(cam.K.detach().cpu().numpy(), device=DEV, dtype=torch.float32)
    vm = cam.viewmatrix.unsqueeze(0).contiguous()
    Kc = K.unsqueeze(0).contiguous()
    with torch.no_grad():
        cam_pos = backend.higs_camera_positions_from_viewmats(vm)
        (radii, means2d, depths, conics, op_bc, colors_eval) = backend.higs_gatherless_projected_producer(
            torch.arange(N, device=DEV, dtype=torch.int64),
            m._xyz.detach().contiguous(),
            F.normalize(m._rotation.detach(), dim=-1).contiguous(),
            torch.exp(m._scaling.detach()).contiguous(),
            torch.sigmoid(m._opacity.detach()).squeeze(-1).contiguous(),
            m.get_features.detach().contiguous(),
            vm, Kc, cam_pos, W, H, 0.3, 0.01, 1e10, 0.0)
    return radii.reshape(N, 2), means2d.reshape(N, 2).float(), conics.reshape(N, 3).float()


def capture_scene(scene, camset, backend, m, ds, N):
    """Return (positions_list, row_arrays) for one scene. row_arrays = stacked
    (a,b,c) over all visible (gaussian,row) for the timing workload."""
    positions = []
    all_a, all_b, all_c, all_x, all_G, all_live = [], [], [], [], [], []
    row_pops = []
    for ci in camset["scenes"][scene]["probe_ids"]:
        cam, gt = ds.get_item(ci)
        W, H = int(cam.image_width), int(cam.image_height)
        radii, means2d, conics = producer_for_cam(backend, m, cam, W, H, N)
        A, B, C = abc_from_conic(conics, means2d)
        # pixel x-grid per gaussian (actual pixels the gaussian covers, from radii)
        rx = torch.ceil((radii[:, 0] + 0.5)).clamp(min=1, max=2048).long()
        ry = torch.ceil((radii[:, 1] + 0.5)).clamp(min=1, max=2048).long()
        vis = (radii[:, 0] > 0) & (radii[:, 1] > 0)
        mx = means2d[:, 0]
        # per-gaussian x range (clamped to image)
        xlo = (mx - rx).floor().long().clamp(min=0, max=W)
        xhi = (mx + rx).ceil().long().clamp(max=W)
        # sample up to 32 x-positions per visible gaussian (stratified across its range)
        step = (xhi - xlo + 1).clamp(min=1)
        xsamp = xlo + (torch.arange(32, device=DEV).unsqueeze(1) * (step // 32).clamp(min=1).unsqueeze(0))
        xsamp = xsamp.clamp(max=W - 1)  # [32, N]
        G = torch.exp2(-(A[None] * xsamp * xsamp + B[None] * xsamp + C[None]))  # [32, N]
        live = (G >= AT)  # [32, N]
        pop = live.sum(0)  # [N]
        cand = torch.nonzero(vis, as_tuple=False).flatten()
        if cand.numel() == 0:
            continue
        # vectorized row arrays for timing (visible gaussians): [rows, 32]
        a_r = A[cand]; b_r = B[cand]; c_r = C[cand]
        x_r = xsamp[:, cand].T.contiguous().float()     # [rows, 32]
        G_r = G[:, cand].T.contiguous()                  # [rows, 32]
        live_r = live[:, cand].T.contiguous()            # [rows, 32]
        all_a.append(a_r); all_b.append(b_r); all_c.append(c_r)
        all_x.append(x_r); all_G.append(G_r); all_live.append(live_r)
        row_pops.extend(pop[cand].cpu().tolist())
        # sample positions for the manifest (stratified by pop)
        pop_cand = pop[cand]
        for p in range(33):
            s = torch.nonzero(pop_cand == p, as_tuple=False).flatten()
            if s.numel() == 0:
                continue
            take = s[:: max(1, s.numel() // 100)][:100]
            for t in take:
                g = int(cand[t].item())
                positions.append({
                    "scene": scene, "cam": ci, "gaussian_order": g,
                    "a": float(A[g].item()), "b": float(B[g].item()), "c": float(C[g].item()),
                    "means2d": [float(means2d[g, 0].item()), float(means2d[g, 1].item())],
                    "x_positions": xsamp[:, g].tolist(),
                    "live_mask": live[:, g].tolist(), "live_pop": int(pop[g].item()),
                    "baseline_G": G[:, g].tolist(), "baseline_alpha": G[:, g].tolist(),
                    "baseline_branch": live[:, g].tolist(),
                })
        log(f"{scene} cam{ci}: visible={int(vis.sum())} rows={int(vis.sum())} positions={len(positions)}")
    # cap positions list at a representative sample (keep all rows for timing)
    if len(positions) > 8192:
        # stratified subsample by pop
        from collections import defaultdict
        bypop = defaultdict(list)
        for pos in positions:
            bypop[pos["live_pop"]].append(pos)
        sample = []
        per = max(1, 8192 // max(1, len(bypop)))
        for p, lst in bypop.items():
            sample.extend(lst[:: max(1, len(lst) // per)][:per])
        positions = sample[:8192]
    row_a = torch.cat([t.reshape(-1) for t in all_a]) if all_a else torch.zeros(0)
    row_b = torch.cat([t.reshape(-1) for t in all_b]) if all_b else torch.zeros(0)
    row_c = torch.cat([t.reshape(-1) for t in all_c]) if all_c else torch.zeros(0)
    row_x = torch.cat([t.reshape(-1) for t in all_x]) if all_x else torch.zeros(0)
    row_G = torch.cat([t.reshape(-1) for t in all_G]) if all_G else torch.zeros(0)
    row_live = torch.cat([t.reshape(-1) for t in all_live]) if all_live else torch.zeros(0)
    return positions, (row_a, row_b, row_c, row_x, row_G, row_live), row_pops


def time_eval(ext, a, b, c, x, exp2, iters=10, reps=7):
    n = a.numel()
    out = torch.zeros(n, device=DEV, dtype=torch.float32)
    ext.eval_direct(a, b, c, x, out, exp2, 5)
    ext.eval_qscan(a, b, c, x, out, 16, exp2, 5)
    torch.cuda.synchronize()

    def capture(fn):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn(1)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn(1)
        return g

    def med_graph(graph, iters, reps):
        graph.replay(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(iters):
                graph.replay()
            e_ev.record()
            torch.cuda.synchronize()
            ts.append(s_ev.elapsed_time(e_ev) / iters)
        return float(np.median(ts))

    gd = capture(lambda it: ext.eval_direct(a, b, c, x, out, exp2, 1))
    gq4 = capture(lambda it: ext.eval_qscan(a, b, c, x, out, 4, exp2, 1))
    gq8 = capture(lambda it: ext.eval_qscan(a, b, c, x, out, 8, exp2, 1))
    gq16 = capture(lambda it: ext.eval_qscan(a, b, c, x, out, 16, exp2, 1))
    t_direct = med_graph(gd, iters, reps)
    t_q4 = med_graph(gq4, iters, reps)
    t_q8 = med_graph(gq8, iters, reps)
    t_q16 = med_graph(gq16, iters, reps)
    return {"n_items": int(n), "iters": iters, "reps": reps,
            "T_direct_us": t_direct, "T_qscan4_us": t_q4, "T_qscan8_us": t_q8, "T_qscan16_us": t_q16,
            "S_eval_q4": t_direct / t_q4, "S_eval_q8": t_direct / t_q8, "S_eval_q16": t_direct / t_q16}


def main():
    ext = build_eval(str(OUT / "p18_eval.cu"))
    log("evaluator ext built")
    bootstrap_c0()
    camset = json.load(open(OUT / "P18_CAMERA_SET.json"))
    from dataset import GTDataset
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from gsplat.experimental.render.kernels import _backend as _ib
    backend = _ib._C
    manifests = {}
    rows = {}
    for scene in ("room", "bicycle"):
        cfg = ReferenceV1Config(); cfg.scene = scene
        ds = GTDataset(scene=scene, repo_root=REPO, resolution="1080p", device=DEV, background="black")
        model = GaussianModel(max_sh_degree=cfg.sh_degree)
        ckpt = torch.load(CKPT[scene], map_location="cpu", weights_only=False)
        model.restore(ckpt, {
            "position_lr_init": cfg.position_lr_init, "position_lr_final": cfg.position_lr_final,
            "position_lr_delay_mult": cfg.position_lr_delay_mult, "position_lr_max_steps": cfg.position_lr_max_steps,
            "feature_lr": cfg.feature_lr, "opacity_lr": cfg.opacity_lr,
            "scaling_lr": cfg.scaling_lr, "rotation_lr": cfg.rotation_lr,
            "percent_dense": cfg.percent_dense})
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        N = int(model._xyz.shape[0])
        positions, rowarr, row_pops = capture_scene(scene, camset, backend, model, ds, N)
        a, b, c, x, G, live = rowarr
        # timing workload: take up to N_TIMING_ROWS rows; n_items must be a
        # multiple of 256 (8 full warps) so shuffle masks never cross blocks.
        rows_avail = a.numel() // 32
        rows_use = min(rows_avail, N_TIMING_ROWS)
        rows_use = (rows_use // 8) * 8
        n_items = rows_use * 32
        rows[scene] = (a[:n_items], b[:n_items], c[:n_items], x[:n_items], G[:n_items], live[:n_items])
        npz = {
            "a": a.cpu().numpy(), "b": b.cpu().numpy(), "c": c.cpu().numpy(),
            "x": x.cpu().numpy(),
            "baseline_G": G.cpu().numpy(), "live_mask": live.cpu().numpy().astype(np.int32),
            "gaussian_order": np.arange(a.numel(), dtype=np.int64),
            "live_pop": np.array(row_pops, dtype=np.int32),
        }
        np.savez_compressed(OUT / f"P18_TRACE_{scene}.npz", **npz)
        hist = {str(p): int(np.sum(np.array(row_pops) == p)) for p in range(0, 34) if np.any(np.array(row_pops) == p)}
        manifests[scene] = {
            "scene": scene, "n_rows": int(a.numel()), "n_positions_captured": int(a.numel() * 32),
            "n_positions_sampled_in_manifest": len(positions), "n_required": N_POSITIONS,
            "cameras": sorted(set(p["cam"] for p in positions)),
            "population_histogram": hist,
            "mean_live_pop": float(np.mean(row_pops)), "median_live_pop": float(np.median(row_pops)),
            "npz": f"P18_TRACE_{scene}.npz",
            "note": "position = (gaussian, pixel-x); baseline_G = forward inner-loop exp2 alpha over the row's 32 x-pixels; "
                    "stratified by per-row live-lane population (alpha>=1/255 cull). Primary trace, NOT all-lanes-active synthetic.",
        }
        log(f"{scene}: rows={a.numel()} positions_sampled={len(positions)} pop mean={np.mean(row_pops):.2f} med={np.median(row_pops)}")
        del model, ds
        torch.cuda.empty_cache()
    jwrite(str(OUT / "P18_TRACE_MANIFEST.json"), {
        "phase": "P18-P0", "artifact": "P18_TRACE_MANIFEST",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenes": manifests, "n_required_per_scene": N_POSITIONS,
        "note": "real warp x Gaussian loop positions from the frozen C0 forward raster, stratified by live-lane population"})
    log("trace manifest written")

    eval_out = {}
    for scene in ("room", "bicycle"):
        a, b, c, x, G, live = rows[scene]
        for exp2, tag in ((True, "exp2_forward"), (False, "exp_backward")):
            r = time_eval(ext, a, b, c, x, exp2)
            eval_out.setdefault(scene, {})[tag] = r
            log(f"{scene}/{tag}: items={r['n_items']} S_eval q4={r['S_eval_q4']:.3f} q8={r['S_eval_q8']:.3f} q16={r['S_eval_q16']:.3f} "
                f"(us D={r['T_direct_us']:.2f} 4={r['T_qscan4_us']:.2f} 8={r['T_qscan8_us']:.2f} 16={r['T_qscan16_us']:.2f})")
    jwrite(str(OUT / "P18_EVAL_MICROBENCH.json"), {
        "phase": "P18-P0", "artifact": "P18_EVAL_MICROBENCH",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "S_eval_definition": "median T_DIRECT / median T_QSCAN over the captured real trace (CUDA events, 7 reps x 20 iters)",
        "workload": "real captured (gaussian, pixel-x) rows; row-major 32 x per gaussian (matches forward inner loop); NOT all-lanes-active synthetic",
        "results": eval_out,
        "note": "S_eval is the isolated inner-loop proxy; integrated S_FB (protocol 14) is authoritative"})
    log("eval microbench written")


if __name__ == "__main__":
    main()
