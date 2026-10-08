#!/usr/bin/env python3
"""DEA-GS M0 instrumented trainer (official graphdeco 3DGS baseline).

Runs REAL short training to produce per-phase CUDA timing (render/loss/backward/
densify/optimizer/total) over a stable window after warmup, streaming visibility
(radii>0), topology counters, and genuine Adam optimizer-state export
(theta,m,v,step,k,lr) for f_dc/f_rest. Does NOT modify optimizer semantics.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from random import randint

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".c0-official-reference"))
sys.path.insert(0, REF)
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--images", default="images_4")
    ap.add_argument("--iterations", type=int, default=1700)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--optimizer-type", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-scalars", type=int, default=150000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.model_out, exist_ok=True)
    safe_state(False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    class MP:
        sh_degree = args.sh_degree
        source_path = args.source; model_path = args.model_out
        images = args.images; depths = ""; resolution = -1
        white_background = False; train_test_exp = False
        data_device = "cuda"; eval = False
    class OP:
        iterations = args.iterations
        position_lr_init = 0.00016; position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01; position_lr_max_steps = 30000
        feature_lr = 0.0025; opacity_lr = 0.025; scaling_lr = 0.005; rotation_lr = 0.001
        exposure_lr_init = 0.01; exposure_lr_final = 0.001
        exposure_lr_delay_steps = 0; exposure_lr_delay_mult = 0.0
        percent_dense = 0.01; lambda_dssim = 0.2
        densification_interval = 100; opacity_reset_interval = 3000
        densify_from_iter = 500; densify_until_iter = 15000
        densify_grad_threshold = 0.0002
        depth_l1_weight_init = 1.0; depth_l1_weight_final = 0.01
        random_background = False; optimizer_type = args.optimizer_type
    class PP:
        convert_SHs_python = False; compute_cov3D_python = False
        debug = False; antialiasing = False

    dataset = MP(); opt = OP(); pipe = PP()
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    n_init = int(gaussians.get_xyz.shape[0])

    last_seen = torch.full((n_init,), -1, dtype=torch.long, device=device)
    pending_k = torch.zeros(n_init, dtype=torch.long, device=device)
    realized_k = torch.zeros(n_init, dtype=torch.long, device=device)
    N = n_init
    streaks = []

    def extend_tracker(new_n):
        nonlocal last_seen, pending_k, realized_k, N
        if new_n > N:
            add = new_n - N
            last_seen = torch.cat([last_seen, torch.full((add,), -1, dtype=torch.long, device=device)])
            pending_k = torch.cat([pending_k, torch.zeros(add, dtype=torch.long, device=device)])
            realized_k = torch.cat([realized_k, torch.zeros(add, dtype=torch.long, device=device)])
            N = new_n

    def prune_tracker(keep_mask):
        nonlocal last_seen, pending_k, realized_k, N
        last_seen = last_seen[keep_mask]; pending_k = pending_k[keep_mask]
        realized_k = realized_k[keep_mask]; N = int(keep_mask.sum().item())

    topology = {"densify_and_prune": 0, "reset_opacity": 0}
    orig_dap = gaussians.densify_and_prune
    def hooked_dap(*a, **k):
        topology["densify_and_prune"] += 1
        r = orig_dap(*a, **k); extend_tracker(int(gaussians.get_xyz.shape[0])); return r
    gaussians.densify_and_prune = hooked_dap
    orig_ro = gaussians.reset_opacity
    def hooked_ro(*a, **k):
        topology["reset_opacity"] += 1; return orig_ro(*a, **k)
    gaussians.reset_opacity = hooked_ro
    orig_prune = gaussians.prune_points
    def hooked_prune(mask):
        prune_tracker((~mask).to(device)); return orig_prune(mask)
    gaussians.prune_points = hooked_prune

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    rec = {k: [] for k in ["render", "loss", "backward", "densify", "optimizer", "total", "wall",
                           "n_visible", "n_total"]}
    prof_start = args.warmup
    peak_mem = 0
    E = lambda: torch.cuda.Event(enable_timing=True)

    for iteration in range(1, args.iterations + 1):
        it_start = time.perf_counter()
        ev_tot_s = E(); ev_tot_s.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        ev_rs = E(); ev_re = E(); ev_le = E(); ev_be = E(); ev_de = E(); ev_dee = E(); ev_oe = E()
        ev_rs.record()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        ev_re.record()
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        ev_le.record()
        loss.backward()
        ev_be.record()

        with torch.no_grad():
            radii = render_pkg["radii"]
            visible_mask = radii > 0
            n_vis = int(visible_mask.sum().item())
            n_tot = int(gaussians.get_xyz.shape[0])
            if iteration >= prof_start:
                ids = visible_mask.nonzero(as_tuple=True)[0]
                if ids.numel():
                    ls = last_seen[ids]; seen = ls >= 0
                    if bool(seen.any()):
                        st = (iteration - ls[seen] - 1)
                        realized_k[ids[seen]] = st
                        streaks.extend(st.detach().cpu().tolist())
                    pending_k[ids] = 0
                    last_seen[ids] = iteration
                inv = ~visible_mask
                pending_k[inv] += 1
                extend_tracker(n_tot)
                rec["n_visible"].append(n_vis); rec["n_total"].append(n_tot)

            ev_de.record()
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visible_mask] = torch.max(
                    gaussians.max_radii2D[visible_mask], radii[visible_mask])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visible_mask)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            ev_dee.record()

            ev_oe.record()
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            if use_sparse_adam:
                vis = radii > 0
                gaussians.optimizer.step(vis, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
        ev_tot_e = E(); ev_tot_e.record()
        torch.cuda.synchronize()

        if iteration >= prof_start:
            rec["render"].append(ev_rs.elapsed_time(ev_re))
            rec["loss"].append(ev_re.elapsed_time(ev_le))
            rec["backward"].append(ev_le.elapsed_time(ev_be))
            rec["densify"].append(ev_de.elapsed_time(ev_dee))
            rec["optimizer"].append(ev_oe.elapsed_time(ev_tot_e))
            rec["total"].append(ev_tot_s.elapsed_time(ev_tot_e))
            rec["wall"].append((time.perf_counter() - it_start) * 1000.0)
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() // (1024 * 1024))
        if iteration % 100 == 0:
            print(f"[iter {iteration}] n_tot={n_tot} n_vis={n_vis} peak_mem={peak_mem}MB", flush=True)

    def grab(name):
        for g in gaussians.optimizer.param_groups:
            if g.get("name") == name:
                p = g["params"][0]; st = gaussians.optimizer.state[p]
                return p.detach(), st["exp_avg"].detach(), st["exp_avg_sq"].detach(), float(g["lr"])
        raise KeyError(name)
    theta_dc, m_dc, v_dc, lr_dc = grab("f_dc")
    theta_rest, m_rest, v_rest, lr_rest = grab("f_rest")
    k_g = torch.where(realized_k > 0, realized_k, pending_k).clamp(min=0).cpu().numpy()

    def sample_scalars(theta, m, v, lr, group_id, n_target):
        Nn, C = theta.shape
        theta_f = theta.reshape(-1).float().cpu().numpy()
        m_f = m.reshape(-1).float().cpu().numpy()
        v_f = v.reshape(-1).float().cpu().numpy()
        k_rep = np.repeat(k_g, C)
        total = theta_f.shape[0]
        if total <= n_target:
            idx = np.arange(total)
        else:
            idx = np.random.default_rng(args.seed).choice(total, size=n_target, replace=False)
        return (theta_f[idx], m_f[idx], v_f[idx], k_rep[idx],
                np.full(idx.shape[0], group_id, dtype=np.int64), np.full(idx.shape[0], lr))

    s_dc = sample_scalars(theta_dc, m_dc, v_dc, lr_dc, 0, args.sample_scalars // 2)
    s_rest = sample_scalars(theta_rest, m_rest, v_rest, lr_rest, 1, args.sample_scalars // 2)
    theta = np.concatenate([s_dc[0], s_rest[0]]); m = np.concatenate([s_dc[1], s_rest[1]])
    v = np.concatenate([s_dc[2], s_rest[2]]); k = np.concatenate([s_dc[3], s_rest[3]])
    grp = np.concatenate([s_dc[4], s_rest[4]]); lr = np.concatenate([s_dc[5], s_rest[5]])
    step_arr = gaussians.optimizer.state[gaussians.optimizer.param_groups[1]["params"][0]]["step"]
    step = int(step_arr.item() if torch.is_tensor(step_arr) else step_arr)

    npz_path = os.path.splitext(args.out)[0] + "_states.npz"
    np.savez(npz_path, theta=theta.astype(np.float64), m=m.astype(np.float64),
             v=v.astype(np.float64), step=step, k=k.astype(np.int64), lr=lr,
             param_group=grp, beta1=0.9, beta2=0.999, eps=1e-15,
             lr_dc=lr_dc, lr_rest=lr_rest)

    mean = lambda x: float(np.mean(x)) if len(x) else 0.0
    std = lambda x: float(np.std(x)) if len(x) else 0.0
    vf = [nv / max(nt, 1) for nv, nt in zip(rec["n_visible"], rec["n_total"])]
    summary = {
        "timing": {
            "iterations_profiled": len(rec["render"]),
            "render_ms_mean": mean(rec["render"]), "render_ms_std": std(rec["render"]),
            "loss_ms_mean": mean(rec["loss"]), "loss_ms_std": std(rec["loss"]),
            "backward_ms_mean": mean(rec["backward"]), "backward_ms_std": std(rec["backward"]),
            "densify_ms_mean": mean(rec["densify"]), "densify_ms_std": std(rec["densify"]),
            "optimizer_ms_mean": mean(rec["optimizer"]), "optimizer_ms_std": std(rec["optimizer"]),
            "total_ms_mean": mean(rec["total"]), "wall_ms_mean": mean(rec["wall"]),
            "gaussian_count_mean": int(np.mean(rec["n_total"])) if rec["n_total"] else n_init,
            "gaussian_count_last": int(rec["n_total"][-1]) if rec["n_total"] else n_init,
            "visible_count_mean": int(np.mean(rec["n_visible"])) if rec["n_visible"] else 0,
            "peak_mem_mb": peak_mem, "n_init": n_init, "topology": topology,
            "npz": os.path.basename(npz_path), "step_at_export": step,
        },
        "visibility": {
            "visible_frac_mean": float(np.mean(vf)) if vf else 0.0,
            "n_streak_samples": len(streaks),
            "streak_p50": float(np.percentile(streaks, 50)) if streaks else 0.0,
            "streak_p90": float(np.percentile(streaks, 90)) if streaks else 0.0,
            "streak_p99": float(np.percentile(streaks, 99)) if streaks else 0.0,
            "streak_max": int(np.max(streaks)) if streaks else 0,
        },
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
