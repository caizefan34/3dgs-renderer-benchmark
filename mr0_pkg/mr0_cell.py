#!/usr/bin/env python3
"""MR0 per-cell pipeline. Modes:
  groups  : build 512 disjoint greedy-NN groups (K=16) + nested K=8/4 subgroups,
            compute per-cam intersection counts, save group structure.
  fit     : fit macros for group range [g0,g1) across K in {4,8,16}, save fitted
            raw params + response_cos/rgb_rmse/alpha_rmse.
  replace : load fitted macros, run full-scene replacement at 25/50/75% coverage
            on 8 train-eval + 8 true-holdout cams, compute PSNR/SSIM delta,
            intersection-count reduction, variable reduction, response-cosine
            aggregate, HM0-predicted speedup. Write cell replacement JSON.
NO production file modified. Runtime monkey-patches only.
"""
import sys, os, json, time, math, argparse
sys.path.insert(0, "/root/mr0_pkg")
import numpy as np
import torch
import torch.nn.functional as F
import mr0_common as M

CELLS = {"room@10000": ("room", 10000), "room@15000": ("room", 15000),
         "bicycle@10000": ("bicycle", 10000), "bicycle@15000": ("bicycle", 15000)}
KS = ("4", "8", "16")
N_GROUPS = 512
KMAX = 16
SEED = 0


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim(x, y):
    # simple mean-SSIM over channels using a 7x7 uniform window (no extra deps)
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    y = y.permute(2, 0, 1).unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    wx = torch.ones(1, 1, 7, 7, device=x.device) / 49.0
    wx = wx.expand(3, 1, 7, 7).contiguous()
    mu_x = F.conv2d(x, wx, padding=3, groups=3)
    mu_y = F.conv2d(y, wx, padding=3, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, wx, padding=3, groups=3) - mu_x2
    sig_y = F.conv2d(y * y, wx, padding=3, groups=3) - mu_y2
    sig_xy = F.conv2d(x * y, wx, padding=3, groups=3) - mu_xy
    s = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
    return float(s.mean())


def render_scene(m, cam, sh):
    """Mirror of hm0_common.render_c0 but with a FRESH _HigsDynamicScene per call
    (render_c0 uses a shared tracker that rejects count changes; the hybrid scene
    has a different primitive count than the baseline)."""
    from gsplat.experimental.render.functional.gaussian_inference import (
        rasterize_gaussian_higs_dynamic, _HigsDynamicScene)
    a = M.C.activated(m)
    out = rasterize_gaussian_higs_dynamic(
        a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
        sh_degree=sh, viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0), width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True, camera_model="pinhole",
        render_mode="RGB", eps2d=0.3, scene=_HigsDynamicScene())
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info, out


def camera_split(n_cams, seed=SEED):
    hold = [i for i in range(n_cams) if i % 8 == 0]
    train = [i for i in range(n_cams) if i % 8 != 0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train), generator=g).tolist()
    train_eval = sorted(train[perm[i]] for i in range(8))
    fit_pool = [c for c in train if c not in train_eval]
    gh = torch.Generator().manual_seed(seed + 1)
    perm_h = torch.randperm(len(hold), generator=gh).tolist()
    hold_eval = sorted(hold[perm_h[i]] for i in range(8))
    return train_eval, fit_pool, hold_eval


N_VIS_CAMS = 32


def build_groups(cell, out_dir, device="cuda"):
    scene, it = CELLS[cell]
    t0 = time.perf_counter()
    ds = M.C.load_dataset(scene, device=device)
    cam_ids = M.C.frozen_camera_ids(len(ds))
    m, _, ident, _, _ = M.C.load_model(scene, it, device=device)
    sh = min(ident["active_sh_degree"], M.C.SH_DEGREE)
    N = ident["num_points"]
    train_eval, fit_pool, hold_eval = camera_split(len(ds))
    # deterministic construction/visibility pool drawn from fit_pool (TRAIN cams
    # disjoint from both eval sets). Used for (a) intersection-weighted grouping,
    # (b) prereg sec.4 per-group "highest visibility" fit-camera selection, and
    # (c) the intersection-reduction sample. No eval camera is ever used here.
    stride = max(1, len(fit_pool) // N_VIS_CAMS)
    vis_pool = fit_pool[::stride][:N_VIS_CAMS]
    info_uni, uninstall = M.A.install_capture_patch(); M.A.A0_CAPTURE["enabled"] = True
    pool_isect = torch.zeros(len(vis_pool), N, device=device)
    for ci, cam_id in enumerate(vis_pool):
        M.A.A0_CAPTURE["records"].clear()
        cam, gt = ds.get_item(cam_id)
        frame, info, out = render_scene(m, cam, sh)
        M.C.train_loss(frame, gt, M.C.LAMBDA_DSSIM).backward()
        cap = M.A.parse_capture(M.A.A0_CAPTURE["records"][-1])
        fif = cap["flatten_ids_f"]
        if fif is None:
            continue
        ids = cap["visible_ids"][fif.reshape(-1)]
        pool_isect[ci] = torch.bincount(ids, minlength=N).float()
    uninstall(); M.sync()
    total_isect = pool_isect.sum(0)
    vis = (total_isect > 0).nonzero().reshape(-1).cpu()
    w = total_isect[vis.to(device)].cpu().double()
    groups = M.disjoint_groups(m["xyz"][vis.to(device)], total_isect[vis.to(device)],
                               N_GROUPS, KMAX, seed=SEED)
    ggl = [vis[g.cpu()] for g in groups]
    members_gpu = torch.stack([g for g in groups]).to(device)      # [G, KMAX]
    group_vis = pool_isect[:, members_gpu].sum(-1)                 # [C, G]
    topk = min(8, len(vis_pool))
    fit_cams_per_group = []
    for gi in range(len(ggl)):
        top = torch.topk(group_vis[:, gi], topk).indices.tolist()
        fit_cams_per_group.append([int(vis_pool[t]) for t in top])
    stats = []
    for gi, g in enumerate(ggl):
        xyz = m["xyz"][g].detach()
        ctr = xyz.mean(0)
        radius = float((xyz - ctr).norm(dim=-1).max())
        scales = torch.exp(m["scaling"][g].detach())
        cov_div = float(scales.std(0).mean() / (scales.mean(0).mean() + 1e-9))
        shs = m["shs"][g].detach()
        sh_div = float(shs.std(0).mean())
        vsum = float(group_vis[:, gi].mean().item())
        stats.append({"gid": gi, "members": [int(x) for x in g.cpu().tolist()],
                      "K": int(g.numel()), "radius": radius, "cov_diversity": cov_div,
                      "sh_diversity": sh_div, "vis_sum": vsum,
                      "isect_weight": float(total_isect[g.cpu()].sum().item()),
                      "fit_cams": fit_cams_per_group[gi]})
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"cell": cell, "scene": scene, "it": it, "N": N, "sh": sh,
               "ckpt_sha256": ident["ckpt_sha256"],
               "train_eval": train_eval, "fit_pool": fit_pool, "hold_eval": hold_eval,
               "isect_cams": [int(c) for c in vis_pool], "n_groups": len(ggl),
               "n_vis_cams": len(vis_pool), "groups": stats},
              open(os.path.join(out_dir, f"{cell}_groups.json"), "w"))
    print(f"[{cell}] groups built: {len(ggl)} in {time.perf_counter()-t0:.1f}s "
          f"vis={vis.numel()} total_hits={int(total_isect.sum())}", flush=True)
    return cell


def fit_shard(cell, out_dir, g0, g1, n_steps=8, n_cam_fit=8, device="cuda", eps=1e-3):
    scene, it = CELLS[cell]
    gj = json.load(open(os.path.join(out_dir, f"{cell}_groups.json")))
    ds = M.C.load_dataset(scene, device=device)
    cam_ids = M.C.frozen_camera_ids(len(ds))
    m, _, ident, _, _ = M.C.load_model(scene, it, device=device)
    sh = min(ident["active_sh_degree"], M.C.SH_DEGREE)
    fit_pool = gj["fit_pool"]
    groups = gj["groups"]
    out = []
    t0 = time.perf_counter()
    for gi in range(g0, min(g1, len(groups))):
        mem = groups[gi]["members"]
        gidx = torch.tensor(mem, device=device)
        g = M.leaves_from_model(m, gidx)
        T = M.tangents(g)
        # prereg sec.4: per-group "highest precomputed visibility" TRAIN cameras
        # (precomputed in build_groups from the construction pool); falls back to
        # fit_pool only if a group somehow has no stored fit cameras.
        fc_ids = (groups[gi].get("fit_cams") or fit_pool)[:n_cam_fit]
        fit_cams = [ds.get_camera(ci) for ci in fc_ids]
        Rg = M.response_exact(g, fit_cams, sh, T).detach()
        rec = {"gid": gi, "K_fits": {}}
        for K in (4, 8, 16):
            sub = gidx[:K]
            gsub = M.leaves_from_model(m, sub)
            Tsub = M.tangents(gsub)
            Rgsub = M.response_exact(gsub, fit_cams, sh, Tsub).detach()
            fit, info = M.fit_macro(gsub, fit_cams, sh, Tsub, Rgsub,
                                    n_steps=n_steps, eps=eps)
            rec["K_fits"][str(K)] = {"raw": {k: v.detach().cpu().tolist() for k, v in fit.items()},
                                     "members": [int(x) for x in sub.cpu().tolist()],
                                     "response_cos": info["response_cos"],
                                     "rgb_rmse": info["rgb_rmse"],
                                     "alpha_rmse": info["alpha_rmse"]}
        out.append(rec)
        if (gi - g0) % 32 == 0:
            print(f"[{cell}] fit {gi}/{min(g1,len(groups))} ({time.perf_counter()-t0:.0f}s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, f"{cell}_fits_{g0}_{g1}.json"), "w"))
    print(f"[{cell}] fit shard [{g0},{g1}) done in {time.perf_counter()-t0:.0f}s", flush=True)




def _intersect_reduction(m, ds, fit_raw, members, cams, sh, device):
    """Tile-intersection reduction of one macro vs its group (BOTH tile-based via
    the a0 capture patch so the units match).
    group_pairs = total tile-gaussian intersection pairs of the K group gaussians;
    macro_pairs = total tile-gaussian intersection pairs of the single macro
                  (1 gaussian -> number of tiles it touches);
    reduction = 1 - macro_pairs / group_pairs  (honest raster-workload reduction).
    NOTE: the macro is NOT measured in pixels; that mismatches the tile-based
    group count and drives reduction strongly negative."""
    def tile_pairs(render_fn, n_ids):
        info_uni, uninstall = M.A.install_capture_patch()
        M.A.A0_CAPTURE["enabled"] = True
        tot = 0.0
        try:
            for cam in cams:
                M.A.A0_CAPTURE["records"].clear()
                fr, al = render_fn(cam)
                (fr.sum() + al.sum()).backward()
                cap = M.A.parse_capture(M.A.A0_CAPTURE["records"][-1])
                fif = cap["flatten_ids_f"]
                if fif is None:
                    tot += float((al > 0).sum().item())
                else:
                    ids = cap["visible_ids"][fif.reshape(-1)]
                    tot += float(torch.bincount(ids, minlength=n_ids).float().sum().item())
        finally:
            uninstall()
        return tot
    gidx = torch.tensor(members, device=device)
    g = M.leaves_from_model(m, gidx)
    def g_render(cam):
        me, qu, sc, op, co = M.activated(g)
        return M.isolated_group_render(me, qu, sc, op, co, [cam], sh)
    g_pairs = tile_pairs(g_render, len(members))
    mp = {k: torch.tensor(v, device=device).float().requires_grad_(True)
          for k, v in fit_raw.items()}
    def m_render(cam):
        me, qu, sc, op, co = M.activated(mp)
        return M.isolated_group_render(me, qu, sc, op, co, [cam], sh)
    m_pairs = tile_pairs(m_render, 1)
    red = 1.0 - m_pairs / max(g_pairs, 1e-9)
    return red, g_pairs, m_pairs


def replace(cell, out_dir, device="cuda", n_isect_sample=128, n_isect_cam=4):
    scene, it = CELLS[cell]
    gj = json.load(open(os.path.join(out_dir, f"{cell}_groups.json")))
    ds = M.C.load_dataset(scene, device=device)
    cam_ids = M.C.frozen_camera_ids(len(ds))
    m, _, ident, _, _ = M.C.load_model(scene, it, device=device)
    sh = gj["sh"]; N = gj["N"]
    train_eval = gj["train_eval"]; hold_eval = gj["hold_eval"]
    eval_cams = train_eval + hold_eval
    prof = json.load(open(f"/root/hm0_out/{scene}_{it}.profile.json"))["phase_frac_of_wall"]
    # load all fit shards
    fits = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith(f"{cell}_fits_") and fn.endswith(".json"):
            for rec in json.load(open(os.path.join(out_dir, fn))):
                fits[rec["gid"]] = rec
    # baseline (full original scene) on eval cams
    base = {}
    for ci in eval_cams:
        cam, gt = ds.get_item(ci)
        frame, info, out = render_scene(m, cam, sh)
        base[ci] = {"psnr": psnr(frame, gt), "ssim": ssim(frame, gt)}
    order = sorted(gj["groups"], key=lambda g: g["isect_weight"], reverse=True)
    # intersection-reduction sample (highest-weight groups, K=16)
    sample = order[:n_isect_sample]
    isect_cams = [ds.get_camera(ci) for ci in gj["isect_cams"][:n_isect_cam]]
    isect_red = {}
    for K in (4, 8, 16):
        reds = []
        for g in sample:
            gid = g["gid"]
            if gid not in fits or str(K) not in fits[gid]["K_fits"]:
                continue
            red, gh, mh = _intersect_reduction(m, ds, fits[gid]["K_fits"][str(K)]["raw"],
                                               fits[gid]["K_fits"][str(K)]["members"],
                                               isect_cams, sh, device)
            reds.append(red)
        isect_red[str(K)] = float(np.median(reds)) if reds else 0.0
        print(f"[{cell}] K={K} intersection_reduction(sample)={isect_red[str(K)]:.4f} n={len(reds)}", flush=True)
    result = {"cell": cell, "scene": scene, "it": it, "N": N, "sh": sh,
              "ckpt_sha256": ident["ckpt_sha256"], "n_groups": len(gj["groups"]),
              "train_eval": train_eval, "hold_eval": hold_eval,
              "baseline_psnr": {str(ci): base[ci]["psnr"] for ci in eval_cams},
              "baseline_ssim": {str(ci): base[ci]["ssim"] for ci in eval_cams},
              "intersection_reduction_sample": isect_red, "K": {}}
    for K in (4, 8, 16):
        Kc = str(K)
        result["K"][Kc] = {}
        avail = [g for g in order if g["gid"] in fits]
        for cov in (0.25, 0.5, 0.75):
            n_replace = min(int(math.ceil(cov * len(gj["groups"]))), len(avail))
            replaced = avail[:n_replace]
            replaced_set = set()
            macro_params = {k: [] for k in ("means", "quats", "scales", "opac", "colors")}
            for g in replaced:
                if Kc not in fits[g["gid"]]["K_fits"]:
                    continue
                rf = fits[g["gid"]]["K_fits"][Kc]
                for mem in rf["members"]:
                    replaced_set.add(int(mem))
                raw = rf["raw"]
                macro_params["means"].append(torch.tensor(raw["means"], device=device).float())
                macro_params["quats"].append(torch.tensor(raw["quats"], device=device).float())
                macro_params["scales"].append(torch.tensor(raw["scales"], device=device).float())
                macro_params["opac"].append(torch.tensor(raw["opac"], device=device).float())
                macro_params["colors"].append(torch.tensor(raw["colors"], device=device).float())
            keep = torch.tensor([i for i in range(N) if i not in replaced_set], device=device)
            hybrid = {
                "xyz": torch.cat([m["xyz"][keep]] + macro_params["means"], 0),
                "shs": torch.cat([m["shs"][keep]] + macro_params["colors"], 0),
                "scaling": torch.cat([m["scaling"][keep]] + macro_params["scales"], 0),
                "rotation": torch.cat([m["rotation"][keep]] + macro_params["quats"], 0),
                "opacity": torch.cat([m["opacity"][keep]] + macro_params["opac"], 0),
            }
            dpsnr_tr = []; dpsnr_hold = []; dssim = []
            nan_inf = 0
            for ci in eval_cams:
                cam, gt = ds.get_item(ci)
                frame, info, out = render_scene(hybrid, cam, sh)
                if not torch.isfinite(frame).all():
                    nan_inf += int((~torch.isfinite(frame)).sum().item())
                dpsnr = psnr(frame, gt) - base[ci]["psnr"]
                dss = ssim(frame, gt) - base[ci]["ssim"]
                dssim.append(dss)
                if ci in train_eval:
                    dpsnr_tr.append(dpsnr)
                else:
                    dpsnr_hold.append(dpsnr)
            n_prim_after = len(keep) + len(replaced)
            prim_red = 1.0 - n_prim_after / N
            cos_rep = [fits[g["gid"]]["K_fits"][Kc]["response_cos"] for g in replaced if Kc in fits[g["gid"]]["K_fits"]]
            result["K"][Kc][f"coverage{int(cov*100)}"] = {
                "train_psnr_delta_median_db": float(np.median(dpsnr_tr)),
                "truehold_psnr_delta_median_db": float(np.median(dpsnr_hold)),
                "truehold_psnr_delta_worst_db": float(np.min(dpsnr_hold)),
                "ssim_delta_median": float(np.median(dssim)),
                "intersection_reduction": isect_red[Kc],
                "optimized_variable_reduction": (K - 1) / K,
                "primitive_reduction": prim_red,
                "response_cos_median": float(np.median(cos_rep)),
                "response_cos_p10": float(np.percentile(cos_rep, 10)),
                "predicted_iteration_speedup": M.predict_speedup(prof, cov, K),
                "nan_inf_count": int(nan_inf),
                "n_replace": n_replace,
            }
            print(f"[{cell}] K={K} cov={cov}: dpsnr_tr={np.median(dpsnr_tr):.3f} "
                  f"dpsnr_hold={np.median(dpsnr_hold):.3f} cos_med={np.median(cos_rep):.3f} "
                  f"speedup={result['K'][Kc][f'coverage{int(cov*100)}']['predicted_iteration_speedup']:.2f}", flush=True)
    json.dump(result, open(os.path.join(out_dir, f"{cell}_replacement.json"), "w"))
    print(f"[{cell}] replacement done -> {cell}_replacement.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["groups", "fit", "replace"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g0", type=int, default=0)
    ap.add_argument("--g1", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--n-cam-fit", type=int, default=8)
    ap.add_argument("--n-isect-sample", type=int, default=128)
    a = ap.parse_args()
    if a.mode == "groups":
        build_groups(a.cell, a.out)
    elif a.mode == "fit":
        fit_shard(a.cell, a.out, a.g0, a.g1, a.n_steps, a.n_cam_fit)
    elif a.mode == "replace":
        replace(a.cell, a.out, n_isect_sample=a.n_isect_sample, n_isect_cam=4)
