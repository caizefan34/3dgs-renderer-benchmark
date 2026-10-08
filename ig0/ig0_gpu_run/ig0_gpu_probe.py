"""InfoGeo-GS IG0 GPU scientific probe harness (EPIC-05 A100, frozen C0).

Reuses the EXACT frozen C0 renderer + fused-ssim loss + Adam optimizer from
publication_trainer_p12_n1snap.py (arm=c0). Does NOT modify prereg/analyzer/
thresholds. No local GPU. All compute on EPIC-05 A100.
"""
import sys, os, math, json, random, argparse, hashlib, platform, time, copy
import numpy as np
import torch
import torch.nn.functional as F
import ig0_fisher_torch as TF

REPO = "/root/3dgs-renderer-benchmark"
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
ACCUTILE = "/root/epic05_frozen/gsplat-true-accutile-v153"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = os.path.join(REPO, "..", "epic05_build_cache", "scene", "gsplat_scene_cuda")
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
P11_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
EPS2D = 0.3  # frozen c0 arm eps2d
RUN_DIR = "/root/ig0_run"
CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bootstrap_c0():
    import importlib.util
    def load_ext(name, so):
        spec = importlib.util.spec_from_file_location(name, so)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    for p in [ACCUTILE, PHASE7_DIR, SRC_DIR, REFERENCE_V1_DIR, C0_WT, SCENE_CUDA_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)
    core = load_ext("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    return rasterize_gaussian_higs_dynamic


class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def compute_scene_extent(dataset, n_samples=50):
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        centers.append(dataset.get_camera(i).camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            extent = max(extent, float(np.linalg.norm(centers[i] - centers[j])))
    return max(extent, 0.1)


def midpoint_stratified(pool, k=8):
    M = len(pool)
    if M <= k:
        return list(pool)
    out = []
    for j in range(k):
        lo = int(round(j * M / k)); hi = int(round((j + 1) * M / k))
        mid = min(max((lo + hi) // 2, 0), M - 1)
        out.append(pool[mid])
    return out


def select_cameras(n):
    holdout_pool = [i for i in range(n) if i % 8 == 0]
    train_pool = [i for i in range(n) if i % 8 != 0]
    return midpoint_stratified(train_pool, 8), midpoint_stratified(holdout_pool, 8)

PARAM_KEYS = ["xyz", "shs", "scaling", "rotation", "opacity"]


class IG0:
    def __init__(self, rasterize, config):
        self.rasterize = rasterize
        self.config = config
        self.ssim_fn = SepSSIM(device="cuda")
        import fused_ssim as fs
        import fused_ssim_cuda as fsc
        self.fused_ssim = fs
        self.fused_so_sha = sha256_file(fsc.__file__)

    def train_loss(self, pred_hwc, gt_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim = self.fused_ssim.fused_ssim(b, g, padding="same", train=True)
        return (1.0 - self.config.lambda_dssim) * L1 + self.config.lambda_dssim * (1.0 - ssim)

    def render(self, model, cam):
        out = self.rasterize(
            model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features,
            sh_degree=model.active_sh_degree,
            viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
            Ks=cam.K.unsqueeze(0).unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            backward_mode="higs_native", enable_culling=True,
            camera_model="pinhole", render_mode="RGB", eps2d=EPS2D)
        return out["frame"].reshape(cam.image_height, cam.image_width, 3).clamp(0, 1)

    def get_state(self, model, cpu=False):
        out = {}
        for k in PARAM_KEYS:
            t = getattr(model, "_" + k).detach()
            out[k] = t.cpu().clone() if cpu else t.clone()
        return out

    def set_state(self, model, state):
        with torch.no_grad():
            for k in PARAM_KEYS:
                getattr(model, "_"+k).data.copy_(state[k])

    def evaluate(self, model, state, cams, dataset):
        self.set_state(model, state)
        res = []
        for ci in cams:
            cam, gt = dataset.get_item(ci)
            with torch.no_grad():
                img = self.render(model, cam)
                L = float(self.train_loss(img, gt))
                psnr = compute_psnr(img, gt)
                ssim_eval = float(self.ssim_fn(img, gt))
            res.append((ci, L, psnr, ssim_eval))
        return res

    def _radius(self, adam_geo, nat_geo, q0, log_scale):
        dmu_a, dls_a, dom_a = adam_geo
        dmu_n, dls_n, dom_n = nat_geo
        q0d = q0.double(); lsd = log_scale.double()
        Si, lam, _ = TF.sigma_inv_batch(q0d, lsd)
        Gr = TF.G_rot_batch(lam)
        rA = TF.radius_geom_batch(dmu_a.double(), dls_a.double(), dom_a.double(), Si, lam, Gr)
        rN = TF.radius_geom_batch(dmu_n.double(), dls_n.double(), dom_n.double(), Si, lam, Gr)
        aG = torch.sqrt(rA.sum() / max(rN.sum().item(), 1e-30))
        rG = (aG * aG) * rN.sum()
        relG = abs(rG.item() - rA.sum().item()) / max(rA.sum().item(), 1e-300)
        aL = torch.where(rN > 0, torch.sqrt(rA / torch.clamp(rN, min=1e-30)), torch.zeros_like(rA))
        rL = (aL * aL) * rN
        mask = rN > 0
        relL = (torch.abs(rL - rA) / torch.clamp(rA, min=1e-300))[mask]
        relL_np = relL.double().cpu().numpy() if relL.numel() else np.array([0.0])
        return float(relG), float(np.median(relL_np)), float(np.percentile(relL_np, 99)), int(mask.sum().item())

    def _nan_check(self, states, grads):
        cnt = 0
        for st in states:
            for k in PARAM_KEYS:
                t = st[k]
                cnt += int((~torch.isfinite(t)).sum().item())
        for gr in grads:
            for k in PARAM_KEYS:
                cnt += int((~torch.isfinite(gr[k])).sum().item())
        return cnt

    def probe(self, model, dataset, theta0, opt0, src_idx, peer_idxs, holdout_idxs, cell, probe_no):
        dev = "cuda"
        # reset to theta0 + opt0
        self.set_state(model, theta0)
        model.optimizer.load_state_dict(opt0)
        model.optimizer.zero_grad(set_to_none=True)
        # Phase A: native gradient at theta0 on source cam
        cam, gt = dataset.get_item(src_idx)
        img = self.render(model, cam)
        loss = self.train_loss(img, gt)
        loss.backward()
        g = {k: getattr(model, "_"+k).grad.detach().clone() for k in PARAM_KEYS}
        # Phase B: native Adam step from theta0
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        theta_adam = self.get_state(model)
        # geometry displacement (tangent)
        d_xyz = theta_adam["xyz"] - theta0["xyz"]
        d_ls = theta_adam["scaling"] - theta0["scaling"]
        d_omega = TF.quat_relative_log_batch(theta0["rotation"], theta_adam["rotation"])
        d_op = theta_adam["opacity"] - theta0["opacity"]
        d_shs = theta_adam["shs"] - theta0["shs"]
        # Fisher natural direction from native gradient
        q0 = theta0["rotation"]
        q0n = TF.normalize_quat(q0)
        g_omega = TF.quat_grad_to_tangent_batch(g["rotation"], q0n)
        dNat_mu, dNat_ls, dNat_om = TF.natural_direction_batch(g["xyz"], g["scaling"], g_omega, q0, theta0["scaling"])
        adam_geo = (d_xyz, d_ls, d_omega)
        nat_geo = (dNat_mu, dNat_ls, dNat_om)
        # FR-GLOBAL
        Si, lam, _ = TF.sigma_inv_batch(q0n, theta0["scaling"])
        Gr = TF.G_rot_batch(lam)
        rA = TF.radius_geom_batch(d_xyz, d_ls, d_omega, Si, lam, Gr).sum()
        rN = TF.radius_geom_batch(dNat_mu, dNat_ls, dNat_om, Si, lam, Gr).sum()
        aG = torch.sqrt(rA / torch.clamp(rN, min=1e-30))
        cg = TF.apply_geometry_batch(theta0["xyz"], theta0["scaling"], q0, aG*dNat_mu, aG*dNat_ls, aG*dNat_om)
        state_g = {"xyz": cg[0], "scaling": cg[1], "rotation": cg[2],
                   "opacity": theta0["opacity"] + d_op, "shs": theta0["shs"] + d_shs}
        # FR-LOCAL
        rA_i = TF.radius_geom_batch(d_xyz, d_ls, d_omega, Si, lam, Gr)
        rN_i = TF.radius_geom_batch(dNat_mu, dNat_ls, dNat_om, Si, lam, Gr)
        aL = torch.where(rN_i > 0, torch.sqrt(rA_i / torch.clamp(rN_i, min=1e-30)), torch.zeros_like(rA_i))
        cl = TF.apply_geometry_batch(theta0["xyz"], theta0["scaling"], q0, aL[:,None]*dNat_mu, aL[:,None]*dNat_ls, aL[:,None]*dNat_om)
        state_l = {"xyz": cl[0], "scaling": cl[1], "rotation": cl[2],
                   "opacity": theta0["opacity"] + d_op, "shs": theta0["shs"] + d_shs}
        # radius parity (float64)
        relG, relL_med, relL_p99, n_cmp = self._radius(adam_geo, nat_geo, q0, theta0["scaling"])
        # NaN / invalid
        nan_inf = self._nan_check([theta0, theta_adam, state_g, state_l], [g])
        qnorm_err = float((TF.normalize_quat(torch.cat([state_g["rotation"], state_l["rotation"]],0)).norm(dim=-1) - 1).abs().max().item()) if state_g["rotation"].shape[0] else 0.0
        scale_min = float(torch.min(torch.exp(state_g["scaling"])).item())
        # evaluation: BASE / ADAM / GLOBAL / LOCAL on source + peer + holdout
        eval_cams = [src_idx] + list(peer_idxs) + list(holdout_idxs)
        ev_base = self.evaluate(model, theta0, eval_cams, dataset)
        ev_adam = self.evaluate(model, theta_adam, eval_cams, dataset)
        ev_g = self.evaluate(model, state_g, eval_cams, dataset)
        ev_l = self.evaluate(model, state_l, eval_cams, dataset)
        def src_L(ev): return ev[0][1]
        def mean_peer(ev): return float(np.mean([r[1] for r in ev[1:9]]))
        def mean_hold(ev): return float(np.mean([r[1] for r in ev[9:17]]))
        def mean_hold_psnr(ev): return float(np.mean([r[2] for r in ev[9:17]]))
        def mean_hold_ssim(ev): return float(np.mean([r[3] for r in ev[9:17]]))
        Lb, La = src_L(ev_base), src_L(ev_adam)
        D_A = Lb - La
        rec = {
            "cell": cell, "probe": probe_no, "source_cam": int(src_idx),
            "n_gaussians": int(theta0["xyz"].shape[0]),
            "L_base_src": Lb, "L_adam_src": La,
            "L_global_src": src_L(ev_g), "L_local_src": src_L(ev_l),
            "D_A": D_A, "adam_source_descent": bool(D_A > 0),
            "peer_mean_L": {"base": mean_peer(ev_base), "adam": mean_peer(ev_adam),
                            "global": mean_peer(ev_g), "local": mean_peer(ev_l)},
            "holdout_mean_L": {"base": mean_hold(ev_base), "adam": mean_hold(ev_adam),
                               "global": mean_hold(ev_g), "local": mean_hold(ev_l)},
            "holdout_mean_psnr": {"base": mean_hold_psnr(ev_base), "adam": mean_hold_psnr(ev_adam),
                                  "global": mean_hold_psnr(ev_g), "local": mean_hold_psnr(ev_l)},
            "holdout_mean_ssim": {"base": mean_hold_ssim(ev_base), "adam": mean_hold_ssim(ev_adam),
                                  "global": mean_hold_ssim(ev_g), "local": mean_hold_ssim(ev_l)},
            "fr_global_radius_rel": relG,
            "fr_local_radius_median": relL_med, "fr_local_radius_p99": relL_p99,
            "fr_local_n_compared": n_cmp,
            "alpha_global": float(aG.item()), "alpha_local_median": float(torch.median(aL).item()),
            "nan_inf_count": nan_inf, "quad_norm_err": qnorm_err, "scale_min": scale_min,
            "grad_norm": {k: float(g[k].norm().item()) for k in PARAM_KEYS},
        }
        rec["R_src_global"] = (Lb - rec["L_global_src"]) / D_A if D_A > 0 else None
        rec["R_src_local"] = (Lb - rec["L_local_src"]) / D_A if D_A > 0 else None
        rec["peer_win_global"] = bool(mean_peer(ev_g) < mean_peer(ev_adam))
        rec["peer_win_local"] = bool(mean_peer(ev_l) < mean_peer(ev_adam))
        rec["holdout_win_global"] = bool(mean_hold(ev_g) < mean_hold(ev_adam))
        rec["holdout_win_local"] = bool(mean_hold(ev_l) < mean_hold(ev_adam))
        rec["holdout_psnr_delta_global"] = mean_hold_psnr(ev_g) - mean_hold_psnr(ev_adam)
        rec["holdout_psnr_delta_local"] = mean_hold_psnr(ev_l) - mean_hold_psnr(ev_adam)
        rec["holdout_ssim_delta_global"] = mean_hold_ssim(ev_g) - mean_hold_ssim(ev_adam)
        rec["holdout_ssim_delta_local"] = mean_hold_ssim(ev_l) - mean_hold_ssim(ev_adam)
        rec["correctness_pass"] = bool(relG <= 1e-5 and relL_med <= 1e-5 and relL_p99 <= 1e-4 and nan_inf == 0)
        # cleanup
        del g, d_xyz, d_ls, d_omega, d_op, d_shs, dNat_mu, dNat_ls, dNat_om, g_omega
        del state_g, state_l, ev_base, ev_adam, ev_g, ev_l, theta_adam
        torch.cuda.empty_cache()
        return rec

    def load_cell(self, cell):
        from gaussian_model import GaussianModel
        from dataset import GTDataset
        from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.reset()  # release prior cell renderer handle (new topology)
        scene, it = cell.split("@")
        ckpt = f"/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{it}.pt"
        dataset = GTDataset(scene=scene, repo_root=REPO, resolution="1080p", device="cuda", background="black")
        sd = torch.load(ckpt, map_location="cuda", weights_only=False)
        model = GaussianModel(max_sh_degree=self.config.sh_degree)
        model.restore(sd, self.config.to_dict())
        model.optimizer.load_state_dict(sd["optimizer_state_dict"])
        src_train, holdout = select_cameras(len(dataset))
        theta0 = self.get_state(model)
        opt0 = copy.deepcopy(model.optimizer.state_dict())
        meta = {"cell": cell, "scene": scene, "iter": int(it), "ckpt": ckpt,
                "iteration": int(sd["iteration"]), "arm": sd["arm"],
                "n_gaussians": int(sd["xyz"].shape[0]),
                "n_cameras": len(dataset), "src_train": [int(x) for x in src_train],
                "holdout": [int(x) for x in holdout]}
        return model, dataset, theta0, opt0, meta, src_train, holdout

    def run_cell(self, cell):
        model, dataset, theta0, opt0, meta, src_train, holdout = self.load_cell(cell)
        peer_map = {s: [x for x in src_train if x != s] for s in src_train}
        recs = []
        for pno, s in enumerate(src_train):
            t0 = time.perf_counter()
            rec = self.probe(model, dataset, theta0, opt0, s, peer_map[s], holdout, cell, pno)
            rec["wall_s"] = round(time.perf_counter() - t0, 2)
            recs.append(rec)
            print(f"  [{cell}] probe {pno} src={s} N={rec['n_gaussians']} "
                  f"D_A={rec['D_A']:.5f} Rg={rec['R_src_global']} Rl={rec['R_src_local']} "
                  f"relG={rec['fr_global_radius_rel']:.2e} nan={rec['nan_inf_count']} "
                  f"correct={rec['correctness_pass']} ({rec['wall_s']}s)", flush=True)
        return meta, recs


def aggregate_cell(meta, recs):
    valid = [r for r in recs if r["adam_source_descent"]]
    out = {"variants": {}}
    for variant, sfx in [("FR-GLOBAL", "global"), ("FR-LOCAL", "local")]:
        R = np.array([r["R_src_" + sfx] for r in valid], dtype=np.float64) if valid else np.array([])
        peer_win = float(np.mean([r["peer_win_" + sfx] for r in valid])) if valid else 0.0
        hold_win = float(np.mean([r["holdout_win_" + sfx] for r in valid])) if valid else 0.0
        psnr_d = np.array([r["holdout_psnr_delta_" + sfx] for r in valid], dtype=np.float64) if valid else np.array([])
        ssim_d = np.array([r["holdout_ssim_delta_" + sfx] for r in valid], dtype=np.float64) if valid else np.array([])
        relG = [r["fr_global_radius_rel"] for r in recs]
        relL_med = [r["fr_local_radius_median"] for r in recs]
        relL_p99 = [r["fr_local_radius_p99"] for r in recs]
        correctness = all(r["correctness_pass"] for r in recs) and max(relG) <= 1e-5 and \
            (np.median(relL_med) <= 1e-5) and (np.percentile(relL_p99, 99) <= 1e-4)
        out["variants"][variant] = {
            "valid_source_descent_count": int(len(valid)),
            "R_src_median": float(np.median(R)) if R.size else None,
            "R_src_p10": float(np.percentile(R, 10)) if R.size else None,
            "peer_win_rate": peer_win,
            "truehold_win_rate": hold_win,
            "truehold_psnr_delta_median_db": float(np.median(psnr_d)) if psnr_d.size else None,
            "truehold_psnr_delta_worst_db": float(np.min(psnr_d)) if psnr_d.size else None,
            "truehold_ssim_delta_median_db": float(np.median(ssim_d)) if ssim_d.size else None,
            "correctness_pass": bool(correctness),
            "nan_inf_count": int(sum(r["nan_inf_count"] for r in recs)),
            "n_probes": len(recs),
            "fr_global_radius_rel_worst": float(max(relG)) if relG else None,
            "fr_local_radius_median_worst": float(max(relL_med)) if relL_med else None,
            "fr_local_radius_p99_worst": float(max(relL_p99)) if relL_p99 else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="room@10000,room@15000,bicycle@10000,bicycle@15000")
    ap.add_argument("--outdir", default=os.path.join(RUN_DIR, "out"))
    ap.add_argument("--smoke", action="store_true", help="room@10000 first source cam only")
    args = ap.parse_args()
    cells = args.cells.split(",")
    os.makedirs(args.outdir, exist_ok=True)
    rasterize = bootstrap_c0()
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    global GTDataset
    from dataset import GTDataset
    config = ReferenceV1Config()
    config.seed = 42
    ig = IG0(rasterize, config)
    print("fused_ssim so sha256", ig.fused_so_sha, "lock_ok", ig.fused_so_sha == P11_FUSED_SO_SHA256, flush=True)
    all_metas = {}; all_recs = []
    for cell in cells:
        meta, recs = ig.run_cell(cell)
        all_metas[cell] = meta; all_recs.extend(recs)
        if args.smoke:
            break
    # summary
    summary = {"schema_version": "IG0-v1", "cells": {}}
    for cell in all_metas:
        recs_c = [r for r in all_recs if r["cell"] == cell]
        summary["cells"][cell] = aggregate_cell(all_metas[cell], recs_c)
    with open(os.path.join(args.outdir, "IG0_PROBES.json"), "w") as f:
        json.dump(all_recs, f, indent=2)
    th = {"per_cell": {c: {"holdout": all_metas[c]["holdout"],
                           "records": [r for r in all_recs if r["cell"] == c]} for c in all_metas}}
    with open(os.path.join(args.outdir, "IG0_TRUE_HOLDOUT.json"), "w") as f:
        json.dump(th, f, indent=2)
    with open(os.path.join(args.outdir, "IG0_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE summary/probes/true_holdout to", args.outdir, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
