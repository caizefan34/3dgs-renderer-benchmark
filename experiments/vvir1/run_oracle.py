"""VRIR1-P0 Incremental Raster Update Oracle runner.

Continues from the official trained (post-15k, frozen-topology) PLY checkpoint and
runs the canonical training loop in the 15001-30000 window with densification OFF
(identical to VRIR0). At fixed iteration checkpoints it snapshots the fresh raster
structure for the deterministic tracked cameras and measures, between consecutive
checkpoints:

  G3 state drift (xyz/radius/depth/means2d) and
  G4 (gaussian, tile) membership delta (unchanged/added/removed).

The baseline still runs the normal fresh intersection/sort/offset path every
iteration; this oracle only OBSERVES (no_grad snapshot renders). It never modifies
training or the rasterizer, and introduces no CUDA. Membership semantics are taken
verbatim from the VRIR0-verified tile_bounds / validate_membership_model (G1 lock).
"""
import sys, os, json, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
VVIR0 = os.path.join(REPO, "experiments", "vvir0")
# G1 source lock: reuse VRIR0-verified modules. vvir0 on path for vvir0_patch.
sys.path.insert(0, VVIR0)
sys.path.insert(0, os.path.join(REPO, "scripts", "epic05", "phase7"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "baseline", "reference_v1"))

import importlib.util as _ilu
def _load(mod_name, path):
    spec = _ilu.spec_from_file_location(mod_name, path)
    m = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m); return m
_cert = _load("vir0_certificate", os.path.join(VVIR0, "oracle", "certificate.py"))
_struct = _load("vir0_structure", os.path.join(VVIR0, "oracle", "structure.py"))
tile_bounds = _cert.tile_bounds
validate_membership_model = _cert.validate_membership_model
select_tracked_views = _struct.select_tracked_views
import vvir0_patch as _v0p; _v0p.apply()

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization

sys.path.insert(0, HERE)
from oracle.drift import state_drift, membership_delta, unaffected_fraction


def build_model(cfg, scene):
    d = load_ply(os.path.join(REPO, "data", "official", "mipnerf360", scene, "point_cloud.ply"), device="cuda")
    m = GaussianModel(max_sh_degree=cfg.sh_degree)
    m.create_from_pcd(d, spatial_lr_scale=1.0)
    m.active_sh_degree = cfg.sh_degree
    m.training_setup({
        "position_lr_init": cfg.position_lr_init,
        "position_lr_final": cfg.position_lr_final,
        "position_lr_delay_mult": cfg.position_lr_delay_mult,
        "position_lr_max_steps": cfg.position_lr_max_steps,
        "feature_lr": cfg.feature_lr, "opacity_lr": cfg.opacity_lr,
        "scaling_lr": cfg.scaling_lr, "rotation_lr": cfg.rotation_lr,
        "percent_dense": cfg.percent_dense,
    })
    return m, d


def render_with_meta(model, cam, sh_degree, accutile=False):
    import inspect
    data = {"xyz": model.get_xyz, "rotations": model.get_rotation, "scales": model.get_scaling,
            "opacity": model.get_opacity, "shs": model.get_features}
    _kw = dict(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
               opacities=data["opacity"], colors=data["shs"],
               viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
               width=cam.image_width, height=cam.image_height, tile_size=16,
               packed=False, sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
               render_mode="RGB", absgrad=True)
    if "accutile" in inspect.signature(rasterization).parameters:
        _kw["accutile"] = accutile
    r, _, meta = rasterization(**_kw)
    return r[0].clamp(0, 1), meta


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2; self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous().permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2
    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2); target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = torch.nn.functional.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = torch.nn.functional.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]; bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()

ssim_fn = SepSSIM(device="cuda")


@torch.no_grad()
def capture_structure(model, cam, sh_degree, width, height, tile_size, accutile=False):
    """Snapshot fresh raster membership for one camera. Returns compact arrays."""
    _, meta = render_with_meta(model, cam, sh_degree, accutile=accutile)
    m2d = meta["means2d"][0].detach().to("cpu", torch.float64).numpy()
    radii = meta["radii"][0].detach().to("cpu", torch.float64).numpy()
    depths = meta["depths"][0].detach().to("cpu", torch.float64).numpy()
    b_minus, b_plus, active = tile_bounds(m2d, radii, width, height, tile_size)
    if not bool(np.all(active)):
        m2d = m2d.copy(); depths = depths.copy()
        m2d[~active] = 0.0; depths[~active] = 0.0
    tw = (width + tile_size - 1) // tile_size
    flat = meta["flatten_ids"].detach().to("cpu", torch.int32).numpy().copy()
    off_raw = meta["isect_offsets"][0].detach().to("cpu", torch.int32).numpy()
    off = np.concatenate([off_raw.reshape(-1), np.array([flat.shape[0]], dtype=np.int32)]).astype(np.int32)
    mm_ok, mm_bad = validate_membership_model(flat, off, b_minus, b_plus, active, tw, m2d.shape[0])
    return {"means2d": m2d, "radii": radii, "depths": depths, "b_minus": b_minus,
            "b_plus": b_plus, "active": active, "n_isects": int(flat.shape[0]),
            "mm_ok": bool(mm_ok), "mm_bad": int(mm_bad)}


def run(cfg, scene, start_iter, end_iter, n_tracked, out_dir, gap, accutile=False):
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "vir1_records.jsonl")
    if os.path.exists(rec_path):
        os.remove(rec_path)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    dataset = GTDataset(scene=scene, repo_root=REPO, resolution=cfg.resolution,
                        device="cuda", background="black")
    n_cameras = len(dataset)
    model, pcd = build_model(cfg, scene)
    N0 = model._xyz.shape[0]
    W, H = dataset.target_width, dataset.target_height
    tracked = select_tracked_views(n_cameras, n_tracked)
    print(f"[VRIR1] scene={scene} N={N0} cameras={n_cameras} tracked={tracked} gap={gap}", flush=True)

    rng = random.Random(cfg.seed)
    stack = list(range(n_cameras))
    for _ in range(start_iter - 1):
        if not stack:
            stack = list(range(n_cameras))
        stack.pop(rng.randint(0, len(stack) - 1))
    seq = []
    for it in range(start_iter, end_iter + 1):
        if not stack:
            stack = list(range(n_cameras))
        seq.append(stack.pop(rng.randint(0, len(stack) - 1)))
    seq = np.array(seq, dtype=np.int32)

    def snap():
        xyz_c = model.get_xyz.detach().to("cpu", torch.float64).numpy()
        cams = {c: capture_structure(model, dataset.get_item(c)[0], model.active_sh_degree,
                                     W, H, cfg.tile_size, accutile=accutile) for c in tracked}
        return xyz_c, cams

    prev_xyz, prev = snap()
    prev_iter = start_iter
    anchor_xyz, anchor = prev_xyz, prev  # window-start reference for cumulative deltas
    for c in tracked:
        if not prev[c]["mm_ok"]:
            print(f"  [WARN] {scene} cam {c} baseline membership mismatch={prev[c]['mm_bad']}", flush=True)

    rec_f = open(rec_path, "w", encoding="utf-8")
    it_s = torch.cuda.Event(enable_timing=True); it_e = torch.cuda.Event(enable_timing=True)
    fs = torch.cuda.Event(enable_timing=True); fe = torch.cuda.Event(enable_timing=True)
    bs = torch.cuda.Event(enable_timing=True); be = torch.cuda.Event(enable_timing=True)
    t_iter = []; t_fwd = []; t_bwd = []
    _ = (it_s, it_e, fs, fe, bs, be)  # keep refs

    for k, iteration in enumerate(range(start_iter, end_iter + 1)):
        cam_idx = int(seq[k])
        cam, gt = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)
        it_s.record(); fs.record()
        image, meta = render_with_meta(model, cam, model.active_sh_degree, accutile=accutile)
        fe.record()
        L1 = F.l1_loss(image, gt); dssim = ssim_fn(image, gt)
        loss = (1.0 - cfg.lambda_dssim) * L1 + cfg.lambda_dssim * dssim
        bs.record(); loss.backward(); be.record()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        it_e.record(); torch.cuda.synchronize()
        t_iter.append(it_s.elapsed_time(it_e)); t_fwd.append(fs.elapsed_time(fe)); t_bwd.append(bs.elapsed_time(be))

        if (iteration - start_iter) % gap == 0 or iteration == end_iter:
            cur_xyz, cur = snap()
            for c in tracked:
                p = prev[c]; q = cur[c]; a = anchor[c]
                # (a) per-gap delta: prev -> cur  (operational refresh cadence)
                drift = state_drift(prev_xyz, cur_xyz, p["means2d"], q["means2d"],
                                    p["radii"], q["radii"], p["depths"], q["depths"],
                                    p["active"], q["active"])
                un, add, rem = membership_delta(p["b_minus"], p["b_plus"], q["b_minus"], q["b_plus"])
                rec_f.write(json.dumps({"scene": scene, "camera_id": int(c), "horizon": "gap",
                       "iter_prev": int(prev_iter), "iter_cur": int(iteration),
                       "gap_iters": int(iteration - prev_iter),
                       "n_isects_prev": p["n_isects"], "n_isects_cur": q["n_isects"],
                       "unchanged": int(un), "added": int(add), "removed": int(rem),
                       "unaffected_uf": unaffected_fraction(un, add, rem),
                       "mm_ok": bool(q["mm_ok"] and p["mm_ok"]), "drift": drift},
                       separators=(",", ":")) + "\n")
                # (b) cumulative delta: anchor(window start) -> cur
                drift_c = state_drift(anchor_xyz, cur_xyz, a["means2d"], q["means2d"],
                                       a["radii"], q["radii"], a["depths"], q["depths"],
                                       a["active"], q["active"])
                unc, addc, remc = membership_delta(a["b_minus"], a["b_plus"], q["b_minus"], q["b_plus"])
                rec_f.write(json.dumps({"scene": scene, "camera_id": int(c), "horizon": "cumulative",
                       "iter_prev": int(start_iter), "iter_cur": int(iteration),
                       "gap_iters": int(iteration - start_iter),
                       "n_isects_prev": a["n_isects"], "n_isects_cur": q["n_isects"],
                       "unchanged": int(unc), "added": int(addc), "removed": int(remc),
                       "unaffected_uf": unaffected_fraction(unc, addc, remc),
                       "mm_ok": bool(q["mm_ok"] and a["mm_ok"]), "drift": drift_c},
                       separators=(",", ":")) + "\n")
            rec_f.flush()
            prev_xyz, prev = cur_xyz, cur; prev_iter = iteration
            print(f"  iter {iteration} snapshot N {model._xyz.shape[0]} "
                  f"fwd {t_fwd[-1]:.1f}ms tot {t_iter[-1]:.1f}ms", flush=True)

    rec_f.close()
    n_snap = int((end_iter - start_iter) // gap + 1)
    summ = {"scene": scene, "start_iter": start_iter, "end_iter": end_iter, "N0": int(N0),
            "n_cameras": int(n_cameras), "tracked_views": [int(t) for t in tracked],
            "gap_iters": int(gap), "n_snapshots": n_snap,
            "t_iter_mean_ms": float(np.mean(t_iter)), "t_fwd_mean_ms": float(np.mean(t_fwd)),
            "t_bwd_mean_ms": float(np.mean(t_bwd)), "accutile": bool(accutile),
            "gsplat_version": "1.5.3"}
    with open(os.path.join(out_dir, "vir1_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    with open(os.path.join(out_dir, "vir1_timing.json"), "w") as f:
        json.dump({"t_iter_mean_ms": float(np.mean(t_iter)), "t_fwd_mean_ms": float(np.mean(t_fwd)),
                   "t_bwd_mean_ms": float(np.mean(t_bwd)), "n_iters": len(t_iter)}, f, indent=2)
    print(f"[VRIR1] DONE scene={scene} snapshots={n_snap} t_iter={summ['t_iter_mean_ms']:.1f}ms", flush=True)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--start_iter", type=int, default=15001)
    ap.add_argument("--end_iter", type=int, default=30000)
    ap.add_argument("--n_tracked", type=int, default=8)
    ap.add_argument("--gap", type=int, default=500)
    ap.add_argument("--accutile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ReferenceV1Config(scene=args.scene)
    out = args.out or os.path.join(REPO, "experiments", "vvir1", "runs",
                                   f"{args.scene}_{args.start_iter}_{args.end_iter}_g{args.gap}")
    run(cfg, args.scene, args.start_iter, args.end_iter, args.n_tracked, out, args.gap, accutile=args.accutile)