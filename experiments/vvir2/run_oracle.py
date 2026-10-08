"""VRIR2 minimal incremental raster-update runner.

Reuses the VRIR1 training scaffolding (same window, densification OFF, same
tracked views). At each refresh snapshot it captures the REAL frozen gsplat
membership (ground truth) for the tracked cameras and, for each cadence in
{500,1000,2000,5000}, runs the incremental repair M_old + Delta_G -> M_new from
the cadence-delayed old snapshot, then:

  - verifies bit-exact equality vs the baseline full rebuild (Phase 3),
  - times baseline full rebuild vs incremental repair (Phase 4),
  - records r_g (changed gaussian ratio) and r_t (changed tile ratio) (Phase 1/5).

Fail-closed: any correctness mismatch is recorded and drives DROP_CORRECTNESS.
No rescue tuning, no threshold search, no scene-specific optimization.
"""
import sys, os, json, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
VVIR0 = os.path.join(REPO, "experiments", "vvir0")
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
_struct = _load("vir0_structure", os.path.join(VVIR0, "oracle", "structure.py"))
select_tracked_views = _struct.select_tracked_views
import vvir0_patch as _v0p; _v0p.apply()

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization

sys.path.insert(0, HERE)
from oracle import incremental as INC


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


def render_with_meta(model, cam, sh_degree):
    data = {"xyz": model.get_xyz, "rotations": model.get_rotation, "scales": model.get_scaling,
            "opacity": model.get_opacity, "shs": model.get_features}
    r, _, meta = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height, tile_size=16,
        packed=False, sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
        render_mode="RGB", absgrad=True)
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


def _sync(): torch.cuda.synchronize()


def _evt():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


def _elapsed(a, b):
    b.synchronize(); return a.elapsed_time(b)


@torch.no_grad()
def capture(model, cam, sh_degree, W, H, tile_size, accutile=False):
    """Capture real baseline membership + descriptor + timing for one camera."""
    _, meta = render_with_meta(model, cam, sh_degree)
    means2d = meta["means2d"][0].to(torch.float32).contiguous()
    radii = meta["radii"][0].contiguous()
    depths = meta["depths"][0].to(torch.float32).contiguous()
    # descriptor
    d = INC.make_descriptor(means2d, radii, depths, W, H, tile_size)
    # baseline membership (timed: full rebuild = isect_tiles + offset encode)
    t0 = _evt(); m_base = INC.baseline_membership(means2d, radii, depths, W, H, tile_size); t1 = _evt()
    t_base_full = _elapsed(t0, t1)
    # emit-only (sort=False) for membership-kernel vs sorting split
    t2 = _evt()
    from gsplat import isect_tiles as _isect_tiles
    tw = (W + tile_size - 1) // tile_size; th = (H + tile_size - 1) // tile_size
    _isect_tiles(means2d.unsqueeze(0), radii.unsqueeze(0), depths.unsqueeze(0),
                 tile_size, tw, th, sort=False, segmented=False, packed=False, n_images=1)
    t3 = _evt()
    t_base_emit = _elapsed(t2, t3)
    return {"means2d": means2d, "radii": radii, "depths": depths,
            "desc": d, "m_base": m_base, "t_base_full": t_base_full,
            "t_base_emit": t_base_emit, "n_isects": int(m_base.flatten_ids.shape[0])}


def run(cfg, scene, start_iter, end_iter, n_tracked, out_dir, gap, cadences,
        accutile=False, smoke=False):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    dataset = GTDataset(scene=scene, repo_root=REPO, resolution=cfg.resolution,
                        device="cuda", background="black")
    n_cameras = len(dataset)
    model, pcd = build_model(cfg, scene)
    N0 = model._xyz.shape[0]
    W, H = dataset.target_width, dataset.target_height
    tracked = select_tracked_views(n_cameras, n_tracked)
    print(f"[VRIR2] scene={scene} N={N0} cams={n_cameras} tracked={tracked} gap={gap} cadences={cadences}", flush=True)

    rng = random.Random(cfg.seed)
    stack = list(range(n_cameras))
    for _ in range(start_iter - 1):
        if not stack: stack = list(range(n_cameras))
        stack.pop(rng.randint(0, len(stack) - 1))
    seq = []
    for it in range(start_iter, end_iter + 1):
        if not stack: stack = list(range(n_cameras))
        seq.append(stack.pop(rng.randint(0, len(stack) - 1)))
    seq = np.array(seq, dtype=np.int32)

    max_cad = max(cadences)
    buf = max_cad // gap  # ring buffer length in snapshots
    # ring[cam] = list of (iter, desc, m_base) most recent `buf` snapshots
    ring = {c: [] for c in tracked}
    rec_path = os.path.join(out_dir, "vir2_records.jsonl")
    if os.path.exists(rec_path): os.remove(rec_path)
    rec_f = open(rec_path, "w", encoding="utf-8")

    def capture_all(it):
        snap = {}
        for c in tracked:
            cap = capture(model, dataset.get_item(c)[0], model.active_sh_degree, W, H,
                          cfg.tile_size, accutile=accutile)
            snap[c] = cap
        return snap

    def push_ring(it, snap):
        for c in tracked:
            ring[c].append((it, snap[c]["desc"], snap[c]["m_base"]))
            if len(ring[c]) > buf: ring[c].pop(0)

    def find_old(c, target_iter):
        for (it, d, m) in ring[c]:
            if it == target_iter:
                return d, m
        return None, None

    t_iter = []; t_fwd = []
    is_s, ie_s = _evt(), _evt()
    fs, fe = _evt(), _evt()

    # initial snapshot at start_iter (push so it is available as the old reference)
    cur = capture_all(start_iter)
    push_ring(start_iter, cur)
    prev_iter = start_iter

    for k, iteration in enumerate(range(start_iter + 1, end_iter + 1)):
        cam_idx = int(seq[k])
        cam, gt = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)
        is_s.record(); fs.record()
        image, meta = render_with_meta(model, cam, model.active_sh_degree)
        fe.record()
        L1 = F.l1_loss(image, gt); dssim = ssim_fn(image, gt)
        loss = (1.0 - cfg.lambda_dssim) * L1 + cfg.lambda_dssim * dssim
        loss.backward(); ie_s.record()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        _sync()
        t_iter.append(_elapsed(is_s, ie_s)); t_fwd.append(_elapsed(fs, fe))

        if (iteration - prev_iter) % gap == 0 or iteration == end_iter:
            snap = capture_all(iteration)
            for c in tracked:
                cap = snap[c]
                d_new = cap["desc"]; m_base = cap["m_base"]
                for C in cadences:
                    if (iteration - start_iter) % C != 0:
                        continue
                    old_iter = iteration - C
                    if old_iter < start_iter:
                        continue
                    d_old, m_old = find_old(c, old_iter)
                    if d_old is None:
                        continue
                    # incremental repair (timed)
                    ts = _evt()
                    m_new, ir = INC.incremental_membership(m_old, d_old, d_new)
                    te = _evt()
                    t_inc = _elapsed(ts, te)
                    ok, detail = INC.exact_match(m_new, m_base)
                    rec_f.write(json.dumps({
                        "scene": scene, "camera_id": int(c), "cadence": int(C),
                        "iter_prev": int(old_iter), "iter_cur": int(iteration),
                        "n_gaussians": int(N0), "n_isects": cap["n_isects"],
                        "r_g": ir.r_g, "r_t": ir.r_t,
                        "n_changed": ir.n_changed, "n_affected_tiles": ir.n_affected_tiles,
                        "correct": bool(ok), "detail": detail,
                        "t_base_full_ms": cap["t_base_full"],
                        "t_base_emit_ms": cap["t_base_emit"],
                        "t_inc_ms": t_inc,
                        "membership_speedup": (cap["t_base_full"] / t_inc) if t_inc > 0 else 0.0,
                    }, separators=(",", ":")) + "\n")
                    if not ok:
                        print(f"  [MISMATCH] {scene} cam{c} C={C} iter{iteration}: {detail}", flush=True)
            rec_f.flush()
            push_ring(iteration, snap)
            prev_iter = iteration
            print(f"  iter {iteration} N {model._xyz.shape[0]} fwd {t_fwd[-1]:.1f}ms tot {t_iter[-1]:.1f}ms", flush=True)

    rec_f.close()
    n_snap = int((end_iter - start_iter) // gap + 1)
    summ = {"scene": scene, "start_iter": start_iter, "end_iter": end_iter, "N0": int(N0),
            "n_cameras": int(n_cameras), "tracked_views": [int(t) for t in tracked],
            "gap_iters": int(gap), "cadences": [int(c) for c in cadences],
            "n_snapshots": n_snap, "t_iter_mean_ms": float(np.mean(t_iter)),
            "t_fwd_mean_ms": float(np.mean(t_fwd)), "accutile": bool(accutile),
            "gsplat_version": "1.5.3", "smoke": bool(smoke)}
    with open(os.path.join(out_dir, "vir2_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[VRIR2] DONE scene={scene} snapshots={n_snap} t_iter={summ['t_iter_mean_ms']:.1f}ms", flush=True)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--start_iter", type=int, default=15001)
    ap.add_argument("--end_iter", type=int, default=30000)
    ap.add_argument("--n_tracked", type=int, default=8)
    ap.add_argument("--gap", type=int, default=500)
    ap.add_argument("--cadences", default="500,1000,2000,5000")
    ap.add_argument("--accutile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ReferenceV1Config(scene=args.scene)
    cadences = [int(x) for x in args.cadences.split(",") if x]
    out = args.out or os.path.join(REPO, "experiments", "vvir2", "runs",
                                   f"{args.scene}_{args.start_iter}_{args.end_iter}_g{args.gap}")
    run(cfg, args.scene, args.start_iter, args.end_iter, args.n_tracked, out, args.gap,
        cadences, accutile=args.accutile, smoke=args.smoke)