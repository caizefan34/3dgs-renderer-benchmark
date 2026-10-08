"""VRIR0 oracle runner.

Continues from an official trained (post-15k, frozen-topology) PLY checkpoint and
runs the canonical training loop in the 15001-30000 window with densification OFF.
The baseline still executes the fresh intersection/sort/offset path every iteration;
the oracle only observes tracked-camera revisits and evaluates certificates.

Does NOT modify baseline source. Imports baseline modules read-only.
"""
import sys, os, json, time, random, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # experiments/vvir0
import vvir0_patch; vvir0_patch.apply()

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# reference_v1 FIRST so our GaussianModel/config take priority (matches trainer.py)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'epic05', 'phase7'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'baseline', 'reference_v1'))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from colmap_reader import read_points3D_binary, sfm_to_pcd_data
from dataset import GTDataset
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization

from oracle.hook import VRIR0Oracle
from oracle.structure import select_tracked_views


def build_model(cfg, scene):
    d = load_ply(os.path.join(REPO, 'data', 'official', 'mipnerf360', scene, 'point_cloud.ply'), device='cuda')
    m = GaussianModel(max_sh_degree=cfg.sh_degree)
    m.create_from_pcd(d, spatial_lr_scale=1.0)
    m.active_sh_degree = cfg.sh_degree  # trained checkpoint has full SH
    m.training_setup({
        'position_lr_init': cfg.position_lr_init,
        'position_lr_final': cfg.position_lr_final,
        'position_lr_delay_mult': cfg.position_lr_delay_mult,
        'position_lr_max_steps': cfg.position_lr_max_steps,
        'feature_lr': cfg.feature_lr, 'opacity_lr': cfg.opacity_lr,
        'scaling_lr': cfg.scaling_lr, 'rotation_lr': cfg.rotation_lr,
        'percent_dense': cfg.percent_dense,
    })
    return m, d


def run(cfg, scene, start_iter, end_iter, n_tracked, out_dir, cache_cap_bytes,
        accutile=False, eval_every=0):
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, 'vir0_records.jsonl')
    if os.path.exists(rec_path):
        os.remove(rec_path)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    dataset = GTDataset(scene=scene, repo_root=REPO, resolution=cfg.resolution,
                        device='cuda', background='black')
    n_cameras = len(dataset)
    model, pcd = build_model(cfg, scene)
    N0 = model._xyz.shape[0]

    tracked = select_tracked_views(n_cameras, n_tracked)
    print(f'[VRIR0] scene={scene} N={N0} cameras={n_cameras} tracked={tracked}')

    # frozen camera sequence (same RNG scheme as baseline, advanced to start_iter)
    rng = random.Random(cfg.seed)
    stack = list(range(n_cameras))
    # advance RNG through (start_iter - 1) iterations to match canonical baseline order
    for _ in range(start_iter - 1):
        if not stack:
            stack = list(range(n_cameras))
        stack.pop(rng.randint(0, len(stack) - 1))
    seq = []
    for it in range(start_iter, end_iter + 1):
        if not stack:
            stack = list(range(n_cameras))
        idx = rng.randint(0, len(stack) - 1)
        seq.append(stack.pop(idx))
    seq = np.array(seq, dtype=np.int32)

    oracle = VRIR0Oracle(scene=scene, width=dataset.target_width, height=dataset.target_height,
                         tile_size=cfg.tile_size, tracked_views=tracked,
                         cache_byte_cap=cache_cap_bytes, record_path=rec_path,
                         epoch_fn=lambda: 0)  # frozen topology -> epoch 0

    # timing
    iter_start = torch.cuda.Event(enable_timing=True); iter_end = torch.cuda.Event(enable_timing=True)
    fwd_s = torch.cuda.Event(enable_timing=True); fwd_e = torch.cuda.Event(enable_timing=True)
    bwd_s = torch.cuda.Event(enable_timing=True); bwd_e = torch.cuda.Event(enable_timing=True)
    t_iter=[]; t_fwd=[]; t_bwd=[]

    # topology frozen in this window -> no densification, no opacity reset
    for k, iteration in enumerate(range(start_iter, end_iter + 1)):
        cam_idx = int(seq[k])
        cam, gt = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)
        iter_start.record(); fwd_s.record()
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree, accutile=accutile)
        fwd_e.record()
        # loss
        import torch.nn.functional as F
        L1 = F.l1_loss(image, gt)
        # SSIM (reuse baseline SepSSIM via import)
        dssim = ssim_fn(image, gt)
        loss = (1.0 - cfg.lambda_dssim) * L1 + cfg.lambda_dssim * dssim
        bwd_s.record(); loss.backward(); bwd_e.record()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        iter_end.record(); torch.cuda.synchronize()
        t_iter.append(iter_start.elapsed_time(iter_end)); t_fwd.append(fwd_s.elapsed_time(fwd_e)); t_bwd.append(bwd_s.elapsed_time(bwd_e))

        # oracle: only acts on tracked-camera visits
        oracle.on_visit(meta, iteration, cam_idx, config_same=True)

        if iteration % 500 == 0:
            print(f'  iter {iteration} cam {cam_idx} loss {loss.item():.5f} N {model._xyz.shape[0]} '
                  f'fwd {t_fwd[-1]:.1f}ms tot {t_iter[-1]:.1f}ms revisits {oracle._revisit_count}')

    # summary
    summ = oracle.summary()
    summ.update({
        'scene': scene, 'start_iter': start_iter, 'end_iter': end_iter,
        'N0': int(N0), 'n_cameras': int(n_cameras), 'tracked_views': [int(t) for t in tracked],
        't_iter_mean_ms': float(np.mean(t_iter)), 't_fwd_mean_ms': float(np.mean(t_fwd)),
        't_bwd_mean_ms': float(np.mean(t_bwd)),
        'accutile': bool(accutile),
        'gsplat_version': '1.4.0',
    })
    with open(os.path.join(out_dir, 'vir0_summary.json'), 'w') as f:
        json.dump(summ, f, indent=2)
    # timing json (baseline T_struct / T_iter provenance)
    timing = {'t_iter_mean_ms': float(np.mean(t_iter)), 't_fwd_mean_ms': float(np.mean(t_fwd)),
              't_bwd_mean_ms': float(np.mean(t_bwd)), 'n_iters': len(t_iter)}
    with open(os.path.join(out_dir, 'vir0_timing.json'), 'w') as f:
        json.dump(timing, f, indent=2)
    print(f'[VRIR0] DONE scene={scene} revisits={summ["revisit_count"]} '
          f't_iter={summ["t_iter_mean_ms"]:.1f}ms t_fwd={summ["t_fwd_mean_ms"]:.1f}ms')
    return summ


def render_with_meta(model, cam, sh_degree, accutile=False):
    data = {'xyz': model.get_xyz, 'rotations': model.get_rotation, 'scales': model.get_scaling,
            'opacity': model.get_opacity, 'shs': model.get_features}
    # gsplat 1.5.3 clean fork has no 'accutile' kwarg; only pass it when supported.
    import inspect
    _kw = dict(means=data['xyz'], quats=data['rotations'], scales=data['scales'],
               opacities=data['opacity'], colors=data['shs'],
               viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
               width=cam.image_width, height=cam.image_height, tile_size=16,
               packed=False, sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
               render_mode='RGB', absgrad=True)
    if 'accutile' in inspect.signature(rasterization).parameters:
        _kw['accutile'] = accutile
    r, _, meta = rasterization(**_kw)
    means2d_full = meta['means2d']
    if means2d_full.requires_grad:
        means2d_full.retain_grad()
    return r[0].clamp(0, 1), meta, means2d_full


# SSIM (separable, same as baseline)
class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device='cuda'):
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

ssim_fn = SepSSIM(device='cuda')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='room')
    ap.add_argument('--start_iter', type=int, default=15001)
    ap.add_argument('--end_iter', type=int, default=30000)
    ap.add_argument('--n_tracked', type=int, default=8)
    ap.add_argument('--cache_cap_gb', type=float, default=4.0)
    ap.add_argument('--accutile', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = ReferenceV1Config(scene=args.scene)
    out = args.out or os.path.join(REPO, 'experiments', 'vvir0', 'runs', f'{args.scene}_{args.start_iter}_{args.end_iter}')
    run(cfg, args.scene, args.start_iter, args.end_iter, args.n_tracked, out,
        int(args.cache_cap_gb * 1e9), accutile=args.accutile)
