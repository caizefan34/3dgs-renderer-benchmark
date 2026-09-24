#!/usr/bin/env python3
"""DSH-E: training-stage representativeness audit — measurement harness.

Frozen C0 V3 renderer (F9 + SCALAR_ADJOINT + H8-MR), composed extension
sha256 7ca1c6bf..., identical bootstrap to c0t1_timing.py / e2p0_harness.py.
1080p (1920 long side), 3 scenes x {5K,15K,30K} checkpoints x 16 stratified
cameras (deterministic IDs).

Per checkpoint/camera:
  - C0 V3 forward: N, n_isects (MEASURED fine-pair count)
  - C0-exact support (fully_fused_projection + AABB tile convention):
      derived pair count (~3.84x over-estimate of n_isects, documented),
      per-Gaussian tile counts, 8x4 macro (macro,gid) mapping ->
      N_unique_(macro,gid), D_macro, multiplicity distribution,
      contention proxies (top 1/5/10% shares)
Per checkpoint, 1 timing camera:
  - stage timing fwd/loss/bwd/consumer/opt/zgrad/total, 20 warmup + 100 samples
    (E2 loss formula, frozen topology, fused Adam 5 groups)

Outputs: raw_measurements.csv (appended), partial_<scene>_<it>.json sidecars.
No production CUDA modifications; no C0 publication timing rebuild.
"""
import os, sys, json, time, argparse, csv, hashlib, statistics, importlib.util
from pathlib import Path
import numpy as np
import torch

DEV = 'cuda'
H2_HELPER = '/tmp/higs_h2_bwd_2r/h2_bwd_2r_exactness.py'
CORE_SO = '/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so'
SCENE_SO = '/tmp/h1_b2_authoritative/gsplat_scene_cuda/gsplat_scene_cuda.so'
WT = '/mnt/storage_pool/liaoyuanjun/higs_c0_worktree'
CACHE = '/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed'
EXP_SO = os.path.join(CACHE, 'experimental_gaussian_render_inference_scene_cuda',
                      'experimental_gaussian_render_inference_scene_cuda.so')
CAMDIR = '/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360'
CK = {
    'room':    '/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints',
    'bicycle': '/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/bicycle/checkpoints',
    'garden':  '/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/garden/checkpoints',
}
MAX_LONG_SIDE = 1920   # 1080p
N_CAMS = 16
N_WARMUP = 20
N_MEASURE = 100
TILE = 16
MACRO_W, MACRO_H = 8, 4   # 8 x 4 fine tiles -> one macro tile (C0 exact)
EPS2D = 0.3
T0 = time.time()

def log(msg):
    print('[%7.1fs] %s' % (time.time() - T0, msg), flush=True)

# ---------------- bootstrap (identical to c0t1_timing.py / e2p0_harness.py) ----
spec = importlib.util.spec_from_file_location('h2_capture', H2_HELPER)
h2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2)
sys.path.insert(0, WT)
core_spec = importlib.util.spec_from_file_location("gsplat_cuda", CORE_SO)
core = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core)
sys.modules["gsplat.csrc"] = core
scene_spec = importlib.util.spec_from_file_location("gsplat_scene_cuda", SCENE_SO)
scene_mod = importlib.util.module_from_spec(scene_spec)
scene_spec.loader.exec_module(scene_mod)
sys.modules["gsplat_scene_cuda"] = scene_mod
exp_spec = importlib.util.spec_from_file_location(
    "experimental_gaussian_render_inference_scene_cuda", EXP_SO)
exp_mod = importlib.util.module_from_spec(exp_spec)
exp_spec.loader.exec_module(exp_mod)
sys.modules["gsplat.experimental.render.kernels.csrc"] = exp_mod
sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp_mod
from gsplat.experimental import rasterize_gaussian_higs_frozen
from gsplat.experimental.render.functional.gaussian_inference import (
    create_higs_renderer, _HIGS_FROZEN_TRACKER)
from gsplat.cuda._wrapper import fully_fused_projection
log('bootstrap done')

os.environ['HIGS_PX_RUNTIME'] = '2'
os.environ['HIGS_BWD_SCALAR_ADJOINT'] = 'scalar_adjoint'
os.environ['HIGS_BWD_H8_MR'] = '1'
os.environ.pop('HIGS_DISABLE_F9', None)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def make_fused_adam(leaves):
    groups = [
        {"params": [leaves[0]], "lr": 1.6e-4},
        {"params": [leaves[1]], "lr": 1e-3},
        {"params": [leaves[2]], "lr": 5e-3},
        {"params": [leaves[3]], "lr": 5e-2},
        {"params": [leaves[4]], "lr": 2.5e-3},
    ]
    return torch.optim.Adam(groups, fused=True)

def cam_set(n_cams, n_want):
    ids = list(range(n_cams))
    idx = [ids[int(round(q * (n_cams - 1) / (n_want - 1)))] for q in range(n_want)]
    return sorted(set(idx))[:n_want]

def load_cam(cam, max_long_side):
    native_w, native_h = int(cam['width']), int(cam['height'])
    scale = min(1.0, max_long_side / max(native_w, native_h))
    width, height = int(round(native_w * scale)), int(round(native_h * scale))
    R = np.asarray(cam['rotation'], dtype=np.float32).T
    p = np.asarray(cam['position'], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ p
    K = np.array([[float(cam['fx']) * width / native_w, 0, (width - 1) / 2],
                  [0, float(cam['fy']) * width / native_w, (height - 1) / 2], [0, 0, 1]], dtype=np.float32)
    return torch.tensor(vm, device=DEV)[None, None], torch.tensor(K, device=DEV)[None, None], width, height

def ckpt_to_params(obj):
    means = obj['xyz'].to(DEV, torch.float32)
    quats = obj['rotation'].to(DEV, torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(obj['scaling'].to(DEV, torch.float32))
    opac = torch.sigmoid(obj['opacity'].to(DEV, torch.float32))
    shs = obj['shs'].to(DEV, torch.float32)
    return means, quats, scales, opac, shs

# ---------------- C0-exact support pass (DERIVED_CONVENTION) ----------------
# Uses C0's own fully_fused_projection (exact radii/means2d) + an AABB
# tile-count convention. The native AccuTile intersection kernel uses tighter
# per-Gaussian tile bounds, so the derived pair count OVERESTIMATES C0's
# n_isects by a measured ~3.84x (room 30K cam150: 12.71M vs 3.31M). D_macro
# and the multiplicity/contention SHAPES are valid as DERIVED (the convention
# applies uniformly); C0 n_isects is the MEASURED authoritative pair count.
def bucketize(multiplicities):
    out = {}
    labels = ['1', '2', '3-4', '5-8', '9-16', '17-32', '33+']
    edges = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 10**9)]
    m = np.asarray(multiplicities, dtype=np.int64)
    for lab, (lo, hi) in zip(labels, edges):
        out[lab] = int(((m > lo) & (m <= hi)).sum())
    return out

def analytic_support(means, quats, scales, vm, K, W, H):
    """Returns dict with derived pair/macro/contention fields (CONVENTION-labeled)."""
    with torch.no_grad():
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means=means.unsqueeze(0).contiguous(), covars=None,
            quats=quats.unsqueeze(0).contiguous(), scales=scales.unsqueeze(0).contiguous(),
            viewmats=vm.contiguous(), Ks=K.contiguous(), width=W, height=H,
            eps2d=EPS2D, packed=False)
    r = radii.squeeze()            # [N,2]
    u = m2d[..., 0].squeeze(0)     # [N]
    v = m2d[..., 1].squeeze(0)
    rx = r[..., 0]; ry = r[..., 1]
    vis = (r > 0).all(dim=-1)
    n_visible = int(vis.sum())
    x0 = (u - rx).floor().long().clamp(0, W - 1)
    x1 = (u + rx).ceil().long().clamp(0, W - 1)
    y0 = (v - ry).floor().long().clamp(0, H - 1)
    y1 = (v + ry).ceil().long().clamp(0, H - 1)
    tx0 = x0 // TILE; tx1 = x1 // TILE
    ty0 = y0 // TILE; ty1 = y1 // TILE
    tc = ((tx1 - tx0 + 1) * (ty1 - ty0 + 1)).clamp_min(0) * vis
    total_pairs = int(tc.sum())
    tcounts = tc.cpu().numpy()
    nmxg = (tx1 // MACRO_W - tx0 // MACRO_W + 1).clamp(min=1)
    nmyg = (ty1 // MACRO_H - ty0 // MACRO_H + 1).clamp(min=1)
    macro_span = (nmxg * nmyg) * (tc > 0)
    n_unique_macro = int((macro_span > 0).sum())
    n_macro_pairs = int(macro_span.sum())
    D_macro = round(total_pairs / n_macro_pairs, 3) if n_macro_pairs else None
    span_vis = macro_span[macro_span > 0].cpu().numpy()
    mult = bucketize(span_vis)
    top = {}
    arr = tcounts[tcounts > 0].astype(np.float64)
    if arr.size:
        tot = arr.sum()
        for q, lab in ((1, 'top1pct'), (5, 'top5pct'), (10, 'top10pct')):
            k = max(1, int(np.ceil(arr.size * q / 100)))
            top[lab] = round(float(np.sort(arr)[-k:].sum() / tot), 4)
        contrib = {'top_shares': top, 'mean': round(float(arr.mean()), 3),
                   'p90': round(float(np.percentile(arr, 90)), 2),
                   'p99': round(float(np.percentile(arr, 99)), 2),
                   'max': round(float(arr.max()), 2), 'n_visible': n_visible}
    else:
        contrib = {'n_visible': 0}
    del radii, m2d, depths, conics, r, u, v, rx, ry, vis, x0, x1, y0, y1
    del tx0, tx1, ty0, ty1, tc, tcounts, nmxg, nmyg, macro_span, span_vis
    return {'total_pairs_derived': total_pairs, 'n_visible_analytic': n_visible,
            'n_unique_macro_gid': n_unique_macro, 'n_macro_pairs': n_macro_pairs,
            'D_macro': D_macro,
            'multiplicity': {k: val for k, val in mult.items() if val},
            'contention': contrib}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--iters', default='5000,15000,30000')
    ap.add_argument('--out', required=True)
    ap.add_argument('--timing-cam', type=int, default=None)
    args = ap.parse_args()

    scene = args.scene
    iters = [int(x) for x in args.iters.split(',')]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    cams = json.loads(Path(f'{CAMDIR}/{scene}/cameras.json').read_text())
    cam_ids = cam_set(len(cams), N_CAMS)
    timing_cam = args.timing_cam if args.timing_cam is not None else cam_ids[len(cam_ids) // 2]
    if timing_cam not in cam_ids:
        # pick the sampled camera nearest to the requested timing cam
        timing_cam = min(cam_ids, key=lambda c: abs(c - timing_cam))
    log(f'{scene}: {len(cams)} cameras; sampled {cam_ids}; timing cam {timing_cam}')

    hashes = {}
    for it in iters:
        hashes[it] = sha256(f'{CK[scene]}/iter_{it}.pt')

    rows = []
    cam_meta = {'scene': scene, 'cameras': cam_ids, 'timing_cam': timing_cam,
                'hashes': {str(k): v for k, v in hashes.items()},
                'resolution_long': MAX_LONG_SIDE, 'macro': [MACRO_W, MACRO_H],
                'eps2d': EPS2D}
    stage_summaries = {}

    for it in iters:
        t0 = time.time()
        obj = torch.load(f'{CK[scene]}/iter_{it}.pt', map_location='cpu', weights_only=False)
        means, quats, scales, opac, shs = ckpt_to_params(obj)
        N = int(means.shape[0])
        log(f'{scene}/{it}: loaded N={N} in {time.time()-t0:.1f}s')
        del obj

        leaves = tuple(x.detach().clone().requires_grad_(True) for x in (means, quats, scales, opac, shs))
        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*leaves, sh_degree=h2.SH_DEGREE)
        opt = make_fused_adam(leaves)
        ck_summary = {'N': N, 'hash': hashes[it], 'cameras': {}}

        for cid in cam_ids:
            cam = cams[cid]
            vm, K, W, H = load_cam(cam, MAX_LONG_SIDE)
            rec = {'scene': scene, 'iter': it, 'cam': cid, 'W': W, 'H': H, 'N': N}
            try:
                with torch.no_grad():
                    o = rasterize_gaussian_higs_frozen(
                        *leaves, backward_mode="higs_native", scene=handle,
                        freeze_topology=True, viewmats=vm, Ks=K, width=W, height=H,
                        sh_degree=h2.SH_DEGREE, use_higs_culling=True, radius_clip=0.0,
                        tile_sampling_ratio=1.0)
                n_isects = int(o['metadata'].get('n_isects', 0))
                n_isects_full = int(o['metadata'].get('n_isects_full', n_isects))
                rec.update(n_visible_measured=n_isects,
                           n_isects_full=n_isects_full)
                del o
            except Exception as ex:
                rec['fwd_error'] = repr(ex)
                log(f'  cam{cid} fwd ERROR {ex!r}')
                rows.append(rec); ck_summary['cameras'][str(cid)] = rec
                torch.cuda.empty_cache(); continue

            # stage timing at one camera
            if cid == timing_cam:
                torch.manual_seed(4200 + it + cid)
                gt = torch.rand(1, H, W, 3, device=DEV) * 0.7
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                def one_iter():
                    ev[0].record()
                    o2 = rasterize_gaussian_higs_frozen(
                        *leaves, backward_mode="higs_native", scene=handle,
                        freeze_topology=True, viewmats=vm, Ks=K, width=W, height=H,
                        sh_degree=h2.SH_DEGREE, use_higs_culling=True, radius_clip=0.0,
                        tile_sampling_ratio=1.0)
                    frame, alpha = o2['frame'], o2['alpha']
                    ev[1].record()
                    loss = 0.8 * (frame - gt).abs().mean() + 0.2 * ((alpha - 0.5) ** 2).mean()
                    ev[2].record()
                    loss.backward()
                    ev[3].record()
                    g = leaves[0].grad.norm(dim=-1).detach()
                    ev[4].record()
                    opt.step()
                    ev[5].record()
                    opt.zero_grad(set_to_none=True)
                    ev[6].record()
                    torch.cuda.synchronize()
                    return dict(fwd_ms=ev[0].elapsed_time(ev[1]),
                                loss_ms=ev[1].elapsed_time(ev[2]),
                                bwd_ms=ev[2].elapsed_time(ev[3]),
                                consumer_ms=ev[3].elapsed_time(ev[4]),
                                opt_ms=ev[4].elapsed_time(ev[5]),
                                zgrad_ms=ev[5].elapsed_time(ev[6]),
                                total_ms=ev[0].elapsed_time(ev[6]))
                for _ in range(N_WARMUP):
                    one_iter()
                s = {k: [] for k in ('fwd_ms', 'loss_ms', 'bwd_ms', 'consumer_ms', 'opt_ms', 'zgrad_ms', 'total_ms')}
                for _ in range(N_MEASURE):
                    r = one_iter()
                    for k in s: s[k].append(r[k])
                summ = {k: {'median': round(statistics.median(v), 4),
                            'p25': round(float(np.percentile(v, 25)), 4),
                            'p75': round(float(np.percentile(v, 75)), 4),
                            'n': len(v)} for k, v in s.items()}
                stage_summaries[f'{it}_cam{cid}'] = summ
                rec.update({f'{k}_median': summ[k]['median'] for k in summ})
                log(f'  cam{cid} timing: total {summ["total_ms"]["median"]}ms')

            # C0-exact support (derived convention)
            try:
                t1 = time.time()
                sup = analytic_support(means, quats, scales, vm, K, W, H)
                rec['n_fine_pairs_derived'] = sup['total_pairs_derived']
                rec['n_visible_analytic'] = sup['n_visible_analytic']
                rec['visible_fraction_analytic'] = round(sup['n_visible_analytic'] / N, 5)
                rec['pairs_per_visible_gs'] = round(sup['total_pairs_derived'] / sup['n_visible_analytic'], 3) if sup['n_visible_analytic'] else None
                rec['pairs_per_master_gs'] = round(sup['total_pairs_derived'] / N, 3) if N else None
                rec['n_unique_macro_gid'] = sup['n_unique_macro_gid']
                rec['n_macro_pairs'] = sup['n_macro_pairs']
                rec['D_macro'] = sup['D_macro']
                rec['multiplicity'] = {k: v for k, v in sup['multiplicity'].items() if v}
                c = sup['contention']
                rec['contrib_top_shares'] = c.get('top_shares')
                rec['contrib_mean'] = c.get('mean')
                rec['contrib_p90'] = c.get('p90')
                rec['contrib_p99'] = c.get('p99')
                rec['contrib_max'] = c.get('max')
                rec['analytic_time_s'] = round(time.time() - t1, 1)
            except Exception as ex:
                rec['analytic_error'] = repr(ex)
                log(f'  cam{cid} analytic ERROR {ex!r}')

            rows.append(rec)
            ck_summary['cameras'][str(cid)] = rec
            torch.cuda.empty_cache()

        del leaves, handle, opt, means, quats, scales, opac, shs
        _HIGS_FROZEN_TRACKER.reset()
        torch.cuda.empty_cache()
        (outdir / f'partial_{scene}_{it}.json').write_text(
            json.dumps({'provenance': cam_meta, 'summary': ck_summary,
                        'stage_timing': stage_summaries}, indent=1, default=str))
        log(f'{scene}/{it}: done cam loop')

    fieldnames = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fieldnames.append(k)
    with open(outdir / 'raw_measurements.csv', 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    log(f'{scene}: WROTE {len(rows)} rows')

if __name__ == '__main__':
    main()
