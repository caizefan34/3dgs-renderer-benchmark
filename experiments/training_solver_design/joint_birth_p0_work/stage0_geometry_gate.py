"""Room-only fresh event-1500 geometry spectral gate (no frequency reconstruction)."""
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gaussian_renderer import render_structgs
from scene import GaussianModel
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                 read_intrinsics_binary, read_intrinsics_text)
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from utils.freq_utils import compute_projected_axes_subset

BASE = Path('/root/sadgs_joint_birth_p0_20260928')
DATA = Path('/root/epic05_p16rb/common_data/mipnerf360')
OUT = Path('/root/sadgs_joint_birth_stage0_20260928/room')
PIPE = Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
BG = torch.zeros(3, dtype=torch.float32, device='cuda')
MULT = 0.25
PARAMS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cameras_train_only():
    root = DATA / 'room'
    sparse = root / 'sparse/0'
    try:
        ext = read_extrinsics_binary(str(sparse / 'images.bin'))
        intr = read_intrinsics_binary(str(sparse / 'cameras.bin'))
    except Exception:
        sparse = root / 'sparse'
        ext = read_extrinsics_text(str(sparse / 'images.txt'))
        intr = read_intrinsics_text(str(sparse / 'cameras.txt'))
    all_ext = sorted(ext.items(), key=lambda kv: os.path.basename(kv[1].name))
    train_ext = {key: value for idx, (key, value) in enumerate(all_ext) if idx % 8 != 0}
    infos = readColmapCameras(train_ext, intr, str(root / 'images'))
    infos.sort(key=lambda c: c.image_name)
    return cameraList_from_camInfos(infos, 1.0, Namespace(resolution=1, data_device='cuda'))


def load_model(entry):
    model = GaussianModel(3, 'hybrid')
    for name in PARAMS:
        setattr(model, name, nn.Parameter(entry['tensors'][name].to('cuda'), requires_grad=False))
    model.filter_3D = entry['tensors']['filter_3D'].to('cuda')
    model.active_sh_degree = int(entry['active_sh_degree'])
    model.spatial_lr_scale = float(entry['spatial_lr_scale'])
    return model


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'count': 0}
    return {'count': int(x.size), 'mean': float(x.mean()),
            'p50': float(np.quantile(x, .5)), 'p90': float(np.quantile(x, .9)),
            'p95': float(np.quantile(x, .95)), 'max': float(x.max())}


def main():
    torch.cuda.set_device(0)
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    capture = BASE / 'room/capture'
    entry_path = capture / 'split_entry.pt'
    meta_path = capture / 'capture_meta.json'
    meta = json.loads(meta_path.read_text())
    blob = torch.load(entry_path, map_location='cpu', mmap=True, weights_only=False)
    entry = blob['model']
    parents = blob['parents'].to(torch.int64)
    counts = blob['child_counts'].to(torch.int64)
    eta = blob['eta']
    if eta is None:
        raise RuntimeError('capture lacks eta')
    native_k = torch.ceil(torch.sqrt(torch.clamp(eta[parents].float(), min=1))).to(torch.int64).clamp(min=1)
    bad = int((native_k.prod(1) != counts).sum())
    if bad or int(counts.sum()) != int(meta['split_child_count']):
        raise RuntimeError(f'native map parity failure: mismatches={bad}')
    three = (native_k > 1).all(1)
    family_ids = parents[three]
    historical_child_counts = counts[three].numpy()
    raw_path = OUT/'stage0_geometry_raw.npz'
    analyze_raw = '--analyze-raw' in sys.argv[1:]
    trans_thresh, opacity_thresh = 0.8, 0.5
    n = len(family_ids)
    if analyze_raw:
        with np.load(raw_path, allow_pickle=False) as raw:
            G = raw['G'].copy()
            active = raw['active'].copy()
            a_names = raw['A_camera_names'].astype(str).tolist()
            b_names = raw['B_camera_names'].astype(str).tolist()
            indices = raw['selected_indices'].tolist()
            train_camera_count = int(raw['train_camera_count'])
        print(f'ANALYZE_RAW {raw_path}', flush=True)
    else:
        model = load_model(entry)
        cams = cameras_train_only()
        train_camera_count = len(cams)
        indices = [round(k * (train_camera_count - 1) / 11) for k in range(12)]
        selected = [cams[i] for i in indices]
        names = [c.image_name for c in selected]
        a_names, b_names = names[::2], names[1::2]
        if a_names != meta['A_camera_names']:
            raise RuntimeError('A fit camera identities disagree with capture metadata')
        G = np.zeros((12, n, 3, 3), dtype=np.float32)
        active = np.zeros((12, n), dtype=bool)
        for vi, camera in enumerate(selected):
            print(f'VIEW {vi+1}/12 {camera.image_name} eligible={n}', flush=True)
            with torch.no_grad():
                pkg = render_structgs(camera, model, PIPE, BG, MULT)
            ids = family_ids.to(device='cuda', dtype=torch.long)
            cov = pkg['cov2D'][ids].detach()
            visible = pkg['radii'][ids] > 0
            opacity = model.get_opacity[ids, 0]
            keep = visible & (cov[:, 6] > trans_thresh) & (opacity > opacity_thresh)
            active[vi] = keep.cpu().numpy()
            idx = torch.nonzero(keep, as_tuple=False).flatten()
            if idx.numel():
                axes = compute_projected_axes_subset(cov[idx, 3:5], cov[idx, 5],
                                                     model.get_scaling[ids[idx]],
                                                     model.get_rotation[ids[idx]], camera)
                a = axes.transpose(1, 2)
                g = a.transpose(1, 2) @ a
                G[vi, idx.cpu().numpy()] = g.cpu().numpy()
            del pkg, cov, visible, opacity, keep
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f'ACTIVE {vi+1}/12 {int(active[vi].sum())}', flush=True)
        np.savez_compressed(raw_path, family_parent_entry_rows=family_ids.numpy(),
                            historical_native_child_counts=historical_child_counts, active=active, G=G,
                            A_camera_names=np.asarray(a_names), B_camera_names=np.asarray(b_names),
                            selected_indices=np.asarray(indices), train_camera_count=np.asarray(train_camera_count))
        print(f'RAW_SAVED {raw_path}', flush=True)

    a_indices = np.arange(0, 12, 2)
    b_indices = np.arange(1, 12, 2)
    a_camera_count = active[a_indices].sum(axis=0)
    b_camera_count = active[b_indices].sum(axis=0)
    fit = active[a_indices].any(axis=0)
    stable_fit = a_camera_count >= 3
    covered = stable_fit & (b_camera_count >= 2)
    gsum = np.where(active[a_indices, :, None, None], G[a_indices], 0).sum(axis=0, dtype=np.float64)
    eig, q = np.linalg.eigh(gsum[fit])
    eig, q = eig[:, ::-1], q[:, :, ::-1]
    trace = np.maximum(eig.sum(axis=1), 1e-30)
    ratios = eig / trace[:, None]
    off = np.stack([gsum[fit,0,1] / np.sqrt(np.maximum(gsum[fit,0,0]*gsum[fit,1,1],1e-30)),
                    gsum[fit,0,2] / np.sqrt(np.maximum(gsum[fit,0,0]*gsum[fit,2,2],1e-30)),
                    gsum[fit,1,2] / np.sqrt(np.maximum(gsum[fit,1,1]*gsum[fit,2,2],1e-30))], axis=1)
    per_view_dropped = {}
    for rank in (1, 2):
        per_view_dropped[str(rank)] = {}
        for vi in range(12):
            e = np.flatnonzero(fit)
            mat = G[vi, e].astype(np.float64)
            projected = np.einsum('nik,nij,njk->nk', q, mat, q, optimize=True)
            den = np.trace(mat, axis1=1, axis2=2)
            valid = active[vi, e] & (den > 0)
            frac = projected[:, rank:][valid].sum(axis=1) / np.maximum(den[valid], 1e-30)
            per_view_dropped[str(rank)][str(vi)] = {
                'valid_parents': int(valid.sum()), 'discarded_trace_fraction': stats(frac),
                'parents_at_or_below_1pct': int((frac <= .01).sum())}
    # Diagnostic count proxy: geometric response eta=sqrt(q^T G q), without the
    # captured structure-tensor frequency multiplier. It is not a replayed birth count.
    def count_proxy(frame, views, matrix_stack, active_stack):
        proj = np.einsum('nik,vnij,njk->vnk', frame, matrix_stack, frame, optimize=True)
        proj = np.maximum(proj, 0)
        proj[~active_stack] = 0
        env = np.sqrt(proj[views]).max(axis=0)
        k = np.ceil(np.sqrt(np.maximum(env, 1))).astype(np.int64)
        return np.prod(k, axis=1)
    G_fit, active_fit = G[:, fit], active[:, fit]
    ident_fit = np.broadcast_to(np.eye(3), (int(fit.sum()), 3, 3))
    c0, cq = count_proxy(ident_fit, a_indices, G_fit, active_fit), count_proxy(q, a_indices, G_fit, active_fit)
    b0, bq = count_proxy(ident_fit, b_indices, G_fit, active_fit), count_proxy(q, b_indices, G_fit, active_fit)
    fit_parent_ids = np.flatnonzero(fit)
    covered_in_fit = covered[fit_parent_ids]
    stable_positions = np.flatnonzero(stable_fit[fit_parent_ids])
    eig_stable, q_stable = np.linalg.eigh(gsum[fit_parent_ids[stable_positions]])
    eig_stable, q_stable = eig_stable[:, ::-1], q_stable[:, :, ::-1]
    G_stable, active_stable = G[:, fit_parent_ids[stable_positions]], active[:, fit_parent_ids[stable_positions]]
    ident_stable = np.broadcast_to(np.eye(3), (len(stable_positions), 3, 3))
    stable_c0 = count_proxy(ident_stable, a_indices, G_stable, active_stable)
    stable_cq = count_proxy(q_stable, a_indices, G_stable, active_stable)
    coverage_in_stable = covered[fit_parent_ids[stable_positions]]
    covered_local = fit_parent_ids[stable_positions][coverage_in_stable]
    covered_trace = np.maximum(np.trace(gsum[covered_local], axis1=1, axis2=2), 1e-30)
    covered_eig = np.linalg.eigvalsh(gsum[covered_local])[:, ::-1]
    proxy = {
        'A_unrotated_children_on_A_active_families': int(c0.sum()),
        'A_G_eigenframe_children_on_A_active_families': int(cq.sum()),
        'A_proxy_reduction_fraction': float(1-cq.sum()/max(1,c0.sum())),
        'B_unrotated_children_on_A_fit_and_B_active_families': int(b0[active[b_indices].any(0)[fit]].sum()),
        'B_G_eigenframe_children_on_A_fit_and_B_active_families': int(bq[active[b_indices].any(0)[fit]].sum()),
        'A_at_least_3_and_B_at_least_2_parent_count': int(covered.sum()),
        'covered_historical_native_children': int(historical_child_counts[covered].sum()),
        'covered_historical_child_share': float(historical_child_counts[covered].sum()/max(1,historical_child_counts.sum())),
        'covered_A_geometric_proxy_unrotated_children': int(stable_c0[coverage_in_stable].sum()),
        'covered_A_geometric_proxy_eigenframe_children': int(stable_cq[coverage_in_stable].sum()),
        'covered_A_geometric_proxy_reduction_fraction': float(1-stable_cq[coverage_in_stable].sum()/max(1,stable_c0[coverage_in_stable].sum())),
    }
    report = {
        'status': 'ROOM_GEOMETRY_ONLY_SPECTRAL_GATE', 'scene': 'room',
        'source_commit': '029814fb5ed773e4a794cec5115951a64f610ab8',
        'capture_iteration': entry.get('context', {}).get('iteration'),
        'family_selection': {'three_axis_native_families': int(three.sum()),
                             'historical_native_children': int(historical_child_counts.sum()),
                             'native_eta_child_count_mismatches': bad},
        'camera_protocol': {'train_camera_count': train_camera_count, 'rule': 'COLMAP basename sort; LLFF holdout index%8==0 filtered before image loading; then sort camera image_name',
             'selected_indices': indices, 'A_fit_camera_indices': a_indices.tolist(), 'B_internal_check_camera_indices': b_indices.tolist(),
             'A_fit_camera_names': a_names, 'B_internal_check_camera_names': b_names,
             'A_and_B_are_training_views_seen_during_prefix': True, 'heldout_images_loaded': False,
             'aggregation': 'equal-weight G_v sum over A indices 0,2,4,6,8,10; Q fit on A only; B indices 1,3,5,7,9,11 excluded from Q'},
        'active_semantics': {'renderer_visible': 'radii>0', 'transmittance': f'cov2D[:,6]>{trans_thresh}',
             'opacity': f'model.get_opacity>{opacity_thresh}', 'gradient_gate': 'omitted; fresh checkpoint has no historical per-view gradients',
             'geometry_only': 'no structure tensor, wavelength/frequency multiplier, or historical observation replay'},
        'geometry_spectrum_A_fit': {'active_family_count': int(fit.sum()),
             'normalized_eigenvalue_ratios': [stats(ratios[:,i]) for i in range(3)],
             'abs_offdiag_correlation': stats(np.abs(off).reshape(-1)),
             'per_view_discarded_trace_fraction_in_A_G_eigenframe': per_view_dropped},
        'visibility_support': {'A_active_camera_count_distribution': {str(k):int((a_camera_count==k).sum()) for k in range(7)},
             'B_active_camera_count_distribution': {str(k):int((b_camera_count==k).sum()) for k in range(7)},
             'A_at_least_3_and_B_at_least_2_count':int(covered.sum()),
             'A_at_least_3_and_B_at_least_2_fraction_of_eligible_families':float(covered.mean()),
             'covered_per_parent_A_sum_G_eigenvalue_ratio':[stats(x) for x in (covered_eig/np.maximum(covered_trace[:,None],1e-30)).T],
             'covered_historical_native_children':int(historical_child_counts[covered].sum()),
             'covered_historical_child_share':float(historical_child_counts[covered].sum()/max(1,historical_child_counts.sum())),
             'minimum_coverage_rule':'at least 3 of 6 A fit cameras and at least 2 of 6 B internal-check cameras; fixed before inspecting output'},
        'geometry_only_child_count_proxy': proxy,
        'interpretation_boundary': 'Fresh event-1500 geometry-only screening statistic on 12 fixed training cameras. The count proxy omits F_v, so it is not eta, native replay, accepted-work, or a full-training-view conservative certificate. Center/visible sampled support only.',
        'accepted_work': {'status': 'NOT_RUN', 'reason': 'frequency-aware spectrum and accepted-work gate not established'},
        'elapsed_seconds': time.time()-start,
    }
    report['provenance'] = {'split_entry_sha256': sha256(entry_path), 'capture_meta_sha256': sha256(meta_path),
        'freq_utils_sha256': sha256(BASE/'utils/freq_utils.py'), 'analysis_script_sha256': sha256(__file__),
        'raw_npz_sha256': sha256(OUT/'stage0_geometry_raw.npz'), 'gpu': torch.cuda.get_device_name(0)}
    (OUT/'stage0_geometry_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'scene':'room', 'proxy_reduction':proxy['A_proxy_reduction_fraction'],
                      'active_fit_families':int(fit.sum()), 'report':str(OUT/'stage0_geometry_report.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
