"""WV1 CPU-only audit; historical aggregates remain labelled as aggregates."""
from pathlib import Path
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
import numpy as np
import torch

torch.set_num_threads(2)
OUT = Path('/mnt/storage_pool/liaoyuanjun/wv1_feasibility_20261008_2138')
RAW = OUT / 'raw'
WV0 = RAW / 'wv0_snapshot' / 'WV_HIER_ORACLE0_MX'
states = json.loads((RAW / 'wv0_state_summaries_exact.json').read_text())
export = json.loads((WV0 / 'WV0_GRADIENT_RAW.json').read_text())
GEOM = ['means', 'quats', 'scales']
GROUPS = GEOM + ['opacities', 'colors']

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def csvout(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

spec = importlib.util.spec_from_file_location('wv0_grad_audited', RAW / 'wv0_live_scripts' / '_wv0_grad.py')
grad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grad)
metric_summaries = 0
raw_metric_arrays = 0
export_matches = True
counts = []
work_rows = []
quality_rows = []
mask_rows = []
bias_rows = []
log_seconds = []
mask_dir = RAW / 'reconstructed_masks'
mask_dir.mkdir(exist_ok=True)
mask_cache = {}
for sid, s in states.items():
    H, W = s['cam_hw']
    ncond = len(s['conditions'])
    counts.append({'state':sid,'cameras_reported':s['n_cams'],'conditions':ncond,
                   'seed_count_in_stats':32,'implied_camera_estimates':s['n_cams'] * ncond * 32,
                   'saved_camera_seed_records':0})
    logpath = RAW / 'wv0_snapshot' / 'logs' / (sid + '.log')
    matches = re.findall(r'WV0_RUN rc (\d+) secs ([0-9.]+)', logpath.read_text())
    log_seconds.append({'state':sid,'reported_rc':int(matches[-1][0]),
                        'reported_process_wall_seconds':float(matches[-1][1]),
                        'evidence':'WV0 launcher log; not measured GPU kernel active time'})
    for ck, c in s['conditions'].items():
        G = int(ck.split('_')[0][1:]); p = float(ck.split('_p')[1])
        for group in GROUPS:
            v = c['groups'][group]
            for metric in ['nmse', 'cos', 'rel_l2']:
                metric_summaries += 1
                raw_metric_arrays += isinstance(v[metric], list)
                export_matches &= v[metric] == export['states'][sid]['conditions'][ck]['groups'][group][metric+'_per_mask']
        mask_key = (H, W, G, p)
        if mask_key not in mask_cache:
            cache = []
            for seed in range(32):
                pixel, nt = grad.tile_pixel_mask_fast(H,W,G,seed,p,device='cpu')
                th,tw=math.ceil(H/G),math.ceil(W/G)
                gen=torch.Generator(device='cpu');gen.manual_seed(seed)
                tile=(torch.rand(th*tw,generator=gen)<p).numpy()
                packed=np.packbits(tile,bitorder='little')
                file=mask_dir/f'H{H}_W{W}_G{G}_p{p}_seed{seed}.bin'
                file.write_bytes(packed.tobytes())
                children=pixel[::4,::4].numpy()
                parents=pixel.reshape(H,W)
                active_parent=0
                for y in range(0,H,16):
                    for x in range(0,W,16):
                        active_parent += bool(parents[y:y+16,x:x+16].any())
                cache.append({'seed':seed,'selected_tiles':nt,'selected_pixels':int(pixel.sum()),
                              'active_parents':active_parent,'parent_count':math.ceil(H/16)*math.ceil(W/16),
                              'mask_file':str(file.relative_to(OUT)),
                              'tile_mask_sha256':hashlib.sha256(packed.tobytes()).hexdigest(),
                              'pixel_mask_sha256':hashlib.sha256(pixel.numpy().tobytes()).hexdigest()})
            mask_cache[mask_key]=cache
        cached=mask_cache[mask_key]
        for row in cached:
            mask_rows.append({'state':sid,'height':H,'width':W,'G':G,'p':p,
                              'provenance':'RECONSTRUCTED_CPU_FROM_SAVED_SOURCE_SEEDS_NOT_HISTORICAL_MASK_FILE',
                              'same_shape_camera_count':s['n_cams'],**row})
        mean_pixels=float(np.mean([r['selected_pixels'] for r in cached]))
        dense=s['work']['per_G'][str(G)]['dense_isects']
        d16=s['work']['per_G']['16']['dense_isects']
        work_rows.append({'state':sid,'camera_index':0,'G':G,'p':p,'height':H,'width':W,
                          'count_evidence':'DERIVED_FROM_WV0_STRUCTURAL_PROXY_NOT_PHYSICAL_CUDA',
                          'dense_structural_proxy':dense,'expected_selected_proxy':p*dense,
                          'expected_sort_entries_if_proxy_valid':p*dense,
                          'entry_bytes_source_verified':12,
                          'expected_key_value_payload_bytes_if_proxy_valid':12*p*dense,
                          'relative_to_G16_sparse_proxy':dense/d16,
                          'normalized_to_own_dense_proxy':p,
                          'normalized_to_G16_dense_proxy':p*dense/d16,
                          'actual_physical_isect_count':'NOT_MEASURED',
                          'actual_sort_bytes':'NOT_MEASURED',
                          'selected_pixels_mean_reconstructed':mean_pixels,
                          'selected_pixels_mean_exported':c['sel_px_mean'],
                          'pixel_mean_matches_export':mean_pixels==c['sel_px_mean'],
                          'expected_selected_pixels':p*H*W,
                          'active_parent_fraction_mean_reconstructed':float(np.mean([r['active_parents']/r['parent_count'] for r in cached]))})
        macro_bias=float(np.mean([c['groups'][g]['bias_rel_l2'] for g in GEOM]))
        mislabeled=float(np.mean([c['groups'][g]['rel_l2']['mean'] for g in GEOM]))
        bias_rows.append({'state':sid,'G':G,'p':p,'reported_bias_column_actual_single_mask_error':mislabeled,
                          'saved_macro_relative_error_of_32_seed_camera_mean':macro_bias,
                          'confidence_interval':'UNAVAILABLE_WITHOUT_RAW_RECORDS',
                          'variance_semantics':'POOLED_CAMERA_AND_MASK_NOT_CONDITIONAL_MASK_VARIANCE'})
    for p in [.25,.4]:
        c16=s['conditions'][f'G16_p{p}']; c4=s['conditions'][f'G4_p{p}']
        n16=float(np.mean([c16['groups'][g]['nmse']['mean'] for g in GEOM]))
        n4=float(np.mean([c4['groups'][g]['nmse']['mean'] for g in GEOM]))
        cos16=float(np.mean([c16['groups'][g]['cos']['mean'] for g in GEOM]))
        cos4=float(np.mean([c4['groups'][g]['cos']['mean'] for g in GEOM]))
        quality_rows.append({'state':sid,'p':p,'nmse_G16':n16,'nmse_G4':n4,'ratio_G4_G16':n4/n16,
                             'cosine_G16':cos16,'cosine_G4':cos4,'cosine_delta':cos4-cos16,
                             'evidence':'RECOMPUTED_FROM_SAVED_AGGREGATES_NOT_RAW_SEED_REVALIDATION'})
csvout('WV1_WORKCOUNTS.csv',work_rows)
csvout('raw/WV1_RECONSTRUCTED_MASKS.csv',mask_rows)
csvout('raw/WV1_BIAS_CORRECTION.csv',bias_rows)
csvout('raw/WV1_GRADIENT_AGGREGATE_RECHECK.csv',quality_rows)
dump('raw/WV1_COUNT_AUDIT.json',{'state_count':len(states),'claimed_estimates':9216,
    'implied_camera_level_estimates':sum(x['implied_camera_estimates'] for x in counts),
    'implied_camera_averaged_seed_units':len(states)*8*32,
    'saved_state_condition_records':len(work_rows),'saved_group_metric_stat_objects':metric_summaries,
    'saved_camera_seed_metric_records':0,'saved_per_seed_metric_arrays':raw_metric_arrays,
    'raw_export_is_renamed_summary':export_matches,'states':counts})
dump('raw/WV1_ORIGINAL_COST_LEDGER.json',{'records':log_seconds,
    'sum_process_wall_seconds':sum(r['reported_process_wall_seconds'] for r in log_seconds),
    'sum_process_wall_hours':sum(r['reported_process_wall_seconds'] for r in log_seconds)/3600,
    'gpu_active_seconds':None,'billing_cost':None,
    'note':'Historical single-GPU process wall allocation sum; includes CPU and I/O. No wall-to-GPU-active-time conversion.'})
dump('raw/WV1_MASK_AUDIT_SUMMARY.json',{'all_reconstructed_selected_pixel_means_match':all(r['pixel_mean_matches_export'] for r in work_rows),
    'mask_rows':len(mask_rows),'unique_masks':sum(len(x) for x in mask_cache.values()),
    'independence_note':'Same CPU seed reused across cameras; masks are identical when dimensions agree. Cameras are not independent mask replicates.',
    'full_parent_activation_theory':{'0.25':1-.75**16,'0.4':1-.6**16},
    'HT_loss':'sum(mask/p * RGB_squared_error)/(3*H*W) for a fixed denominator; WV0 omitted /(3*H*W) in both dense and sparse paths.',
    'normalization_effect':'Per-camera NMSE, cosine and relative L2 unchanged by the same fixed nonzero scalar. Absolute variance changes by scalar squared.',
    'edge_case':{'H':540,'widths':sorted(set(s['cam_hw'][1] for s in states.values())),
                'old_proxy_example':{'mean_xy':[8,8],'radius_xy':[8,8],'H':64,'W':64,'G':16,'old_count':4,'cuda_exclusive_max_count':1}}})
print(json.dumps({'states':len(states),'implied_camera_estimates':sum(x['implied_camera_estimates'] for x in counts),
                  'raw_seed_arrays':raw_metric_arrays,'renamed_summary':export_matches,
                  'mask_means_match':all(r['pixel_mean_matches_export'] for r in work_rows),
                  'G4_proxy_cost_ratio_range':[min(r['relative_to_G16_sparse_proxy'] for r in work_rows if r['G']==4),max(r['relative_to_G16_sparse_proxy'] for r in work_rows if r['G']==4)],
                  'G4_NMSE_ratio_range':[min(r['ratio_G4_G16'] for r in quality_rows),max(r['ratio_G4_G16'] for r in quality_rows)],
                  'bias_corrected_range':[min(r['saved_macro_relative_error_of_32_seed_camera_mean'] for r in bias_rows),max(r['saved_macro_relative_error_of_32_seed_camera_mean'] for r in bias_rows)]},indent=2))
