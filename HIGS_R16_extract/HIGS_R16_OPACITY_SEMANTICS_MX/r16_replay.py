#!/usr/bin/env python3
"""R16: Controlled opacity-semantics replay using an actual R15 projection capture.

This script does not change renderer sources, binary, checkpoint, conics or camera.
It alters ONLY the opacity argument (and, in one explicit diagnostic branch,
sets radii=0 for primitives whose activated alpha is <= kernel threshold).
That last branch is a diagnostic skip control, not a proposed renderer patch.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from oracle_higs import parse_value, run_oracle


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(x):
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    return dict(min=float(np.min(v)), median=float(np.median(v)), p90=float(np.quantile(v, .9)),
                p99=float(np.quantile(v, .99)), max=float(np.max(v)), mean=float(np.mean(v)))


def safe_cpu_validation(data, alpha):
    import torch
    for name in ('means2d', 'radii', 'depths', 'conics', 'opacities', 'gpu_counts', 'sample_indices', 'meta'):
        if name not in data or data[name] is None:
            raise ValueError('Missing captured input: ' + name)
    o = data['opacities']
    if not o.is_floating_point():
        raise ValueError('Expected floating-point opacities')
    a = torch.sigmoid(o)
    rad = data['radii']
    g = data['gpu_counts']
    return {
        'raw_opacity': stats(o.numpy()), 'activated_opacity': stats(a.numpy()),
        'raw_negative_count': int((o < 0).sum().item()),
        'raw_nonpositive_count': int((o <= 0).sum().item()),
        'raw_at_or_below_threshold_count': int((o <= alpha).sum().item()),
        'activated_at_or_below_threshold_count': int((a <= alpha).sum().item()),
        'raw_nan_or_inf_count': int((~torch.isfinite(o)).sum().item()),
        'nonpositive_radius_count': int(((rad[..., 0] <= 0) | (rad[..., 1] <= 0)).sum().item()),
        'captured_gpu_count_stats': stats(g.numpy()),
        'captured_gpu_total': int(g.to(torch.int64).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True,
                    help='Original MX directory containing gsplat, frozen csrc.so, and gsplat/cuda/include/Common.h')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--extend', type=float, default=None)
    ap.add_argument('--skip-gpu', action='store_true', help='Run CPU audit only (no CUDA replays)')
    args = ap.parse_args()
    source = args.source_root.resolve()
    header = source / 'gsplat/cuda/include/Common.h'
    alpha, extend = args.alpha, args.extend
    if header.exists():
        text = header.read_text(errors='replace')
        if alpha is None: alpha = parse_value(text, 'ALPHA_THRESHOLD')
        if extend is None: extend = parse_value(text, 'GAUSSIAN_EXTEND')
    if alpha is None or extend is None:
        ap.error('Cannot find actual CUDA ALPHA_THRESHOLD / GAUSSIAN_EXTEND. Inspect source header and pass --alpha/--extend.')
    if not 0 < alpha < 1 or not extend > 0:
        ap.error('Invalid alpha or extend')

    import torch
    args.out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.capture, map_location='cpu', weights_only=False)
    md = data['meta']
    summary = {
        'capture_path': str(args.capture.resolve()),
        'capture_sha256': sha256(args.capture),
        'source_root': str(source),
        'alpha_threshold': alpha, 'gaussian_extend': extend,
        'original_metadata': md,
        'cpu_audit': safe_cpu_validation(data, alpha),
        'modes': {}, 'status': 'CPU_AUDIT_ONLY',
        'method_note': 'Only 1k captured projected primitives, not full R3/R8 workload; not a performance benchmark.',
    }
    op = data['opacities']
    activated = torch.sigmoid(op)
    cases = {
        'raw': (op, data['radii']),
        'sigmoid': (activated, data['radii']),
        'sigmoid_threshold_skip': (
            activated,
            torch.where((activated > alpha)[..., None], data['radii'], torch.zeros_like(data['radii'])),
        ),
    }
    if not args.skip_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('No NVIDIA GPU detected; rerun on MX or use --skip-gpu')
        sys.path.insert(0, str(source))
        import gsplat.cuda._wrapper as w
        from gsplat.cuda._wrapper import isect_tiles
        device = torch.device('cuda:0')
        rows = []
        for mode, (opacities, radii) in cases.items():
            try:
                with torch.no_grad():
                    counts, isect_ids, flatten_ids = isect_tiles(
                        data['means2d'].to(device), radii.to(device), data['depths'].to(device),
                        int(md['tile_size']), int(md['width_in_tiles']), int(md['height_in_tiles']),
                        sort=False, segmented=False, packed=False, n_images=int(md['n_images']),
                        conics=data['conics'].to(device), opacities=opacities.to(device),
                        tile_mask=(data['tile_mask'].to(device) if data.get('tile_mask') is not None else None),
                    )
                    torch.cuda.synchronize()
                cpu_counts = counts.detach().cpu().contiguous()
                case_data = dict(data)
                case_data['gpu_counts'] = cpu_counts
                case_data['opacities'] = opacities.cpu().contiguous()
                case_data['radii'] = radii.cpu().contiguous()
                case_data['meta'] = dict(md, r16_opacity_mode=mode, r16_diagnostic=True)
                saved = args.out / f'{mode}_replay_capture.pt'
                torch.save(case_data, saved)
                oracle = run_oracle(saved, args.out / f'oracle_{mode}', alpha, extend)
                baseline_gpu = data['gpu_counts'].to(torch.int64)
                case_gpu = cpu_counts.to(torch.int64)
                counts_equal = bool(torch.equal(cpu_counts, data['gpu_counts']))
                mode_report = {
                    'status': 'EXECUTED',
                    'gpu_count_sum': int(case_gpu.sum().item()),
                    'gpu_emitted_size': int(isect_ids.numel()),
                    'gpu_flatten_size': int(flatten_ids.numel()),
                    'count_emit_equal': bool(case_gpu.sum().item() == isect_ids.numel() == flatten_ids.numel()),
                    'gpu_count_stats': stats(cpu_counts.numpy()),
                    'captured_gpu_exact_match': counts_equal if mode == 'raw' else None,
                    'compared_to_original': {'sum_abs_diff': int((case_gpu - baseline_gpu).abs().sum().item()),
                                             'changed_pairs': int((case_gpu != baseline_gpu).sum().item())},
                    'oracle_result': str((args.out / f'oracle_{mode}' / 'oracle_summary.json')),
                    'oracle_sum': oracle['reference_sum_valid'],
                    'gpu_oracle_mismatch_pairs': oracle['mismatch_pairs'],
                }
                summary['modes'][mode] = mode_report
                originals = data['sample_indices'].numpy().tolist()
                for cam in range(cpu_counts.reshape(-1, len(originals)).shape[0]):
                    a = cpu_counts.reshape(-1, len(originals))[cam]
                    for j, orig in enumerate(originals):
                        rows.append({'mode': mode, 'camera': cam, 'gaussian_id': orig,
                                     'count': int(a[j]), 'raw_opacity': float(op.reshape(-1,len(originals))[cam,j]),
                                     'activated_opacity': float(activated.reshape(-1,len(originals))[cam,j]),
                                     'radii_skipped': bool((radii.reshape(-1,len(originals),2)[cam,j] == 0).all())})
            except Exception as e:
                summary['modes'][mode] = {'status': 'ERROR', 'error': repr(e)}
                print(f'{mode} ERROR: {e!r}', flush=True)
        with (args.out / 'R16_REPLAY_COUNTS.csv').open('w', newline='') as f:
            ww = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['mode','camera','gaussian_id','count'])
            ww.writeheader(); ww.writerows(rows)
        summary['status'] = ('REPLAY_COMPLETED' if all(v.get('status') == 'EXECUTED' for v in summary['modes'].values())
                             else 'REPLAY_PARTIAL')
        if summary['modes'].get('raw', {}).get('captured_gpu_exact_match') is False:
            summary['status'] = 'PROTOCOL_MISMATCH'
    (args.out / 'R16_OPACITY_SEMANTICS_REPORT.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k:v for k,v in summary.items() if k!='original_metadata'}, indent=2, ensure_ascii=False, default=str))
    if summary['status'] in ('REPLAY_PARTIAL', 'PROTOCOL_MISMATCH'):
        raise SystemExit(2)

if __name__ == '__main__':
    main()
