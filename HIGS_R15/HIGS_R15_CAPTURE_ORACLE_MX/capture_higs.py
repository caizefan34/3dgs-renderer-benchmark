#!/usr/bin/env python3
"""Intercept existing gsplat isect_tiles at the *actual* HiGS native capture call.
No CUDA sources or frozen shared libraries are modified. A deliberate SystemExit(0)
after saving the sample prevents a huge full-scene intersection allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def choose_ids(n, k, seed):
    import numpy as np
    k = min(n, k)
    if k == n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def capture_intersection_args(orig, args, kwargs, out, samples, seed):
    import torch
    means2d, radii, depths, tile_size, tile_width, tile_height = args[:6]
    conics = kwargs.get('conics')
    opacities = kwargs.get('opacities')
    mask = kwargs.get('tile_mask')
    if kwargs.get('packed', False):
        raise RuntimeError('This diagnostic currently requires HiGS packed=False')
    if means2d.device.type != 'cuda':
        raise RuntimeError(f'Unexpected projected tensor device {means2d.device}')
    if means2d.ndim < 3 or means2d.shape[-1] != 2:
        raise RuntimeError(f'Unexpected projection shape {means2d.shape}')
    full_n = int(means2d.shape[-2])
    ids = choose_ids(full_n, samples, seed)
    index = torch.as_tensor(ids, device=means2d.device, dtype=torch.long)
    mean_small = means2d.index_select(-2, index).contiguous()
    radii_small = radii.index_select(-2, index).contiguous()
    depths_small = depths.index_select(-1, index).contiguous()
    conics_small = conics.index_select(-2, index).contiguous() if conics is not None else None
    opacities_small = opacities.index_select(-1, index).contiguous() if opacities is not None else None
    gpu_kwargs = dict(kwargs)
    gpu_kwargs['sort'] = False
    gpu_kwargs['segmented'] = False
    gpu_kwargs['conics'] = conics_small
    gpu_kwargs['opacities'] = opacities_small
    # Run unmodified compiled CUDA op on a subset of already-projected data.
    gpu_counts, gpu_ids, gpu_flatten = orig(
        mean_small, radii_small, depths_small,
        int(tile_size), int(tile_width), int(tile_height), **gpu_kwargs
    )
    torch.cuda.synchronize(means2d.device)
    count_cpu = gpu_counts.detach().cpu()
    total_gpu_count = int(count_cpu.to(dtype=torch.int64).sum().item())
    emitted = int(gpu_ids.numel())
    count_emit_equal = emitted == total_gpu_count and int(gpu_flatten.numel()) == emitted
    if not count_emit_equal:
        print(f'WARNING: GPU count/emit mismatch: sum={total_gpu_count} ids={emitted} flatten={gpu_flatten.numel()}', flush=True)
    def to_cpu(x):
        return x.detach().cpu().contiguous() if x is not None else None
    pkg = {
        'means2d': to_cpu(mean_small),
        'radii': to_cpu(radii_small),
        'depths': to_cpu(depths_small),
        'conics': to_cpu(conics_small),
        'opacities': to_cpu(opacities_small),
        'tile_mask': to_cpu(mask),
        'gpu_counts': count_cpu,
        'sample_indices': torch.as_tensor(ids, dtype=torch.int64),
        'meta': {
            'source': 'real HiGS _native_forward_capture => gsplat.cuda._wrapper.isect_tiles',
            'full_projected_gaussian_count_per_image': full_n,
            'sample_gaussian_count_per_image': int(index.numel()),
            'projected_tensor_shape': list(means2d.shape),
            'sampled_tensor_shape': list(mean_small.shape),
            'sample_seed': seed,
            'width_in_tiles': int(tile_width),
            'height_in_tiles': int(tile_height),
            'tile_size': int(tile_size),
            'n_images': int(mean_small.numel() // (index.numel() * 2)),
            'device_type': means2d.device.type,
            'cuda_device': str(means2d.device),
            'gpu_count_sum': total_gpu_count,
            'gpu_emitted_intersection_count': emitted,
            'gpu_count_and_emit_agree': count_emit_equal,
            'has_conics': conics_small is not None,
            'has_opacities': opacities_small is not None,
            'has_tile_mask': mask is not None,
            'torch_version': torch.__version__,
            'dispatch_key_set': str(torch._C._dispatch_keys(means2d)),
            'candidate_binary_expected_sha256': 'b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3',
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pkg, out)
    summary = dict(pkg['meta'])
    summary['capture_path'] = str(out)
    summary['capture_sha256'] = sha256(out)
    maps_file = Path('/proc/self/maps')
    if maps_file.exists():
        so_paths = sorted({line.split()[-1] for line in maps_file.read_text(errors='replace').splitlines()
                           if '/csrc.so' in line and line.split()[-1].startswith('/')})
        summary['loaded_csrc_shared_libraries'] = [{'path': p, 'sha256': sha256(p)} for p in so_paths]
    summary['per_primitive_max_gpu_count'] = int(count_cpu.max().item()) if count_cpu.numel() else 0
    summary['per_primitive_image_tile_bound'] = int(tile_width * tile_height)
    summary['above_tile_bound_count'] = int((count_cpu > tile_width * tile_height).sum().item())
    with out.with_suffix('.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('HIGS_ORACLE_CAPTURED', json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True,
        help='MX root containing gsplat/ and the original compiled csrc.so')
    parser.add_argument('--entrypoint', required=True,
        help='Already verified HiGS test script (R8 smoke or original R3 reproduction)')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0: parser.error('--samples must be > 0')
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import gsplat.cuda._wrapper as w
    orig = w.isect_tiles
    entry = Path(args.entrypoint).resolve()
    intercepted = {'count': 0}
    def intercept(*opargs, **opkwargs):
        if len(opargs) < 6 or opkwargs.get('conics') is None or opkwargs.get('opacities') is None:
            # Do not capture unrelated ops that might be called before HiGS native forward.
            return orig(*opargs, **opkwargs)
        intercepted['count'] += 1
        if intercepted['count'] != 1:
            raise RuntimeError('Unexpected repeated capture attempt')
        capture_intersection_args(orig, opargs, opkwargs, args.out, args.samples, args.seed)
        raise SystemExit(0)  # intentional diagnostic short circuit after verified capture
    w.isect_tiles = intercept
    print(f'Installing guarded interceptor for isect_tiles; executing {entry}', flush=True)
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        w.isect_tiles = orig
    if intercepted['count'] == 0 or not args.out.exists():
        raise RuntimeError('HIGS_CAPTURE_NOT_WRITTEN: check entrypoint, import paths, and traceback')


if __name__ == '__main__':
    main()
