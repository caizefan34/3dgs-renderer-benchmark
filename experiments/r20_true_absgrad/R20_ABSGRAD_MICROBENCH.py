#!/usr/bin/env python3
"""Status: IMPLEMENTED; MX timing NOT_EXECUTED.

Fixed input/topology; separate forward/backward CUDA events; no optimizer or
densification. Raw iteration CSV is retained; p99 uses linear interpolation.
Run each backend/mode in an isolated process with a verified import root.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from R20_GRADIENT_PARITY_TESTS import (STD_SHA, sha, activate, render,
                                      backward_loss, synthetic_fixtures, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--backend', choices=['std', 'higs'], required=True)
    p.add_argument('--absgrad', action='store_true'); p.add_argument('--real-fixture')
    p.add_argument('--warmup', type=int, default=20); p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.warmup < 1 or a.iterations < 2:
        p.error('warmup >= 1 and iterations >= 2 required')
    root = Path(a.root).resolve(); out = Path(a.output).resolve()
    if not torch.cuda.is_available() or not (root/'gsplat/csrc.so').is_file():
        write_json(out, dict(status='NOT_EXECUTED', reason='MX CUDA/root missing')); return 2
    if sha(root/'gsplat/csrc.so') != STD_SHA:
        raise RuntimeError('std_ll frozen binary hash mismatch')
    sys.path.insert(0, str(root))
    import gsplat
    if root not in Path(gsplat.__file__).resolve().parents:
        raise RuntimeError('wrong import root')
    binary = root/'gsplat/csrc.so'
    if a.backend == 'higs':
        from gsplat.experimental.render.kernels import _backend
        ext = _backend._C
        if getattr(ext, 'higs_true_absgrad_version', 0) != 1:
            raise RuntimeError('R20 experimental binary required')
        binary = Path(ext.__file__)
    binary_sha = sha(binary)
    fixture = (torch.load(a.real_fixture, map_location='cpu', weights_only=True)
               if a.real_fixture else synthetic_fixtures()['overlap'])
    # Cotangents are moved once, outside every event interval.
    fixture['v_rgb'] = fixture['v_rgb'].cuda(); fixture['v_alpha'] = fixture['v_alpha'].cuda()
    fixture['viewmats'] = fixture['viewmats'].cuda(); fixture['Ks'] = fixture['Ks'].cuda()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    rows = []
    for i in range(-a.warmup, a.iterations):
        raw, act = activate(fixture)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated(); before_reserved = torch.cuda.memory_reserved()
        started = time.perf_counter()
        events[0].record()
        rgb, alpha, m2d, radii = render(fixture, act, a.backend, a.absgrad)
        events[1].record()
        loss = backward_loss(fixture, rgb, alpha)
        events[2].record()
        loss.backward()
        events[3].record()
        events[3].synchronize()
        wall_ms = (time.perf_counter()-started)*1000
        if a.absgrad and getattr(m2d, 'absgrad', None) is None:
            raise RuntimeError('absgrad was not produced')
        if i >= 0:
            rows.append(dict(iteration=i, forward_ms=events[0].elapsed_time(events[1]),
                backward_ms=events[2].elapsed_time(events[3]),
                forward_loss_backward_ms=events[0].elapsed_time(events[3]), wall_ms=wall_ms,
                allocated_before=before_alloc, reserved_before=before_reserved,
                allocated_after=torch.cuda.memory_allocated(), reserved_after=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
        # Drop all graph references BETWEEN iterations, after event completion.
        del loss, rgb, alpha, m2d, radii, act, raw
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in ('forward_ms', 'backward_ms', 'forward_loss_backward_ms', 'wall_ms'):
        values = np.array([r[key] for r in rows])
        summary[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                            p90=float(np.percentile(values, 90, method='linear')),
                            p99=float(np.percentile(values, 99, method='linear')), max=float(values.max()),
                            n=len(values), top_outlier_iterations=np.argsort(values)[-10:][::-1].tolist())
    if sha(binary) != binary_sha or sha(root/'gsplat/csrc.so') != STD_SHA:
        raise RuntimeError('binary changed during measurement')
    write_json(out, dict(status='EXECUTED', backend=a.backend, absgrad=a.absgrad, summary=summary,
        device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        root=str(root), binary=str(binary), binary_sha256=binary_sha,
        fixture_sha256=sha(a.real_fixture) if a.real_fixture else 'synthetic_seed_42_overlap',
        n_gaussians=len(fixture['xyz']), n_cameras=len(fixture['viewmats']),
        width=fixture['width'], height=fixture['height'], warmup=a.warmup,
        px=os.getenv('HIGS_PX_RUNTIME', '2'), no_optimizer=True, no_densification=True,
        percentile_method='linear', raw_csv=str(out.with_suffix('.csv'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
