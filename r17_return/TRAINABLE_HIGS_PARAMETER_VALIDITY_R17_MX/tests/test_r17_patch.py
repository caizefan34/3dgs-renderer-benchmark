"""CPU-only regression tests for MX package control flow and interception math.
Run: python3 tests/test_r17_patch.py
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from capture_intersections import IsectRecorder
from r17_p3_controlled import run_cell


def test_camera_dimension_in_subset():
    c, n, k = 4, 100, 10
    means2d = torch.zeros(c, n, 2)
    radii = torch.ones(c, n, 2)
    depths = torch.ones(c, n)
    kwargs = {'conics': torch.ones(c, n, 3), 'opacities': torch.ones(c, n), 'packed': False}
    def stub(*args, **kw):
        sampled_count = args[0].shape[-2]
        cams = args[0].shape[0]
        return (torch.ones(cams, sampled_count, dtype=torch.int32),
                torch.zeros(cams * sampled_count, dtype=torch.int64),
                torch.zeros(cams * sampled_count, dtype=torch.int32))
    rec = IsectRecorder(stub, mode='stats')
    with patch('torch.cuda.synchronize', lambda *a, **kw: None):
        pkg = rec._subset_rerun((means2d, radii, depths, 16, 60, 34), kwargs, k, 42)
    assert pkg['meta']['n_images'] == 4, pkg['meta']['n_images']
    assert pkg['meta']['gpu_count_sum'] == 40
    assert list(pkg['gpu_counts'].shape) == [c, k]
    print('PASS: subset n_images is camera batch=4, not N/k amplified')


def test_no_camera_graceful_row():
    cfg = dict(desc='real-camera test', opacity='sigmoid', scale='exp', quat='normalize', camera='real')
    with tempfile.TemporaryDirectory() as td:
        row, _ = run_cell('D', cfg, None, [], torch.arange(5).numpy(), None, 0.00392, 3.33,
                          Path(td), min_free_gb=0.0)
        assert row['fwd_status'] == 'ERROR'
        assert 'cameras.json' in row['error']
        assert (Path(td) / 'raw' / 'D_cell.json').is_file()
    print('PASS: real-camera unavailable yields explicit failed row, no crash')


def test_oom_safe_cleanup_source():
    src=(ROOT/'scripts'/'r17_p3_controlled.py').read_text()
    assert src.index('    result = None') < src.index('    try:\n        start = Event')
    assert 'del state, frame, render_alpha, result' in src
    print('PASS: P3 result cleanup initialized before possible OOM')


def test_safety_estimate_camera_factor():
    c, n, k = 4, 100, 10
    means2d = torch.zeros(c, n, 2)
    radii = torch.ones(c, n, 2)
    depths = torch.ones(c, n)
    kwargs = {'conics': torch.ones(c, n, 3), 'opacities': torch.ones(c, n), 'packed': False}
    def stub(*args, **kw):
        sample_n, cams = args[0].shape[-2], args[0].shape[0]
        counts = torch.full((cams, sample_n), 2, dtype=torch.int32)
        return counts, torch.zeros(cams*sample_n*2,dtype=torch.int64), torch.zeros(cams*sample_n*2,dtype=torch.int32)
    with tempfile.TemporaryDirectory() as td:
        rec = IsectRecorder(stub, mode='estimate_only', save_dir=Path(td))
        with patch('torch.cuda.synchronize', lambda *a, **kw: None):
            pkg = rec._subset_rerun((means2d, radii, depths, 16, 60, 34), kwargs, k, 42)
        with patch('torch.cuda.mem_get_info', return_value=(50<<30, 80<<30)), \
             patch('torch.cuda.memory_allocated', return_value=1000):
            info = rec._estimate_and_gate(pkg, n, None)
    assert info['call_n_images']==c
    assert info['est_isects_mean_scaled']==2*c*n
    print('PASS: safety estimate scales counts by actual 4 cameras, not N/k')

if __name__=='__main__':
    test_camera_dimension_in_subset()
    test_no_camera_graceful_row()
    test_oom_safe_cleanup_source()
    test_safety_estimate_camera_factor()
