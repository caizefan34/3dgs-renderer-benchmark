"""Profile four windows on an unchanged author FastGS training trajectory."""

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function


REPO = Path('/root/codex_fastgs_audit_20260927')
WINDOWS = (4096, 10048, 16000, 24064)
LENGTH = 128
PIN = '44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f'


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=('room', 'garden'), required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise RuntimeError('Wrong author source commit')
    subprocess.run(['git', 'diff', '--exit-code', '--', 'train.py', 'scene',
                    'gaussian_renderer', 'utils', 'arguments'], check=True)
    from scene import GaussianModel
    import gaussian_renderer
    import fused_ssim
    from utils import loss_utils

    output = REPO / 'output' / f'{args.scene}_native_stage_profile_20260927'
    output.mkdir(exist_ok=False)
    manifest = {
        'scene': args.scene, 'source_commit': PIN,
        'train_sha256': digest(REPO / 'train.py'),
        'probe_sha256': digest(Path(__file__).resolve()),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scope': 'Native 30K trajectory with instrumented windows. Profiled wall is not a speed baseline.',
        'windows': [],
    }
    state = {'iteration': 0, 'profiler': None, 'start': None,
             'last_time': None, 'walls': [], 'optimizer_calls': {},
             'window_start': None}
    originals = []

    def save():
        (output / 'stage_profile.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def patch(obj, attr, value):
        old = getattr(obj, attr)
        originals.append((obj, attr, old))
        setattr(obj, attr, value)
        return old

    def wrap(obj, attr, label):
        original = getattr(obj, attr)

        @functools.wraps(original)
        def wrapped(*call_args, **kwargs):
            if state['profiler'] is None:
                return original(*call_args, **kwargs)
            if label == 'phase.adam':
                opt = call_args[0]
                key = ','.join(group.get('name', '?') for group in opt.param_groups)
                state['optimizer_calls'][key] = state['optimizer_calls'].get(key, 0) + 1
            with record_function(label):
                return original(*call_args, **kwargs)

        patch(obj, attr, wrapped)

    original_lr = GaussianModel.update_learning_rate

    def update_learning_rate(model, iteration):
        now = time.perf_counter()
        if state['profiler'] is not None:
            state['walls'].append(now - state['last_time'])
            if iteration == state['window_start'] + LENGTH:
                prof = state['profiler']
                prof.stop()
                rows = []
                for event in prof.key_averages():
                    rows.append({
                        'name': event.key, 'count': event.count,
                        'self_cpu_us': event.self_cpu_time_total,
                        'cpu_total_us': event.cpu_time_total,
                        'self_device_us': event.self_device_time_total,
                        'device_total_us': event.device_time_total,
                        'device_type': str(event.device_type),
                    })
                window = {
                    'first_iteration': state['window_start'],
                    'last_iteration': iteration - 1,
                    'gaussians_start': state['n_start'],
                    'gaussians_end': len(model.get_xyz),
                    'wall_seconds': state['walls'],
                    'optimizer_calls': state['optimizer_calls'],
                    'rows': rows,
                }
                manifest['windows'].append(window)
                state['profiler'] = None
                save()
                print('STAGE_PROFILE ' + json.dumps({
                    key: value for key, value in window.items()
                    if key not in ('rows', 'wall_seconds')
                }), flush=True)
        state['iteration'] = iteration
        if iteration in WINDOWS:
            torch.cuda.synchronize()
            state.update(window_start=iteration, walls=[], optimizer_calls={},
                         n_start=len(model.get_xyz))
            state['profiler'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            state['profiler'].start()
        state['last_time'] = time.perf_counter()
        return original_lr(model, iteration)

    patch(GaussianModel, 'update_learning_rate', update_learning_rate)
    wrap(gaussian_renderer, 'render_fastgs', 'phase.forward')
    wrap(loss_utils, 'l1_loss', 'phase.loss_l1')
    wrap(fused_ssim, 'fused_ssim', 'phase.loss_ssim')
    wrap(torch.Tensor, 'backward', 'phase.backward')
    wrap(GaussianModel, 'optimizer_step', 'phase.optimizer')
    wrap(torch.optim.Adam, 'step', 'phase.adam')
    wrap(GaussianModel, 'densify_and_prune_fastgs', 'phase.topology')
    wrap(GaussianModel, 'add_densification_stats', 'phase.density_stats')
    wrap(GaussianModel, 'reset_opacity', 'phase.opacity_reset')
    wrap(GaussianModel, 'final_prune_fastgs', 'phase.final_prune')
    # fast_utils imports render_fastgs from the wrapped renderer module.
    from utils import fast_utils
    wrap(fast_utils, 'compute_gaussian_score_fastgs', 'phase.score')

    sys.argv = [
        'train.py', '-s', '/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/' + args.scene,
        '-i', 'images', '--eval', '--iterations', '30000', '--densification_interval', '100',
        '--optimizer_type', 'default', '--highfeature_lr', '0.02',
        '--grad_abs_thresh', '0.0004' if args.scene == 'room' else '0.0003',
        '--test_iterations', '999999', '--save_iterations', '30000', '-m', str(output),
    ]
    if args.scene == 'garden':
        sys.argv += ['--loss_thresh', '0.06']
    manifest['argv'] = sys.argv[:]
    save()
    start = time.perf_counter()
    try:
        runpy.run_path(str(REPO / 'train.py'), run_name='__main__')
        manifest['trajectory_wall_seconds_including_profile_and_loading_saving'] = time.perf_counter() - start
        manifest['final_ply_sha256'] = digest(output / 'point_cloud/iteration_30000/point_cloud.ply')
        manifest['status'] = 'completed'
        save()
    finally:
        for obj, attr, original in reversed(originals):
            setattr(obj, attr, original)


if __name__ == '__main__':
    main()
