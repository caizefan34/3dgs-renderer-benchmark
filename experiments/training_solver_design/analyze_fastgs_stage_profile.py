"""Summarize disjoint phase timings without double-counting profiler kernels."""

import argparse
import json
from pathlib import Path
import statistics


PHASES = {
    'forward': 'phase.forward', 'loss_l1': 'phase.loss_l1',
    'loss_ssim': 'phase.loss_ssim',
    'raster_backward': '_RasterizeGaussiansBackward',
    'ssim_backward': 'FusedSSIMMapBackward',
    'optimizer': 'phase.optimizer', 'topology': 'phase.topology',
    'density_stats': 'phase.density_stats',
    'opacity_reset': 'phase.opacity_reset', 'final_prune': 'phase.final_prune',
}


def summarize(source):
    result = {'scene': source['scene'], 'windows': []}
    for window in source['windows']:
        rows = window['rows']
        # Kineto also emits CUDA user-annotation ranges with CPU event names.
        # They include idle spans/children and must not be added to kernels.
        cpu_names = {row['name'] for row in rows if row['device_type'] == 'DeviceType.CPU'}
        kernels = [row for row in rows if row['device_type'] == 'DeviceType.CUDA'
                   and row['name'] not in cpu_names]
        if any('(' not in row['name'] for row in kernels):
            raise ValueError('Unexpected non-kernel CUDA event; inspect before counting')
        kernel_us = sum(row['self_device_us'] for row in kernels)
        phases = {}
        for name, event_name in PHASES.items():
            # Autograd executes on another CPU thread; phase.backward's child
            # device total is incomplete. Use the native autograd operator.
            matches = [row for row in rows if row['name'] == event_name
                       and row['device_type'] == 'DeviceType.CPU']
            if matches:
                assert len(matches) == 1
                row = matches[0]
                phases[name] = {
                    'calls': row['count'], 'cuda_ms': row['device_total_us'] / 1000,
                    'fraction_active_cuda': row['device_total_us'] / kernel_us,
                }
        phase_us = sum(row['cuda_ms'] * 1000 for row in phases.values())
        # Allow rounding, but reject an invalid overlapping phase partition.
        if phase_us > kernel_us * 1.001:
            raise ValueError(('Phase overlap or unexpected profiler accounting', phase_us, kernel_us))
        walls = window['wall_seconds']
        result['windows'].append({
            'first_iteration': window['first_iteration'],
            'last_iteration': window['last_iteration'],
            'steps': len(walls),
            'gaussians_start': window['gaussians_start'],
            'gaussians_end': window['gaussians_end'],
            'optimizer_calls': window['optimizer_calls'],
            'cuda_ms_per_step': kernel_us / 1000 / len(walls),
            'profiled_wall_ms_per_step': 1000 * statistics.mean(walls),
            'profiled_wall_median_ms': 1000 * statistics.median(walls),
            'phases': phases,
            'unassigned_cuda_fraction': max(0, 1 - phase_us / kernel_us),
            'top_kernels': sorted(kernels, key=lambda row: row['self_device_us'], reverse=True)[:12],
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(json.loads(path.read_text())) for path in args.inputs]
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        for window in result['windows']:
            print(result['scene'], window['first_iteration'],
                  'CUDA ms/step', round(window['cuda_ms_per_step'], 3),
                  'profiled wall ms/step', round(window['profiled_wall_ms_per_step'], 3),
                  {key: round(value['fraction_active_cuda'] * 100, 1)
                   for key, value in window['phases'].items()})


if __name__ == '__main__':
    main()
