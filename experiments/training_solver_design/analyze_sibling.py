"""Analyze all predeclared sibling policies without selecting a winning seed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', choices=('room', 'garden'), default=['room', 'garden'])
    parser.add_argument('--out', default='sibling_summary.json')
    args = parser.parse_args()
    root = Path(__file__).parent
    records = []
    for scene in args.scenes:
        path = root/f'{scene}_sibling_256.json'
        data = json.loads(path.read_text())
        assert data['completed']
        for key, file in (('source_sha256', data.get('source_file', 'sibling_probe.py')),
                          ('update_helper_sha256', 'newborn_updates.py'),
                          ('coordinate_helper_sha256', 'sibling_coordinates.py')):
            assert data[key] == sha(root/file), (scene, key)
        if 'numerical_evidence_sha256' in data['parity_audit']:
            assert data['parity_audit']['numerical_evidence_sha256'] == sha(root/f'{scene}_sibling_numerics.json')
        arms = {a['name']: a for a in data['arms']}
        assert len(arms) == 26
        policies = data['protocol']['policies']
        seeds = data['protocol']['seeds']
        for seed in seeds:
            ref = arms[f'native_{seed}']
            for policy in list(policies)+['native_repeat']:
                arm = arms[f'{policy}_{seed}']
                assert arm['initial_children_sha256'] == ref['initial_children_sha256']
                for view in ('A', 'B'):
                    assert arm['history']['0'][view] == ref['history']['0'][view], (scene, seed, policy, view)
        per_horizon = {}
        for horizon in data['protocol']['horizons']:
            h = str(horizon)
            views = {}
            for view in ('A', 'B'):
                def loss(policy, seed):
                    return arms[f'{policy}_{seed}']['history'][h][view]['loss']
                means = [(loss('native', s)+loss('native_repeat', s))/2 for s in seeds]
                gaps = [abs(loss('native', s)-loss('native_repeat', s)) for s in seeds]
                results = {}
                for policy in policies[1:]:
                    gains = [m-loss(policy, s) for m, s in zip(means, seeds)]
                    psnr_gain = [arms[f'{policy}_{s}']['history'][h][view]['mean_view_psnr'] -
                                 (arms[f'native_{s}']['history'][h][view]['mean_view_psnr']+
                                  arms[f'native_repeat_{s}']['history'][h][view]['mean_view_psnr'])/2 for s in seeds]
                    results[policy] = {'paired_gains': gains, 'mean_gain': float(np.mean(gains)),
                                       'positive_draws': sum(g > 0 for g in gains),
                                       'over_repeat_gap': sum(g > d for g, d in zip(gains, gaps)),
                                       'mean_view_psnr_gain': float(np.mean(psnr_gain))}
                contrasts = {}
                for target, control in (('pair_full', 'pair_shared'), ('pair_shared', 'pair_zero'),
                                        ('pair_full', 'inherit_05'), ('pair_full', 'inherit_02'),
                                        ('pair_full', 'physical_lr2'), ('pair_zero', 'physical_lr2')):
                    values = [loss(control, s)-loss(target, s) for s in seeds]
                    contrasts[f'{target}_vs_{control}'] = {'paired_gains': values, 'mean_gain': float(np.mean(values))}
                views[view] = {'native_repeat_gaps': gaps, 'native_mean_loss': float(np.mean(means)),
                               'policies': results, 'contrasts': contrasts}
            per_horizon[h] = views
        times = {p: float(np.mean([a['train_seconds_with_sync'] for a in data['arms']
                                  if a['seed'] is not None and a['policy'] == p])) for p in policies}
        records.append({'scene': scene, 'file_sha256': sha(path), 'per_horizon': per_horizon,
                        'mean_train_seconds': times, 'prototype_time_ratio_vs_native': {p: t/times['native'] for p, t in times.items()},
                        'total_seconds': data['total_seconds_including_setup'], 'parity_audit': data['parity_audit']})
        print(scene, 'total seconds', round(data['total_seconds_including_setup'], 2))
        for h in ('32', '128', '256'):
            print('horizon', h)
            for p in policies[1:]:
                a, b = [per_horizon[h][v]['policies'][p] for v in ('A', 'B')]
                print(p, 'A/B loss gain x1e6', round(a['mean_gain']*1e6, 3), round(b['mean_gain']*1e6, 3),
                      'B wins/over replay', b['positive_draws'], b['over_repeat_gap'])
        print('time ratios', records[-1]['prototype_time_ratio_vs_native'])
    (root/args.out).write_text(json.dumps({'scope': 'Single-birth development evidence; no full-training or novelty claim',
                                                       'analyzer_sha256': sha(Path(__file__)), 'records': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
