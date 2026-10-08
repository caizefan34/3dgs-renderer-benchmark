"""Verify and summarize all fixed 32-step handoff development draws."""

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
    parser.add_argument('--out', default='transient_summary.json')
    args = parser.parse_args()
    root = Path(__file__).parent
    records = []
    for scene in args.scenes:
        path = root/f'{scene}_transient_256.json'
        data = json.loads(path.read_text())
        assert data['completed'] and len(data['arms']) == 20
        for key, file in (('source_sha256', 'transient_probe.py'),
                          ('update_helper_sha256', 'newborn_updates.py'),
                          ('coordinate_helper_sha256', 'sibling_coordinates.py'),
                          ('transport_helper_sha256', 'sibling_transport.py')):
            assert data[key] == sha(root/file), (scene, key)
        numeric_sha = data['parity_audit']['numerical_evidence_sha256']
        if numeric_sha:
            assert numeric_sha == sha(root/f'{scene}_sibling_numerics.json')
        arms = {a['name']: a for a in data['arms']}
        policies, seeds = data['protocol']['policies'], data['protocol']['seeds']
        for seed in seeds:
            ref = arms[f'native_{seed}']
            for policy in list(policies)+['native_repeat']:
                arm = arms[f'{policy}_{seed}']
                assert arm['initial_children_sha256'] == ref['initial_children_sha256']
                assert arm['history']['0'] == ref['history']['0']
            assert arms[f'pair_transient32_{seed}']['handoff']['exact_image_and_parameter_identity']
        horizons = {}
        for horizon in data['protocol']['horizons']:
            views = {}
            for view in ('A', 'B'):
                def loss(policy, seed):
                    return arms[f'{policy}_{seed}']['history'][str(horizon)][view]['loss']
                midpoint = [(loss('native', s)+loss('native_repeat', s))/2 for s in seeds]
                gaps = [abs(loss('native', s)-loss('native_repeat', s)) for s in seeds]
                summaries = {}
                for policy in policies[1:]:
                    gains = [m-loss(policy, s) for m, s in zip(midpoint, seeds)]
                    summaries[policy] = {'paired_gains': gains, 'mean_gain': float(np.mean(gains)),
                                         'positive_draws': sum(g > 0 for g in gains),
                                         'over_native_repeat_gap': sum(g > gap for g, gap in zip(gains, gaps))}
                contrasts = {control: [loss(control, s)-loss('pair_transient32', s) for s in seeds]
                             for control in ('pair_full', 'inherit_05', 'inherit_02')}
                views[view] = {'native_repeat_gaps': gaps, 'policies': summaries,
                               'transient_gain_vs_control': contrasts}
            horizons[str(horizon)] = views
        times = {policy: float(np.mean([a['train_seconds_with_sync'] for a in data['arms']
                                        if a['seed'] is not None and a['policy'] == policy])) for policy in policies}
        record = {'scene': scene, 'file_sha256': sha(path), 'horizons': horizons,
                  'mean_train_seconds': times, 'time_ratio_vs_native': {p: t/times['native'] for p,t in times.items()},
                  'total_seconds': data['total_seconds_including_setup']}
        records.append(record)
        print(scene, 'wall seconds', round(record['total_seconds'], 2))
        for h in ('32', '128', '256'):
            print('horizon', h)
            for policy in policies[1:]:
                a,b = [horizons[h][view]['policies'][policy] for view in ('A','B')]
                print(policy, 'A/B gain x1e6', round(a['mean_gain']*1e6,3), round(b['mean_gain']*1e6,3),
                      'B wins/over replay', b['positive_draws'], b['over_native_repeat_gap'])
        print('time ratios', record['time_ratio_vs_native'])
    (root/args.out).write_text(json.dumps({'scope': '32-step lifetime selected after development evidence; no independent generalization claim',
                                         'analyzer_sha256': sha(Path(__file__)), 'records': records},indent=2)+'\n')


if __name__ == '__main__':
    main()
