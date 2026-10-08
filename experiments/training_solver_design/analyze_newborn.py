"""Matched-policy summaries for fixed newborn controls; no score selection."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).parent
    source = hashlib.sha256((root/'newborn_probe.py').read_bytes()).hexdigest()
    helper = hashlib.sha256((root/'newborn_updates.py').read_bytes()).hexdigest()
    records = []
    for path in sorted(root.glob('*_newborn_256.json')):
        data = json.loads(path.read_text())
        assert data['completed'] and data['source_sha256'] == source and data['update_helper_sha256'] == helper
        arms = {a['name']: a for a in data['arms']}
        assert len(arms) == 17
        per_horizon = {}
        for horizon in data['protocol']['horizons']:
            h = str(horizon)
            views = {}
            for view in ('A', 'B'):
                def loss(name):
                    return arms[name]['history'][h][view]['loss']
                pairs = []
                for seed in data['protocol']['seeds']:
                    native = loss(f'native_{seed}')
                    repeat = loss(f'native_repeat_{seed}')
                    midpoint = (native+repeat)/2
                    init_hash = arms[f'native_{seed}']['initial_children_sha256']
                    for policy in ('fresh_age', 'inherit_02', 'lr_scale'):
                        arm = arms[f'{policy}_{seed}']
                        assert arm['initial_children_sha256'] == init_hash
                        assert abs(arm['history']['0'][view]['loss'] - arms[f'native_{seed}']['history']['0'][view]['loss']) < 1e-10
                    pairs.append({'seed': seed, 'native_repeat_gap': abs(native-repeat),
                                  'native_mean_loss': midpoint,
                                  'policy_gain': {p: midpoint-loss(f'{p}_{seed}') for p in ('fresh_age', 'inherit_02', 'lr_scale')}})
                policies = {}
                for policy in ('fresh_age', 'inherit_02', 'lr_scale'):
                    gains = np.array([p['policy_gain'][policy] for p in pairs])
                    policies[policy] = {'paired_gains': gains.tolist(), 'mean_gain': float(gains.mean()),
                                         'positive_draws': int(np.sum(gains > 0)),
                                         'positive_over_native_repeat_gap': int(sum(g > p['native_repeat_gap'] for g, p in zip(gains, pairs)))}
                views[view] = {'pairs': pairs, 'policies': policies,
                               'no_edit_repeat_gap': abs(loss('no_edit') - loss('no_edit_repeat')),
                               'native_mean_vs_no_edit_gain': (loss('no_edit')+loss('no_edit_repeat'))/2 - float(np.mean([p['native_mean_loss'] for p in pairs]))}
            per_horizon[h] = views
        record = {'file': path.name, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scene': data['scene'], 'N': data['N'], 'nparents': data['nparents'],
                  'total_seconds': data['total_seconds_including_setup'], 'per_horizon': per_horizon,
                  'mean_train_seconds': {p: float(np.mean([a['train_seconds_with_sync'] for a in data['arms'] if a['seed'] is not None and a['policy'] == p]))
                                         for p in ('native', 'fresh_age', 'inherit_02', 'lr_scale')}}
        records.append(record)
        print(path.name, 'seconds', round(record['total_seconds'], 2))
        for h in ('1','8','32','128','256'):
            print('horizon',h)
            for p in ('fresh_age','inherit_02','lr_scale'):
                a=per_horizon[h]['A']['policies'][p]
                b=per_horizon[h]['B']['policies'][p]
                print(' ',p,'A/B mean gain x1e6',round(a['mean_gain']*1e6,3),round(b['mean_gain']*1e6,3),
                      'B positive',b['positive_draws'],'B above repeat gap',b['positive_over_native_repeat_gap'])
    (root/'newborn_summary.json').write_text(json.dumps({'scope':'Known-mechanism development controls, not new method or full training',
                                                       'source_sha256':source,'helper_sha256':helper,'records':records},indent=2)+'\n')


if __name__ == '__main__':
    main()
