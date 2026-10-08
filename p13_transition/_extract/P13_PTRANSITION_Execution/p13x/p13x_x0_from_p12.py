#!/usr/bin/env python3
import argparse, json, statistics, zipfile

ap = argparse.ArgumentParser()
ap.add_argument('p12_zip')
ap.add_argument('-o','--output', default='P13X_X0_P12_PRIOR.json')
a = ap.parse_args()

with zipfile.ZipFile(a.p12_zip) as z:
    traj = json.loads(z.read('p12_artifacts/P12A_SYSTEM_TRAJECTORY.json'))
    runs = json.loads(z.read('p12_artifacts/P12A_SYSTEM_RUNS.json'))

rows = {}
scene_means = []
for scene, d in traj['visible_gaussian_trajectories'].items():
    xs = [r for r in d['den'] if 15000 <= r[0] <= 30000]
    ratios = [r[2]/r[1] for r in xs]
    Ns = sorted(set(r[1] for r in xs))
    rows[scene] = {
        'n_samples': len(xs),
        'N_unique_count': len(Ns),
        'N_values': Ns,
        'visible_fraction_mean': statistics.mean(ratios),
        'visible_fraction_median': statistics.median(ratios),
        'visible_fraction_min': min(ratios),
        'visible_fraction_max': max(ratios),
    }
    scene_means.append(statistics.mean(ratios))

phase = {}
for scene, arms in runs['per_scene'].items():
    c = arms['C0_FUSED']
    phase[scene] = {k: c['phase_ms'][k]['mean_ms'] for k in ['forward','loss','backward','densify','optimizer']}
    phase[scene]['mean_iter_ms'] = c['mean_iter_ms']

keys = ['forward','loss','backward','densify','optimizer','mean_iter_ms']
phase_mean = {k: statistics.mean(phase[s][k] for s in phase) for k in keys}

out = {
    'phase':'P13-X0 existing-evidence extraction',
    'source':'P12A_SYSTEM_TRAJECTORY.json + P12A_SYSTEM_RUNS.json from supplied P12 zip',
    'window':[15000,30000],
    'all_13_scenes_constant_N': all(v['N_unique_count']==1 for v in rows.values()),
    'mean_of_scene_visible_fractions': statistics.mean(scene_means),
    'per_scene': rows,
    'c0_phase_ms_unweighted_scene_mean': phase_mean,
    'interpretation': 'Supports testing persistent topology/active worksets only; does not establish exact membership/order/dirty-work stability.'
}
with open(a.output,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
print(json.dumps(out, indent=2))
