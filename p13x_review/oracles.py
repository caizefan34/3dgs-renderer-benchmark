"""CPU algebra counterexamples and explicitly hypothetical Amdahl scenarios.

This is not a 3DGS benchmark and does not run or modify a CUDA renderer.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
KS = [2, 4, 8, 16, 32, 64]
MS = [1, 2, 4, 8, 16, 32]
SCENARIOS = {
    'pessimistic': dict(u=.65, eta=2., B=32, G=.90, C=.15, setup=.30, pair=.70),
    'realistic_assumption': dict(u=.50, eta=1.5, B=8, G=.75, C=.10, setup=.20, pair=.40),
    'optimistic': dict(u=.10, eta=1., B=4, G=.55, C=.04, setup=.10, pair=.12),
}


def speedup(k, m, s):
    window = s['G'] + s['B'] * (s['C'] + s['setup'] + m * s['pair'])
    return 1 / (s['u'] + (1 - s['u']) * s['eta'] * window / k)


def pcg_diagonal(a, y, max_iter=2000, tol=1e-10):
    h = a.T @ a
    b = a.T @ y
    x = np.zeros_like(b)
    r = b.copy()
    z = r / np.diag(h)
    p = z.copy()
    rz = r @ z
    history = []
    for it in range(1, max_iter + 1):
        hp = a.T @ (a @ p)
        alpha = rz / (p @ hp)
        x += alpha * p
        r -= alpha * hp
        # Log true residual separately from the recursively updated residual.
        actual = np.linalg.norm(a.T @ (y - a @ x)) / np.linalg.norm(b)
        history.append(dict(iteration=it, normal_relative=float(actual),
                            image_relative=float(np.linalg.norm(a @ x - y) / np.linalg.norm(y))))
        if actual < tol:
            break
        z = r / np.diag(h)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return history


def main():
    rows = []
    tables = ['All numbers below are MODEL OUTPUT, not measured training speedups.\n']
    for name, scenario in SCENARIOS.items():
        tables += [f'## {name}', '', '| K \\ m | 1 | 2 | 4 | 8 | 16 | 32 |',
                   '|---|---:|---:|---:|---:|---:|---:|']
        for k in KS:
            values = [speedup(k, m, scenario) for m in MS]
            tables.append('| ' + str(k) + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
            rows += [dict(scenario=name, K=k, iterations=m, speedup=v, **scenario)
                     for m, v in zip(MS, values)]
        tables.append('')
    (ROOT / 'cost_tables.md').write_text('\n'.join(tables), encoding='utf-8')
    with (ROOT / 'cost_grid.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # An alpha-compositing weight operator: nonnegative rows with sums below 1.
    # Any such row has alpha_i = w_i / (1 - sum_{j<i} w_j).
    # This establishes an algebraic compositing counterexample, not a claim
    # that these rows come from a particular multi-view projected 3D scene.
    small = .98 * np.array([[.8, .2], [.2, .8]])
    truth = np.array([1., 0.])
    y = small @ truth
    independent = (small.T @ y) / small.sum(axis=0)
    eps = np.geomspace(.001, .2, 128)
    a = np.zeros((256, 256))
    for i, e in enumerate(eps):
        a[2*i:2*i+2, 2*i:2*i+2] = .98 * np.array([[.5+e,.5-e],[.5-e,.5+e]])
    rng = np.random.default_rng(13)
    hist = pcg_diagonal(a, a @ rng.uniform(0, 1, 256))
    two = .98 * np.array([[.501, .499], [.499, .501]])
    h = two.T @ two
    jac = np.eye(2) - h / np.diag(h)[:, None]
    result = {
        'scope': 'CPU float64 algebra only; no scene, no renderer, no GPU timing',
        'independent_fit': {'A': small.tolist(), 'target': y.tolist(),
                            'global_beta': truth.tolist(), 'row_sum_beta': independent.tolist(),
                            'row_sum_residual_norm': float(np.linalg.norm(small @ independent-y))},
        'two_column_condition_A': float(np.linalg.cond(two)),
        'two_column_condition_ATA': float(np.linalg.cond(h)),
        'two_column_jacobi_spectral_radius': float(max(abs(np.linalg.eigvals(jac)))),
        'multiblock_condition_A': float(np.linalg.cond(a)),
        'multiblock_condition_ATA': float(np.linalg.cond(a.T @ a)),
        'pcg_iterations_to_1e_10_normal_residual': len(hist),
        'pcg_selected': [x for x in hist if x['iteration'] in [1,2,4,8,16,32,64,128,256,512,len(hist)]],
        'scenario_assumptions': SCENARIOS,
    }
    (ROOT / 'oracle_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('\n'.join(tables))


if __name__ == '__main__':
    main()
