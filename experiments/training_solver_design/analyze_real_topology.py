"""Summarize existing real-state probe evidence without choosing new scores."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).parent
    paths = sorted(root.glob('*_2000_topology*.json'))
    source_sha = hashlib.sha256((root / 'real_topology_probe.py').read_bytes()).hexdigest()
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        assert data['completed'] and data['source_sha256'] == source_sha
        assert len(data['optimizer_survivor_and_newborn_checks_passed']) == len(data['candidates'])
        horizon = str(max(map(int, data['analysis'])))
        base = data['arms'][0]
        edits = data['arms'][2:]
        gains_a = np.array([base['history'][horizon]['A']['loss'] - a['history'][horizon]['A']['loss'] for a in edits])
        gains_b = np.array(data['analysis'][horizon]['B_loss_gains'])
        zero_b = np.array(data['analysis']['0']['B_loss_gains'])
        noise = data['analysis'][horizon]['base_repeat_absolute_difference']
        rows = {}
        for score, item in data['analysis'][horizon]['rankings'].items():
            indices = item['top3_indices']
            rows[score] = dict(item, top3_A_gain=float(gains_a[indices].mean()),
                               top3_equal_A_B_gain=float(((gains_a + gains_b)[indices] / 2).mean()))
        records.append({
            'file': path.name, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scene': data['scene'], 'N': data['N'], 'horizon': int(horizon),
            'candidate_count': len(edits), 'groups_of': len(data['candidates'][0]['ids']),
            'wall_seconds_including_setup': data['elapsed_seconds_including_setup'],
            'base_continuation_plus_evaluation_seconds': base['continuation_and_evaluation_seconds'],
            'median_local_basis_seconds': float(np.median([a['local_basis_seconds'] for a in edits])),
            'repeat_B_noise': noise, 'ranking_rows': rows,
            'initially_B_harmful_later_B_helpful_indices': np.flatnonzero((zero_b < -noise) & (gains_b > noise)).tolist(),
            'final_A_and_B_both_improve_indices': np.flatnonzero((gains_a > noise) & (gains_b > noise)).tolist(),
            'B_best_index': int(gains_b.argmax()), 'B_best_gain': float(gains_b.max()),
            'B_best_candidate_A_gain': float(gains_a[gains_b.argmax()]),
            'local_explained_fraction_range': [float(min(a['scores']['local_edit_fraction_explained'] for a in edits)),
                                              float(max(a['scores']['local_edit_fraction_explained'] for a in edits))],
        })
    result = {'scope': 'Offline development diagnostics, not E2E training or official method comparison',
              'source_sha256_verified': source_sha, 'records': records,
              'end_to_end_speedup_verified': False, 'novelty_established': False}
    destination = root / 'real_topology_summary.json'
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for r in records:
        print(r['file'], 'wall', round(r['wall_seconds_including_setup'], 2),
              'noise', r['repeat_B_noise'], 'both_improve', r['final_A_and_B_both_improve_indices'])
        for score, row in r['ranking_rows'].items():
            print(' ', score, 'B_gain', round(row['top3_mean_gain'] * 1e6, 3),
                  'A+B_gain', round(row['top3_equal_A_B_gain'] * 1e6, 3))


if __name__ == '__main__':
    main()
