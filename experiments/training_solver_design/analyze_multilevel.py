"""Summarize completed diagnostic evidence, checking executed source identities."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = {'scope': 'Fixed-checkpoint diagnostics; no trained-quality, end-to-end speedup or novelty claim',
               'gradients': {}, 'costs': {}, 'source_checks': {}}
    for scene in ('room', 'garden'):
        for prefix, script, helper in (
            ('multilevel', 'multilevel_gradient_probe.py', None),
            ('shared', 'shared_gradient_probe.py', 'shared_proxy_renderer.py'),
            ('scaled', 'scaled_gradient_probe.py', 'scaled_proxy_renderer.py'),
        ):
            path = ROOT/f'{scene}_{prefix}_gradients.json'
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            assert d['completed']
            assert sha(ROOT/script) == d['source_sha256']
            assert sha(ROOT/'multilevel_loss.py') == d['helper_sha256']
            if helper:
                assert sha(ROOT/helper) == d['shared_helper_sha256']
            summary['source_checks'][path.name] = True
            factors = {}
            for factor in d['protocol']['coarse_factors']:
                records = [r for r in d['records'] if r['coarse_factor'] == factor]
                factors[factor] = {
                    group: {
                        key: [min(r['summary'][group][key] for r in records),
                              max(r['summary'][group][key] for r in records)]
                        for key in ('beta1_variance_ratio', 'beta1_adam_error_ratio', 'cv_mean_error_relative')
                    }
                    for group in records[0]['summary']
                }
            summary['gradients'][path.name] = factors
        for suffix, script, helper, shared in (
            ('masked_work', 'masked_work_probe.py', 'masked_renderer.py', None),
            ('masked_work_v2', 'masked_work_probe_v2.py', 'masked_renderer_v2.py', None),
            ('shared_work', 'shared_work_probe.py', 'masked_renderer_v2.py', 'shared_proxy_renderer.py'),
            ('shared_patch_work', 'shared_patch_work_probe.py', 'masked_renderer_v2.py', 'shared_proxy_renderer.py'),
            ('scaled_patch_work', 'scaled_patch_work_probe.py', 'masked_renderer_v2.py', 'scaled_proxy_renderer.py'),
        ):
            path = ROOT/f'{scene}_{suffix}.json'
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            assert d['completed']
            assert sha(ROOT/script) == d['source_sha256']
            assert sha(ROOT/helper) == d['masked_helper_sha256']
            if shared:
                assert sha(ROOT/shared) == d['shared_helper_sha256']
            if 'patch_helper_sha256' in d:
                assert sha(ROOT/'patch_loss.py') == d['patch_helper_sha256']
            summary['source_checks'][path.name] = True
            summary['costs'][path.name] = [
                {'camera': r['camera_id'], 'wall_ms_per_step': {
                    mode: sum(v['wall_ms_per_step'] for v in times)/len(times)
                    for mode, times in r['timings'].items()},
                 'max_halo_gradient_relative': max(v['relative_l2'] for check in r['halo_checks'] for v in check['gradient_parity'].values()),
                 'active_tile_fraction': [check['active_tile_fraction'] for check in r['halo_checks']]}
                for r in d['records']
            ]
    (ROOT/'multilevel_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Verified completed raw files:', len(summary['source_checks']))
    for file, records in summary['costs'].items():
        print(file, [(r['camera'], {k: round(v, 3) for k, v in r['wall_ms_per_step'].items()}) for r in records])


if __name__ == '__main__':
    main()
