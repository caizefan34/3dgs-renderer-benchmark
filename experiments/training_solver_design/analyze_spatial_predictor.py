"""Verify executed identities and summarize spatial predictor diagnostics."""

import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).parent


def sha(file):
    return hashlib.sha256((ROOT/file).read_bytes()).hexdigest()


def main():
    report={'scope':'Diagnostics only; no completed training speed/quality or novelty claim','records':{}}
    configurations=(
        ('garden_sort_reuse.json','sort_reuse_probe.py','sort_reuse_renderer.py'),
        ('garden_history_predictor.json','history_predictor_probe.py','block_predictor.py'),
        ('garden_spatial_moment.json','spatial_moment_probe.py','spatial_moment_predictor.py'),
        ('room_spatial_moment_validation.json','spatial_moment_validation_probe.py','spatial_moment_predictor.py'),
        ('garden_spatial_cache_staleness.json','spatial_cache_staleness_probe.py','spatial_moment_predictor.py'),
        ('garden_transported_cache.json','transported_cache_probe.py','spatial_moment_predictor.py'),
    )
    for file,script,helper in configurations:
        d=json.loads((ROOT/file).read_text())
        assert d['completed'] and d['source_sha256']==sha(script)
        key='shared_helper_sha256' if 'sort_reuse' in file else 'predictor_helper_sha256'
        assert d[key]==sha(helper)
        if 'projection_helper_sha256' in d:
            assert d['projection_helper_sha256']==sha('sort_reuse_renderer.py')
        if 'transport_helper_sha256' in d:
            assert d['transport_helper_sha256']==sha('transported_adjoint_predictor.py')
        rows=[]
        for r in d['records']:
            row={'camera':r['camera_id']}
            if 'timings' in r:
                row['wall_ms']={k:sum(x['wall_ms_per_step'] for x in v)/len(v) for k,v in r['timings'].items()}
                row['all_lists_equal']=all(v['equal'] for v in r['list_identity'].values())
            else:
                row.update(predictor=r['predictor'],groups={},cache_age=r.get('cache_age'))
                for k,v in r['summary'].items():
                    ss=v['sums']
                    row['groups'][k]={
                        'reported_beta':d['protocol']['beta'],
                        'reported_variance_ratio':v.get('beta1_variance_ratio',v.get('beta_half_variance_ratio')),
                        'reported_adam_error_ratio':v.get('beta1_adam_error_ratio',v.get('beta_half_adam_error_ratio')),
                        'beta_half_variance_from_sums':(ss['base_sq']-ss['cross']+.25*ss['proxy_sq'])/ss['base_sq'],
                        'cv_mean_error_relative':v['cv_mean_error_relative'],
                    }
            rows.append(row)
        report['records'][file]=rows
    (ROOT/'spatial_predictor_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified completed probe identities:',len(configurations))


if __name__=='__main__':
    main()
