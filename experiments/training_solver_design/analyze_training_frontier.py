"""Audit matched 30K A100 results; no cross-protocol SOTA claim."""

import hashlib
import json
from pathlib import Path
import statistics


ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'artifacts'/'training-all'/'results'
SCENES=('mipnerf360-room','mipnerf360-garden','mipnerf360-bicycle','tanks_and_temples-train')
METHODS=('gsplat','higs_full','higs_proposed')


def main():
    protocol_path=ROOT/'benchmark'/'higs-paper-protocol.json'
    protocol=json.loads(protocol_path.read_text())
    proposed=protocol['methods']['higs_proposed']['algorithm']
    cfg=proposed['trainer_cfg']
    assert cfg['higs_train_res_scale']==0.5 and cfg['higs_full_res_step']==24000
    assert cfg['visible_adam'] is True
    report={'scope':'Project 30K A100 results, matched within each scene/seed; not independent TurboGS reproduction',
            'protocol_source':str(protocol_path.relative_to(ROOT).as_posix()),
            'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'method_protocol':{method:protocol['methods'][method].get('algorithm') for method in METHODS},
            'records':[]}
    for scene in SCENES:
        rows={}
        for seed in range(3):
            reference=None
            for method in METHODS:
                path=RESULTS/f'primary_full_convergence--{method}--{scene}--a100--s{seed}.json'
                data=json.loads(path.read_text())
                assert data['status']=='complete'
                assert data['scene']==scene.replace('-', '/', 1) and data['method']==method and data['seed']==seed
                assert data['hardware']=='a100' and data['training']['iterations']==30000
                key=(data['provenance']['dataset_inventory_sha256'],data['hardware'],data['training']['iterations'])
                if reference is None: reference=key
                assert key==reference
                curve={r['iteration']:r for r in data['quality_curve']}
                assert 15000 in curve and 30000 in curve
                record={'scene':scene,'seed':seed,'method':method,
                        'source_result':path.relative_to(ROOT).as_posix(),
                        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'dataset_inventory_sha256':key[0],
                        'wall_seconds':data['performance']['wall_time_seconds'],
                        'psnr_db':data['quality']['psnr_db'],
                        'ssim':data['quality']['ssim'],
                        'lpips':data['quality']['lpips'],
                        'gaussians_15000':curve[15000]['num_GS'],
                        'gaussians_final':data['resources']['final_gaussian_count'],
                        'psnr_15000':curve[15000]['psnr_db'],
                        'lpips_15000':curve[15000]['lpips']}
                report['records'].append(record)
                rows[seed,method]=record
        def mean(method,key):
            return statistics.mean(rows[seed,method][key] for seed in range(3))
        summary={'scene':scene,'matched_seeds':[0,1,2],'methods':{},'proposed_minus_gsplat':{}}
        keys=('wall_seconds','psnr_db','ssim','lpips','gaussians_15000','gaussians_final','psnr_15000','lpips_15000')
        for method in METHODS: summary['methods'][method]={key:mean(method,key) for key in keys}
        for key in keys:
            deltas=[rows[seed,'higs_proposed'][key]-rows[seed,'gsplat'][key] for seed in range(3)]
            summary['proposed_minus_gsplat'][key]={'mean':statistics.mean(deltas),
                                                  'range':[min(deltas),max(deltas)]}
        report.setdefault('summary',[]).append(summary)
    out=Path(__file__).with_name('training_frontier_audit.json')
    out.write_text(json.dumps(report,indent=2)+'\n')
    for r in report['summary']:
        print(r['scene'],{k:round(v['mean'],3) for k,v in r['proposed_minus_gsplat'].items()})


if __name__=='__main__':
    main()
