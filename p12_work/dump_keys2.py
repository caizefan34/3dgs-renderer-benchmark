import json
A = '/root/epic05_p12/artifacts/'
for camp in ('P12A_SYSTEM', 'P12B_F9'):
    d = json.load(open(A + camp + '_QUALITY.json'))
    print('==', camp, '_QUALITY.json keys:', list(d.keys()))
    for k, v in d.items():
        if k != 'per_scene':
            print('  ', k, '=', json.dumps(v))
    print()
