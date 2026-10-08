import json
A = '/root/epic05_p12/artifacts/'
for name in ('P12A_SYSTEM_STATS.json', 'P12A_SYSTEM_QUALITY.json',
             'P12A_SYSTEM_TRAJECTORY.json', 'P12A_SYSTEM_VERDICT.json',
             'P12_FUSED_LOSS_LOCK.json'):
    d = json.load(open(A + name))
    print('==', name)
    print(json.dumps(d, indent=1)[:2200])
    print()
