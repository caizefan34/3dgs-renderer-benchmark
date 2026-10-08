import json
d=json.load(open('/root/epic05_dpc0/artifacts/DPC0_RESULTS.json'))
print('type:', type(d))
if isinstance(d, dict):
    for k in d:
        v=d[k]
        if isinstance(v, dict):
            print('cell', k, '-> keys:', list(v.keys()))
            for kk in v:
                vv=v[kk]
                if isinstance(vv, dict):
                    print('   ', kk, '->', list(vv.keys())[:15])