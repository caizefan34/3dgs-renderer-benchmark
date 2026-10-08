import json
c = json.load(open('/root/epic05_p12/artifacts/P12_CONTAMINATION.json'))
runs = c['campaigns']['p12a']['runs']
print(f"{'run_id':32s} {'gpu':>3s} {'samples':>7s} {'own_share':>9s} {'wall_min':>8s}")
bad = []
for rid, r in sorted(runs.items()):
    share = r['own_samples'] / r['n_samples'] if r['n_samples'] else 0
    wall = (r['ts_end'] - r['ts_start']) / 60
    print(f"{rid:32s} {r['gpu']:>3d} {r['n_samples']:>7d} {share:>9.2f} {wall:>8.1f}")
    if r['n_samples'] < 10 or share < 0.8:
        bad.append(rid)
print('LOW-COVERAGE RUNS:', bad or 'none')
