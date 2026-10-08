import json
c = json.load(open('/root/epic05_p12/artifacts/P12_CONTAMINATION.json'))
for camp in ('p12a', 'p12b'):
    runs = c['campaigns'][camp]['runs']
    print(camp, 'n_runs =', len(runs))
    for rid in sorted(runs):
        r = runs[rid]
        share = r['own_samples'] / r['n_samples'] if r['n_samples'] else 0
        wall = (r['ts_end'] - r['ts_start']) / 60
        if r['n_samples'] < 10 or share < 0.8 or not r['clean']:
            print(f"  FLAG {rid} gpu{r['gpu']} samples={r['n_samples']} own_share={share:.2f} clean={r['clean']}")
    # garden rerun specifically
    for rid in ('p12b_a1f_s42_garden', 'p12b_a0f_s42_garden'):
        r = runs.get(rid)
        if r:
            share = r['own_samples'] / r['n_samples'] if r['n_samples'] else 0
            print(f"  RERUN {rid} gpu{r['gpu']} samples={r['n_samples']} own_share={share:.2f} clean={r['clean']}")
    print('  preserved:', c['campaigns'][camp].get('preserved_contaminated_attempts'))
