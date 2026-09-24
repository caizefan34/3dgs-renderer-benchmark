import json
t = json.load(open('/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p1_b1_vs_b1a.json'))
a = t['aggregate']
print("b1 vs b1a (13 scenes): pairs=%d geomean=%.4f faster=%d slower=%d mean_dPSNR=%+.3f N_geo=%.4f" % (
    a['n_pairs'], a['speedup_geomean'], a['n_faster'], a['n_slower'],
    a['mean_d_psnr'], a.get('n_geo_ratio', 0)))
for r in t.get('pairs', []):
    print("  %-11s ratio=%.4f dPSNR=%+.3f" % (
        r['scene'], r.get('speedup', r.get('ratio', 0)), r['d_psnr']))
