import json
t = json.load(open('/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p1_b0_vs_b1a.json'))
a = t['aggregate']
print("b1a vs b0 (candidate b0; speedup = wall_b1a/wall_b0, <1 = B0 slower):")
print("  pairs=%d geomean=%.4f faster=%d slower=%d mean_dPSNR=%+.3f" % (
    a['n_pairs'], a['speedup_geomean'], a['n_faster'], a['n_slower'], a['mean_d_psnr']))
for r in t.get('rows', []):
    if r.get('status') == 'ok':
        print("  %-11s B1A=%.1f B0=%.1f ratio=%.4f dPSNR=%+.3f N_ratio=%.4f" % (
            r['scene'], r['wall_base'], r['wall_cand'], r['speedup'],
            r['d_psnr'], r['n_ratio']))
    else:
        print("  %-11s pending" % r['scene'])
