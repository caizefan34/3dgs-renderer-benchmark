/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
recs=d['records']
print('scene',d['scene'],'n',d['n_records'],'gate',d['gate'])
for rank in [4,8,16]:
    rows=[r for r in recs if r['rank']==rank]
    psnr=np.array([r['psnr'] for r in rows]); mx=np.array([r['max_abs'] for r in rows])
    jacrel=np.array([r['jac_rel'] for r in rows]); jacwp=np.array([r['jac_wp'] for r in rows])
    jn=np.array([r['teacher_jac_norm'] for r in rows]); rr=np.array([r['resp_range'] for r in rows])
    ng=np.array([r['n_gauss'] for r in rows]); tw=np.array([r['tile_work'] for r in rows])
    print(f"\n== rank {rank}: n={len(rows)} n_gauss[min/med/max]={ng.min()}/{int(np.median(ng))}/{ng.max()} tile_work[min/med/max]={tw.min()}/{int(np.median(tw))}/{tw.max()}")
    print(f"   psnr   med={np.median(psnr):.1f} min={psnr.min():.1f} max={psnr.max():.1f}")
    print(f"   max_abs med={np.median(mx):.4f} p90={np.quantile(mx,0.9):.4f} max={mx.max():.4f}")
    print(f"   teacher_jac_norm med={np.median(jn):.4f} min={jn.min():.4f} max={jn.max():.4f}")
    print(f"   resp_range      med={np.median(rr):.4f} min={rr.min():.4f} max={rr.max():.4f}")
    print(f"   jac_rel (raw)   med={np.median(jacrel):.3f} min={jacrel.min():.3f} max={jacrel.max():.3f}")
    print(f"   jac_wp (scale-aware) med={np.median(jacwp if False else jacwp):.3f} min={jacwp.min():.3f} max={jacwp.max():.3f}" if False else f"   jac_wp med={np.median(jacwp):.3f} min={jacwp.min():.3f} max={jacwp.max():.3f}")
    # combined gate pass
    ok=np.array([r['psnr']>=30 and r['max_abs']<=0.02 and r['jac_wp']<=0.5 for r in rows])
    print(f"   GATE(psnr>=30 & max_abs<=0.02 & jac_wp<=0.5): {int(ok.sum())}/{len(rows)} = {ok.mean()*100:.0f}%")
    ok2=np.array([r['psnr']>=30 and r['max_abs']<=0.02 for r in rows])
    print(f"   VALUE-ONLY gate(psnr>=30 & max_abs<=0.02): {int(ok2.sum())}/{len(rows)} = {ok2.mean()*100:.0f}%")
PYEOF