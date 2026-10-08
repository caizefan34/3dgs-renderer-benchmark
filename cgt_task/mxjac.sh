/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/smoke.json'))
recs=d['records']
# jac_abs is RMS per-element Jacobian error (well-posed absolute). Inspect distribution.
for rank in [4,8,16]:
    rows=[r for r in recs if r['rank']==rank]
    ja=np.array([r['jac_abs'] for r in rows])
    rr=np.array([r['resp_range'] for r in rows])
    ng=np.array([r['n_gauss'] for r in rows])
    psnr=np.array([r['psnr'] for r in rows]); mx=np.array([r['max_abs'] for r in rows])
    print(f"\n== rank {rank} ==")
    print(f"  jac_abs (RMS/elem) med={np.median(ja):.2e} p25={np.quantile(ja,.25):.2e} p75={np.quantile(ja,.75):.2e} min={ja.min():.2e} max={ja.max():.2e}")
    print(f"  resp_range         med={np.median(rr):.2e} min={rr.min():.2e} max={rr.max():.2e}")
    # relationship: jac_abs vs resp_range/px_extent
    px_ext=14.0
    nat=rr/px_ext
    print(f"  natural_grad(resp/pxext) med={np.median(nat):.2e}")
    ratio=ja/np.maximum(nat,1e-12)
    print(f"  jac_abs/natural_grad med={np.median(rat):.3f} p75={np.quantile(rat,.75):.3f} max={ratio.max():.3f}")
    # candidate well-posed gate: jac_abs <= 0.02/14 (value tol / px extent)
    for thr in [0.02/14, 0.005, 0.002, 0.001]:
        ok=np.mean(ja<=thr)*100
        print(f"    jac_abs<={thr:.4f}: {ok:.0f}% pass")
PYEOF