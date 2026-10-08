/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import json, numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_realscene/logs/g1/g1_room_r15k.json'))
recs=[r for r in d['records'] if 'psnr' in r and r['rank']==16]
ja=np.array([r['jac_abs'] for r in recs])           # RMS/elem student-teacher grad err
tjn=np.array([r['teacher_jac_norm'] for r in recs]) # teacher grad Frobenius
rr=np.array([r['resp_range'] for r in recs])
ng=np.array([r['n_gauss'] for r in recs])
tw=np.array([r['tile_work'] for r in recs])
psnr=np.array([r['psnr'] for r in recs]); mx=np.array([r['max_abs'] for r in recs])
# per-element teacher grad RMS
te=tjn/np.sqrt(128*8)
# well-posed RELATIVE gradient error: student err / max(teacher per-elem grad, floor)
floor=0.02/16
rel_wp = ja/np.maximum(te, floor)
print("rank16 room r15k: n=",len(recs))
print(f"  psnr med={np.median(psnr):.1f} min={psnr.min():.1f} | max_abs med={np.median(mx):.4f} max={mx.max():.4f}")
print(f"  jac_abs(RMS/elem) med={np.median(ja):.2e} p90={np.quantile(ja,.9):.2e} max={ja.max():.2e}")
print(f"  teacher per-elem grad RMS med={np.median(te):.2e} p90={np.quantile(te,.9):.2e} max={te.max():.2e}")
print(f"  REL grad err (ja/max(te,floor)) med={np.median(rel_wp):.3f} p75={np.quantile(rel_wp,.75):.3f} p90={np.quantile(rel_wp,.9):.3f} max={rel_wp.max():.3f}")
# strict relative gate: ja <= 0.5*te  (i.e. rel_wp<=0.5), with floor for flat
strict = ja <= np.maximum(0.5*te, floor)
print(f"  STRICT rel gate (ja<=max(0.5*te,{floor:.1e})) pass: {strict.sum()}/{len(recs)} = {strict.mean()*100:.0f}%  work-w={tw[strict].sum()/tw.sum():.3f}")
# value gate
vg = (mx<=0.02)&(psnr>=30)
print(f"  value gate pass: {vg.sum()}/{len(recs)} = {vg.mean()*100:.0f}% work-w={tw[vg].sum()/tw.sum():.3f}")
# combined strict
both = vg & strict
print(f"  VALUE+STRICT pass: {both.sum()}/{len(recs)} = {both.mean()*100:.0f}% work-w={tw[both].sum()/tw.sum():.3f}")
# by work decile
order=np.argsort(tw); rank=np.empty(len(tw),int); rank[order]=np.arange(len(tw))
dec=np.minimum((rank*10)//len(tw),9)
print("  by work decile (strict-rel pass rate / work-w):")
for dd in range(10):
    m=dec==dd
    print(f"    d{dd}: n={m.sum():3d} ng_med={int(np.median(ng[m])):4d} strict={strict[m].mean()*100:3.0f}% val={vg[m].mean()*100:3.0f}% both={both[m].mean()*100:3.0f}%")
PYEOF