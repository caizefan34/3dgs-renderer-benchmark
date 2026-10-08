#!/bin/bash
# assemble_g1.sh : build deliverables from G1 raw outputs (run on MX after jobs done)
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
ROOT=/home/liaoyuanjun/cgt_realscene
OUT=$ROOT/logs/g1
FIN=/home/liaoyuanjun/cgt_realscene/final
rm -rf "$FIN"; mkdir -p "$FIN/G1_RAW_RECORDS" "$FIN/SCRIPTS" "$FIN/PROVENANCE"
# 1. analysis
cd "$ROOT/src"
$PY analyze_g1.py --inputs "$OUT/g1_*.json" --out "$FIN/G1_REAL_SCENE_COMPRESSIBILITY.json" 2>&1
# 2. copy raw records + logs
cp "$OUT"/g1_*.json "$FIN/G1_RAW_RECORDS/" 2>/dev/null
cp "$OUT"/g1_*.log "$FIN/G1_RAW_RECORDS/" 2>/dev/null
cp "$OUT"/g1_test.json "$FIN/G1_RAW_RECORDS/" 2>/dev/null
cp "$OUT"/smoke.json "$FIN/G1_RAW_RECORDS/g1_smoke_20tiles.json" 2>/dev/null
cp g1_compressibility.py analyze_g1.py "$FIN/SCRIPTS/" 2>/dev/null
cp run_g1.sh launch_all.sh "$FIN/SCRIPTS/" 2>/dev/null
# 3. per-view CSV summary
$PY - <<'PYEOF'
import json, glob, csv, os
fin=os.environ.get('FIN','/home/liaoyuanjun/cgt_realscene/final')
rows=[]
for fp in sorted(glob.glob(f'{fin}/G1_RAW_RECORDS/g1_*.json')):
    if 'smoke' in fp or 'test' in fp: continue
    d=json.load(open(fp))
    for r in d['records']:
        if 'psnr' not in r: continue
        rows.append(dict(scene=d['scene'],ckpt=d['ckpt'].split('/')[-1],view=r['view'],rank=r['rank'],
                         tile=f"({r['tile_ti']},{r['tile_tj']})",n_gauss=r['n_gauss'],tile_work=r['tile_work'],
                         psnr=round(r['psnr'],2),max_abs=round(r['max_abs'],5),jac_abs=f"{r['jac_abs']:.3e}",
                         resp_range=f"{r['resp_range']:.3e}"))
with open(f'{fin}/G1_PER_VIEW_SUMMARY.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('CSV rows',len(rows))
PYEOF
# 4. provenance
{
echo "host: $(hostname)"
echo "date_utc: $(date -u +%ISO8601)"
echo "python: $($PY --version 2>&1)"
$PY -c "import torch,gsplat,numpy,scipy;print('torch',torch.__version__,'gsplat',gsplat.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"
echo "gpu_used: 5,6,7 (verified idle before run)"
nvidia-smi --query-gpu=index,memory.used,name --format=csv,noheader
} > "$FIN/PROVENANCE/ENVIRONMENT.txt" 2>&1
echo "ASSEMBLE DONE -> $FIN"
ls -la "$FIN"