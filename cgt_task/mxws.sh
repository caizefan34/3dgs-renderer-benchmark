echo "=== workspace ==="
ls -la /home/liaoyuanjun/cgt_realscene/ 2>&1
echo "--- src ---"
ls -la /home/liaoyuanjun/cgt_realscene/src/ 2>&1
echo "--- logs ---"
ls -la /home/liaoyuanjun/cgt_realscene/logs/ 2>&1
echo "=== c53 room30k full ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import json
d=json.load(open('/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/c53_workload_statistics.json'))
print('cohort_size',d.get('cohort_size'))
for k,c in d['checkpoints'].items():
    print(k,'n_gauss',c['n_gauss'],'n_visible',c['n_visible'],'tiles_mean',round(c['tiles_mean'],2),'tiles_median',c['tiles_median'],'tiles_p99',c['tiles_p99'],'tiles_max',c['tiles_max'],'top10_mass',round(c['tiles_top10_mass'],3))
PYEOF