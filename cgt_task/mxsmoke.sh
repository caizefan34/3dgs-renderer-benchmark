echo "=== g1 file info ==="
wc -c /home/liaoyuanjun/cgt_realscene/src/g1_compressibility.py
md5sum /home/liaoyuanjun/cgt_realscene/src/g1_compressibility.py
echo "=== has new metrics? ==="
grep -c "jac_wp\|resp_range\|grad_scale\|teacher_jac_norm" /home/liaoyuanjun/cgt_realscene/src/g1_compressibility.py
echo "=== run smoke (20 tiles, room 30k) ==="
cd /home/liaoyuanjun/cgt_realscene/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
timeout 400 $PY g1_compressibility.py --scene room --ckpt /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt --data /tmp/3dgs-renderer-benchmark/data --views 1 --iterations 80 --max-tiles-per-view 20 --out /home/liaoyuanjun/cgt_realscene/logs/smoke.json 2>&1 | tail -25