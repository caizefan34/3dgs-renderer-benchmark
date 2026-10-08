cd /home/liaoyuanjun/cgt_realscene/final/SCRIPTS
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
echo "=== G2 dense aggregate ==="
$PY build_g2_agg.py 2>&1
echo "=== G2 rigorous aggregate ==="
$PY build_g2_rig_agg.py 2>&1
echo "=== G3 aggregate ==="
$PY build_g3_agg.py 2>&1
