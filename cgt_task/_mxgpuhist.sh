echo "=== g1 log: any GPU mention ==="
grep -i "cuda\|gpu\|device" /home/liaoyuanjun/cgt_realscene/final/logs/g1_garden_garden.log 2>/dev/null | head -3
echo "=== find rigorous launcher in bash history / scripts ==="
grep -rn "g2_rigorous_certificate" /home/liaoyuanjun/cgt_realscene/final/*.sh /home/liaoyuanjun/cgt_realscene/*.sh 2>/dev/null | head
echo "=== check /tmp launchers ==="
ls -la /tmp/launch_*.sh /tmp/*rig*.sh 2>/dev/null | head
echo "=== rigorous log GPU (g2_certificate prints?) ==="
head -3 /home/liaoyuanjun/cgt_realscene/final/logs/g3r_garden_r30k.log
echo "=== ps history: what launched rigorous? check g2_rigorous output header ==="
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G2_CERT/G2_RIGOROUS_room_r15k.json'));print({k:d[k] for k in d if k not in ['records','per_rank']})"
