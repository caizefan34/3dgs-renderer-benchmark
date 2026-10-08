PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("===== RIGOROUS bicycle =====")
d=json.load(open(base+"/G2_CERT/G2_RIGOROUS_bicycle_r30k.json"))
print("keys:",list(d.keys()))
print(json.dumps(d,indent=1)[:1400])
print()
print("===== EXISTING AGGREGATE G2_CONTINUOUS_DOMAIN_CERTIFICATE.json =====")
d=json.load(open(base+"/G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"))
print("keys:",list(d.keys()))
print(json.dumps(d,indent=1)[:1800])
PYEOF
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r5k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r30k.json | awk "{print \$6,\$7,\$8}"
