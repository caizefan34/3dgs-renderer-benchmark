echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
ls -la --time-style=+%H:%M:%S /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r5k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r15k.json /home/liaoyuanjun/cgt_realscene/final/G1_RAW_RECORDS/g1_room_r30k.json | awk "{print \$6,\$7,\$8}"
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
# Save full existing aggregates locally for schema reference
for f in ["G2_CONTINUOUS_DOMAIN_CERTIFICATE.json","G3_GPU_MICROBENCH.json"]:
    d=json.load(open(base+"/"+f))
    open("/tmp/REF_"+f,"w").write(json.dumps(d,indent=1))
    print(f,"-> keys:",list(d.keys()))
print("=== G2 aggregate pooled + monotonicity + worst_tiles ===")
d=json.load(open(base+"/G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"))
print("pooled_all_scenes:",json.dumps(d.get("pooled_all_scenes"),indent=1)[:800])
print("monotonicity:",json.dumps(d.get("monotonicity"),indent=1)[:600])
print("worst_tiles:",json.dumps(d.get("worst_tiles"),indent=1)[:600])
PYEOF
