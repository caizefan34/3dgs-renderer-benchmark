PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("===== G3 bicycle fresh =====")
d=json.load(open(base+"/G1_RAW_RECORDS/g3_bicycle_r30k.json"))
print("keys:",list(d.keys()))
print("per_rank type:",type(d.get("per_rank")))
print("per_rank:",json.dumps(d.get("per_rank"),indent=1)[:600])
print("headline:",json.dumps(d.get("headline"),indent=1)[:400])
print()
print("===== DENSE bicycle =====")
d=json.load(open(base+"/G2_CERT/G2_CONTINUOUS_DOMAIN_CERTIFICATE_bicycle_r30k.json"))
print("keys:",list(d.keys()))
print(json.dumps(d,indent=1)[:1200])
PYEOF
