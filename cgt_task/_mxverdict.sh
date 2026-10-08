PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("===== CGT_MASTER_VERDICT.json =====")
print(json.dumps(json.load(open(base+"/CGT_MASTER_VERDICT.json")),indent=1)[:2500])
print("===== G2_G5_STATUS.json =====")
print(json.dumps(json.load(open(base+"/G2_G5_STATUS.json")),indent=1)[:2000])
PYEOF
