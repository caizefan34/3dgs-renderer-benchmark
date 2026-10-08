PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("===== FULL CGT_MASTER_VERDICT.json =====")
print(json.dumps(json.load(open(base+"/CGT_MASTER_VERDICT.json")),indent=1))
PYEOF
