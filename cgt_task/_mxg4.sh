PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("===== G4_BATCHING_CEILING.json =====")
print(json.dumps(json.load(open(base+"/G4_BATCHING_CEILING.json")),indent=1)[:1500])
PYEOF
