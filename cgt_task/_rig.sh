PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
d=json.load(open("/home/liaoyuanjun/cgt_realscene/final/G2_CERT/G2_RIGOROUS_bicycle_r30k.json"))
print("record[0] keys:",list(d["records"][0].keys()))
print("record[0]:",json.dumps(d["records"][0],indent=1))
print("per_rank[4] keys:",list(d["per_rank"]["4"].keys()))
print("headline-ish top:",{k:d[k] for k in ["gate","n_tiles","value_gate_max_abs","validated"] if k in d})
PYEOF
