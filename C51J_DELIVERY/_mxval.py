import json
for f in ["RUN_STATUS.json","MANIFEST_SHA256.json","C51J_FINAL_VERDICT.json","C51J_FROZEN_PROTOCOL.json","C51J_1K_SMOKE.json"]:
    json.load(open(f))
print("MX_ALL_JSON_OK")
