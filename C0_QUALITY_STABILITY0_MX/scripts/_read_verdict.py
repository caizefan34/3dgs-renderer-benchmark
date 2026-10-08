import json
v = json.load(open("/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX/QUALITY_VERDICT.json"))
print(json.dumps(v, indent=2))
