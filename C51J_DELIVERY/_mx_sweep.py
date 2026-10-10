import json, hashlib, os, sys
AUDIT = "/mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/audit"
MAN = os.path.join(AUDIT, "MANIFEST_SHA256.json")
man = json.load(open(MAN))
files = man["files"]
print(f"manifest lists {len(files)} files")
ok = bad = missing = 0
for name, expected in sorted(files.items()):
    p = os.path.join(AUDIT, name)
    if not os.path.exists(p):
        print(f"  MISSING on MX: {name}"); missing += 1; continue
    h = hashlib.sha256(open(p,'rb').read()).hexdigest()
    if h == expected:
        ok += 1
    else:
        print(f"  HASH MISMATCH: {name}"); bad += 1
# also verify the txt manifest lines up
print(f"RESULT ok={ok} bad={bad} missing={missing}")
# validate all json files parse
for name in files:
    if name.endswith(".json"):
        try:
            json.load(open(os.path.join(AUDIT,name)))
        except Exception as e:
            print(f"  JSON PARSE FAIL {name}: {e}")
print("MX_INTEGRITY_SWEEP_DONE")
