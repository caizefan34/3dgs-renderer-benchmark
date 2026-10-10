#!/usr/bin/env python3
import json, os
DST = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_deliverables"
FREEZE = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_ENV_FREEZE.json"
ht = json.load(open(f"{DST}/S1A_HT_CORRECTNESS.json"))
print("=== HT gates ===")
print(json.dumps(ht.get("gates", {}), indent=2)[:1500])
print("overall_pass:", ht.get("overall_pass"))
print("ht_p1_equals_dense:", ht.get("ht_p1_equals_dense"))
print("partition_additivity:", ht.get("partition_additivity"))
print("mc_consistency:", json.dumps(ht.get("mc_consistency",{}))[:400])
print("unbiasedness:", json.dumps(ht.get("unbiasedness",{}))[:400])
print("render_quality_note:", ht.get("render_quality_note"))

# gpu_snapshots
gsnap = f"{DST}/gpu_snapshots"
print("\n=== gpu_snapshots ===", os.listdir(gsnap) if os.path.isdir(gsnap) else "MISSING")
print("=== logs ===", os.listdir(f"{DST}/logs"))
print("=== raw ===", os.listdir(f"{DST}/raw"))
print("=== commands ===", os.listdir(f"{DST}/commands"))
print("=== top-level files ===")
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST,f)
    if os.path.isfile(p): print(f"  {f} ({os.path.getsize(p)}B)")
