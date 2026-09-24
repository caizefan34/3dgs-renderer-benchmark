#!/usr/bin/env python3
import json
d = json.load(open("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c42/c42_p2_training_validation_30k.json"))
v = d["variant_A_baseline"]
print("Keys:", list(v.keys()))
fe = v.get("final_eval")
print("final_eval type:", type(fe), "value:", fe)
eps = v.get("eval_points", [])
print(f"eval_points: {len(eps)} entries")
for ep in eps[:5]:
    print(f"  {ep}")
print("...")
for ep in eps[-3:]:
    print(f"  {ep}")
