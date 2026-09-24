#!/usr/bin/env python3
import json

# Check s23 garden for provenance and C42 info
path = "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/garden/s0.750_30k.json"
d = json.load(open(path))
print("Keys:", list(d.keys()))
for k in d:
    if isinstance(v := d[k], str):
        print(f"  {k}: {v}")
    elif isinstance(v, dict) and len(v) < 20:
        print(f"  {k}: {json.dumps(v, indent=2)[:300]}")
    elif isinstance(v, (int, float, bool)):
        print(f"  {k}: {v}")

# Check training_metrics for PSNR/SSIM trajectory
if "training_metrics" in d:
    tm = d["training_metrics"]
    ckpts = tm.get("checkpoints", {})
    for it in sorted(ckpts.keys(), key=int):
        if int(it) in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
            c = ckpts[it]
            print(f"  iter={it}: PSNR={c.get('psnr','?')} SSIM={c.get('ssim','?')} L1={c.get('l1','?')}")

# Check for config/trainer info
if "config" in d:
    print(f"\nConfig: {d['config']}")
if "mode" in d:
    print(f"Mode: {d['mode']}")
if "loss" in d:
    print(f"Loss: {d['loss']}")
