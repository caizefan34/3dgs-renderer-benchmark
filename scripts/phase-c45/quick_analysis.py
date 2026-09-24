#!/usr/bin/env python3
import json
from pathlib import Path

d = Path("results/a100/phase-c45")
print(f"{'File':<50} {'Time(s)':>8} {'PSNR':>8} {'SSIM':>8} {'GS':>10}")
print("-" * 90)
for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    if "eval_points" in data and data["eval_points"]:
        final = data["eval_points"][-1]
        tt = data.get("total_train_time_s", 0)
        name = f.name
        psnr = final["psnr"]
        ssim = final["ssim"]
        gs = final["gaussians"]
        print(f"{name:<50} {tt:>8.1f} {psnr:>8.2f} {ssim:>8.4f} {gs:>10,}")
    elif "post_c44" in str(f):
        # Profile data
        t = data.get("post_c44", {})
        c = data.get("comparison", {})
        print(f"{f.name:<50} PROFILE: total_sep={t.get('total_ms',0):.1f}ms speedup={c.get('speedup_x',0):.2f}x")
