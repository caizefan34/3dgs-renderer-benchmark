import json, os, glob
log = "clockgs/runs/c0/c0_log.jsonl"
rows = {}
for line in open(log, encoding="utf-8"):
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get("type") == "EVAL":
        key = f"{o['scene']}_kgeo{o['k_geo']}"
        rows[key] = o
# also scan per-run drift.json + out files for any missing
print("keys:", sorted(rows.keys()))
for sc in ["room", "garden", "bicycle"]:
    c = rows.get(f"{sc}_kgeo1")
    if not c:
        print(f"== {sc}: NO CONTROL YET")
        continue
    print(f"== {sc} control: PSNR={c['psnr']:.4f} SSIM={c['ssim']:.4f} N={c['N_final']}")
    for k in [2, 4]:
        m = rows.get(f"{sc}_kgeo{k}")
        if not m:
            print(f"  k_geo={k}: MISSING")
            continue
        dp = m["psnr"] - c["psnr"]; ds = m["ssim"] - c["ssim"]
        gate = "PASS" if (abs(dp) <= 0.15 and abs(ds) <= 0.005) else "FAIL"
        print(f"  k_geo={k}: PSNR={m['psnr']:.4f} dPSNR={dp:+.4f} SSIM={m['ssim']:.4f} dSSIM={ds:+.4f} -> {gate}")
json.dump(rows, open("clockgs/runs/c0/c0_eval_all.json", "w"), indent=2)
print("saved c0_eval_all.json")
