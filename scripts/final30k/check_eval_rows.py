import json

for run in ("c0_bonsai", "c0_bicycle"):
    r = json.load(open(f"/mnt/storage_pool/liaoyuanjun/final30k_runs/{run}/results.json"))
    rows = r["eval_rows"]
    print(run, "eval_rows:", len(rows))
    print("  keys:", sorted(rows[0].keys()))
    for row in rows[:3]:
        print("   step=%s tag=%s psnr=%.2f lpips=%s" % (
            row["step"], row["tag"], row["psnr"], row.get("lpips")))
