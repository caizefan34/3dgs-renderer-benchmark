"""Check drjohnson's contribution to the A0->C0 ladder dPSNR mean (+0.079)."""
import json

t = json.load(open("/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/p2_c0_vs_a0.json"))
rows = t["pairs"] if "pairs" in t else t["rows"]
ds = [(r["scene"], r["d_psnr"]) for r in rows]
ds.sort(key=lambda x: -abs(x[1]))
print("per-scene dPSNR (c0 vs a0), sorted by |d|:")
for s, d in ds:
    print(f"  {s:12s} {d:+.3f}")
mean = sum(d for _, d in ds) / len(ds)
without_dj = sum(d for s, d in ds if s != "drjohnson") / (len(ds) - 1)
print(f"\nmean all 13 = {mean:+.4f}")
print(f"mean without drjohnson = {without_dj:+.4f}")
dj = dict(ds)["drjohnson"]
print(f"drjohnson dPSNR = {dj:+.3f}")
