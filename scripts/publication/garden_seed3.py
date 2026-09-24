"""Garden accutile effect across seeds 42/43/44 (C-11 evidence).
b1a = gsplat+accutile, b1 = pristine gsplat; effect = b1a - b1 (negative = accutile helps).
Also the wall-time edge per seed.
"""
import json

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

def res(path):
    d = json.load(open(f"{path}/results.json"))
    return {"psnr": d["final_eval"]["psnr"],
            "ssim": d["final_eval"]["ssim"],
            "lpips": d["final_eval"].get("lpips"),
            "wall": d["timing"]["total_wall_s"]}

rows = {}
rows[42] = (res(f"{F30K}/b1a_garden"), res(f"{PUB}/b1_garden"))
rows[43] = (res(f"{PUB}/b1as43_garden"), res(f"{PUB}/b1s43_garden"))
rows[44] = (res(f"{PUB}/b1as44_garden"), res(f"{PUB}/b1s44_garden"))

print(f"{'seed':>4s} {'b1a PSNR':>9s} {'b1 PSNR':>9s} {'effect':>8s} {'b1a wall':>10s} {'b1 wall':>10s} {'wall edge':>9s}")
effs, edges = [], []
for s, (a, b) in rows.items():
    eff = a["psnr"] - b["psnr"]  # positive = accutile helps quality
    edge = (b["wall"] - a["wall"]) / b["wall"] * 100  # positive = accutile faster
    effs.append(eff)
    edges.append(edge)
    print(f"{s:4d} {a['psnr']:9.3f} {b['psnr']:9.3f} {eff:+8.3f} {a['wall']:10.1f} {b['wall']:10.1f} {edge:+8.2f}%")

print(f"\naccutile garden effect (b1a-b1, positive = accutile helps): mean {sum(effs)/3:+.3f} dB, range [{min(effs):+.3f}, {max(effs):+.3f}]")
print(f"accutile garden wall edge (positive = accutile faster): mean {sum(edges)/3:+.2f}%, range [{min(edges):+.2f}%, {max(edges):+.2f}%]")
sign_consistent = all(e > 0 for e in effs) and all(e > 0 for e in edges)
print(f"sign-consistent across seeds (quality AND wall): {sign_consistent}")

out = {"claim": "C-11 garden accutile effect, seeds 42/43/44",
       "per_seed": {str(s): {"b1a_psnr": a["psnr"], "b1_psnr": b["psnr"],
                             "effect_db": a["psnr"] - b["psnr"],
                             "wall_edge_pct": (b["wall"] - a["wall"]) / b["wall"] * 100}
                    for s, (a, b) in rows.items()},
       "effect_mean_db": sum(effs) / 3,
       "effect_range_db": [min(effs), max(effs)],
       "sign_consistent": sign_consistent}
json.dump(out, open(f"{PUB}/garden_seed3_table.json", "w"), indent=1)
print("WROTE garden_seed3_table.json")
