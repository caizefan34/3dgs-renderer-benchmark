import json

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

def get(path):
    d = json.load(open(path))
    fe = d["final_eval"]
    return {"wall": d["timing"]["total_wall_s"], "psnr": fe["psnr"],
            "ssim": fe["ssim"], "n": fe["n_gaussians"],
            "grade": d.get("timing_grade"), "seed": d.get("seed")}

print("== GARDEN seed-43 pair (the decisive B1-gate evidence) ==")
pairs = {
    "b1a s42": f"{F30K}/b1a_garden/results.json",
    "b1a s43": f"{PUB}/b1as43_garden/results.json",
    "b1  s42": f"{PUB}/b1_garden/results.json",
    "b1  s43": f"{PUB}/b1s43_garden/results.json",
}
r = {k: get(v) for k, v in pairs.items()}
for k, v in r.items():
    print(f"  {k}: PSNR={v['psnr']:.3f} wall={v['wall']:.0f}s N={v['n']:,} "
          f"grade={v['grade']} seed_field={v['seed']}")
d42 = r["b1  s42"]["psnr"] - r["b1a s42"]["psnr"]
d43 = r["b1  s43"]["psnr"] - r["b1a s43"]["psnr"]
print(f"  b1-b1a delta @s42: {d42:+.3f} dB   @s43: {d43:+.3f} dB")
print(f"  seed-to-seed spread: b1a {r['b1a s43']['psnr']-r['b1a s42']['psnr']:+.3f}, "
      f"b1 {r['b1  s43']['psnr']-r['b1  s42']['psnr']:+.3f}")
w42 = r["b1a s42"]["wall"] / r["b1  s42"]["wall"]
w43 = r["b1a s43"]["wall"] / r["b1  s43"]["wall"]
print(f"  accutile wall edge @s42: {1-w42:+.2%}  @s43: {1-w43:+.2%}")

print("\n== B0 gate (fixed trainer) ==")
for s in ("room", "bicycle", "garden"):
    try:
        b0 = get(f"{PUB}/b0_{s}/results.json")
        b1 = get(f"{PUB}/b1_{s}/results.json")
        b1a = get(f"{F30K}/b1a_{s}/results.json")
        print(f"  {s:9s} b0: wall={b0['wall']/60:.1f}min PSNR={b0['psnr']:.3f} N={b0['n']:,} "
              f"grade={b0['grade']}")
        print(f"  {'':9s}   b1a/b0 wall ratio={b1a['wall']/b0['wall']:.3f}  "
              f"b1/b0={b1['wall']/b0['wall']:.3f}  dPSNR(b0-b1a)={b0['psnr']-b1a['psnr']:+.3f}")
    except FileNotFoundError as e:
        print(f"  {s}: pending ({e.filename})")
