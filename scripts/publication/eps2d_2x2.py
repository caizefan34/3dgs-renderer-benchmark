"""P4 eps2d 2x2 assembly: {C0, B1A} x {eps2d 0.1, 0.3} on room/bicycle/garden.
Reuse corners: FINAL-30K c0 (0.3) + b1a (0.1); new corners from pub_runs.
Pre-registered conclusion rule (variant_definitions.json):
  EPS2D_NEGLIGIBLE: geomean wall delta <= 1% AND |dPSNR| <= 0.10 dB on >= 2/3 scenes, both stacks
  MATERIAL_CONFOUND: eps2d effect on C0 wall > half the C0-vs-B1A speedup, or dPSNR shifts
                     beyond FINAL-30K mean
  SMALL_BUT_PRESENT: otherwise
"""
import json, math, os

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
SCENES = ["room", "bicycle", "garden"]


def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    fe = d["final_eval"]
    return {"wall_s": d["timing"]["total_wall_s"], "psnr": fe["psnr"],
            "ssim": fe["ssim"], "lpips": fe["lpips"], "n": fe["n_gaussians"],
            "arm": d.get("arm"), "eps2d_override": d.get("eps2d_override")}


cells = {}
for s in SCENES:
    cells[(s, "b1a", 0.1)] = load(f"{F30K}/b1a_{s}/results.json")          # FINAL-30K reuse
    cells[(s, "c0", 0.3)] = load(f"{F30K}/c0_{s}/results.json")            # FINAL-30K reuse
    cells[(s, "b1a", 0.3)] = load(f"{PUB}/b1ae03_{s}/results.json")        # new corner
    cells[(s, "c0", 0.1)] = load(f"{PUB}/c0e01_{s}/results.json")          # new corner

print(f"{'scene':9s} {'cell':16s} {'wall_s':>7s} {'PSNR':>7s} {'LPIPS':>7s} {'N':>9s} {'src':>6s}")
for (s, arm, e), v in cells.items():
    if v is None:
        print(f"{s:9s} {arm+'@'+str(e):16s} {'PENDING':>7s}")
        continue
    src = "F30K" if (s, arm, e) in {(x, a, ee) for x in SCENES for a, ee in (("b1a", 0.1), ("c0", 0.3))} else "new"
    print(f"{s:9s} {arm+'@'+str(e):16s} {v['wall_s']:7.0f} {v['psnr']:7.3f} {v['lpips']:7.4f} {v['n']:9d} {src:>6s}")

print("\n== eps2d effect per stack (complete scenes only) ==")
results = {}
for s in SCENES:
    for arm in ("b1a", "c0"):
        lo = cells[(s, arm, 0.1)]   # eps2d 0.1
        hi = cells[(s, arm, 0.3)]   # eps2d 0.3
        if lo and hi:
            wall_ratio = hi["wall_s"] / lo["wall_s"]
            dpsnr = hi["psnr"] - lo["psnr"]
            results[(s, arm)] = {"wall_ratio_03_over_01": wall_ratio,
                                 "dpsnr_03_minus_01": dpsnr}
            print(f"  {s:9s} {arm}: wall 0.3/0.1 = {wall_ratio:.4f}  "
                  f"dPSNR(0.3-0.1) = {dpsnr:+.3f} dB")

# geomean wall effect + PSNR test per stack
for arm in ("b1a", "c0"):
    rs = [v for (s, a), v in results.items() if a == arm]
    if len(rs) >= 2:
        gm = math.exp(sum(math.log(v["wall_ratio_03_over_01"]) for v in rs) / len(rs))
        n_within = sum(1 for v in rs if abs(v["dpsnr_03_minus_01"]) <= 0.10)
        print(f"\n{arm}: geomean wall(0.3/0.1) = {gm:.4f} over {len(rs)} scenes; "
              f"|dPSNR|<=0.10 on {n_within}/{len(rs)}")
