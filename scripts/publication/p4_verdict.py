import json, math

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

def load(path):
    d = json.load(open(path))
    fe = d["final_eval"]
    return {"wall": d["timing"]["total_wall_s"], "psnr": fe["psnr"],
            "ssim": fe["ssim"], "lpips": fe["lpips"], "n": fe["n_gaussians"],
            "grade": d.get("timing_grade")}

cells = {}
for s in ("room", "bicycle", "garden"):
    cells[(s, "b1a", 0.1)] = load(f"{F30K}/b1a_{s}/results.json")
    cells[(s, "b1a", 0.3)] = load(f"{PUB}/b1ae03_{s}/results.json")
    cells[(s, "c0", 0.1)] = load(f"{PUB}/c0e01_{s}/results.json")
    cells[(s, "c0", 0.3)] = load(f"{F30K}/c0_{s}/results.json")

print(f"{'scene':9s} {'stack':5s} {'wall@0.1':>9s} {'wall@0.3':>9s} {'ratio':>7s} "
      f"{'dPSNR':>7s} {'dSSIM':>8s} {'dLPIPS':>8s} {'dN':>7s}")
summary = {}
for arm in ("b1a", "c0"):
    ratios, dpsnrs, nwithin = [], [], 0
    for s in ("room", "bicycle", "garden"):
        lo, hi = cells[(s, arm, 0.1)], cells[(s, arm, 0.3)]
        r = hi["wall"] / lo["wall"]
        dp = hi["psnr"] - lo["psnr"]
        ratios.append(r)
        dpsnrs.append(dp)
        if abs(dp) <= 0.10:
            nwithin += 1
        print(f"{s:9s} {arm:5s} {lo['wall']:9.0f} {hi['wall']:9.0f} {r:7.4f} "
              f"{dp:+7.3f} {hi['ssim']-lo['ssim']:+8.4f} {hi['lpips']-lo['lpips']:+8.4f} "
              f"{hi['n']/lo['n']:7.3f}")
    gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    summary[arm] = {"geomean_wall_ratio_03_over_01": gm,
                    "wall_effect_pct": abs(gm - 1) * 100,
                    "dpsnr_mean": sum(dpsnrs) / len(dpsnrs),
                    "n_within_010": nwithin}
    print(f"  -> {arm}: geomean ratio {gm:.4f} (effect {abs(gm-1)*100:.2f}%), "
          f"mean dPSNR {sum(dpsnrs)/len(dpsnrs):+.3f}, within 0.10 on {nwithin}/3\n")

# pre-registered rule
b1a_s, c0_s = summary["b1a"], summary["c0"]
half_speedup = (1.0685 - 1) / 2 * 100  # half the C0-vs-B1A speedup, in pct points
negligible = (b1a_s["wall_effect_pct"] <= 1.0 and c0_s["wall_effect_pct"] <= 1.0
              and b1a_s["n_within_010"] >= 2 and c0_s["n_within_010"] >= 2)
material_wall = c0_s["wall_effect_pct"] > half_speedup
f30k_mean_dpsnr = 0.070
material_quality = abs(c0_s["dpsnr_mean"]) > abs(f30k_mean_dpsnr)
verdict = ("EPS2D_NEGLIGIBLE" if negligible else
           "MATERIAL_CONFOUND" if (material_wall or material_quality) else
           "SMALL_BUT_PRESENT")
out = {
    "verdict": verdict,
    "cells": {f"{s}_{a}_{e}": v for (s, a, e), v in cells.items()},
    "summary": summary,
    "rule": {
        "EPS2D_NEGLIGIBLE": "geomean wall <=1% AND |dPSNR|<=0.10 on >=2/3 scenes both stacks",
        "MATERIAL_CONFOUND": f"C0 wall effect > {half_speedup:.2f}pct (half the 6.85pct speedup) "
                             f"or |C0 mean dPSNR| > {f30k_mean_dpsnr} (FINAL-30K mean)",
        "SMALL_BUT_PRESENT": "otherwise",
    },
    "rule_evaluation": {
        "b1a_wall_effect_pct": b1a_s["wall_effect_pct"],
        "c0_wall_effect_pct": c0_s["wall_effect_pct"],
        "c0_wall_material": material_wall,
        "c0_dpsnr_mean": c0_s["dpsnr_mean"],
        "c0_quality_material": material_quality,
    },
    "headline_implication": {
        "speed_component_from_eps2d": f"C0@0.3 is {c0_s['wall_effect_pct']:.2f}% faster than C0@0.1; "
                                      f"the 1.0685x headline (C0@0.3 vs B1A@0.1) would be "
                                      f"~{1.0685 * c0_s['geomean_wall_ratio_03_over_01']:.4f}x at matched "
                                      f"eps2d=0.1 (the asymmetry INFLATES the headline speedup by ~1.1%)",
        "quality_component": f"C0@0.3 is {c0_s['dpsnr_mean']:+.3f} dB vs C0@0.1 on average; at matched "
                             f"eps2d=0.1 the headline mean dPSNR +0.070 would become "
                             f"~{0.070 - c0_s['dpsnr_mean']:+.3f} (the asymmetry DEFLATES C0's measured "
                             f"quality; it is CONSERVATIVE against C0)",
        "conclusion_robustness": "the eps2d axis does not flip 'faster at quality-neutral': "
                                 "matched-eps2d would show a slightly smaller speedup (~1.057x) "
                                 "with slightly better C0 quality (~+0.170 dB); the two biases "
                                 "point in opposite directions and partially offset",
    },
}
with open(f"{PUB}/p4_eps2d_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
print("VERDICT:", verdict)
print(json.dumps(out["rule_evaluation"], indent=1))
print(json.dumps(out["headline_implication"], indent=1))
print("WROTE", f"{PUB}/p4_eps2d_verdict.json")
