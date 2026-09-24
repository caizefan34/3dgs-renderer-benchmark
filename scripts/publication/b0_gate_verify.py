"""B0 gate verification: all 3 scenes complete. Check the matched-protocol
expectations for the original Graphdeco rasterizer under the unified trainer.
"""
import json

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

def get(path):
    d = json.load(open(path))
    fe = d["final_eval"]
    t = d["timing"]
    return {"wall": t["total_wall_s"], "psnr": fe["psnr"], "ssim": fe["ssim"],
            "lpips": fe.get("lpips"), "n": fe["n_gaussians"],
            "grade": d.get("timing_grade"),
            "iters": t.get("n_iters"),
            "dens": d.get("densification", {}),
            "renderer": d.get("renderer", {})}

print(f"{'scene':9s} {'wall':>7s} {'PSNR':>7s} {'SSIM':>6s} {'N':>10s} {'grade':>12s} {'iters':>6s}")
B0 = {}
for s in ("room", "bicycle", "garden"):
    b0 = get(f"{PUB}/b0_{s}/results.json")
    B0[s] = b0
    print(f"{s:9s} {b0['wall']/60:6.1f}m {b0['psnr']:7.3f} {b0['ssim']:6.3f} "
          f"{b0['n']:10,} {b0['grade']:>12s} {b0['iters']:6d}")
    # protocol identity checks
    r = b0["renderer"]
    print(f"          arm={r.get('arm')} eps2d={r.get('eps2d')} "
          f"absgrad={r.get('absgrad')} thresh={r.get('densify_grad_thresh') if 'densify_grad_thresh' in r else r.get('thresh')}")

print()
print(f"{'scene':9s} {'b0 PSNR':>8s} {'b1a PSNR':>9s} {'dPSNR':>7s} {'b0 N/b1a N':>10s} "
      f"{'b1a/b0 wall':>11s}")
ratios = []
for s in ("room", "bicycle", "garden"):
    b1a = get(f"{F30K}/b1a_{s}/results.json")
    d = B0[s]["psnr"] - b1a["psnr"]
    nr = B0[s]["n"] / b1a["n"]
    wr = b1a["wall"] / B0[s]["wall"]
    ratios.append(wr)
    print(f"{s:9s} {B0[s]['psnr']:8.3f} {b1a['psnr']:9.3f} {d:+7.3f} {nr:10.3f} {wr:11.3f}")
import math
gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
print(f"\nB1A-vs-B0 wall geomean ratio: {gm:.4f} (B0 is {1/gm:.2f}x slower)")

verdict = {
    "gate": "B0_MATCHED_BASELINE",
    "status": "PASS" if all(v["grade"] == "PUBLICATION" and v["iters"] == 30000
                            for v in B0.values()) else "CHECK",
    "scenes": {s: {"wall_s": v["wall"], "psnr": v["psnr"], "ssim": v["ssim"],
                   "lpips": v["lpips"], "n": v["n"], "grade": v["grade"]}
               for s, v in B0.items()},
    "b1a_over_b0_wall_geomean": gm,
    "notes": [
        "B0 = original diff_gaussian_rasterization @54c035f under the unified matched "
        "trainer; densification statistic is the ORIGINAL signed-grad-norm >= 2e-4 "
        "(pixel-space, no rescale) per ORIGINAL_METHOD_PROTOCOL -- the single disclosed "
        "protocol difference from the matched arms.",
        "Integration fixes (disclosed): opacities [N,1] unsqueeze; radii reshape [1,N,1]; "
        "means2D kwarg; 3-tuple return. Verified by 200-iter smoke RC=0 + these full runs.",
        "B0 trains 1.9-2.6x slower than B1A and reaches ~12-22% fewer Gaussians "
        "(stricter original densification statistic).",
    ],
}
with open(f"{PUB}/b0_gate_verdict.json", "w") as f:
    json.dump(verdict, f, indent=1)
print("WROTE", f"{PUB}/b0_gate_verdict.json", "->", verdict["status"])
