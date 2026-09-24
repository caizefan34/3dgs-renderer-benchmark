import json

paths = {
    "b1_garden (pub)": "/mnt/storage_pool/liaoyuanjun/pub_runs/b1_garden/results.json",
    "b1a_garden (FINAL-30K)": "/mnt/storage_pool/liaoyuanjun/final30k_runs/b1a_garden/results.json",
}
for name, path in paths.items():
    d = json.load(open(path))
    fe = d["final_eval"]
    print(f"== {name} ==")
    print("  grid evals:", {row["step"]: round(row["psnr"], 3) for row in d["eval_rows"]})
    print(f"  final_eval: psnr={fe['psnr']:.3f} ssim={fe['ssim']:.4f} "
          f"n_cam={fe.get('n_eval_cameras')} N={fe['n_gaussians']}")
    # eval protocol of grid rows
    if d["eval_rows"]:
        r = d["eval_rows"][-1]
        print(f"  last grid row keys: {sorted(r.keys())}")
        print(f"  last grid row: {r}")
    print()
