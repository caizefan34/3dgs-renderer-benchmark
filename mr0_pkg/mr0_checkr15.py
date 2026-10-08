import json
d = json.load(open("/root/mr0_out/room@15000_replacement.json"))
print("room@15000 N", d["N"], "ckpt", d["ckpt_sha256"][:12])
print("isect_red", {k: round(v,3) for k,v in d["intersection_reduction_sample"].items()})
for K in ("4","8","16"):
    x = d["K"][K]["coverage75"]
    print(f"K={K}: dpsnr_tr={x['train_psnr_delta_median_db']:+.3f} hold={x['truehold_psnr_delta_median_db']:+.3f} "
          f"worst={x['truehold_psnr_delta_worst_db']:+.3f} ssim={x['ssim_delta_median']:+.4f} "
          f"cos={x['response_cos_median']:.3f}/{x['response_cos_p10']:.3f} isect={x['intersection_reduction']:.3f} "
          f"spd={x['predicted_iteration_speedup']:.3f}")
