import json
d = json.load(open("/root/mr0_out/room@10000_replacement.json"))
print("ckpt", d["ckpt_sha256"][:16], "N", d["N"], "n_groups", d["n_groups"])
print("baseline_psnr sample", {k: round(v,2) for k,v in list(d["baseline_psnr"].items())[:4]})
print("isect_red_sample", {k: round(v,3) for k,v in d["intersection_reduction_sample"].items()})
for K in ("4","8","16"):
    x = d["K"][K]["coverage75"]
    print(f"K={K}: dpsnr_tr={x['train_psnr_delta_median_db']:+.3f} dpsnr_hold={x['truehold_psnr_delta_median_db']:+.3f} "
          f"worst={x['truehold_psnr_delta_worst_db']:+.3f} ssim={x['ssim_delta_median']:+.4f} "
          f"cos_med={x['response_cos_median']:.3f} cos_p10={x['response_cos_p10']:.3f} "
          f"isect={x['intersection_reduction']:.3f} var={x['optimized_variable_reduction']:.3f} "
          f"speedup={x['predicted_iteration_speedup']:.3f} nan={x['nan_inf_count']} nrep={x['n_replace']}")
