import json

for run in ("c0_bonsai", "c0_bicycle"):
    d = f"/mnt/storage_pool/liaoyuanjun/final30k_runs/{run}"
    print(f"===== {run} =====")
    fs = json.load(open(f"{d}/final_status.json"))
    print("final_status:", {k: fs[k] for k in ("status", "arm", "scene", "iterations",
                                               "seed", "all_losses_finite", "contaminated", "timing_grade")})
    t = json.load(open(f"{d}/timing.json"))
    print("timing: total_wall_s=%.1f mean_iter_ms=%.2f median=%.2f" % (
        t["total_wall_s"], t["mean_iter_ms"], t["median_iter_ms"]))
    print("  top-level phases: fwd=%.2f bwd=%.2f loss=%.2f densify=%.2f opt=%.2f nested_fb=%.2f" % (
        t["forward_ms"], t["backward_ms"], t["loss_ms"],
        t.get("phase_ms", {}).get("densify", {}).get("mean_ms", -1),
        t["optimizer_ms"], t["nested_fb_ms"]))
    q = json.load(open(f"{d}/quality.json"))
    print("quality: psnr=%.3f ssim=%.4f lpips=%.4f n_eval_cameras=%s" % (
        q["psnr"], q["ssim"], q["lpips"], q.get("n_eval_cameras")))
    m = json.load(open(f"{d}/memory.json"))
    print("memory: peak_vram_gb=%.2f" % m["peak_vram_gb"])
    tr = json.load(open(f"{d}/training_results.json"))
    print("training_results: initial_N=%s final_N=%s clones=%s splits=%s prunes=%s resets=%s" % (
        tr["initial_N"], tr["final_N"], tr["total_clones"], tr["total_splits"],
        tr["total_prunes"], tr["total_opacity_resets"]))
    csv = open(f"{d}/training_curve.csv").read().strip().splitlines()
    print("curve rows:", len(csv) - 1, "| header:", csv[0])
    print("  first:", csv[1])
    print("  last:", csv[-1])
