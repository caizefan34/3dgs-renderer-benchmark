import json

d = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
# First complete matched pair: bicycle
for run in ("b1a_bicycle", "c0_bicycle"):
    t = json.load(open(f"{d}/{run}/timing.json"))
    q = json.load(open(f"{d}/{run}/quality.json"))
    fs = json.load(open(f"{d}/{run}/final_status.json"))
    tr = json.load(open(f"{d}/{run}/training_results.json"))
    print("%-12s wall=%.1fs (%.1f min) mean_iter=%.2fms psnr=%.3f ssim=%.4f lpips=%.4f N=%s contaminated=%s" % (
        run, t["total_wall_s"], t["total_wall_s"] / 60, t["mean_iter_ms"],
        q["psnr"], q["ssim"], q["lpips"], tr["final_N"], fs.get("contaminated")))
tb = json.load(open(f"{d}/b1a_bicycle/timing.json"))["total_wall_s"]
tc = json.load(open(f"{d}/c0_bicycle/timing.json"))["total_wall_s"]
qb = json.load(open(f"{d}/b1a_bicycle/quality.json"))
qc = json.load(open(f"{d}/c0_bicycle/quality.json"))
print("\nPAIR bicycle: speedup = %.4fx  reduction = %.2f%%  dPSNR = %+.3f dB  dSSIM = %+.4f  dLPIPS = %+.4f" % (
    tb / tc, (1 - tc / tb) * 100, qc["psnr"] - qb["psnr"], qc["ssim"] - qb["ssim"],
    qc["lpips"] - qb["lpips"]))
# also show which runs are done so far
import os
print("\nruns with results.json:",
      sorted(x for x in os.listdir(d) if os.path.exists(f"{d}/{x}/results.json")))
