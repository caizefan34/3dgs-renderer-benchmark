import json, os

d = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
# check the five newly completed runs
for run in ("b1a_kitchen", "c0_kitchen", "b1a_playroom", "c0_playroom", "b1a_room"):
    rj = f"{d}/{run}/results.json"
    if not os.path.exists(rj):
        print(f"{run}: NO results.json yet")
        continue
    t = json.load(open(f"{d}/{run}/timing.json"))
    q = json.load(open(f"{d}/{run}/quality.json"))
    tr = json.load(open(f"{d}/{run}/training_results.json"))
    print("%-12s wall=%7.1fs psnr=%.3f ssim=%.4f lpips=%.4f N=%9d" % (
        run, t["total_wall_s"], q["psnr"], q["ssim"], q["lpips"], tr["final_N"]))
