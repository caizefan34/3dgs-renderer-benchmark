import json, os

d = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
pairs = {}
for run in ("b1a_bonsai", "b1a_flowers", "c0_flowers", "b1a_garden", "c0_garden"):
    t = json.load(open(f"{d}/{run}/timing.json"))
    q = json.load(open(f"{d}/{run}/quality.json"))
    fs = json.load(open(f"{d}/{run}/final_status.json"))
    tr = json.load(open(f"{d}/{run}/training_results.json"))
    arm, scene = run.split("_", 1)
    pairs.setdefault(scene, {})[arm] = (t, q, fs, tr)
    print("%-12s wall=%7.1fs (%5.1f min) psnr=%.3f ssim=%.4f lpips=%.4f N=%9d contaminated=%s" % (
        run, t["total_wall_s"], t["total_wall_s"] / 60, q["psnr"], q["ssim"], q["lpips"],
        tr["final_N"], fs.get("contaminated", False)))

print()
for scene, arms in pairs.items():
    if "b1a" in arms and "c0" in arms:
        (tb, qb, _, trb), (tc, qc, _, trc) = arms["b1a"], arms["c0"]
        print("PAIR %-9s speedup=%.4fx reduction=%+.2f%% dPSNR=%+.3f dSSIM=%+.4f dLPIPS=%+.4f N_ratio=%.3f" % (
            scene, tb["total_wall_s"] / tc["total_wall_s"],
            (1 - tc["total_wall_s"] / tb["total_wall_s"]) * 100,
            qc["psnr"] - qb["psnr"], qc["ssim"] - qb["ssim"],
            qc["lpips"] - qb["lpips"], trc["final_N"] / trb["final_N"]))

st = json.load(open(f"{d}/scheduler_state.json"))
print("\nactive:", {g: (i["arm"], i["scene"]) for g, i in st["active"].items()})
print("done:", len(st["done"]), "queue:", 26 - len(st["done"]) - len(st["active"]))
