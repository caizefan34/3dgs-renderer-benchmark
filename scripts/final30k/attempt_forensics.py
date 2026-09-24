import os, json, time

dirs = "/mnt/storage_pool/liaoyuanjun/final30k_runs"

for run in ("b1a_bicycle", "c0_counter", "b1a_counter", "b1a_bonsai", "b1a_drjohnson"):
    rd = f"{dirs}/{run}"
    rj = f"{rd}/results.json"
    if not os.path.exists(rj):
        print(f"{run}: NO results.json")
        # any files?
        files = sorted(os.listdir(rd)) if os.path.isdir(rd) else []
        print(f"   dir files: {files[:8]}")
        continue
    mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(rj)))
    t = json.load(open(f"{rd}/timing.json"))
    q = json.load(open(f"{rd}/quality.json"))
    print(f"{run}: results.json mtime={mt} wall={t['total_wall_s']:.1f}s "
          f"({t['total_wall_s']/60:.1f} min) psnr={q['psnr']:.3f}")

print()
# b1a_drjohnson log tail
p = f"{dirs}/log_b1a_drjohnson.txt"
if os.path.exists(p):
    lines = open(p, errors="replace").read().strip().splitlines()
    print(f"b1a_drjohnson log: {len(lines)} lines; last: {lines[-1][:120]}")
else:
    print("b1a_drjohnson log: MISSING")
