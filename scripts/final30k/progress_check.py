import os, datetime

dirs = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
print("now:", datetime.datetime.now().strftime("%H:%M:%S"))
for f in ("log_b1a_bonsai.txt", "log_b1a_flowers.txt", "log_c0_flowers.txt",
          "log_b1a_garden.txt", "log_c0_garden.txt"):
    p = f"{dirs}/{f}"
    if os.path.exists(p):
        lines = open(p, errors="replace").read().strip().splitlines()
        print(f"{f}: {lines[-1][:100] if lines else '(empty)'}")
    else:
        print(f"{f}: MISSING")
