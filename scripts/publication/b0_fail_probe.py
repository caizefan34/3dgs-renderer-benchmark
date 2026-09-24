import os

for rid in ("b0_room", "b0_bicycle", "b0_garden"):
    p = f"/mnt/storage_pool/liaoyuanjun/pub_runs/log_{rid}.txt"
    if not os.path.exists(p):
        print(f"== {rid}: NO LOG")
        continue
    lines = open(p, errors="replace").read().strip().splitlines()
    print(f"== {rid} ({len(lines)} lines; last 12) ==")
    for ln in lines[-12:]:
        print("  ", ln[:170])
    print()
