import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr)

# 1. actual last lines of the failed run log (unclipped)
print("== b1ae03_room log last 15 lines (raw) ==")
print(run("tail -15 /mnt/storage_pool/liaoyuanjun/pub_runs/log_b1ae03_room.txt"))
print("== a0_room (rc=0) log last 8 lines ==")
print(run("tail -8 /mnt/storage_pool/liaoyuanjun/pub_runs/log_a0_room.txt"))
