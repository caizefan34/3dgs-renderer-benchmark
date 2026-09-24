import os
import subprocess

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"

def du(path):
    try:
        out = subprocess.check_output(["du", "-sh", path], text=True).split()[0]
        return out
    except Exception:
        return "?"

print("== per-run dir sizes (pub_runs) ==")
for d in sorted(os.listdir(PUB))[:12]:
    p = os.path.join(PUB, d)
    if os.path.isdir(p):
        print(f"  {d:24s} {du(p)}")

print("\n== what's inside a completed run dir ==")
d = os.path.join(PUB, "b0_room")
for root, dirs, files in os.walk(d):
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        if sz > 50 * 1024 * 1024:
            print(f"  {os.path.relpath(fp, d):50s} {sz/1e6:8.0f} MB")

print("\n== biggest consumers of the pool (top level) ==")
out = subprocess.check_output(
    ["bash", "-c",
     "du -sh /mnt/storage_pool/liaoyuanjun/* 2>/dev/null | sort -rh | head -15"],
    text=True)
print(out)
