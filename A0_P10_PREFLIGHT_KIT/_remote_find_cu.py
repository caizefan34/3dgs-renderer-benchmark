import os, hashlib, subprocess
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
base = "/root/epic05_frozen/higs_c0_final30k_worktree"
print("--- csrc dir listing ---")
for p in [base + "/gsplat/cuda/csrc"]:
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)):
            if f.endswith((".cu", ".cuh")):
                print(f"  {f:42s} {sha(os.path.join(p,f))[:24]}  {os.path.getsize(os.path.join(p,f))}")
    else:
        print("MISSING dir", p)
print("--- find HigsNativeBackward.cu anywhere under worktree ---")
r = subprocess.run(["find", base, "-name", "HigsNativeBackward.cu", "-type", "f"],
                   capture_output=True, text=True)
for line in r.stdout.strip().splitlines():
    print("  FOUND", sha(line)[:24], line)
print("--- find RasterizeToPixels3DGSDevice.cuh anywhere ---")
r2 = subprocess.run(["find", base, "-name", "RasterizeToPixels3DGSDevice.cuh", "-type", "f"],
                    capture_output=True, text=True)
for line in r2.stdout.strip().splitlines():
    print("  FOUND", sha(line)[:24], line)
