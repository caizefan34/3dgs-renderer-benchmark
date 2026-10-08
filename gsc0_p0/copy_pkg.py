import os, shutil, subprocess
TREE = "/root/epic05_frozen/gsplat-true-accutile-v153"
DEST = "/root/gsc0_p0/gsplat_pkg"
# size
total = 0
for r, d, f in os.walk(TREE):
    for fn in f:
        total += os.path.getsize(os.path.join(r, fn))
print(f"tree size: {total/1e6:.1f} MB", flush=True)
if os.path.exists(DEST):
    shutil.rmtree(DEST)
shutil.copytree(TREE, DEST)
# remove stale prebuilt csrc.so so `from gsplat import csrc` fails cleanly -> JIT path
csrc = os.path.join(DEST, "gsplat", "csrc.so")
if os.path.exists(csrc):
    os.remove(csrc); print("removed stale csrc.so", flush=True)
# replace _backend.py with patched version
shutil.copy("/root/gsc0_p0/_backend_patched.py", os.path.join(DEST, "gsplat", "cuda", "_backend.py"))
print("patched _backend.py in place", flush=True)
# verify no csrc.so remains
print("csrc.so gone?", not os.path.exists(os.path.join(DEST,"gsplat","csrc.so")), flush=True)
print("backend flags line fixed?", '"+ ["-use_fast_math", "--extended-lambda"]' in open(os.path.join(DEST,"gsplat","cuda","_backend.py")).read(), flush=True)