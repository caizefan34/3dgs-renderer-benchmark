import os

print("=== gsplat csrc structure ===")
for root, dirs, files in os.walk("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc"):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), "/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc")
        print(f"  {rel}")

print("\n=== gsplat top-level .py files ===")
for f in sorted(os.listdir("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/")):
    if f.endswith(".py"):
        sz = os.path.getsize(os.path.join("/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/", f))
        print(f"  {f} ({sz} bytes)")

print("\n=== include files ===")
inc = "/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/include"
for f in sorted(os.listdir(inc)):
    sz = os.path.getsize(os.path.join(inc, f))
    print(f"  {f} ({sz} bytes)")
