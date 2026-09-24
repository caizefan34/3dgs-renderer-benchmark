import re, glob, os

cands = glob.glob("/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_*train_benchmark.py")
P = cands[0]
src = open(P).read()
lines = src.splitlines()

print("=== create_higs / handle / HigsScene ===")
for i, l in enumerate(lines):
    if any(k in l for k in ["create_higs", "create_himsg", "HigsScene", "HimsgScene", "handle =", "handle=", "build_scene", "make_scene", "HigsRendererHandle", "RendererHandle"]):
        print(f"{i+1}: {l.strip()[:200]}")

print("\n=== imports from experimental ===")
for i, l in enumerate(lines):
    if "experimental" in l and ("import" in l or "from" in l):
        print(f"{i+1}: {l.strip()[:200]}")