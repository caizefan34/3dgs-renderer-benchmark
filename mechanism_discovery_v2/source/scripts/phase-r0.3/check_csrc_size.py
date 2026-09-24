import os

csrc = "/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc"
# Only .cu, .cuh, .h, .cpp files NOT in third_party
total = 0
for root, dirs, files in os.walk(csrc):
    # Skip third_party
    if "third_party" in root:
        continue
    rel = os.path.relpath(root, csrc)
    for f in files:
        if any(f.endswith(x) for x in [".cu", ".cuh", ".h", ".cpp", ".hpp"]):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total += sz
            if rel == ".":
                print(f"  {f}  ({sz} bytes)")
            else:
                print(f"  {rel}/{f}  ({sz} bytes)")

print(f"\nTotal CUDA source size (no third_party): {total} bytes ({total/1024:.1f} KB)")

# Also check how big each dir is
print("\n=== Directory sizes (excluding third_party) ===")
for root, dirs, files in os.walk(csrc):
    if "third_party" in root:
        continue
    rel = os.path.relpath(root, csrc)
    dir_sz = sum(os.path.getsize(os.path.join(root, f)) for f in files if not any(f.endswith(x) for x in [".pyc"]))
    if dir_sz > 0:
        print(f"  {rel if rel != '.' else '.'}: {dir_sz/1024:.1f} KB")
