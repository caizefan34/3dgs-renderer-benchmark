"""Resolve gsplat path at runtime, then print _wrapper.py around 'viewmats'."""
import gsplat, os

GSPLAT_ROOT = os.path.dirname(gsplat.__file__)
print("GSPLAT_ROOT =", GSPLAT_ROOT)

wrapper = os.path.join(GSPLAT_ROOT, "cuda", "_wrapper.py")
print("wrapper exists:", os.path.isfile(wrapper))
lines = open(wrapper, errors="replace").read().splitlines()

idx = next((i for i, l in enumerate(lines) if "def fully_fused_projection" in l), None)
if idx is not None:
    print(f"\n=== fully_fused_projection @L{idx+1} ===")
    for j in range(idx, min(idx + 100, len(lines))):
        print(f"{j+1}: {lines[j]}")
else:
    print("\nNo def fully_fused_projection found! Listing defs:")
    for i, l in enumerate(lines):
        if l.startswith("def "):
            print(f"  L{i+1}: {l.strip()}")

idx2 = next((i for i, l in enumerate(lines) if "def rasterize_gaussian_higs_frozen" in l), None)
if idx2 is not None:
    print(f"\n=== rasterize_gaussian_higs_frozen @L{idx2+1} ===")
    for j in range(idx2, min(idx2 + 80, len(lines))):
        print(f"{j+1}: {lines[j]}")
