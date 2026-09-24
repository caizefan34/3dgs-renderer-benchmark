"""Extract benchmark _std_ll_forward + view shaping, discovering paths dynamically."""
import os, subprocess, re

cand_root = "/mnt/storage_pool"
envdirs = []
for d in os.listdir(cand_root):
    full = os.path.join(cand_root, d)
    if d.endswith("-env") and os.path.isdir(os.path.join(full, "bin")):
        envdirs.append(full)
print("ENVDIRS:", envdirs)

PY = None
for e in envdirs:
    for cand in ("python", "python3"):
        p = os.path.join(e, "bin", cand)
        if os.path.exists(p):
            PY = p
            break
    if PY:
        break
print("PY:", PY)
if not PY:
    raise SystemExit(1)

out = subprocess.check_output(
    [PY, "-c", "import gsplat, os; print(os.path.dirname(gsplat.__file__))"],
    text=True).strip()
GSPLAT_ROOT = out
print("GSPLAT_ROOT =", GSPLAT_ROOT)

wrapper = os.path.join(GSPLAT_ROOT, "cuda", "_wrapper.py")
if not os.path.exists(wrapper):
    print("NO wrapper at", wrapper)
    raise SystemExit(1)

lines = open(wrapper, errors="replace").read().splitlines()
idx = next((i for i, l in enumerate(lines) if "def fully_fused_projection" in l), None)
if idx is not None:
    print(f"\n=== fully_fused_projection @L{idx+1} ===")
    for j in range(idx, min(idx + 120, len(lines))):
        print(f"{j+1}: {lines[j]}")
else:
    print("\nNo def fully_fused_projection found! Listing defs:")
    for i, l in enumerate(lines):
        if l.startswith("def "):
            print(f"  L{i+1}: {l.strip()[:130]}")
