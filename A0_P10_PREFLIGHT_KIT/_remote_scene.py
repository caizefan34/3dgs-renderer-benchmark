import re
p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
src = open(p).read()
lines = src.splitlines()
print("=== occurrences of _HIGS_DYNAMIC_SCENE ===")
for i,l in enumerate(lines,1):
    if "_HIGS_DYNAMIC_SCENE" in l:
        print(f"{i:5d} {l.rstrip()[:150]}")
print("\n=== defs of scene/capture/state functions ===")
for i,l in enumerate(lines,1):
    if re.match(r"\s*(def|class)\s", l) and any(k in l for k in ("capture","scene","forward","isect","state","def rasterize","def _higs")):
        print(f"{i:5d} {l.rstrip()[:120]}")
