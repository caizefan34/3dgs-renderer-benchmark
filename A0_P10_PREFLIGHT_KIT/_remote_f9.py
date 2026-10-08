p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
lines = open(p).read().splitlines()
print("=== f9 / HIGS_DISABLE_F9 / gatherless references ===")
for i,l in enumerate(lines,1):
    if any(k in l for k in ("f9_enabled","HIGS_DISABLE_F9","HIGS_F9","gatherless","_f9","f9 ")):
        print(f"{i:5d} {l.rstrip()[:135]}")
