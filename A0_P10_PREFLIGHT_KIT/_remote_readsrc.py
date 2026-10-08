import os
p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
lines = open(p).read().splitlines()
def show(a,b,tag):
    print(f"===== {tag} (lines {a}-{b}) =====")
    for i in range(a-1, min(b,len(lines))):
        print(f"{i+1:5d} {lines[i]}")
# native forward capture around the intersect_tile call (line ~2168)
show(2100, 2200, "_native_forward_capture: intersect_tile + surroundings")
