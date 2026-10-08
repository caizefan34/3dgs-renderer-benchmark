p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
lines = open(p).read().splitlines()
def show(a,b,tag):
    print(f"===== {tag} (lines {a}-{b}) =====")
    for i in range(a-1, min(b,len(lines))):
        print(f"{i:5d} {lines[i]}")
show(1740, 1815, "f9_enabled + F9 branch entry")
show(2184, 2280, "_native_forward_capture: post-intersect + render + captured return")
