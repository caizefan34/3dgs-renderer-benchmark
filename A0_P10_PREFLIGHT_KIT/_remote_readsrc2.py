p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
lines = open(p).read().splitlines()
def show(a,b,tag):
    print(f"===== {tag} (lines {a}-{b}) =====")
    for i in range(a-1, min(b,len(lines))):
        print(f"{i+1:5d} {lines[i]}")
# _native_forward_capture signature + the means2d/radii/depths computation before intersect_tile
show(2033, 2100, "_native_forward_capture: signature + pre-intersect setup")
# the _higs_dynamic_forward (the function that calls _native_forward_capture)
show(2860, 2990, "_higs_dynamic_forward")
