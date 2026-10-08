p = "/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/functional/gaussian_inference.py"
lines = open(p).read().splitlines()
def show(a,b,tag):
    print(f"===== {tag} (lines {a}-{b}) =====")
    for i in range(a-1, min(b,len(lines))):
        print(f"{i+1:5d} {lines[i]}")
show(1621, 1720, "forward: signature + entry")
show(1920, 2033, "forward: capture call + ctx saves + return")
