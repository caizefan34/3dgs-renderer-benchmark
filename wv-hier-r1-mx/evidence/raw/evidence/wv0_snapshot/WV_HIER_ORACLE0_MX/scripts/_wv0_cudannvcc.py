import glob, os, subprocess
# conda cuda 12.4 nvcc
cands = glob.glob("/home/liaoyuanjun/miniforge3/envs/anysplat/bin/nvcc")
cands += glob.glob("/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/nvcc")
cands += glob.glob("/home/liaoyuanjun/miniforge3/envs/anysplat/**/nvcc", recursive=True)
for c in cands:
    if os.path.exists(c):
        r=subprocess.run([c,"--version"],capture_output=True,text=True)
        print(c, "->", r.stdout.strip().split("\n")[-1] if r.stdout else r.stderr[:80])
# also check the runtime/cudart
print("=== nvidia libs in env ===")
for p in glob.glob("/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages/nvidia/*/lib*/libcudart*"):
    print(p)
